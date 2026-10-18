"""DA3 checkpoint loading without overwritten large-weight random fills.

Only the calling thread's large Linear/Conv weight initialization is skipped.
Biases, tokens, normalizations and buffers retain their native initialization.
The CPU model is never returned unless checkpoint coverage and loading succeed.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)
MIN_SKIPPED_WEIGHT_NUMEL = 65_536
_INIT_LOCK = threading.Lock()
_SAFETENSOR_DTYPES = {
    'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16,
    'BF16': torch.bfloat16, 'I64': torch.int64, 'I32': torch.int32,
    'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool,
}


@contextmanager
def skip_large_weight_initialization(min_numel: int = MIN_SKIPPED_WEIGHT_NUMEL):
    """Temporarily bypass only large Linear/Conv weight Kaiming fills.

    This is intended for the isolated DA3 loader process. Patched reset methods
    preserve other threads' initialization and all original bias initialization.
    Returned records contain identities/scalars, never tensor references.
    """
    if min_numel < 1:
        raise ValueError('min_numel must be positive')
    skipped: dict[int, dict[str, Any]] = {}
    owner = threading.get_ident()
    with _INIT_LOCK:
        original_linear = nn.Linear.reset_parameters
        original_conv = nn.modules.conv._ConvNd.reset_parameters
        original_kaiming = nn.init.kaiming_uniform_

        def wrap_reset(original):
            def reset(module):
                weight = module.weight
                if threading.get_ident() != owner or weight.numel() < min_numel:
                    return original(module)

                def selective_kaiming(tensor, *args, **kwargs):
                    if threading.get_ident() == owner and tensor is weight:
                        skipped[id(tensor)] = dict(shape=list(tensor.shape), numel=tensor.numel(),
                                                   bytes=tensor.numel()*tensor.element_size(),
                                                   module_type=type(module).__name__)
                        return tensor
                    return original_kaiming(tensor, *args, **kwargs)

                nn.init.kaiming_uniform_ = selective_kaiming
                try:
                    return original(module)
                finally:
                    nn.init.kaiming_uniform_ = original_kaiming
            return reset

        nn.Linear.reset_parameters = wrap_reset(original_linear)
        nn.modules.conv._ConvNd.reset_parameters = wrap_reset(original_conv)
        try:
            yield skipped
        finally:
            nn.init.kaiming_uniform_ = original_kaiming
            nn.Linear.reset_parameters = original_linear
            nn.modules.conv._ConvNd.reset_parameters = original_conv


def audit_checkpoint(model: nn.Module, checkpoint: str | Path,
                     skipped: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Read metadata; require physical tensor coverage, resolving identity aliases.

    Reused modules can expose several state keys for the very same Parameter.
    Safetensors stores one of those names; resolve only identical tensor objects,
    never infer coverage from a shared pointer or an overlapping strided view.
    Nonpersistent buffers retain their native constructor values.
    """
    from safetensors import safe_open

    state = model.state_dict(keep_vars=True)
    with safe_open(str(checkpoint), framework='pt', device='cpu') as stream:
        checkpoint_keys = set(stream.keys())
        raw_missing = sorted(set(state) - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - set(state))
        mismatched = []
        for name in sorted(checkpoint_keys & set(state)):
            metadata = stream.get_slice(name)
            shape, dtype = metadata.get_shape(), metadata.get_dtype()
            if list(state[name].shape) != shape or state[name].dtype != _SAFETENSOR_DTYPES.get(dtype):
                mismatched.append(dict(name=name, model_shape=list(state[name].shape), checkpoint_shape=shape,
                                       model_dtype=str(state[name].dtype), checkpoint_dtype=dtype))
    checkpoint_names_by_identity: dict[int, str] = {}
    for name in sorted(checkpoint_keys & set(state)):
        checkpoint_names_by_identity.setdefault(id(state[name]), name)
    covered_aliases = {name: checkpoint_names_by_identity[id(state[name])]
                       for name in raw_missing if id(state[name]) in checkpoint_names_by_identity}
    missing = [name for name in raw_missing if name not in covered_aliases]
    if missing or unexpected or mismatched:
        raise ValueError(f'DA3 checkpoint coverage failed: missing={missing}, unexpected={unexpected}, mismatched={mismatched}; only identical tensor-object aliases are supported, not distinct storage/stride views')
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    skipped = skipped or {}
    unknown = [identity for identity in skipped if identity not in names_by_id]
    if unknown:
        raise ValueError('Skipped weights are absent from the completed model')
    skipped_names = sorted(names_by_id[identity] for identity in skipped)
    if any(name not in checkpoint_keys and name not in covered_aliases for name in skipped_names):
        raise ValueError('Skipped parameters are not completely covered by checkpoint')
    nonpersistent = []
    persistent_buffer_ids = set()
    nonpersistent_buffer_ids = set()
    for prefix, module in model.named_modules():
        persistent_buffer_ids.update(id(tensor) for name, tensor in module._buffers.items()
                                     if tensor is not None and name not in module._non_persistent_buffers_set)
        for name in sorted(module._non_persistent_buffers_set):
            tensor = module._buffers.get(name)
            if tensor is not None:
                nonpersistent_buffer_ids.add(id(tensor))
                nonpersistent.append(dict(name=(prefix + '.' if prefix else '') + name,
                                          shape=list(tensor.shape), dtype=str(tensor.dtype),
                                          bytes=tensor.numel()*tensor.element_size()))
    return dict(checkpoint=str(checkpoint), state_keys=len(state), checkpoint_keys=len(checkpoint_keys),
                parameter_tensors=len(list(model.parameters())),
                physical_state_tensors=len({id(tensor) for tensor in state.values()}),
                physical_persistent_buffers=len(persistent_buffer_ids),
                physical_nonpersistent_buffers=len(nonpersistent_buffer_ids),
                nonpersistent_buffers=nonpersistent, raw_missing=raw_missing,
                covered_aliases=covered_aliases, uncovered_missing=missing,
                missing=[], unexpected=[], mismatched=[], skipped_parameter_names=skipped_names,
                skipped_parameter_tensors=len(skipped), skipped_numel=sum(row['numel'] for row in skipped.values()),
                skipped_bytes=sum(row['bytes'] for row in skipped.values()), min_skipped_numel=MIN_SKIPPED_WEIGHT_NUMEL)


def load_da3_model(model_path: str):
    """Use native HF configuration/loading, auditing before any weight copy.

    CPU construction only; normal runner/standby owns the later device move.
    Both incomplete coverage and loading errors fail without returning a model.
    """
    from depth_anything_3.api import DepthAnything3

    started = time.perf_counter()
    report: dict[str, Any] = {}
    with skip_large_weight_initialization() as skipped:
        class CoveredDA3(DepthAnything3):
            @classmethod
            def _load_as_safetensor(cls, model, model_file, map_location, strict):
                report.update(audit_checkpoint(model, model_file, skipped))
                # No assign/meta path: retain native safetensors copy semantics.
                return super()._load_as_safetensor(model, model_file, map_location, True)

        model = CoveredDA3.from_pretrained(model_path, strict=True)
    if not report:
        raise RuntimeError('DA3 loading did not pass the safetensors coverage audit')
    report['cpu_construct_audit_load_s'] = time.perf_counter() - started
    model._da3_loading_report = report
    logger.info('DA3 checkpoint coverage: state=%d checkpoint=%d skipped_weights=%d skipped_numel=%d skipped_bytes=%d nonpersistent_buffers=%d CPU_construct_audit_load=%.3fs raw_missing=%d covered_aliases=%d uncovered_missing=%d',
                report['state_keys'], report['checkpoint_keys'], report['skipped_parameter_tensors'],
                report['skipped_numel'], report['skipped_bytes'], len(report['nonpersistent_buffers']),
                report['cpu_construct_audit_load_s'], len(report['raw_missing']),
                len(report['covered_aliases']), len(report['uncovered_missing']))
    if report['covered_aliases']:
        logger.info('DA3 checkpoint state-key aliases (same tensor object): %s', report['covered_aliases'])
    return model
