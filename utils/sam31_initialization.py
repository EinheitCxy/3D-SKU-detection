"""Selective PVS weight initialization, gated by checkpoint coverage.

Only large Linear/Conv weights in the constructing thread skip their native
Kaiming fill and subsequent weight trunc-normal fill. All other initialization, including CUDA position caches,
runs normally. The caller must audit and strictly load before using the model.
"""

from contextlib import contextmanager
import threading
from typing import Any, Iterator, Mapping

import torch
from torch import nn


# The native tracker deep-copies its 256x2048 transformer/fuser templates.
# Keep those templates' initialization: skipped identities must survive into
# the completed model. The vision backbone's large matrices exceed this size.
MIN_SKIPPED_WEIGHT_NUMEL = 1_048_576
_INITIALIZATION_LOCK = threading.Lock()


@contextmanager
def skip_large_weight_initialization(
    min_numel: int = MIN_SKIPPED_WEIGHT_NUMEL,
    *, random_initializer_modules: tuple[Any, ...] = (),
) -> Iterator[dict[int, nn.Parameter]]:
    """Skip only a reset's large weight fill; preserve other threads' resets."""
    if min_numel < 1:
        raise ValueError("min_numel must be positive")
    owner = threading.get_ident()
    skipped: dict[int, nn.Parameter] = {}
    active_weights: list[nn.Parameter] = []
    with _INITIALIZATION_LOCK:
        original_linear = nn.Linear.reset_parameters
        original_conv = nn.modules.conv._ConvNd.reset_parameters
        initializer_originals = []

        def selective_fill(original):
            def fill(tensor, *args, **kwargs):
                is_active_weight = active_weights and tensor is active_weights[-1]
                is_recorded_weight = skipped.get(id(tensor)) is tensor
                if threading.get_ident() == owner and (is_active_weight or is_recorded_weight):
                    skipped[id(tensor)] = tensor
                    return tensor
                return original(tensor, *args, **kwargs)

            return fill

        def wrap_reset(original):
            def reset(module):
                if threading.get_ident() != owner or module.weight.numel() < min_numel:
                    return original(module)
                active_weights.append(module.weight)
                try:
                    return original(module)
                finally:
                    active_weights.pop()

            return reset

        try:
            nn.Linear.reset_parameters = wrap_reset(original_linear)
            nn.modules.conv._ConvNd.reset_parameters = wrap_reset(original_conv)
            targets = [(nn.init, "kaiming_uniform_"), (nn.init, "trunc_normal_")]
            targets.extend((module, "trunc_normal_") for module in random_initializer_modules)
            for module, name in targets:
                original = getattr(module, name)
                initializer_originals.append((module, name, original))
                setattr(module, name, selective_fill(original))
            yield skipped
        finally:
            for module, name, original in reversed(initializer_originals):
                setattr(module, name, original)
            nn.modules.conv._ConvNd.reset_parameters = original_conv
            nn.Linear.reset_parameters = original_linear


def audit_skipped_weights(
    model: nn.Module,
    weights: Mapping[str, torch.Tensor],
    skipped: Mapping[int, nn.Parameter],
) -> dict[str, Any]:
    """Require each skipped Parameter's identity and compatible checkpoint key.

    Aliases are names for the exact same Parameter, not storage-sharing views.
    Native strict loading still checks the complete model state afterwards.
    Ordinary dict caches and nonpersistent buffers are never skipped or loaded.
    """
    state = model.state_dict(keep_vars=True)
    names_by_identity: dict[int, list[str]] = {}
    for name, tensor in state.items():
        names_by_identity.setdefault(id(tensor), []).append(name)
    rows = []
    for identity, parameter in skipped.items():
        names = names_by_identity.get(identity, [])
        if not names or any(state[name] is not parameter for name in names):
            raise ValueError("PVS skipped weight is absent from the completed model")
        covered = [name for name in names if name in weights]
        if not covered:
            raise ValueError(f"PVS skipped weight has no checkpoint coverage: {names}")
        for name in covered:
            checkpoint_tensor = weights[name]
            if checkpoint_tensor.shape != parameter.shape or checkpoint_tensor.dtype != parameter.dtype:
                raise ValueError(
                    f"PVS skipped weight checkpoint shape/dtype mismatch: {name}: "
                    f"model={tuple(parameter.shape)}/{parameter.dtype}, "
                    f"checkpoint={tuple(checkpoint_tensor.shape)}/{checkpoint_tensor.dtype}"
                )
        rows.append({
            "names": names,
            "checkpoint_names": covered,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "bytes": parameter.numel() * parameter.element_size(),
        })
    nonpersistent = [
        {"name": f"{prefix}.{name}" if prefix else name,
         "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for prefix, module in model.named_modules()
        for name, tensor in module._buffers.items()
        if tensor is not None and name in module._non_persistent_buffers_set
    ]
    return {
        "skipped_parameter_tensors": len(rows),
        "min_skipped_weight_numel": MIN_SKIPPED_WEIGHT_NUMEL,
        "skipped_numel": sum(row["numel"] for row in rows),
        "skipped_bytes": sum(row["bytes"] for row in rows),
        "skipped_weights": rows,
        "nonpersistent_buffers": nonpersistent,
    }
