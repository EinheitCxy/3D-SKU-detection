"""Isolated PromptDA skip-init experiment; import performs no model/GPU work.

Use load_promptda_model() for a checkpoint-audited candidate. The lower-level
context manager alone does NOT make skipped allocations safe to consume: call
its audit_checkpoint() before native strict load, then finish() after loading.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import weakref

_MIN_NUMEL = 65_536
_LOCK = threading.Lock()


class InitializationAudit:
    def __init__(self):
        self.weights = {}
        self.metadata = None
        self.loaded = False

    def record(self, tensor, module, operation):
        identity = id(tensor)
        row = self.weights.setdefault(identity, {
            "ref": weakref.ref(tensor), "shape": list(tensor.shape),
            "dtype": str(tensor.dtype), "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
            "module_type": type(module).__name__, "operations": [],
        })
        if row["ref"]() is not tensor:
            raise RuntimeError("Skipped parameter identity reused")
        row["operations"].append(operation)

    def audit_checkpoint(self, model, state):
        """Validate actual .ckpt state already decoded by the official loader."""
        expected = model.state_dict(keep_vars=True)
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = [name for name in set(expected) & set(state)
                      if expected[name].shape != state[name].shape
                      or expected[name].dtype != state[name].dtype]
        if missing or unexpected or mismatched:
            raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, shape_or_dtype={mismatched}")
        names = {}
        for name, tensor in expected.items():
            names.setdefault(id(tensor), []).append(name)
        rows = []
        for identity, row in self.weights.items():
            tensor = row["ref"]()
            if tensor is None or identity not in names:
                raise ValueError("Skipped parameter is absent from completed model")
            aliases = names[identity]
            if not all(name in state for name in aliases):
                raise ValueError(f"Uncovered skipped parameter: {aliases}")
            rows.append({k: v for k, v in row.items() if k != "ref"} | {
                "name": aliases[0], "state_names": aliases,
                "checkpoint_covered": True,
                "checkpoint_shape": list(state[aliases[0]].shape),
                "checkpoint_dtype": str(state[aliases[0]].dtype),
            })
        self.metadata = {
            "state_keys": len(expected), "checkpoint_keys": len(state),
            "checkpoint_format": "torch .ckpt state_dict; official key[9:] mapping",
            "min_numel": _MIN_NUMEL, "skipped_weights": rows,
            "skipped_numel": sum(row["numel"] for row in rows),
            "skipped_bytes": sum(row["bytes"] for row in rows),
        }
        return self.metadata

    def finish(self):
        if self.metadata is None or not self.loaded:
            raise RuntimeError("Skipped model did not pass checkpoint audit and strict loading")
        return self.metadata


@contextmanager
def skip_promptda_initialization():
    """Skip large real Linear/Conv weight Kaiming and DINO trunc-normal fills.

    Leaves biases, tokens, norms, buffers and other threads' initialization intact.
    Uses PyTorch's underlying trunc-normal helper because DINO binds the public
    trunc_normal_ symbol at import time. Intended only for isolated loader runs;
    global patches are restored on exceptions. Do not nest this context.
    """
    import torch
    from torch import nn

    owner = threading.get_ident()
    audit = InitializationAudit()
    with _LOCK:
        linear_reset = nn.Linear.reset_parameters
        conv_reset = nn.modules.conv._ConvNd.reset_parameters
        kaiming = nn.init.kaiming_uniform_
        trunc_normal = nn.init._no_grad_trunc_normal_
        active_weights = {}

        def wrap_reset(original):
            def reset(module):
                weight = module.weight
                if threading.get_ident() != owner or weight.numel() < _MIN_NUMEL:
                    return original(module)
                active_weights[id(weight)] = (weakref.ref(weight), weakref.ref(module))
                return original(module)
            return reset

        def target(tensor):
            if threading.get_ident() != owner:
                return None
            entry = active_weights.get(id(tensor))
            if entry is not None and entry[0]() is tensor:
                return entry[1]()
            return None

        def selective_kaiming(tensor, *args, **kwargs):
            module = target(tensor)
            if module is not None:
                audit.record(tensor, module, "kaiming_uniform_")
                return tensor
            return kaiming(tensor, *args, **kwargs)

        def selective_trunc_normal(tensor, *args, **kwargs):
            module = target(tensor)
            if module is not None:
                audit.record(tensor, module, "trunc_normal_")
                return tensor
            return trunc_normal(tensor, *args, **kwargs)

        nn.Linear.reset_parameters = wrap_reset(linear_reset)
        nn.modules.conv._ConvNd.reset_parameters = wrap_reset(conv_reset)
        nn.init.kaiming_uniform_ = selective_kaiming
        nn.init._no_grad_trunc_normal_ = selective_trunc_normal
        try:
            yield audit
        finally:
            nn.Linear.reset_parameters = linear_reset
            nn.modules.conv._ConvNd.reset_parameters = conv_reset
            nn.init.kaiming_uniform_ = kaiming
            nn.init._no_grad_trunc_normal_ = trunc_normal


def load_promptda_model(repo, checkpoint, *, skip=True):
    """Official constructor/checkpoint reader, with strict pre-copy coverage audit.

    No mmap, assign, dtype conversion, safetensors or alternative checkpoint path.
    Returns CPU eval model and scalar/metadata report. Timing includes audit cost.
    """
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from promptda.promptda import PromptDA

    if not skip:
        started = time.perf_counter()
        model = PromptDA(encoder="vitl", ckpt_path=str(checkpoint)).eval()
        return model, {"construct_load_s": time.perf_counter() - started}

    with skip_promptda_initialization() as audit:
        class AuditedPromptDA(PromptDA):
            def load_state_dict(self, state_dict, strict=True, assign=False):
                if not strict or assign:
                    raise ValueError("Probe requires native strict copy loading")
                audit_started = time.perf_counter()
                audit.audit_checkpoint(self, state_dict)
                self._probe_audit_s = time.perf_counter() - audit_started
                result = super().load_state_dict(state_dict, strict=True, assign=False)
                audit.loaded = True
                return result

        started = time.perf_counter()
        model = AuditedPromptDA(encoder="vitl", ckpt_path=str(checkpoint)).eval()
        elapsed = time.perf_counter() - started
        metadata = audit.finish()
    metadata.update(construct_load_s=elapsed, coverage_audit_s=model._probe_audit_s)
    return model, metadata


def compare_models(native, candidate):
    """Direct tensor equality; all parameters and persistent/nonpersistent buffers."""
    import torch

    groups = {
        "state": (native.state_dict(), candidate.state_dict()),
        "buffers": (dict(native.named_buffers(remove_duplicate=False)),
                    dict(candidate.named_buffers(remove_duplicate=False))),
    }
    result = {}
    for group, (left, right) in groups.items():
        if left.keys() != right.keys():
            raise AssertionError(f"{group} keys differ")
        for name, tensor in left.items():
            other = right[name]
            if tensor.shape != other.shape or tensor.dtype != other.dtype or not torch.equal(tensor, other):
                raise AssertionError(f"{group} differs: {name}")
        result[group + "_equal_count"] = len(left)
    result["nonpersistent_buffers"] = [
        (prefix + "." if prefix else "") + name
        for prefix, module in native.named_modules()
        for name in sorted(module._non_persistent_buffers_set)
        if module._buffers.get(name) is not None
    ]
    return result


def main():
    import argparse
    import gc
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output)
    report = {"runs": [], "status": "running", "timing_scope":
              "CPU construction + checkpoint read + strict copy; skip includes coverage audit; comparisons excluded"}
    try:
        started = time.perf_counter()
        import torch
        report["torch_import_s"] = time.perf_counter() - started
        sys.path.insert(0, str(repo))
        started = time.perf_counter()
        from promptda.promptda import PromptDA  # noqa: F401
        report["promptda_import_s"] = time.perf_counter() - started
        # hubconf's _make_dinov2_model imports vision_transformer lazily inside
        # the factory. Executing hubconf alone does not warm DINO/xformers.
        hubdir = repo / "torchhub/facebookresearch_dinov2_main"
        sys.path.insert(0, str(hubdir))
        started = time.perf_counter()
        spec = importlib.util.spec_from_file_location("_probe_hubconf", hubdir / "hubconf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hubconf_import_s = time.perf_counter() - started
        dependency_started = time.perf_counter()
        vision_transformer = importlib.import_module("vision_transformer")
        report["hubconf_import_s"] = hubconf_import_s
        report["vision_transformer_dependency_import_s"] = time.perf_counter() - dependency_started
        report["vision_transformer_module_file"] = str(vision_transformer.__file__)
        report["hub_dependency_import_s"] = time.perf_counter() - started
        sys.path.remove(str(hubdir))
        report["torch_version"] = torch.__version__
        report["cpu_threads"] = torch.get_num_threads()
        for repeat, order in enumerate(((False, True), (True, False)), start=1):
            models = {}
            for skip in order:
                # Same starting seed; skipped RNG consumption can change retained
                # random tensors. Final direct comparison deliberately detects this.
                torch.manual_seed(20261020)
                model, metadata = load_promptda_model(repo, args.checkpoint, skip=skip)
                label = "skip" if skip else "native"
                models[label] = model
                report["runs"].append({"repeat": repeat, "mode": label, **metadata})
            comparison_started = time.perf_counter()
            compared = compare_models(models["native"], models["skip"])
            report.setdefault("comparisons", []).append({"repeat": repeat, **compared,
                "comparison_s": time.perf_counter() - comparison_started})
            del model, models
            gc.collect()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
