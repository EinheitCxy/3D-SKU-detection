"""PVS selective initialization must retain all state outside loaded weights."""

import threading
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from utils.sam31_initialization import audit_skipped_weights, skip_large_weight_initialization


class TinyPVS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)
        self.alias = self.linear
        self.conv = torch.nn.Conv2d(1024, 1024, 1)
        self.small = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)
        self.token = torch.nn.Parameter(torch.ones(1, 2))
        self.register_buffer("nonpersistent", torch.tensor([0.25]), persistent=False)
        self.position_cache = {(2, 2): torch.arange(4).reshape(2, 2)}


def test_checkpoint_load_retains_aliases_buffers_tokens_and_outputs(monkeypatch):
    baseline = TinyPVS()
    checkpoint = baseline.state_dict()
    fills = []
    kaiming = torch.nn.init.kaiming_uniform_

    def record_fill(tensor, *args, **kwargs):
        fills.append(tuple(tensor.shape))
        return kaiming(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", record_fill)
    with skip_large_weight_initialization() as skipped:
        candidate = TinyPVS()
    assert fills == [(2, 2)]
    report = audit_skipped_weights(candidate, checkpoint, skipped)
    assert report["skipped_parameter_tensors"] == 2
    linear = next(row for row in report["skipped_weights"] if "linear.weight" in row["names"])
    assert linear["names"] == ["linear.weight", "alias.weight"]
    assert torch.isfinite(candidate.linear.bias).all()
    assert torch.equal(candidate.norm.weight, baseline.norm.weight)
    assert torch.equal(candidate.token, baseline.token)
    assert torch.equal(candidate.nonpersistent, baseline.nonpersistent)
    assert torch.equal(candidate.position_cache[(2, 2)], baseline.position_cache[(2, 2)])
    candidate.load_state_dict(checkpoint, strict=True)
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, checkpoint[name])
    x = torch.ones(1, 1024)
    assert torch.equal(candidate.linear(x), baseline.linear(x))
    assert candidate.alias.weight is candidate.linear.weight


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "replaced"])
def test_candidate_rejects_uncovered_or_replaced_parameter(fault):
    with skip_large_weight_initialization() as skipped:
        model = TinyPVS()
    checkpoint = dict(model.state_dict())
    if fault == "missing":
        del checkpoint["linear.weight"], checkpoint["alias.weight"]
    elif fault == "shape":
        checkpoint["linear.weight"] = torch.zeros(1)
    elif fault == "dtype":
        checkpoint["linear.weight"] = checkpoint["linear.weight"].double()
    else:
        model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
    with pytest.raises(ValueError, match="PVS skipped weight"):
        audit_skipped_weights(model, checkpoint, skipped)


def test_patch_preserves_other_threads_and_restores_after_failure(monkeypatch):
    fills = []
    kaiming = torch.nn.init.kaiming_uniform_
    resets = (torch.nn.Linear.reset_parameters, torch.nn.modules.conv._ConvNd.reset_parameters)

    def record_fill(tensor, *args, **kwargs):
        fills.append(threading.get_ident())
        return kaiming(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", record_fill)
    other_models = []
    owner = threading.get_ident()
    with pytest.raises(RuntimeError, match="constructor failure"):
        with skip_large_weight_initialization(min_numel=65_536) as skipped:
            local = torch.nn.Linear(256, 256)
            worker = threading.Thread(target=lambda: other_models.append(torch.nn.Linear(256, 256)))
            worker.start()
            worker.join()
            assert set(skipped) == {id(local.weight)}
            raise RuntimeError("constructor failure")
    assert len(other_models) == 1
    assert len(fills) == 1 and fills[0] != owner
    assert torch.count_nonzero(other_models[0].weight) > 0
    assert torch.nn.init.kaiming_uniform_ is record_fill
    assert (torch.nn.Linear.reset_parameters, torch.nn.modules.conv._ConvNd.reset_parameters) == resets


def test_pvs_loader_defaults_to_audited_skip_and_retains_explicit_native(tmp_path, monkeypatch):
    from utils import sam31_pvs

    class CPUTracker(TinyPVS):
        def to(self, device):
            assert device == "cuda:0"
            return self

    expected = TinyPVS().state_dict()
    checkpoint = tmp_path / "pvs.pt"
    torch.save({"model": {f"tracker.model.{name}": value for name, value in expected.items()}}, checkpoint)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setitem(sys.modules, "sam3.model_builder", SimpleNamespace(
        build_sam3_multiplex_video_model=lambda **_kwargs: CPUTracker(),
    ))
    monkeypatch.setitem(sys.modules, "sam3.model.vitdet", SimpleNamespace(trunc_normal_=torch.nn.init.trunc_normal_))
    native, native_keys = sam31_pvs.load_model(checkpoint, "cuda:0", skip_random_init=False)
    candidate, candidate_keys = sam31_pvs.load_model(checkpoint, "cuda:0")
    assert native._pvs_loading_report["skip_random_init"] is False
    assert native._pvs_loading_report["skipped_parameter_tensors"] == 0
    assert candidate._pvs_loading_report["skip_random_init"] is True
    assert candidate._pvs_loading_report["skipped_parameter_tensors"] == 2
    assert native_keys == candidate_keys == len(expected)
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, native.state_dict()[name])


def test_vit_reinitialization_skips_only_recorded_weights_preserves_token(monkeypatch):
    calls = []
    native_trunc = torch.nn.init.trunc_normal_

    def record_trunc(tensor, *args, **kwargs):
        calls.append(id(tensor))
        return native_trunc(tensor, *args, **kwargs)

    vitdet = SimpleNamespace(trunc_normal_=record_trunc)
    with skip_large_weight_initialization(random_initializer_modules=(vitdet,)) as skipped:
        model = TinyPVS()
        vitdet.trunc_normal_(model.linear.weight)
        vitdet.trunc_normal_(model.token)
        torch.nn.init.trunc_normal_(model.conv.weight)
    assert calls == [id(model.token)]
    assert set(skipped) == {id(model.linear.weight), id(model.conv.weight)}
    assert vitdet.trunc_normal_ is record_trunc
    assert torch.nn.init.trunc_normal_ is native_trunc
