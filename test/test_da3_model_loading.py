"""Focused CPU contracts for selective DA3 initialization and checkpoint coverage."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file, save_model, load_model as load_safetensors
from huggingface_hub import PyTorchModelHubMixin

_PATH = Path(__file__).resolve().parents[1] / 'src/da3_model_loading.py'
_SPEC = importlib.util.spec_from_file_location('tested_da3_model_loading', _PATH)
loading = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loading)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.large = torch.nn.Linear(256, 256)
        self.conv = torch.nn.Conv2d(256, 256, 1)
        self.small = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)
        self.token = torch.nn.Parameter(torch.zeros(1, 2))
        self.register_buffer('nonpersistent', torch.tensor([1., .25]), persistent=False)


def test_selective_skip_and_restore_preserves_other_initialization(monkeypatch):
    called = []
    original = torch.nn.init.kaiming_uniform_
    def recording(tensor, *args, **kwargs):
        called.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.nn.init, 'kaiming_uniform_', recording)
    resets = (torch.nn.Linear.reset_parameters, torch.nn.modules.conv._ConvNd.reset_parameters)
    with loading.skip_large_weight_initialization() as skipped:
        model = TinyModel()
    assert len(skipped) == 2
    assert called == [(2, 2)]
    assert torch.nn.init.kaiming_uniform_ is recording
    assert (torch.nn.Linear.reset_parameters, torch.nn.modules.conv._ConvNd.reset_parameters) == resets
    assert torch.equal(model.norm.weight, torch.ones(2))
    assert torch.equal(model.norm.bias, torch.zeros(2))
    assert torch.equal(model.token, torch.zeros(1, 2))
    assert torch.equal(model.nonpersistent, torch.tensor([1., .25]))
    assert torch.isfinite(model.large.bias).all()


def test_patch_restores_after_constructor_failure():
    original = (torch.nn.Linear.reset_parameters, torch.nn.modules.conv._ConvNd.reset_parameters,
                torch.nn.init.kaiming_uniform_)
    with pytest.raises(RuntimeError, match='construction failed'):
        with loading.skip_large_weight_initialization():
            torch.nn.Linear(256, 256)
            raise RuntimeError('construction failed')
    assert original == (torch.nn.Linear.reset_parameters, torch.nn.modules.conv._ConvNd.reset_parameters,
                        torch.nn.init.kaiming_uniform_)


def test_other_thread_retains_native_random_fill(monkeypatch):
    calls = []
    original = torch.nn.init.kaiming_uniform_
    def recording(tensor, *args, **kwargs):
        calls.append(threading.get_ident())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.nn.init, 'kaiming_uniform_', recording)
    owner = threading.get_ident()
    with loading.skip_large_weight_initialization() as skipped:
        local = torch.nn.Linear(256, 256)
        worker = threading.Thread(target=lambda: torch.nn.Linear(256, 256))
        worker.start()
        worker.join()
    assert set(skipped) == {id(local.weight)}
    assert len(calls) == 1 and calls[0] != owner


def test_real_safetensors_metadata_audit_and_loaded_equivalence(tmp_path):
    torch.manual_seed(42)
    baseline = TinyModel()
    checkpoint = tmp_path / 'model.safetensors'
    save_file(baseline.state_dict(), checkpoint)
    with loading.skip_large_weight_initialization() as skipped:
        candidate = TinyModel()
    report = loading.audit_checkpoint(candidate, checkpoint, skipped)
    assert report['skipped_parameter_names'] == ['conv.weight', 'large.weight']
    assert report['skipped_numel'] == 2 * 256 * 256
    assert report['state_keys'] == report['checkpoint_keys'] == len(baseline.state_dict())
    assert report['nonpersistent_buffers'][0]['name'] == 'nonpersistent'
    load_safetensors(candidate, checkpoint, strict=True)
    x = torch.tensor([[.25, -.5]])
    assert torch.equal(baseline.small(x), candidate.small(x))
    x = torch.zeros(1, 256)
    assert torch.equal(baseline.large(x), candidate.large(x))
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, baseline.state_dict()[name])
    assert torch.equal(baseline.nonpersistent, candidate.nonpersistent)


@pytest.mark.parametrize('fault', ['missing', 'unexpected', 'shape', 'dtype'])
def test_audit_rejects_incomplete_checkpoint(tmp_path, fault):
    model = TinyModel()
    state = model.state_dict()
    if fault == 'missing':
        del state['large.weight']
    elif fault == 'unexpected':
        state['unexpected'] = torch.zeros(1)
    elif fault == 'shape':
        state['large.weight'] = torch.zeros(1)
    else:
        state['large.weight'] = state['large.weight'].double()
    path = tmp_path / 'bad.safetensors'
    save_file(state, path)
    with pytest.raises(ValueError, match='coverage failed'):
        loading.audit_checkpoint(model, path)


def test_hf_subclass_resolves_config_and_audits_before_native_copy(monkeypatch, tmp_path):
    class FakeDA3(torch.nn.Module, PyTorchModelHubMixin):
        def __init__(self, width=256, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(width, width)
            self.register_buffer('buffer', torch.tensor([.25]), persistent=False)
    baseline = FakeDA3()
    baseline.save_pretrained(tmp_path)
    monkeypatch.setitem(sys.modules, 'depth_anything_3.api', SimpleNamespace(DepthAnything3=FakeDA3))
    model = loading.load_da3_model(str(tmp_path))
    assert model._da3_loading_report['skipped_parameter_names'] == ['linear.weight']
    assert torch.equal(model.linear.weight, baseline.linear.weight)
    assert torch.equal(model.buffer, baseline.buffer)
    save_file({'linear.bias': baseline.linear.bias}, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError, match='coverage failed'):
        loading.load_da3_model(str(tmp_path))


def test_shared_layernorm_state_aliases_are_covered_and_native_strict_loads(tmp_path):
    class SharedNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            norm = torch.nn.LayerNorm(32)
            # The same LayerNorm object is reused, as in DA3 DualDPT ln_seq.
            self.heads = torch.nn.ModuleList([torch.nn.Sequential(norm) for _ in range(4)])
            self.large = torch.nn.Linear(256, 256)
    baseline = SharedNorm()
    with torch.no_grad():
        baseline.heads[0][0].weight.fill_(2.)
        baseline.heads[0][0].bias.fill_(.5)
    path = tmp_path / 'shared.safetensors'
    save_model(baseline, path)
    with loading.skip_large_weight_initialization() as skipped:
        candidate = SharedNorm()
    report = loading.audit_checkpoint(candidate, path, skipped)
    assert len(report['raw_missing']) == len(report['covered_aliases']) == 6
    assert report['uncovered_missing'] == []
    assert report['state_keys'] == 10 and report['checkpoint_keys'] == 4
    assert report['physical_state_tensors'] == report['parameter_tensors'] == 4
    assert report['physical_persistent_buffers'] == report['physical_nonpersistent_buffers'] == 0
    assert report['skipped_parameter_names'] == ['large.weight']
    missing, unexpected = load_safetensors(candidate, path, strict=True)
    assert not missing and not unexpected
    assert candidate.heads[0][0] is candidate.heads[-1][0]
    x = torch.linspace(-1, 1, 32).reshape(1, 32)
    assert torch.equal(candidate.heads[-1](x), baseline.heads[-1](x))


def test_distinct_parameter_storage_views_do_not_imply_alias_coverage(tmp_path):
    class Views(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = torch.zeros(2, 3)
            self.left = torch.nn.Parameter(base)
            self.right = torch.nn.Parameter(base.T)
    model = Views()
    path = tmp_path / 'views.safetensors'
    save_file({'left': model.left}, path)
    with pytest.raises(ValueError, match='not distinct storage/stride views'):
        loading.audit_checkpoint(model, path)
