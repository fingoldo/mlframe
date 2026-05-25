"""Wave 10e monolith-split sensor for ``mlframe.training.neural.flat``."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training.neural import flat
    return flat


@pytest.fixture(scope="module")
def sibling():
    from mlframe.training.neural import _flat_torch_module
    return _flat_torch_module


def test_mlp_torch_model_identity(parent_module, sibling):
    assert parent_module.MLPTorchModel is sibling.MLPTorchModel


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= 600, f"flat.py facade is {n_lines} LOC, expected <= 600"


def test_smoke_generate_mlp_returns_module(parent_module):
    import torch.nn as nn
    model = parent_module.generate_mlp(num_features=4, num_classes=1, nlayers=2)
    assert isinstance(model, nn.Module)


def test_arch_enum_present(parent_module):
    e = parent_module.MLPNeuronsByLayerArchitecture
    assert e.Constant.name == "Constant"
    assert e.Declining.name == "Declining"
