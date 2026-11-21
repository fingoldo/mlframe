"""Regression tests for the audit findings in this module."""
import pytest
"""Audit 2026-11-20 finding 39-F54: an orthogonal initialiser applied to a BatchNorm model does not crash on the 1-D gamma."""

import torch
from torch import nn

pytest.importorskip("lightning")
from mlframe.training.neural._flat_generate_helpers import _generate_mlp_weights_init_fcn


def test_orthogonal_init_on_a_batchnorm_layer_sets_gamma_to_one_and_keeps_linear_orthogonal():
    """orthogonal_ refuses the 1-D BatchNorm gamma; that layer gets the identity 1.0 while the Linear weight is still initialised."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    _generate_mlp_weights_init_fcn(nn.init.orthogonal_, model)
    bn = model[1]
    assert torch.allclose(bn.weight.detach(), torch.ones(4))
    lin = model[0].weight.detach()
    assert torch.allclose(lin @ lin.T, torch.eye(4), atol=1e-5)
