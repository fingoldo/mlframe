"""Audit 2026-11-20 finding 39-F67: the L1 norm-parameter exclusion is not stored on the module, so a pickle cannot carry stale ids."""

import pickle

import torch
from torch import nn

from mlframe.training.neural._flat_torch_module import MLPTorchModel


def _model_with_norm_and_l1():
    """A small MLP with BatchNorm and an active L1 penalty, so the exclusion path runs in training_step."""
    network = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1))
    return MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=network, l1_alpha=0.01)


def test_training_step_does_not_store_the_id_set_on_the_module():
    """After a training step the module holds no exclusion id set, and its pickle does not contain one."""
    torch.manual_seed(0)
    model = _model_with_norm_and_l1()
    batch = (torch.randn(8, 4), torch.randn(8, 1))
    model.training_step(batch, 0)
    assert "_l1_excluded_param_ids" not in model.__dict__
    assert b"_l1_excluded_param_ids" not in pickle.dumps(model.__getstate__())
