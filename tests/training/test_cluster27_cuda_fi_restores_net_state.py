"""Cluster 27 regression: the CUDA permutation-importance path leaves the caller's network in its original mode and device (audit 2026-11-20, finding 27-F26)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mlframe.training import _feature_importances as fi


def test_failed_cuda_setup_leaves_net_training_mode_and_device():
    """Without a usable CUDA setup the function falls back and the caller's net keeps its training flag and device (27-F26)."""
    net = torch.nn.Linear(3, 1)
    net.train()
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.random.default_rng(1).normal(size=40)
    fi._cuda_batched_permutation_importance(net, X, y)
    assert net.training is True
    assert next(net.parameters()).device.type == "cpu"
