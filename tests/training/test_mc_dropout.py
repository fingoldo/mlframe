"""Unit tests for MC-dropout predictive spread (`_mc_dropout.py`, Workstream B1)."""

from __future__ import annotations

import numpy as np
import pytest


def test_mc_dropout_spread_positive_with_dropout_zero_without():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.standard_normal((50, 8)).astype("float32"))

    from mlframe.training._mc_dropout import mc_dropout_predict

    with_drop = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(16, 1))
    mean, std, n_drop = mc_dropout_predict(with_drop, X, n=32)
    assert n_drop == 1
    assert mean.shape == (50, 1)
    assert float(std.mean()) > 0.0  # dropout -> stochastic passes -> positive spread

    no_drop = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    _mean2, std2, n_drop2 = mc_dropout_predict(no_drop, X, n=8)
    assert n_drop2 == 0
    assert float(std2.max()) < 1e-5  # no dropout -> passes identical up to float32 noise -> ~zero spread
    assert float(std.mean()) > 100 * float(std2.mean() + 1e-12)  # dropout spread dwarfs the no-dropout floor


def test_mc_dropout_restores_module_mode():
    torch = pytest.importorskip("torch")
    from mlframe.training._mc_dropout import mc_dropout_predict

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.3))
    m.train()  # leave it in train mode
    X = torch.zeros((3, 4))
    mc_dropout_predict(m, X, n=4)
    assert m.training is True  # original mode restored
    m.eval()
    mc_dropout_predict(m, X, n=4)
    assert m.training is False


def test_predictive_entropy_peaks_for_uniform_min_for_confident():
    from mlframe.training._mc_dropout import predictive_entropy

    uniform = np.full((1, 4), 0.25)
    confident = np.array([[0.97, 0.01, 0.01, 0.01]])
    h_uniform = predictive_entropy(uniform)[0]
    h_conf = predictive_entropy(confident)[0]
    assert h_uniform > h_conf
    assert abs(h_uniform - np.log(4)) < 1e-6  # uniform over 4 classes -> ln(4)
