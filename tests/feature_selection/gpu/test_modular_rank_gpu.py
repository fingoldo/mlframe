"""The GPU ranking of the modular-scan pairs and triples matches the njit kernels, and the pair/triple selection does not depend on the backend."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _pairwise_modular_fe as pm
from mlframe.feature_selection.filters import _pairwise_modular_rank_gpu as rg

_MODS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 23, 24, 32], dtype=np.int64)


def _needs_gpu():
    """Skip when no usable CUDA device is visible to the project."""
    if not rg._gpu_available():
        pytest.skip("no usable CUDA device")


def _frame(p: int, n: int = 3000, seed: int = 1):
    """Integer columns with a planted (x0 + x5) mod 3 target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.integers(0, 30, size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = ((X["x0"].to_numpy() + X["x5"].to_numpy()) % 3).astype(int)
    return X, y


def test_gpu_pair_scores_match_the_cpu_kernel():
    """Pair scores from the CUDA kernel equal the njit kernel's up to float summation order, and the top pairs agree."""
    _needs_gpu()
    X, y = _frame(40)
    cols = list(X.columns)
    arrs = {c: np.asarray(X[c]) for c in cols}
    mat, ysub = pm._rank_subsample(cols, arrs, pm.encode_y_for_classif_mi(y), 0)
    pairs = np.array(list(itertools.combinations(range(len(cols)), 2)), dtype=np.int64)
    ia, ib = pairs[:, 0].copy(), pairs[:, 1].copy()
    n_cls = int(ysub.max()) + 1
    cpu = pm._residue_rank_scores(mat, ia, ib, _MODS, ysub, n_cls)
    gpu = rg.residue_rank_scores_gpu(mat, ia, ib, None, _MODS, ysub, n_cls)
    assert gpu is not None
    assert float(np.max(np.abs(cpu - gpu))) < 1e-10
    assert set(np.argsort(-cpu, kind="stable")[:10]) == set(np.argsort(-gpu, kind="stable")[:10])


def test_gpu_triple_scores_match_the_cpu_kernel():
    """Triple scores from the CUDA kernel equal the njit kernel's up to float summation order."""
    _needs_gpu()
    X, y = _frame(16)
    cols = list(X.columns)
    arrs = {c: np.asarray(X[c]) for c in cols}
    mat, ysub = pm._rank_subsample(cols, arrs, pm.encode_y_for_classif_mi(y), 0)
    tr = np.array(list(itertools.combinations(range(len(cols)), 3)), dtype=np.int64)
    ia, ib, ic = tr[:, 0].copy(), tr[:, 1].copy(), tr[:, 2].copy()
    n_cls = int(ysub.max()) + 1
    cpu = pm._residue_rank_scores3(mat, ia, ib, ic, _MODS, ysub, n_cls)
    gpu = rg.residue_rank_scores_gpu(mat, ia, ib, ic, _MODS, ysub, n_cls)
    assert gpu is not None
    assert float(np.max(np.abs(cpu - gpu))) < 1e-10


def test_selection_is_the_same_on_both_backends(monkeypatch):
    """The selected pairs and triples are the same whichever backend ranks them, and the planted pair is among them."""
    _needs_gpu()
    X, y = _frame(60)
    cols = list(X.columns)
    arrs = {c: np.asarray(X[c]) for c in cols}
    yi = pm.encode_y_for_classif_mi(y)
    picks = {}
    for backend in ("cpu", "gpu"):
        monkeypatch.setattr(rg, "rank_backend_choice", lambda n_work, _b=backend: _b)
        picks[backend] = (
            pm._select_pairs_by_residue_score(cols, arrs, yi, pm.COARSE_MODULI, 0, 24),
            pm._select_triples_by_residue_score(cols[:20], arrs, yi, pm.COARSE_MODULI, 0, 50),
        )
    assert picks["cpu"] == picks["gpu"]
    assert ("x0", "x5") in picks["gpu"][0]


def test_backend_heuristic_switches_at_the_measured_crossover():
    """Below the measured crossover the heuristic routes to the CPU kernels, above it to the GPU."""
    assert rg._rank_fallback_choice(100) == "cpu"
    assert rg._rank_fallback_choice(250_000) == "gpu"
