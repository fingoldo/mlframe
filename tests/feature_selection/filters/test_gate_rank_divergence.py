"""The gate rank walk equals the per-column loop of batched plug-in MIs it replaces, on the host and on the device."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gate_rank_divergence as gd
from tests.feature_selection.gpu._gpu_support import has_device


def _frame(n: int, p: int, ties: bool, seed: int = 0):
    """``(mat, y)``: float32 columns (optionally rounded to one decimal so values tie) and a 3-class target the first columns drive."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(n, p)).astype(np.float32)
    if ties:
        mat = np.round(mat, 1)
    y = ((mat[:, 0] > 0).astype(np.int64) + (mat[:, 1] > 0.3) + 2 * (mat[:, 2] < -0.5)) % 3
    return mat, y.astype(np.int64)


def _loop_reference(mat: np.ndarray, y: np.ndarray, probe_idx: list, nbins: int) -> np.ndarray:
    """The per-column loop of ``_rank_and_prune``: split the probes at the column median, rank-binned plug-in MI per half."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    probe = mat[:, probe_idx]
    out = np.zeros(mat.shape[1])
    for g in range(mat.shape[1]):
        cv = mat[:, g]
        hi = cv > np.median(cv)
        lo = ~hi
        if hi.sum() < nbins or lo.sum() < nbins:
            continue
        mh = np.asarray(_mi_classif_batch(np.ascontiguousarray(probe[hi]), y[hi], nbins=nbins, rank_binning=True))
        ml = np.asarray(_mi_classif_batch(np.ascontiguousarray(probe[lo]), y[lo], nbins=nbins, rank_binning=True))
        out[g] = np.abs(mh - ml).sum()
    return out


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("n,p", [(3000, 7), (6000, 33)])
def test_walk_equals_the_loop(n, p, ties):
    """Same bins, same tie rule, same plug-in: the divergences agree to round-off (a partial last chunk of gate columns included)."""
    mat, y = _frame(n, p, ties)
    got = gd.gate_divergence(mat, y, [0, 1, 2], 10)
    np.testing.assert_allclose(got, _loop_reference(mat, y, [0, 1, 2], 10), atol=1e-12)


def test_degenerate_split_scores_zero():
    """A near-constant gate column has fewer than ``nbins`` rows on one side of its median and scores 0, as the loop's ``continue`` leaves it."""
    mat, y = _frame(2000, 5, False)
    mat[:, 4] = 1.0
    mat[:3, 4] = 2.0  # median 1.0: three rows above it, fewer than nbins
    got = gd.gate_divergence(mat, y, [0, 1, 2], 10)
    assert got[4] == 0.0


def test_non_finite_matrix_is_left_to_the_loop():
    """The walk does not reproduce the partial-NaN handling: it declines (``None``) and the caller keeps the per-column path."""
    mat, y = _frame(1000, 4, False)
    mat[5, 3] = np.nan
    assert gd.gate_divergence(mat, y, [0, 1, 2], 10) is None


@pytest.mark.parametrize("ties", [False, True])
def test_device_walk_equals_the_host_walk(ties):
    """The cupy form (stable argsort, run starts, prefix counts, one bincount) gives the host walk's values."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no CUDA device")
    mat, y = _frame(4000, 21, ties)
    got = gd.gate_divergence_device(cp, cp.asarray(mat), cp.asarray(y), [0, 1, 2], 10)
    want = gd.gate_divergence(mat, y, [0, 1, 2], 10)
    assert got is not None and want is not None and np.shape(got) == np.shape(want)
    assert float(np.max(np.abs(np.asarray(got) - np.asarray(want)))) <= 1e-12


def test_rank_and_prune_ranking_is_the_loops(monkeypatch):
    """``_rank_and_prune`` returns the same gate and operand pools with the walk and with ``MLFRAME_GATE_RANK_LOOP=1``."""
    import pandas as pd

    from mlframe.feature_selection.filters import _conditional_gate_fe as g

    mat, y = _frame(4000, 16, False, seed=3)
    X = pd.DataFrame(mat, columns=[f"f{i}" for i in range(16)])
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    monkeypatch.delenv("MLFRAME_GATE_RANK_LOOP", raising=False)
    fast = g._rank_and_prune(X, list(X.columns), y, 10, 6, 6)
    monkeypatch.setenv("MLFRAME_GATE_RANK_LOOP", "1")
    loop = g._rank_and_prune(X, list(X.columns), y, 10, 6, 6)
    assert fast == loop


def test_device_choice_is_forced_by_the_environment(monkeypatch):
    """``MLFRAME_GATE_DIVERGENCE_DEVICE`` forces either side of the tuned host / device choice."""
    monkeypatch.setenv("MLFRAME_GATE_DIVERGENCE_DEVICE", "1")
    assert gd.gate_divergence_on_device(10) is True
    monkeypatch.setenv("MLFRAME_GATE_DIVERGENCE_DEVICE", "0")
    assert gd.gate_divergence_on_device(10**9) is False


def test_sweep_inputs_take_the_row_axis_and_the_two_walks_agree():
    """The sweep builds ``n_samples`` rows by ``_SWEEP_COLS`` columns, and its host and device callables agree to the sweep's tolerance."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no CUDA device")
    mat, y = gd._sweep_inputs({"n_samples": 3000})
    assert mat.shape == (3000, gd._SWEEP_COLS)
    np.testing.assert_allclose(gd._sweep_device(mat, y), gd._sweep_host(mat, y), rtol=1e-9, atol=1e-9)
