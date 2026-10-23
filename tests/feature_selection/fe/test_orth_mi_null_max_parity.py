"""The per-permutation null maximum with the matrix binned once equals the repeated per-permutation MI calls."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_null_max import _null_max_streamed_path, _null_max_strict_gpu, _takes_cpu_rank_path, null_max_mi


def _frame(n: int = 4000, k: int = 12, seed: int = 0):
    """Continuous candidates with ties and a binary target that depends on two of them, plus a set of permutations."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)).astype(np.float32)
    X[:, 3] = np.round(X[:, 3] * 2)  # tie-heavy column
    y = ((X[:, 0] + X[:, 1] + rng.standard_normal(n)) > 0).astype(np.int64)
    perms = [rng.permutation(n) for _ in range(9)]
    return X, y, perms


def _repeated(X, y, perms, nbins):
    """Reference: one full ``_mi_classif_batch`` call per permutation."""
    return np.array([float(np.nanmax(np.asarray(_mi_classif_batch(X, y[p], nbins=nbins), dtype=np.float64))) for p in perms])


def test_cpu_rank_shortcut_is_bit_identical_to_repeated_calls(monkeypatch):
    """Default CPU configuration: same maxima, bit for bit."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_IMBALANCE_MI", raising=False)
    X, y, perms = _frame()
    assert _takes_cpu_rank_path(X, 10)  # the shortcut, not the fallback loop, produced the values
    np.testing.assert_array_equal(null_max_mi(X, y, perms, 10, _mi_classif_batch), _repeated(X, y, perms, 10))


def test_a_nan_column_takes_the_repeated_calls_and_matches(monkeypatch):
    """A partly non-finite matrix is outside the shortcut; the result still equals the reference."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    X, y, perms = _frame()
    X[5, 2] = np.nan
    assert not _takes_cpu_rank_path(X, 10)
    np.testing.assert_array_equal(null_max_mi(X, y, perms, 10, _mi_classif_batch), _repeated(X, y, perms, 10))


def test_strict_gpu_branch_equals_repeated_strict_calls(monkeypatch):
    """STRICT residency: one upload and one edge computation per matrix give the same maxima as the per-call STRICT route."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    from mlframe.feature_selection.filters._fe_gpu_strict import fe_gpu_strict_enabled

    X, y, perms = _frame()
    if not fe_gpu_strict_enabled(n=X.shape[0], p=X.shape[1]):
        pytest.skip("STRICT GPU mode is not available for this shape on this host")
    assert _null_max_strict_gpu(X, y, perms, 10) is not None  # the resident branch ran, it did not fall back to the loop
    np.testing.assert_allclose(null_max_mi(X, y, perms, 10, _mi_classif_batch), _repeated(X, y, perms, 10), rtol=0, atol=0)


def test_streamed_gpu_branch_equals_repeated_streamed_calls(monkeypatch):
    """A matrix routed through the streamed GPU path (several narrow blocks): one upload and one edge computation per block give the repeated calls' maxima."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as be
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as sg

    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_IMBALANCE_MI", raising=False)
    monkeypatch.setattr(be, "_mi_path_for", lambda n, p, dtype, rank_binning: "streamed")
    real_plan = sg.plan_streamed_mi

    def narrow(n_rows, devices, host_free, dtype):
        """The real plan with the block width cut to five columns."""
        plan = real_plan(n_rows, devices, host_free, dtype)
        if plan["width"] <= 0:
            return plan
        width = min(plan["width"], 5)
        return {**plan, "width": width, "block_bytes": width * n_rows * int(np.dtype(dtype).itemsize)}

    monkeypatch.setattr(sg, "plan_streamed_mi", narrow)
    X, y, perms = _frame()
    fast = _null_max_streamed_path(X, y, perms, 10)
    if fast is None:
        pytest.skip("the streamed GPU path is not usable on this host")
    np.testing.assert_allclose(fast, _repeated(X, y, perms, 10), rtol=0, atol=0)
    np.testing.assert_allclose(null_max_mi(X, y, perms, 10, _mi_classif_batch), fast, rtol=0, atol=0)


def test_a_substituted_scorer_is_honoured_not_shortcut(monkeypatch):
    """A caller's own scorer (a test double) is called once per permutation; the shortcuts only stand in for the genuine batch MI."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    X, y, perms = _frame()
    seen = []

    def own(arr, y_arg, nbins=10):
        """Records the permuted labels it is given and returns a fixed MI vector."""
        seen.append(np.asarray(y_arg))
        return np.linspace(0.0, 1.0, arr.shape[1])

    out = null_max_mi(X, y, perms, 10, own)
    assert len(seen) == len(perms) and np.all(out == 1.0)
