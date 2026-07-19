"""Regression: ``_mi_classif_batch`` must accept a genuinely 1-D resident cupy candidate column.

``_mi()`` (``_pairwise_modular_fe.py``) documents that a device-born caller may hand it an ALREADY-RESIDENT
1-D cupy candidate (e.g. the row-argmax scorer's ``cp.argmax(...)`` index column, or the conditional-gate
scorer) and that ``_mi_classif_batch`` reshapes 1-D inputs to a single-column batch. That reshape only actually
ran INSIDE the GPU-STRICT-enabled branch (``if Xd.ndim == 1: Xd = Xd[:, None]``); with GPU-STRICT off (the
default), a 1-D array fell through unreshaped to ``_mi_classif_batch_numba``, whose ``_n, p = X.shape`` assumes
2-D and raised ``ValueError: not enough values to unpack (expected 2, got 1)``.

Surfaced by profile_fuzz_chains.py on a binary_classification combo (200k rows, xgb, cats=15): MRMR's row-argmax
FE logged "row-argmax FE raised ValueError: not enough values to unpack (expected 2, got 1); continuing without
row-argmax columns" -- a caught, non-fatal degradation (the row-argmax features were silently skipped for the
whole fit), not a process crash, but a real correctness/coverage bug.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is available (used to skip the module when it is not)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def test_mi_classif_batch_accepts_1d_cupy_column_gpu_strict_off(monkeypatch):
    """A 1-D resident cupy column must not raise, regardless of the GPU-STRICT flag."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch

    rng = np.random.default_rng(0)
    n = 500
    col_host = rng.integers(0, 3, size=n).astype(np.float64)
    y_host = rng.integers(0, 2, size=n).astype(np.int64)
    col_gpu = cp.asarray(col_host)

    mis = _mi_classif_batch(col_gpu, y_host, nbins=12)
    assert mis.shape == (1,)
    assert np.isfinite(mis[0])

    mis_host = _mi_classif_batch(col_host.reshape(-1, 1), y_host, nbins=12)
    assert mis[0] == pytest.approx(mis_host[0], abs=1e-9)


def test_row_argmax_mi_call_with_1d_resident_cupy_feature_does_not_raise(monkeypatch):
    """The exact call shape row-argmax scoring uses: ``_mi(1d_cupy_feature, y, nbins)``."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    from mlframe.feature_selection.filters._pairwise_modular_fe import _mi

    rng = np.random.default_rng(1)
    n = 500
    feat_host = rng.integers(0, 3, size=n).astype(np.float64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    feat_gpu = cp.asarray(feat_host)

    result = _mi(feat_gpu, y, nbins=12)
    assert isinstance(result, float)
    assert np.isfinite(result)
