"""The tiled MM kernel and the coalesced column min-max equal the one-column-per-block path, bit for bit.

``binned_mm_mi_from_values_gpu`` runs the tiled kernel by default; the other route is forced by making the launcher refuse. Inputs cover duplicate edges
(a column with a shorter valid prefix ``ne_k``), NaN values, a column count that the group size does not divide, and float32 X.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi as M


def _y_stats(y: np.ndarray, ky: int) -> tuple:
    """The target's plug-in entropy and occupied-class count, which the MM correction takes as arguments."""
    p = np.bincount(y, minlength=ky) / len(y)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()), int((p > 0).sum())


def _run(X, edges, y, nbins, ky, monkeypatch, tiled: bool):
    """The MM MI for one route: the tiled launcher as is, or forced to refuse so the per-column kernel runs."""
    if not tiled:
        monkeypatch.setattr(M, "_launch_mm_nek_tile", lambda *a, **k: False)
    h_y, k_y = _y_stats(cp.asnumpy(y), ky)
    return cp.asnumpy(M.binned_mm_mi_from_values_gpu(X, edges, y, nbins, ky, h_y, k_y, return_device=True))


def _case(rng, n, K, nbins, ky, dtype, dup_cols=()):
    """Device inputs with NaN rows, sorted edges (optionally duplicated in ``dup_cols``) and a target that depends on X."""
    X = rng.standard_normal((n, K)).astype(dtype)
    if n > 40:
        X[7, 0] = np.nan
    edges = np.sort(rng.standard_normal((nbins - 1, K)), axis=0)
    for c in dup_cols:
        edges[:, c] = edges[0, c]  # every interior edge equal: dedup leaves one valid edge for this column
    # y depends on the first and last columns, so each column's MI is positive (independent y would give all-zero MI and no sensitivity)
    y = ((X[:, 0] > 0).astype(np.int64) + 2 * (X[:, -1] > 0)) % ky
    return cp.asarray(X), cp.asarray(edges), cp.asarray(y)


@pytest.mark.parametrize("K", [1, 3, 4, 9, 37])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tiled_equals_per_column_kernel(K, dtype, monkeypatch):
    """Both routes agree exactly, and the MI is positive somewhere so a wrong column cannot hide behind zeros."""
    rng = np.random.default_rng(K)
    X, edges, y = _case(rng, 3000, K, nbins=10, ky=4, dtype=dtype, dup_cols=(0,) if K > 1 else ())
    tiled = _run(X, edges, y, 10, 4, monkeypatch, tiled=True)
    plain = _run(X, edges, y, 10, 4, monkeypatch, tiled=False)
    assert float(tiled.max()) > 0.0, "all-zero MI would hide a wrong column"
    np.testing.assert_array_equal(tiled, plain)


def test_column_min_max_equals_cupy_including_nan():
    """The coalesced per-column bounds equal cupy min and max, float64 and float32, with a NaN in one column."""
    rng = np.random.default_rng(3)
    X = cp.asarray(rng.standard_normal((5000, 13)))
    X[17, 3] = np.nan
    cmin, cmax = M._column_min_max(X, 5000, 13, False)
    assert cp.array_equal(cmin, X.min(axis=0), equal_nan=True)
    assert cp.array_equal(cmax, X.max(axis=0), equal_nan=True)
    X32 = X.astype(cp.float32)
    cmin32, cmax32 = M._column_min_max(X32, 5000, 13, True)
    assert cp.array_equal(cmin32, X32.min(axis=0).astype(cp.float64), equal_nan=True)
    assert cp.array_equal(cmax32, X32.max(axis=0).astype(cp.float64), equal_nan=True)
