"""The one-sort ``_quantile_codes`` gives the codes of the two-pass original."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _orthogonal_triplet_fe as T


def _original(values, nbins):
    """The implementation this replaced: ``np.unique`` for the levels, ``np.quantile`` for the edges."""
    v = np.asarray(values, dtype=T._crit_np_dtype())
    finite = np.isfinite(v)
    codes = np.full(v.shape[0], nbins, dtype=np.int64)
    if finite.any():
        uniq = np.unique(v[finite])
        if uniq.size <= nbins:
            codes[finite] = np.searchsorted(uniq, v[finite])
        else:
            edges = np.unique(np.quantile(v[finite], np.linspace(0.0, 1.0, nbins + 1)[1:-1]))
            codes[finite] = np.searchsorted(edges, v[finite], side="right")
    return codes


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_codes_equal_the_original(dtype):
    """Continuous, tied, few-level, NaN-bearing, all-NaN and constant columns, at 3 and 4 bins."""
    rng = np.random.default_rng(3)
    cols = [rng.normal(size=5000), np.round(rng.normal(size=5000), 1), rng.integers(0, 3, 5000), rng.integers(0, 40, 5000), np.ones(500), np.full(300, np.nan)]
    cols.append(np.where(rng.random(5000) < 0.1, np.nan, rng.normal(size=5000)))
    cols.append(np.where(rng.random(2000) < 0.2, np.inf, rng.exponential(size=2000)))
    for col in cols:
        for nbins in (3, 4, 10):
            x = col.astype(dtype)
            np.testing.assert_array_equal(T._quantile_codes(x, nbins), _original(x, nbins))


def test_enumeration_is_the_same_with_the_column_pool(monkeypatch):
    """The ranked triples do not depend on the pool size."""
    import pandas as pd

    from mlframe.feature_selection.filters import _adaptive_nbins

    rng = np.random.default_rng(9)
    X = pd.DataFrame({f"x{i}": rng.normal(size=3000) for i in range(8)})
    y = (X["x1"] * X["x2"] > 0).astype(int).to_numpy()
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = T._enumerate_triplet_candidates(X, list(X.columns), y)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    assert serial == T._enumerate_triplet_candidates(X, list(X.columns), y)
