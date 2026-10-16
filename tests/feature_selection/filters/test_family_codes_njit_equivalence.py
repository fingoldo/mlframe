"""The parallel block coder returns the codes of the per-column numpy coder for every shape of column the family test meets."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._family_maxt_gate import _family_codes, _quantile_codes


def _columns(n, seed):
    """Continuous, heavy-tailed, tied, few-level, binary, constant and outlier columns."""
    rng = np.random.default_rng(seed)
    cols = [
        rng.normal(size=n),
        rng.standard_cauchy(size=n),
        np.round(rng.normal(size=n) * 3) / 3,
        rng.integers(0, 5, n).astype(float),
        rng.integers(0, 2, n).astype(float),
        np.full(n, 3.25),
        np.where(rng.random(n) < 0.02, 1e6, rng.normal(size=n)),
        rng.random(n) ** 6,
        np.sort(rng.normal(size=n)),
        np.where(rng.random(n) < 0.7, 0.0, rng.normal(size=n)),
        rng.integers(0, 11, n).astype(float),
        rng.integers(0, 10, n).astype(float),
    ]
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [37, 1000, 6001])
def test_block_coder_equals_the_numpy_coder(n, dtype, monkeypatch):
    """Codes and the number of levels agree column by column, in the fit's float width."""
    from mlframe.feature_selection.filters import _family_maxt_gate as G

    monkeypatch.setattr(G, "_crit_np_dtype", lambda: dtype)
    X = _columns(n, seed=n)
    codes, nx = _family_codes(X, 10)
    for j in range(X.shape[1]):
        ref_c, ref_k = _quantile_codes(np.asarray(X[:, j], dtype=dtype), 10)
        assert nx[j] == ref_k, j
        assert np.array_equal(codes[j].astype(np.int64), ref_c), j


def test_non_finite_values_are_replaced_by_the_median_before_coding():
    """A NaN / inf cell is coded as the column median is, as before."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    X[::17, 0] = np.nan
    X[5, 1] = np.inf
    codes, nx = _family_codes(X, 10)
    filled = X.copy()
    for j in (0, 1):
        bad = ~np.isfinite(filled[:, j])
        filled[bad, j] = np.median(filled[~bad, j])
    ref, k = _family_codes(filled, 10)
    assert np.array_equal(codes, ref) and np.array_equal(nx, k)
