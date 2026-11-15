"""The njit edge count and the early-exit distinct check of ``_fe_gain_stats`` equal the numpy forms they replaced, value for value, and are faster."""

import time

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_gain_stats import N_BINS, has_few_distinct, quantile_codes


def _reference_quantile_codes(x, n_bins=N_BINS):
    """The numpy form that ``quantile_codes`` had before the njit edge count."""
    x = np.where(np.isfinite(x), x, -np.inf)
    finite = x[np.isfinite(x)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]) if finite.size else np.zeros(n_bins - 1)
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def _columns():
    """Float32 and float64 columns: continuous, tied, heavy-tailed, with NaN and infinities, constant, discrete, tiny."""
    rng = np.random.default_rng(3)
    n = 30_001
    out = {
        "normal32": rng.standard_normal(n).astype(np.float32),
        "normal64": rng.standard_normal(n),
        "ties": np.round(rng.standard_normal(n) * 3).astype(np.float64),
        "ties32": np.round(rng.standard_normal(n) * 5).astype(np.float32),
        "lognormal": np.exp(3 * rng.standard_normal(n)),
        "constant": np.full(n, 2.5),
        "binary": (rng.random(n) < 0.3).astype(np.float64),
        "tiny": rng.standard_normal(7),
        "int": rng.integers(0, 40, n).astype(np.float64),
    }
    nonfinite = rng.standard_normal(n).astype(np.float32)
    nonfinite[rng.random(n) < 0.1] = np.nan
    nonfinite[5], nonfinite[9] = np.inf, -np.inf
    out["nonfinite"] = nonfinite
    return out


@pytest.mark.parametrize("n_bins", [2, 3, 5, 10, 20])
def test_quantile_codes_equal_the_numpy_form(n_bins):
    """Every column kind and bin count gives exactly the codes of the ``searchsorted`` form (and the same dtype)."""
    for name, x in _columns().items():
        got, want = quantile_codes(x, n_bins), _reference_quantile_codes(x, n_bins)
        assert got.dtype == want.dtype == np.int64, name
        assert np.array_equal(got, want), f"{name}: {int((got != want).sum())} codes differ"


@pytest.mark.parametrize("cap", [1, 2, 5, 20])
def test_has_few_distinct_equals_unique_size(cap):
    """The early-exit check agrees with ``np.unique(v).size <= cap`` on columns with fewer, exactly cap and more distinct values."""
    rng = np.random.default_rng(cap)
    cases = [rng.standard_normal(5000), np.round(rng.standard_normal(5000) * 2), np.zeros(10), rng.integers(0, cap, 3000).astype(np.float32), rng.integers(0, cap + 1, 3000).astype(np.float64), np.array([0.0, -0.0, 1.0]), np.array([], dtype=np.float64)]
    for v in cases:
        assert has_few_distinct(v, cap) == bool(np.unique(v).size <= cap), (cap, np.unique(v).size)


def test_fast_paths_are_faster_than_the_numpy_forms():
    """On a 50000-row float32 column the two replaced calls together cost less than half of the numpy forms."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50_000).astype(np.float32)
    quantile_codes(x), has_few_distinct(x, 20)

    def best(f):
        """The fastest of five timings of ten calls of ``f``."""
        r = []
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(10):
                f()
            r.append(time.perf_counter() - t)
        return min(r)

    new = best(lambda: (quantile_codes(x), has_few_distinct(x, 20)))
    old = best(lambda: (_reference_quantile_codes(x), np.unique(x).size <= 20))
    assert new < 0.5 * old, f"new {new:.4f} s against old {old:.4f} s"
