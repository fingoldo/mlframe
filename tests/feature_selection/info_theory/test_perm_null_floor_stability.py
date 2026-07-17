"""The order-1 maxT noise floor must be STABLE run-to-run: its 95th-percentile of a per-shuffle MAX is an extreme upper-tail order statistic that needs enough
permutations to estimate. The default was raised 25 -> 200 because K=25 places only ~1.25 draws above the 95th percentile, giving a high-variance floor that
wobbles across permutation seeds. This pins that the floor's across-seed std at the new default is materially below the std K=25 produces, so a future revert to
a small K (which would re-introduce a noisy, unreliable noise floor) fails here.
"""

import numpy as np

from mlframe.feature_selection.filters._permutation_null import (
    pooled_permutation_null_gain_floor,
)


def _floor_std_over_seeds(k: int, n_seeds: int = 30) -> float:
    """Across-seed std of the 95th-pct maxT floor on a FIXED pure-noise pool (true MI = 0), varying only the permutation RNG seed."""
    n, p, nb = 1500, 40, 8
    rng = np.random.default_rng(12345)
    data = rng.integers(0, nb, size=(n, p + 1)).astype(np.int64)
    nbins = np.full(p + 1, nb, dtype=np.int64)
    cand = np.arange(p)
    floors = np.array([pooled_permutation_null_gain_floor(data, nbins, cand, p, n_permutations=k, quantile=0.95, random_seed=1000 + s) for s in range(n_seeds)])
    return float(floors.std(ddof=1))


def test_default_permutation_count_is_200():
    """The corrective mechanism (lower-variance noise floor) is ON by default: the function default must be the stabilised 200, not the legacy high-variance 25."""
    import inspect

    sig = inspect.signature(pooled_permutation_null_gain_floor)
    assert sig.parameters["n_permutations"].default == 200


def test_floor_variance_at_default_below_k25():
    """K=200 yields a several-fold lower run-to-run floor std than K=25 on a fixed null -- the statistical justification for the default flip. The threshold (1.5x
    lower) is comfortably cleared by the measured ~2.2x and is one K=25 categorically fails (its own std is the numerator)."""
    std_25 = _floor_std_over_seeds(25)
    std_200 = _floor_std_over_seeds(200)
    assert std_200 < std_25, f"K=200 floor std {std_200:.3e} not below K=25 {std_25:.3e}"
    assert std_200 <= std_25 / 1.5, f"K=200 floor std {std_200:.3e} not >=1.5x below K=25 {std_25:.3e} (ratio {std_25 / std_200:.2f}x)"
