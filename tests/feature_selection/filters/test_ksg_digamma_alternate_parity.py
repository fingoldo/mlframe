"""The array-digamma KSG aggregation is an opt-in alternate; the scalar aggregation stays the default and the reference."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import digamma

from mlframe.feature_selection.filters import _ksg


def test_default_path_is_the_scalar_aggregation() -> None:
    """The flag is off by default, so the production estimator runs the scalar loop."""
    assert _ksg.KSG_DIGAMMA_VECTORISED is False


def test_digamma_vec_matches_scipy() -> None:
    """The vector digamma agrees with scipy to the scalar form's accuracy, across small and large arguments."""
    x = np.concatenate([np.linspace(1e-3, 20.0, 400), np.array([1.0, 2.0, 3.0, 100.0, 1e4])])
    np.testing.assert_allclose(_ksg._digamma_vec(x), digamma(x), rtol=0, atol=5e-8)


@pytest.mark.parametrize("n", [1_000, 50_000])
@pytest.mark.parametrize("seed", [0, 1])
def test_vector_aggregation_matches_scalar_within_tolerance(n: int, seed: int) -> None:
    """The alternate aggregation gives the scalar aggregation's value to 1e-9 relative on the same counts."""
    rng = np.random.default_rng(seed)
    n_x = rng.integers(1, 200, n).astype(np.int64)
    n_y = rng.integers(1, 200, n).astype(np.int64)
    ref = _ksg._mixed_ksg_aggregate(5, n, n_x, n_y)
    alt = _ksg._mixed_ksg_aggregate_vec(5, n, n_x, n_y)
    assert abs(alt - ref) <= 1e-9 * max(1.0, abs(ref))


def test_flag_switches_the_public_estimator_only_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning the flag on changes the public mixed-KSG estimate by no more than the stated tolerance."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=3_000)
    y = x + 0.5 * rng.normal(size=3_000)
    off = _ksg.mixed_ksg_mi(x, y, k=5)
    monkeypatch.setattr(_ksg, "KSG_DIGAMMA_VECTORISED", True)
    on = _ksg.mixed_ksg_mi(x, y, k=5)
    assert abs(on - off) <= 1e-9 * max(1.0, abs(off))
