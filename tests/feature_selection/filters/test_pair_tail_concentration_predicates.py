"""pair_is_tail_concentrated is the linear half of pair_is_tail_concentrated_rankaware, the predicate the FE pre-scan calls."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import pair_is_tail_concentrated, pair_is_tail_concentrated_rankaware, usability_form_corrs

_KW = dict(min_corr=0.5, pairness_margin=1.2)


def _pair(seed: int, kind: str, n: int = 600):
    """Return (y, x0, x1) for a clean product signal, pure noise, or a heavy-tailed ratio target."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    if kind == "product":
        y = x0 * x1 + rng.normal(scale=0.05, size=n)
    elif kind == "noise":
        y = rng.normal(size=n)
    else:
        x1 = rng.pareto(1.5, n) + 0.2
        y = (x0**2) / x1 + rng.normal(scale=0.1, size=n)
    return y, x0, x1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_half_matches_its_definition(seed):
    """The predicate is exactly |corr(best pair form)| >= min_corr and >= margin * best single-operand corr."""
    for kind in ("product", "noise", "ratio"):
        y, x0, x1 = _pair(seed, kind)
        cp, cs = usability_form_corrs(y, x0, x1)
        expect = bool(cp >= _KW["min_corr"] and cp >= _KW["pairness_margin"] * cs)
        assert pair_is_tail_concentrated(y, x0, x1, **_KW) is expect


def test_product_is_linear_half_positive_and_noise_is_negative():
    """Teeth both ways: a genuine pairwise product passes the linear half, pure noise does not."""
    assert pair_is_tail_concentrated(*_pair(0, "product"), **_KW) is True
    assert pair_is_tail_concentrated(*_pair(0, "noise"), **_KW) is False


@pytest.mark.parametrize("kind", ["product", "noise", "ratio"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_rank_aware_true_implies_linear_half_true(kind, seed):
    """The rank-aware predicate adds a rank-collapse leg on top of the linear half, never replaces it."""
    y, x0, x1 = _pair(seed, kind)
    if pair_is_tail_concentrated_rankaware(y, x0, x1, **_KW):
        assert pair_is_tail_concentrated(y, x0, x1, **_KW)
