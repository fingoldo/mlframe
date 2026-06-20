"""biz_value tests for the EXPERIMENTAL extra-basis prototype: Bernstein / Jacobi / Gegenbauer.

These pin a MEASURED NEGATIVE (per "REJECTED != DELETED"): on every natural target these three tempting "missing basis"
families are REDUNDANT with the existing Chebyshev/Legendre/RBF/spline catalog -- lift ~1.0, never >= 1.20. The test
guards against a future contributor wiring Bernstein/Jacobi/Gegenbauer into prod believing they fill a gap they do not.

Measured (n=2000, seed=0, degree=8, bench_extra_basis_fe): all three families lift 0.98-1.00 on bounded-saturation,
smooth-poly, and endpoint-monotone targets -- the existing panel spans them all.
"""
import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import legvander

from mlframe.feature_selection.filters._extra_basis_fe_proto import (
    bernstein_design,
    fit_basis_mi,
    gegenbauer_design,
    jacobi_design,
)

_NBINS = 12
_DEG = 8


def _to_pm1(x):
    c = x.astype(np.float64)
    return 2 * (c - c.min()) / (np.ptp(c) + 1e-12) - 1


def _best_existing_mi(x, y):
    return max(
        fit_basis_mi(chebvander(_to_pm1(x), _DEG), y, nbins=_NBINS),
        fit_basis_mi(legvander(_to_pm1(x), _DEG), y, nbins=_NBINS),
    )


@pytest.fixture(scope="module")
def targets():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-3, 3, 2000))
    return {
        "bounded_saturation": (x, (1.0 / (1.0 + np.exp(-3.0 * x)) > 0.5).astype(np.int64)),
        "smooth_poly": (x, ((x**3 - 2 * x) > 0).astype(np.int64)),
        "endpoint_monotone": (x, (np.tanh(2 * x) > 0).astype(np.int64)),
    }


@pytest.mark.parametrize("target", ["bounded_saturation", "smooth_poly", "endpoint_monotone"])
@pytest.mark.parametrize("family", ["bernstein", "jacobi", "gegenbauer"])
def test_biz_val_extra_basis_is_redundant_with_existing(targets, target, family):
    """Pinned non-edge: each candidate basis lift over Chebyshev/Legendre stays below the 1.20x gap bar."""
    x, y = targets[target]
    if family == "bernstein":
        mi = fit_basis_mi(bernstein_design(x, _DEG), y, nbins=_NBINS)
    elif family == "jacobi":
        mi = max(fit_basis_mi(jacobi_design(x, _DEG, a, b), y, nbins=_NBINS)
                 for a in (-0.5, 0.0, 1.0) for b in (-0.5, 0.0, 1.0))
    else:
        mi = max(fit_basis_mi(gegenbauer_design(x, _DEG, lam), y, nbins=_NBINS) for lam in (0.25, 0.5, 1.0, 2.0))
    lift = mi / max(_best_existing_mi(x, y), 1e-6)
    assert lift < 1.20, f"{family} on {target} unexpectedly an EDGE (lift {lift:.2f}); re-evaluate wiring it in"
