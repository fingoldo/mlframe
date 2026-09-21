"""Each base transform absorbs the base relation it is designed for, on its canonical DGP, at small and large n.

Absorption is measured scale-free: inverting the constant median T at every row's base must explain at least 90% of y's
variance (the transform captured the base relation, so a model predicting a constant T already reproduces y). The additive
family is also checked with the base offset by 1e4. This found ``quantile_residual`` ignoring the base entirely at n=300
(ten bins of 30 rows under a 50-row minimum all fell back to the global median).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

from .test_composite_transforms_registry_contract import _base_for, _call_domain, _call_fit, _call_forward, _call_inverse

_CANONICAL_DGP = {
    "additive_residual": "additive",
    "asinh_residual": "multiplicative",
    "asinh_residual_multi": "multiplicative",
    "causal_anchor_residual": "additive",
    "centered_ratio": "multiplicative",
    "chain_linres_cbrt": "linear",
    "chain_linres_cbrt_qn": "linear",
    "chain_linres_yj": "linear",
    "chain_monres_cbrt": "saturating",
    "chain_monres_yj": "saturating",
    "diff": "additive",
    "ewma_residual": "additive",
    "ewma_residual_grouped": "additive",
    "gaussian_copula_residual": "saturating",
    "geometric_mean_residual": "geometric",
    "linear_residual": "linear",
    "linear_residual_grouped": "linear",
    "linear_residual_multi": "linear",
    "linear_residual_multi_robust": "linear",
    "linear_residual_robust": "linear",
    "logratio": "multiplicative",
    "median_residual": "saturating",
    "monotonic_residual": "saturating",
    "monotonic_residual_grouped": "saturating",
    "nadaraya_watson_residual": "saturating",
    "pairwise_interaction_residual": "product",
    "polynomial_residual_deg2": "saturating",
    "quantile_residual": "saturating",
    "quantile_residual_grouped": "saturating",
    "rank_ecdf_residual": "saturating",
    "rank_residual": "saturating",
    "ratio": "multiplicative",
    "reciprocal_residual": "multiplicative",
    "rolling_quantile_ratio": "multiplicative",
    "rolling_quantile_ratio_centered": "multiplicative",
    "rolling_quantile_ratio_grouped": "multiplicative",
    "second_diff": "second_difference",
    "smoothing_spline_residual": "saturating",
    "theilsen_residual": "linear",
    "volatility_normalized_residual": "multiplicative",
}
# The product relation is absorbed only up to the noise the product amplifies; everything else clears 0.9.
_MIN_R2 = {"product": 0.8}


def _dgp(kind: str, n: int, offset: float, seed: int = 0):
    """``(y, base, base2)`` for a DGP family; ``offset`` shifts the bases (not the multiplicative family's level)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 10.0, n) + rng.normal(0.0, 0.05, n)
    b2 = rng.uniform(1.0, 5.0, n)
    e = rng.normal(0.0, 0.1, n)
    base = offset + x
    y = {
        "additive": base + 3.0 + e,
        "linear": 2.0 * base + 0.5 * b2 + 3.0 + e,
        "second_difference": 2.0 * base - (offset + b2) + e,
        "product": x * b2 + e,
        "multiplicative": x * np.exp(0.02 * e),
        "geometric": np.sqrt(x * b2) * np.exp(0.02 * e),
        "saturating": 10.0 * np.log1p(x) + e,
    }[kind]
    if kind in ("multiplicative", "product", "geometric"):
        return y, x, b2
    return y, base, offset + b2


def test_every_base_transform_has_a_canonical_dgp():
    """A new base transform must declare the relation it absorbs."""
    base_transforms = {n for n, t in TRANSFORMS_REGISTRY.items() if t.requires_base}
    assert set(_CANONICAL_DGP) == base_transforms, sorted(base_transforms ^ set(_CANONICAL_DGP))


@pytest.mark.parametrize("n", [300, 2000])
@pytest.mark.parametrize("name", sorted(_CANONICAL_DGP))
def test_a_transform_absorbs_its_canonical_base_relation(name: str, n: int):
    """Inverting the median T at each row's base explains >= 90% of var(y) (offset bases too, where they apply)."""
    t = TRANSFORMS_REGISTRY[name]
    kind = _CANONICAL_DGP[name]
    offsets = (0.0,) if kind in ("multiplicative", "product", "geometric") else (0.0, 1e4)
    for offset in offsets:
        y, base, base2 = _dgp(kind, n, offset)
        g = (np.arange(n) * 4 // n).astype(np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = _base_for(name, base, base2)
            d = np.asarray(_call_domain(t, y, b), dtype=bool)
            params = _call_fit(t, y[d], b[d], g[d])
            t_fit = _call_forward(t, y[d], b[d], params, g[d])
            y_hat = _call_inverse(t, np.full(int(d.sum()), float(np.nanmedian(t_fit))), b[d], params, g[d])
        r2 = 1.0 - float(np.nanvar(y[d] - y_hat)) / float(np.var(y[d]))
        assert r2 >= _MIN_R2.get(kind, 0.9), f"{name} on its {kind} DGP (n={n}, offset={offset}): the base explains only {r2:.3f}"
