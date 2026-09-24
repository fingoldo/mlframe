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


# ---------------------------------------------------------------------------
# Smoothers are consistent: the fitted curve gets closer to the truth with more data.
# ---------------------------------------------------------------------------

# Nonparametric fits of g(base). The parametric residuals (linear, polynomial, the Gaussian-copula regression in normal
# scores) have a fixed functional form whose bias on a curved relation does not shrink with n, so they are not here.
_SMOOTHERS = ("nadaraya_watson_residual", "smoothing_spline_residual", "monotonic_residual", "quantile_residual", "median_residual",
              "rank_residual", "rank_ecdf_residual")
_NOT_SMOOTHERS = {
    "polynomial_residual_deg2": "a fixed quadratic: on a log curve its bias is the approximation error, flat in n",
    "gaussian_copula_residual": "a linear regression in normal-score space: its shrinkage (slope < 1) is structural, flat in n",
}


def _curve_error(name: str, n: int, seed: int) -> float:
    """RMSE, up to a constant, of the reconstruction at constant T against the noise-free curve ``10 log(1 + x)``."""
    t = TRANSFORMS_REGISTRY[name]
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 10.0, n)
    y = 10.0 * np.log1p(x) + rng.normal(0.0, 0.5, n)
    xs = np.linspace(1.5, 9.5, 400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = _call_fit(t, y, x, None)
        c = float(np.median(_call_forward(t, y, x, p, None)))
        y_hat = _call_inverse(t, np.full(xs.size, c), xs, p, None)
    d = np.asarray(y_hat, dtype=float) - 10.0 * np.log1p(xs)
    d -= d.mean()
    return float(np.sqrt(np.mean(d * d)))


def test_every_saturating_transform_is_classed_as_smoother_or_not():
    """A new transform on the saturating DGP must join the consistency leg or say why its bias is structural."""
    saturating = {n for n, k in _CANONICAL_DGP.items() if k == "saturating" and not TRANSFORMS_REGISTRY[n].requires_groups and not n.startswith("chain_")}
    assert saturating == set(_SMOOTHERS) | set(_NOT_SMOOTHERS), sorted(saturating ^ (set(_SMOOTHERS) | set(_NOT_SMOOTHERS)))


@pytest.mark.parametrize("name", _SMOOTHERS)
def test_a_smoother_improves_with_ten_times_the_data(name: str):
    """At n=20k the curve error is at most 0.7x its n=2k value (mean of three seeds).

    The binned residuals used a fixed 10-20 bins, so their step-function bias stayed at 0.51 / 0.25 from 2k to 20k rows (ratio
    0.97); the bin count now grows with n (~100 rows per bin) and the ratio is 0.26.
    """
    small = np.mean([_curve_error(name, 2_000, s) for s in range(3)])
    large = np.mean([_curve_error(name, 20_000, s) for s in range(3)])
    assert large <= 0.7 * small, f"{name}: curve error {small:.4f} at n=2k, {large:.4f} at n=20k (ratio {large / small:.2f})"


# ---------------------------------------------------------------------------
# Grouped transforms on identical groups: no systematic per-group level offset.
# ---------------------------------------------------------------------------

# Grouped transforms with an ungrouped twin whose constant-T reconstruction is a level. The recurrent frac_diff is excluded:
# its inverse from a constant T is a recursion, not a level, and its unseen-group seed is pinned in test_unseen_key_fallback.py.
_LEVEL_TWINS = {
    "ewma_residual_grouped": "ewma_residual",
    "linear_residual_grouped": "linear_residual",
    "monotonic_residual_grouped": "monotonic_residual",
    "quantile_residual_grouped": "quantile_residual",
    "rolling_quantile_ratio_grouped": "rolling_quantile_ratio",
}


def test_every_grouped_transform_is_in_the_level_leg_or_excluded():
    """The level leg covers every grouped transform with an ungrouped twin, except the recurrent one."""
    grouped = {n for n, t in TRANSFORMS_REGISTRY.items() if t.requires_groups and n.replace("_grouped", "") in TRANSFORMS_REGISTRY
               and not TRANSFORMS_REGISTRY[n.replace("_grouped", "")].requires_groups}
    assert grouped == set(_LEVEL_TWINS) | {"frac_diff_grouped"}, sorted(grouped ^ (set(_LEVEL_TWINS) | {"frac_diff_grouped"}))


@pytest.mark.parametrize("name", sorted(_LEVEL_TWINS))
def test_identical_groups_get_no_systematic_level_offset(name: str):
    """Twenty groups drawn from one skewed (lognormal) population: the groups' reconstructions at constant T sit on the pooled
    fit's level on average, within 0.05 sd of y.

    The monotone knot values were made monotone by a cumulative max, which lifts every dip in noisy knot medians to the running
    maximum: on ~200-row groups every group landed 0.07-0.08 sd above the pooled fit. Weighted isotonic regression leaves 0.02.
    """
    tg, tu = TRANSFORMS_REGISTRY[name], TRANSFORMS_REGISTRY[_LEVEL_TWINS[name]]
    offsets = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        n = 4000
        g = rng.integers(0, 20, n).astype(np.int64)
        b = rng.uniform(1.0, 10.0, n)
        y = b * np.exp(rng.normal(0.0, 0.5, n))
        base = b if tg.requires_base else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pu = _call_fit(tu, y, base, None)
            c = float(np.median(_call_forward(tu, y, base, pu, None)))
            pg = _call_fit(tg, y, base, g)
            y_u = np.asarray(_call_inverse(tu, np.full(n, c), base, pu, None), dtype=float)
            y_g = np.asarray(_call_inverse(tg, np.full(n, c), base, pg, g), dtype=float)
        offsets.append(np.mean([np.mean(y_g[g == k] - y_u[g == k]) for k in range(20)]) / float(np.std(y)))
    assert abs(float(np.mean(offsets))) < 0.05, f"{name}: identical groups sit {np.mean(offsets):+.3f} sd off the pooled fit"
