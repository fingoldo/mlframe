"""Per-Transform contract tests for the composite-target registry.

A2#20 backed the audit gap: per-transform unit suites were missing -- the
public contract (fit returns JSON-serialisable dict, forward / inverse are
shape-preserving, domain_check rejects non-finite, round-trip on training
data is identity within epsilon) was only verified through
``CompositeTargetEstimator`` integration. This file exercises every entry in
``TRANSFORMS_REGISTRY`` against that contract with cheap synthetic data, so
adding a new transform to the registry without a unit either passes the
shared contract or fails fast.

Why one file with parametrised tests rather than one file per transform:
the contract is identical across all 24 entries; per-transform files would
multiply boilerplate without adding signal. Per-transform behavioural
asserts (e.g. ``y_quantile_clip`` clips to bounds, ``ratio`` divides by
base) already exist in the older ``test_composite_*`` files.
"""
from __future__ import annotations

import orjson
from typing import Any, Mapping

import numpy as np
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY


_RNG = np.random.default_rng(0)
_N = 200
_BASE = np.linspace(1.0, 10.0, _N)
_Y = 0.5 * _BASE + 1.0 + _RNG.standard_normal(_N) * 0.1
_GROUPS = (np.arange(_N) // 50).astype(np.int64)


# Tolerance per transform. ``diff`` / ``additive_residual`` / linear_residual
# are exact within float-64 ULP; quantile / monotonic / chain paths use
# binning / spline / clip operations that lose precision on the inverse
# leg, so we relax. Unary y-only transforms (cbrt / yj / qn / log) have
# exact analytic inverses but accumulate small float error on extreme tails.
_TRANSFORM_RTOL: dict[str, float] = {
    "diff": 1e-9,
    "additive_residual": 1e-9,
    "linear_residual": 1e-9,
    "linear_residual_multi": 1e-9,
    "linear_residual_grouped": 1e-9,
    "linear_residual_robust": 1e-9,
    "ratio": 1e-9,
    "logratio": 5e-1,  # MAD soft-cap clips extreme values
    "median_residual": 1.0,  # per-bin median lookup, not invertible row-wise
    "quantile_residual": 1.0,  # binned IQR division, lossy
    "monotonic_residual": 0.5,  # PCHIP interpolation
    "ewma_residual": 1e-6,
    "frac_diff": 1e-6,
    "rolling_quantile_ratio": 1e-6,
    "y_quantile_clip": 1.0,  # clips by definition; inverse is identity in [q_lo, q_hi]
    "cbrt_y": 1e-8,
    "signed_power_y": 1e-6,  # fitted p; 1/p up to 10 amplifies tail float error
    "log_y": 1e-8,
    "yeo_johnson_y": 1e-6,
    "quantile_normal_y": 0.5,  # spline-based, lossy at edges
    "chain_linres_cbrt": 1e-6,
    "chain_linres_yj": 1e-5,
    "chain_monres_cbrt": 0.5,  # composes monotonic_residual + cbrt
    "chain_monres_yj": 0.5,
    "chain_linres_cbrt_qn": 0.5,  # composes 3 stages, includes quantile_normal
    # Pack L (2026-05-26).
    "asinh_residual": 1e-9,           # exact OLS in arcsinh space
    "centered_ratio": 1e-9,           # exact algebra
    "polynomial_residual_deg2": 1e-9, # exact OLS solve
    "rank_residual": 1.0,             # ECDF quantize loss like quantile_normal_y
    "smoothing_spline_residual": 0.5, # smoothing-spline approximation lossy
    "reciprocal_residual": 1e-6,
    "geometric_mean_residual": 1e-9,  # exact: y / g * g = y
    "pairwise_interaction_residual": 1e-9,
}


def _call_fit(t, y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    if not t.requires_base:
        return t.fit(y, None)
    if t.requires_groups:
        return t.fit(y, base, groups=_GROUPS[: len(y)])
    return t.fit(y, base)


def _call_forward(t, y: np.ndarray, base: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    if not t.requires_base:
        return t.forward(y, None, params)
    if t.requires_groups:
        return t.forward(y, base, params, groups=_GROUPS[: len(y)])
    return t.forward(y, base, params)


def _call_inverse(t, t_hat: np.ndarray, base: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    if not t.requires_base:
        return t.inverse(t_hat, None, params)
    if t.requires_groups:
        return t.inverse(t_hat, base, params, groups=_GROUPS[: len(t_hat)])
    return t.inverse(t_hat, base, params)


def _call_domain(t, y: np.ndarray, base: np.ndarray) -> np.ndarray:
    if not t.requires_base:
        return t.domain_check(y, None)
    return t.domain_check(y, base)


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_transform_fit_returns_json_serializable_dict(name: str):
    """The Transform contract docstring requires ``fit`` to return a
    JSON-serialisable dict so model state survives pickle / round-trip."""
    t = TRANSFORMS_REGISTRY[name]
    domain = _call_domain(t, _Y, _BASE)
    params = _call_fit(t, _Y[domain], _BASE[domain])
    assert isinstance(params, dict)
    # Numpy arrays are not directly JSON-serialisable but Transform contract
    # allows ndarray-valued params (bin_edges, knots_y, ...). Round-trip
    # through a thin coercion to numpy -> list so we exercise the same
    # encoder path used downstream by io.save_mlframe_model.
    def _coerce(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, dict):
            return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_coerce(x) for x in o]
        return o

    encoded = orjson.dumps(_coerce(params), option=orjson.OPT_SORT_KEYS)
    decoded = orjson.loads(encoded)
    assert isinstance(decoded, dict)
    assert set(decoded) == set(params)


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_transform_forward_inverse_round_trip(name: str):
    """``inverse(forward(y, base, params), base, params)`` must equal
    ``y`` within the transform-specific tolerance on training data."""
    t = TRANSFORMS_REGISTRY[name]
    domain = _call_domain(t, _Y, _BASE)
    y_train = _Y[domain]
    base_train = _BASE[domain]
    params = _call_fit(t, y_train, base_train)
    t_train = _call_forward(t, y_train, base_train, params)
    assert t_train.shape == y_train.shape
    assert np.all(np.isfinite(t_train) | ~np.isfinite(y_train))
    y_back = _call_inverse(t, t_train, base_train, params)
    assert y_back.shape == y_train.shape
    rtol = _TRANSFORM_RTOL.get(name, 1e-6)
    # Use median absolute deviation rather than max-abs so an isolated edge
    # case at the spline boundary doesn't fail the whole transform; for
    # tight-tolerance transforms (diff, ratio, linear_residual) the median
    # is dominated by the row-wise error and matches max-abs to within ULP.
    err = np.median(np.abs(y_back - y_train))
    assert err <= rtol, (
        f"transform={name!r} round-trip median err {err:.3e} > rtol {rtol:.3e}"
    )


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_transform_domain_check_rejects_non_finite(name: str):
    """The wrapper relies on domain_check to drop NaN / Inf rows. If a
    transform's domain_check ever returns ``True`` for a non-finite y
    or base, the inverse path can produce silent NaN outputs that bypass
    the y_train_median fallback."""
    t = TRANSFORMS_REGISTRY[name]
    y_with_nan = _Y.copy()
    y_with_nan[0] = np.nan
    y_with_nan[1] = np.inf
    base_with_nan = _BASE.copy()
    base_with_nan[2] = np.nan
    domain = _call_domain(t, y_with_nan, base_with_nan)
    # At minimum rows 0 + 1 must be rejected (non-finite y). For transforms
    # that require base, row 2 must also be rejected. The unary transforms
    # do not consume base so they may legitimately keep row 2.
    assert not domain[0], f"transform={name!r} domain_check accepted NaN y"
    assert not domain[1], f"transform={name!r} domain_check accepted inf y"
    if t.requires_base:
        assert not domain[2], f"transform={name!r} domain_check accepted NaN base"


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_transform_fit_does_not_mutate_inputs(name: str):
    """``fit`` is contractually pure; mutating y_train / base_train would
    leak per-target state across the suite's per-target loop."""
    t = TRANSFORMS_REGISTRY[name]
    domain = _call_domain(t, _Y, _BASE)
    y_train = _Y[domain].copy()
    base_train = _BASE[domain].copy()
    y_snapshot = y_train.copy()
    base_snapshot = base_train.copy()
    _ = _call_fit(t, y_train, base_train)
    np.testing.assert_array_equal(y_train, y_snapshot, err_msg=f"{name} mutated y_train")
    np.testing.assert_array_equal(base_train, base_snapshot, err_msg=f"{name} mutated base_train")
