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
from typing import Any
from collections.abc import Mapping

import numpy as np
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

_RNG = np.random.default_rng(0)
_N = 200
_BASE = np.linspace(1.0, 10.0, _N)
_Y = 0.5 * _BASE + 1.0 + _RNG.standard_normal(_N) * 0.1
_BASE2 = _RNG.uniform(1.0, 5.0, _N)
_GROUPS = (np.arange(_N) // 50).astype(np.int64)


# One tolerance for every transform: max absolute round-trip error within ``_REL_TOL * max(1, max|y|)``. Every exact
# transform round-trips to float64 round-off; 1e-8 (not 1e-9) leaves room for the power transforms' conditioning on a
# large offset (yeo_johnson_y at |y| ~ 5e3 loses 1.5e-9 relative). The old per-name table accepted up to 1.0 on a fixture
# whose median |y - mean| is 1.08 - an inverse close to a constant passed - and had drifted out of sync with the registry.
_REL_TOL = 1e-8

# Transforms that lose information by definition, each with its asserted loss below. Nothing else may be lossy.
_LOSSY_BY_DEFINITION = {"y_quantile_clip"}


def _base_for(name: str, base: np.ndarray, base2: np.ndarray) -> np.ndarray:
    """The base a transform takes: a two-column matrix for the multi-base family, the 1-D base otherwise.

    Handing a multi-base transform a 1-D base silently exercised its degenerate single-column path as if it were valid.
    """
    return np.column_stack([base, base2]) if TRANSFORMS_REGISTRY[name].n_bases > 1 else base


def _grid_data(n: int, scale: float, offset: float, seed: int = 0):
    """``y``, base, second base and groups at a given size, scale and base offset."""
    rng = np.random.default_rng(seed)
    base = offset + scale * np.linspace(1.0, 10.0, n)
    y = 0.5 * base + scale * (1.0 + 0.1 * rng.standard_normal(n))
    base2 = offset + scale * rng.uniform(1.0, 5.0, n)
    groups = (np.arange(n) * 4 // n).astype(np.int64)
    return y, base, base2, groups


# Groups is checked FIRST: a transform can be requires_groups=True AND
# requires_base=False (target_encoding_residual -- the category column carries
# the signal, no numeric base), in which case it still needs groups and takes
# base=None.
def _call_fit(t, y: np.ndarray, base: np.ndarray, groups: np.ndarray | None = None) -> dict[str, Any]:
    """Call fit."""
    if t.requires_groups:
        return t.fit(y, base if t.requires_base else None, groups=_GROUPS[: len(y)] if groups is None else groups)
    if not t.requires_base:
        return t.fit(y, None)
    return t.fit(y, base)


def _call_forward(t, y: np.ndarray, base: np.ndarray, params: Mapping[str, Any], groups: np.ndarray | None = None) -> np.ndarray:
    """Call forward."""
    if t.requires_groups:
        return t.forward(y, base if t.requires_base else None, params, groups=_GROUPS[: len(y)] if groups is None else groups)
    if not t.requires_base:
        return t.forward(y, None, params)
    return t.forward(y, base, params)


def _call_inverse(t, t_hat: np.ndarray, base: np.ndarray, params: Mapping[str, Any], groups: np.ndarray | None = None) -> np.ndarray:
    """Call inverse."""
    if t.requires_groups:
        return t.inverse(t_hat, base if t.requires_base else None, params, groups=_GROUPS[: len(t_hat)] if groups is None else groups)
    if not t.requires_base:
        return t.inverse(t_hat, None, params)
    return t.inverse(t_hat, base, params)


def _call_domain(t, y: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Call domain."""
    if not t.requires_base:
        return t.domain_check(y, None)
    return t.domain_check(y, base)


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_transform_fit_returns_json_serializable_dict(name: str):
    """The Transform contract docstring requires ``fit`` to return a
    JSON-serialisable dict so model state survives pickle / round-trip."""
    t = TRANSFORMS_REGISTRY[name]
    base = _base_for(name, _BASE, _BASE2)
    domain = _call_domain(t, _Y, base)
    params = _call_fit(t, _Y[domain], base[domain])
    assert isinstance(params, dict)

    # Numpy arrays are not directly JSON-serialisable but Transform contract
    # allows ndarray-valued params (bin_edges, knots_y, ...). Round-trip
    # through a thin coercion to numpy -> list so we exercise the same
    # encoder path used downstream by io.save_mlframe_model.
    def _coerce(o):
        """Coerce."""
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


def _fit_on(t, name, y, base, base2, groups):
    """Fit ``t`` on its domain rows; returns (params, domain mask, the base it takes)."""
    b = _base_for(name, base, base2)
    domain = np.asarray(_call_domain(t, y, b), dtype=bool)
    params = _call_fit(t, y[domain], b[domain] if t.requires_base else b, groups[domain])
    return params, domain, b


@pytest.mark.parametrize("n", [50, 3000])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e6])
@pytest.mark.parametrize("offset", [0.0, 1e4])
@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_transform_forward_inverse_round_trip(name: str, n: int, scale: float, offset: float):
    """``inverse(forward(y))`` recovers ``y`` to float64 round-off on every row, at every scale, size and base offset.

    The MAX error is checked, not the median: a median passes while up to half the rows are wrong, which is the shape of
    the tail- and level-only defects. A transform declaring ``oof_train_forward`` (target encoding) answers a forward on
    its own fit rows out-of-fold, so it is checked on a slice, which takes the full-train statistics as predict does.
    """
    t = TRANSFORMS_REGISTRY[name]
    y, base, base2, groups = _grid_data(n, scale, offset)
    params, domain, b = _fit_on(t, name, y, base, base2, groups)
    y_fit, b_fit, g_fit = y[domain], (b[domain] if t.requires_base else b), groups[domain]
    if getattr(t, "oof_train_forward", False):
        half = len(y_fit) // 2
        y_fit, g_fit = y_fit[:half].copy(), g_fit[:half]
        b_fit = b_fit[:half].copy() if t.requires_base else b_fit
    t_fit = _call_forward(t, y_fit, b_fit, params, g_fit)
    assert t_fit.shape == y_fit.shape
    assert np.all(np.isfinite(t_fit) | ~np.isfinite(y_fit))
    y_back = _call_inverse(t, t_fit, b_fit, params, g_fit)
    assert y_back.shape == y_fit.shape
    tol = _REL_TOL * max(1.0, float(np.max(np.abs(y_fit))))
    if name == "y_quantile_clip":
        # Lossy by definition: the inverse is exactly the clip of y to the fitted quantile band.
        np.testing.assert_allclose(y_back, np.clip(y_fit, params["q_lo"], params["q_hi"]), rtol=0, atol=tol)
        return
    err = float(np.max(np.abs(y_back - y_fit)))
    assert err <= tol, f"transform={name!r} n={n} scale={scale} offset={offset}: max round-trip err {err:.3e} > {tol:.3e}"


def test_the_lossy_exception_list_is_part_of_the_registry():
    """A lossy exception naming a transform that no longer exists would drift out of sync like the old table did."""
    assert _LOSSY_BY_DEFINITION <= set(TRANSFORMS_REGISTRY)


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_inverse_on_a_base_beyond_the_train_range_stays_finite_and_positive(name: str):
    """A base 5% above anything seen in training must still invert to finite y with the sign training y had."""
    t = TRANSFORMS_REGISTRY[name]
    y, base, base2, groups = _grid_data(600, 1.0, 0.0)
    params, _, _ = _fit_on(t, name, y, base, base2, groups)
    rng = np.random.default_rng(1)
    base_out = np.linspace(10.0, 10.45, 200)
    y_out = 0.5 * base_out + 1.0 + 0.1 * rng.standard_normal(200)
    b_out = _base_for(name, base_out, rng.uniform(1.0, 5.0, 200))
    g_out = (np.arange(200) * 4 // 200).astype(np.int64)
    t_out = _call_forward(t, y_out, b_out, params, g_out)
    y_back = _call_inverse(t, t_out, b_out, params, g_out)
    assert np.all(np.isfinite(y_back)), f"{name}: {int((~np.isfinite(y_back)).sum())} non-finite rows beyond the train base range"
    assert np.all(y_back > 0), f"{name}: {int((y_back <= 0).sum())} rows flipped sign beyond the train base range"


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_a_second_batch_round_trips_with_the_first_batchs_params(name: str):
    """Params fitted on one batch must invert a disjoint batch exactly (inside the train y range): no batch-bound state.

    Rows beyond the train y range are left out on purpose: the rank, ECDF and power transforms clamp to the training
    envelope there by design, and the out-of-range behaviour is covered by the finite-and-positive test above.
    """
    t = TRANSFORMS_REGISTRY[name]
    y, base, base2, groups = _grid_data(600, 1.0, 0.0)
    params, domain, _ = _fit_on(t, name, y, base, base2, groups)
    y2, base_b, base2_b, groups2 = _grid_data(300, 1.0, 0.0, seed=7)
    b2 = _base_for(name, base_b, base2_b)
    keep = np.asarray(_call_domain(t, y2, b2), dtype=bool) & (y2 >= y[domain].min()) & (y2 <= y[domain].max())
    y2k, g2k = y2[keep], groups2[keep]
    b2k = b2[keep] if t.requires_base else b2
    y_back = _call_inverse(t, _call_forward(t, y2k, b2k, params, g2k), b2k, params, g2k)
    tol = _REL_TOL * max(1.0, float(np.max(np.abs(y2k))))
    if name == "y_quantile_clip":
        np.testing.assert_allclose(y_back, np.clip(y2k, params["q_lo"], params["q_hi"]), rtol=0, atol=tol)
        return
    err = float(np.max(np.abs(y_back - y2k)))
    assert err <= tol, f"{name}: second-batch round-trip err {err:.3e} > {tol:.3e}"


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
    base_with_nan = _base_for(name, _BASE, _BASE2).copy()
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
    base = _base_for(name, _BASE, _BASE2)
    domain = _call_domain(t, _Y, base)
    y_train = _Y[domain].copy()
    base_train = base[domain].copy()
    y_snapshot = y_train.copy()
    base_snapshot = base_train.copy()
    _ = _call_fit(t, y_train, base_train)
    np.testing.assert_array_equal(y_train, y_snapshot, err_msg=f"{name} mutated y_train")
    np.testing.assert_array_equal(base_train, base_snapshot, err_msg=f"{name} mutated base_train")
