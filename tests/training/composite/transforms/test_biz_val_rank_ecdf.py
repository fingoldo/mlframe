"""Unit + biz_value tests for the ``rank_ecdf_residual`` composite transform.

``rank_ecdf_residual`` maps ``y`` and ``base`` into their common TRAIN empirical-
CDF (rank) space, ``T = ecdf_y(y) - ecdf_base(base)``, and inverts through the
stored quantile function ``y = quantile_y(T_hat + ecdf_base(base))``. Any monotone
/ heavy-tailed distortion where a ``linear_residual`` line leaves structure and
extrapolates on the tails is collapsed to the identity in rank space; the quantile
inverse cannot leave the train y-support.

biz_value assertion pins the measured win with margin (per CLAUDE.md): on a
heavy-tailed monotone-warp synthetic (``y = sinh(2.5*latent)``) the rank-space
reconstruction median-abs error crushes ``linear_residual`` (measured ratio
~0.0014; floor 0.5) whose line inverse explodes on the tails.
"""

from __future__ import annotations

import warnings

import numpy as np
import orjson
import pytest

warnings.filterwarnings("ignore")

from mlframe.training.composite.transforms import (
    TRANSFORMS_REGISTRY,
    compose_target_name,
    get_transform,
    is_composite_target_name,
)


# ---------------------------------------------------------------------------
# Unit: registry wiring, round-trip, out-of-support clamp, ties, domain, serde.
# ---------------------------------------------------------------------------


def test_rank_ecdf_registered_contract():
    t = get_transform("rank_ecdf_residual")
    assert t is TRANSFORMS_REGISTRY["rank_ecdf_residual"]
    assert t.requires_base is True
    assert t.requires_groups is False
    assert not t.recurrent


def test_rank_ecdf_not_in_default_transform_list():
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )

    cfg = CompositeTargetDiscoveryConfig()
    assert "rank_ecdf_residual" not in cfg.transforms


def test_rank_ecdf_round_trip_identity_on_training_points():
    """On the train points (which ARE ECDF knots) the forward/inverse maps are
    exact inverses, so inverse(forward(y)) == y within interpolation tol."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(2000) * 2.0
    y = np.sinh(base + rng.standard_normal(2000))
    t = get_transform("rank_ecdf_residual")
    p = t.fit(y, base)
    T = t.forward(y, base, p)
    y_back = t.inverse(T, base, p)
    assert np.median(np.abs(y_back - y)) <= 1e-9
    assert np.max(np.abs(y_back - y)) <= 1e-6


def test_rank_ecdf_residual_in_rank_range():
    """T = ecdf_y - ecdf_base lives in [-1, 1] (difference of two CDFs)."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(1000)
    y = rng.standard_normal(1000)
    t = get_transform("rank_ecdf_residual")
    T = t.forward(y, base, t.fit(y, base))
    assert T.min() >= -1.0 - 1e-12 and T.max() <= 1.0 + 1e-12


def test_rank_ecdf_out_of_support_base_clamps():
    """A predict-time base beyond the train range clamps to the edge CDF (no
    extrapolation blow-up); the recovered y stays inside the train y-support."""
    rng = np.random.default_rng(2)
    base_tr = rng.uniform(-1.0, 1.0, 1500)
    y_tr = 3.0 * base_tr + rng.standard_normal(1500) * 0.1
    t = get_transform("rank_ecdf_residual")
    p = t.fit(y_tr, base_tr)
    base_oos = np.array([-50.0, 50.0])
    # T_hat = 0 -> recovered rank == ecdf_base(base), inverse must stay bounded.
    y_hat = t.inverse(np.zeros(2), base_oos, p)
    assert y_hat.min() >= y_tr.min() - 1e-9
    assert y_hat.max() <= y_tr.max() + 1e-9


def test_rank_ecdf_handles_ties():
    """Heavily-tied y/base (few distinct values) must still fit invertible knots
    and round-trip the distinct levels."""
    y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    base = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0])
    t = get_transform("rank_ecdf_residual")
    p = t.fit(y, base)
    T = t.forward(y, base, p)
    assert np.all(np.isfinite(T))
    y_back = t.inverse(T, base, p)
    # Distinct y levels recover to themselves (knot-exact).
    assert np.median(np.abs(y_back - y)) <= 1e-9


def test_rank_ecdf_constant_y_stays_finite():
    """Degenerate constant y -> a 2-knot ramp keeps interp invertible (no /0)."""
    y = np.full(40, 7.0)
    base = np.arange(40.0)
    t = get_transform("rank_ecdf_residual")
    p = t.fit(y, base)
    T = t.forward(y, base, p)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(t.inverse(T, base, p)))


def test_rank_ecdf_single_row():
    t = get_transform("rank_ecdf_residual")
    y = np.array([3.0])
    base = np.array([2.0])
    p = t.fit(y, base)
    T = t.forward(y, base, p)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(t.inverse(T, base, p)))


def test_rank_ecdf_params_serialize():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(500)
    y = np.sinh(base)
    p = get_transform("rank_ecdf_residual").fit(y, base)
    assert set(p) == {"y_knots", "y_cdf", "base_knots", "base_cdf"}

    def _coerce(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    encoded = orjson.dumps({k: _coerce(v) for k, v in p.items()}, option=orjson.OPT_SORT_KEYS)
    decoded = orjson.loads(encoded)
    assert set(decoded) == set(p)
    assert all(isinstance(decoded[k], list) for k in p)


def test_rank_ecdf_domain_rejects_non_finite():
    t = get_transform("rank_ecdf_residual")
    y = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    base = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    mask = t.domain_check(y, base)
    assert mask.tolist() == [True, False, False, False, True]
    assert t.domain_check(None, base).tolist() == [True, True, False, True, True]


def test_rank_ecdf_fit_does_not_mutate_inputs():
    rng = np.random.default_rng(4)
    base = rng.standard_normal(400)
    y = np.sinh(base)
    b_snap, y_snap = base.copy(), y.copy()
    get_transform("rank_ecdf_residual").fit(y, base)
    np.testing.assert_array_equal(base, b_snap)
    np.testing.assert_array_equal(y, y_snap)


def test_rank_ecdf_composite_name_recognised():
    name = compose_target_name("y", "rank_ecdf_residual", "b")
    assert name == "y-rankecdf-b"
    assert is_composite_target_name(name)


# ---------------------------------------------------------------------------
# biz_value: heavy-tailed monotone warp -> rank space beats linear_residual.
# ---------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _medae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(np.abs(a - b)))


def _gen_warped(n: int, seed: int):
    """Heavy-tailed monotone warp: y = sinh(2.5*(base + 0.7*sig + noise)). The
    generalisable driver ``sig`` maps ~linearly in RANK space but the raw y~base
    relation is explosively nonlinear, so a linear_residual line leaves structure
    and its inverse blows up on the tails."""
    r = np.random.default_rng(seed)
    base = r.standard_normal(n) * 1.5
    sig = r.standard_normal(n)
    latent = base + 0.7 * sig + r.standard_normal(n) * 0.3
    y = np.sinh(2.5 * latent)
    return base, sig, y


def test_biz_val_rank_ecdf_beats_linear_residual_on_heavy_tailed_warp():
    re = get_transform("rank_ecdf_residual")
    lin = get_transform("linear_residual")
    re_errs, lin_errs = [], []
    for s in range(6):
        base_tr, sig_tr, y_tr = _gen_warped(2000, s + 10)
        base_te, sig_te, y_te = _gen_warped(2000, 100 + s)
        # rank_ecdf reconstruction.
        p = re.fit(y_tr, base_tr)
        T = re.forward(y_tr, base_tr, p)
        c = _ols(sig_tr, T)
        yhat = re.inverse(c[0] * sig_te + c[1], base_te, p)
        re_errs.append(_medae(yhat, y_te))
        # linear_residual reconstruction.
        pl = lin.fit(y_tr, base_tr)
        Tl = lin.forward(y_tr, base_tr, pl)
        cl = _ols(sig_tr, Tl)
        ylhat = lin.inverse(cl[0] * sig_te + cl[1], base_te, pl)
        lin_errs.append(_medae(ylhat, y_te))
    re_err = float(np.median(re_errs))
    lin_err = float(np.median(lin_errs))
    assert re_err <= lin_err * 0.5, f"rank_ecdf medAE {re_err:.4f} should be <=0.5x linear_residual medAE {lin_err:.4f} (ratio {re_err / lin_err:.4f})"
