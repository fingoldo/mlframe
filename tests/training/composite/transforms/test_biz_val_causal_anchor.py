"""Unit + biz_value tests for the ``causal_anchor_residual`` composite transform.

``causal_anchor_residual`` fits ``T = y - alpha*base`` with ``alpha`` a robust
SHRINK coefficient clamped to ``[0, 1]``. It is the safe middle ground between
``diff`` (implicit ``alpha=1``, over-commits to a noisy anchor) and a free
``linear_residual`` (can fit a fragile large ``alpha`` that extrapolates badly on
unseen groups). The clamp bounds the additive inverse relative to the anchor.

biz_value assertions pin measured wins with margin (per CLAUDE.md):
* mean-reverting anchor -> reconstruction RMSE beats ``diff`` (measured ratio
  ~0.385; floor 0.55).
* group-disjoint OOD holdout -> does NOT blow up like an unconstrained
  large-alpha ``linear_residual`` (measured ratio ~0.32; floor 0.6).
* strong-AR anchor (true ``alpha~1``) -> ties ``diff`` (floor 1.05) -- safe to
  include in a transform set.
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
from mlframe.training.composite.transforms._causal_anchor import (
    _CAUSAL_ANCHOR_ALPHA_PRIOR,
    _causal_anchor_residual_fit,
)

# ---------------------------------------------------------------------------
# Unit: registry wiring, round-trip, [0,1] clamp, domain, serialisation.
# ---------------------------------------------------------------------------


def test_causal_anchor_registered_contract():
    """Causal anchor registered contract."""
    t = get_transform("causal_anchor_residual")
    assert t is TRANSFORMS_REGISTRY["causal_anchor_residual"]
    assert t.requires_base is True
    assert t.requires_groups is False
    assert t.requires_base and not t.recurrent


def test_causal_anchor_not_in_default_transform_list():
    """Must be reachable ONLY via explicit config, mirroring linear_residual_grouped."""
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )

    cfg = CompositeTargetDiscoveryConfig()
    assert "causal_anchor_residual" not in cfg.transforms


def test_causal_anchor_round_trip_identity_when_t_hat_exact():
    """inverse(forward(y)) == y exactly (pure additive inverse, no lossy step)."""
    rng = np.random.default_rng(0)
    anchor = rng.standard_normal(1500) * 3.0
    y = 0.6 * anchor + rng.standard_normal(1500)
    t = get_transform("causal_anchor_residual")
    p = t.fit(y, anchor)
    T = t.forward(y, anchor, p)
    y_back = t.inverse(T, anchor, p)
    assert np.allclose(y_back, y, atol=1e-12)


def test_causal_anchor_alpha_in_unit_interval_and_recovers_true():
    """Causal anchor alpha in unit interval and recovers true."""
    rng = np.random.default_rng(1)
    anchor = rng.standard_normal(3000) * 3.0
    y = 0.6 * anchor + rng.standard_normal(3000) + rng.standard_normal(3000) * 0.5
    p = _causal_anchor_residual_fit(y, anchor)
    assert 0.0 <= p["alpha"] <= 1.0
    assert abs(p["alpha"] - 0.6) < 0.06, f"alpha {p['alpha']:.4f} should recover ~0.6"


def test_causal_anchor_clamps_would_be_alpha_above_one_to_one():
    """A would-be slope > 1 (over-commit) is clamped DOWN to exactly 1.0."""
    rng = np.random.default_rng(2)
    anchor = rng.standard_normal(3000) * 3.0
    y = 2.5 * anchor + rng.standard_normal(3000) * 0.5
    p = _causal_anchor_residual_fit(y, anchor)
    assert p["alpha"] == 1.0


def test_causal_anchor_clamps_would_be_alpha_below_zero_to_zero():
    """A would-be negative slope (sign inversion) is clamped UP to exactly 0.0."""
    rng = np.random.default_rng(3)
    anchor = rng.standard_normal(3000) * 3.0
    y = -1.5 * anchor + rng.standard_normal(3000) * 0.5
    p = _causal_anchor_residual_fit(y, anchor)
    assert p["alpha"] == 0.0


def test_causal_anchor_robust_to_vertical_outliers():
    """MAD-trim rejects vertical (y) outliers so alpha still recovers ~0.6."""
    rng = np.random.default_rng(4)
    anchor = rng.standard_normal(2000) * 3.0
    y = 0.6 * anchor + rng.standard_normal(2000) * 0.5
    idx = rng.choice(2000, 120, replace=False)
    y[idx] += rng.standard_normal(120) * 100.0 + 80.0
    p = _causal_anchor_residual_fit(y, anchor)
    assert abs(p["alpha"] - 0.6) < 0.08, f"robust alpha {p['alpha']:.4f} not ~0.6"


def test_causal_anchor_scarce_data_defaults_toward_prior():
    """Below MIN_N the prior is returned outright; a tiny n with a moderate
    would-be slope is pulled toward the 0.5 prior (not left at the raw slope)."""
    tiny = _causal_anchor_residual_fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert tiny["alpha"] == pytest.approx(_CAUSAL_ANCHOR_ALPHA_PRIOR)

    # n=8, raw slope ~0.9 -> blended toward 0.5 (lands strictly between).
    x = np.arange(8.0)
    y = 0.9 * x + np.random.default_rng(5).standard_normal(8) * 0.01
    a = _causal_anchor_residual_fit(y, x)["alpha"]
    assert _CAUSAL_ANCHOR_ALPHA_PRIOR < a < 0.9, f"scarce alpha {a:.4f} not pulled toward prior"


def test_causal_anchor_domain_rejects_non_finite():
    """Causal anchor domain rejects non finite."""
    t = get_transform("causal_anchor_residual")
    y = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    base = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    mask = t.domain_check(y, base)
    assert mask.tolist() == [True, False, False, False, True]
    # Predict-time (y is None) gates only the base side.
    assert t.domain_check(None, base).tolist() == [True, True, False, True, True]


def test_causal_anchor_params_serialize():
    """Causal anchor params serialize."""
    rng = np.random.default_rng(6)
    anchor = rng.standard_normal(500) * 3.0
    y = 0.6 * anchor + rng.standard_normal(500)
    p = _causal_anchor_residual_fit(y, anchor)
    encoded = orjson.dumps(p, option=orjson.OPT_SORT_KEYS)
    decoded = orjson.loads(encoded)
    assert set(decoded) == {"alpha"}
    assert isinstance(decoded["alpha"], float)


def test_causal_anchor_fit_does_not_mutate_inputs():
    """Causal anchor fit does not mutate inputs."""
    rng = np.random.default_rng(7)
    anchor = rng.standard_normal(400) * 3.0
    y = 0.6 * anchor + rng.standard_normal(400)
    a_snap, y_snap = anchor.copy(), y.copy()
    _causal_anchor_residual_fit(y, anchor)
    np.testing.assert_array_equal(anchor, a_snap)
    np.testing.assert_array_equal(y, y_snap)


def test_causal_anchor_constant_base_stays_bounded():
    """Degenerate (zero-variance) base carries no slope info; alpha must stay a
    finite value in [0, 1] rather than NaN / blow up."""
    a = _causal_anchor_residual_fit(np.arange(50.0), np.full(50, 5.0))["alpha"]
    assert 0.0 <= a <= 1.0 and np.isfinite(a)


def test_causal_anchor_composite_name_recognised():
    """Causal anchor composite name recognised."""
    name = compose_target_name("y", "causal_anchor_residual", "anchor")
    assert name == "y-canchor-anchor"
    assert is_composite_target_name(name)


# ---------------------------------------------------------------------------
# biz_value helpers.
# ---------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ols."""
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Rmse."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _reconstruct(t, anchor_tr, feat_tr, y_tr, anchor_te, feat_te):
    """Fit transform -> forward -> fit a downstream OLS(T ~ residual feature) ->
    inverse on the holdout. The downstream sees ONLY the generalisable
    mean-reversion feature (NOT the anchor), so a bad alpha leaves anchor
    structure in T the downstream cannot recover and the inverse must extrapolate."""
    p = t.fit(y_tr, anchor_tr)
    T_tr = t.forward(y_tr, anchor_tr, p)
    coef = _ols(feat_tr, T_tr)
    T_hat_te = coef[0] * feat_te + coef[1]
    return t.inverse(T_hat_te, anchor_te, p), p


def _gen_mean_revert(n, seed):
    """Gen mean revert."""
    r = np.random.default_rng(seed)
    anchor = r.standard_normal(n) * 3.0
    mr = r.standard_normal(n)
    y = 0.6 * anchor + mr + r.standard_normal(n) * 0.5
    return anchor, mr, y


# ---------------------------------------------------------------------------
# biz_value: mean-reverting anchor -> beats diff.
# ---------------------------------------------------------------------------


def test_biz_val_causal_anchor_beats_diff_on_mean_reverting_anchor():
    """True y = 0.6*anchor + mean_revert + noise: the honest shrink is ~0.6, so
    diff (alpha=1) over-commits and leaves -0.4*anchor in its residual that the
    base-agnostic downstream cannot recover. causal_anchor's reconstruction RMSE
    must beat diff by a clear margin (measured ratio ~0.385; floor 0.55)."""
    ca = get_transform("causal_anchor_residual")
    diff = get_transform("diff")
    ca_rmses, diff_rmses, alphas = [], [], []
    for s in range(6):
        atr, mtr, ytr = _gen_mean_revert(4000, s)
        ate, mte, yte = _gen_mean_revert(4000, 100 + s)
        yhat_ca, p = _reconstruct(ca, atr, mtr, ytr, ate, mte)
        yhat_diff, _ = _reconstruct(diff, atr, mtr, ytr, ate, mte)
        ca_rmses.append(_rmse(yhat_ca, yte))
        diff_rmses.append(_rmse(yhat_diff, yte))
        alphas.append(p["alpha"])
    ca_rmse = float(np.median(ca_rmses))
    diff_rmse = float(np.median(diff_rmses))
    assert 0.45 <= np.median(alphas) <= 0.75, f"alpha {np.median(alphas):.3f} off ~0.6"
    assert ca_rmse <= diff_rmse * 0.55, f"causal RMSE {ca_rmse:.4f} should be <=0.55x diff RMSE {diff_rmse:.4f} (ratio {ca_rmse / diff_rmse:.3f})"


# ---------------------------------------------------------------------------
# biz_value: group-disjoint OOD holdout -> bounded, unlike large-alpha linear.
# ---------------------------------------------------------------------------


def _gen_train_leverage_outliers(n, seed):
    """Train group: clean mean-revert relation PLUS a few high-leverage outliers
    (anchor far out of range, steep slope) that drag an unconstrained OLS
    (linear_residual) toward a fragile large alpha."""
    r = np.random.default_rng(seed)
    anchor = r.uniform(-3.0, 3.0, n)
    mr = r.standard_normal(n)
    y = 0.6 * anchor + mr + r.standard_normal(n) * 0.5
    n_out = int(0.04 * n)
    idx = r.choice(n, n_out, replace=False)
    anchor[idx] = r.uniform(15.0, 25.0, n_out)
    y[idx] = 2.5 * anchor[idx] + r.standard_normal(n_out) * 2.0
    return anchor, mr, y


def _gen_ood_group(n, seed):
    """Disjoint holdout group whose anchor sits OUT of the main train range."""
    r = np.random.default_rng(seed)
    anchor = r.uniform(8.0, 12.0, n)
    mr = r.standard_normal(n)
    y = 0.6 * anchor + mr + r.standard_normal(n) * 0.5
    return anchor, mr, y


def test_biz_val_causal_anchor_bounded_on_ood_group_vs_linear_residual():
    """linear_residual's free OLS is dragged to alpha>1.5 by leverage outliers;
    on a group-disjoint holdout with the anchor out of range its inverse
    extrapolates and blows up. causal_anchor's alpha is clamped to <=1 so the
    additive inverse stays bounded -> far lower OOD RMSE (measured ratio ~0.32;
    floor 0.6)."""
    ca = get_transform("causal_anchor_residual")
    lin = get_transform("linear_residual")
    ca_rmses, lin_rmses, ca_alphas, lin_alphas = [], [], [], []
    for s in range(6):
        atr, mtr, ytr = _gen_train_leverage_outliers(3000, s)
        ate, mte, yte = _gen_ood_group(2000, 100 + s)
        yhat_ca, pca = _reconstruct(ca, atr, mtr, ytr, ate, mte)
        yhat_lin, plin = _reconstruct(lin, atr, mtr, ytr, ate, mte)
        ca_rmses.append(_rmse(yhat_ca, yte))
        lin_rmses.append(_rmse(yhat_lin, yte))
        ca_alphas.append(pca["alpha"])
        lin_alphas.append(plin["alpha"])
    ca_rmse = float(np.median(ca_rmses))
    lin_rmse = float(np.median(lin_rmses))
    # The clamp holds and linear went fragile-large.
    assert np.median(ca_alphas) <= 1.0
    assert np.median(lin_alphas) > 1.5, f"linear alpha {np.median(lin_alphas):.3f} expected fragile-large (>1.5)"
    assert ca_rmse <= lin_rmse * 0.6, f"causal OOD RMSE {ca_rmse:.3f} should be <=0.6x linear {lin_rmse:.3f} (ratio {ca_rmse / lin_rmse:.3f})"


# ---------------------------------------------------------------------------
# biz_value: strong-AR anchor (true alpha~1) -> ties diff (does not hurt).
# ---------------------------------------------------------------------------


def _gen_strong_ar(n, seed):
    """Gen strong ar."""
    r = np.random.default_rng(seed)
    anchor = r.standard_normal(n) * 3.0
    mr = r.standard_normal(n) * 0.3
    y = 1.0 * anchor + mr + r.standard_normal(n) * 0.3
    return anchor, mr, y


def test_biz_val_causal_anchor_ties_diff_on_strong_ar():
    """When the true relation is a strong AR (alpha~1), causal_anchor's clamp
    lets alpha reach ~1 so it reconstructs as well as diff (does not hurt) --
    the transform is therefore safe to add to any transform set (floor 1.05)."""
    ca = get_transform("causal_anchor_residual")
    diff = get_transform("diff")
    ca_rmses, diff_rmses, alphas = [], [], []
    for s in range(6):
        atr, mtr, ytr = _gen_strong_ar(4000, s)
        ate, mte, yte = _gen_strong_ar(4000, 100 + s)
        yhat_ca, p = _reconstruct(ca, atr, mtr, ytr, ate, mte)
        yhat_diff, _ = _reconstruct(diff, atr, mtr, ytr, ate, mte)
        ca_rmses.append(_rmse(yhat_ca, yte))
        diff_rmses.append(_rmse(yhat_diff, yte))
        alphas.append(p["alpha"])
    ca_rmse = float(np.median(ca_rmses))
    diff_rmse = float(np.median(diff_rmses))
    assert np.median(alphas) >= 0.9, f"alpha {np.median(alphas):.3f} should reach ~1 on strong AR"
    assert ca_rmse <= diff_rmse * 1.05, f"causal RMSE {ca_rmse:.4f} must not hurt vs diff {diff_rmse:.4f} (ratio {ca_rmse / diff_rmse:.3f})"
