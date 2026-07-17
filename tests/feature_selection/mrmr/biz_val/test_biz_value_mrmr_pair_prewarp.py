"""biz_value: per-operand PRE-WARP recovery through the unary/binary pair path.

Companion to ``test_biz_value_mrmr_pre_distortion.py``. That file pins the
TRUE-NEGATIVE result that the elementary unary/binary pair search cannot recover
the non-monotone pre-distortion ``y = (a**3-2a)*(b**2-b)`` (``F-POLY``) -- no
single library unary equals ``a**3-2a``, so the search engineers nothing
(corr ~0). Recovery there was the orthogonal-poly (smart_polynom) path's job.

This file pins the 2026-06-02 capability that gives the unary/binary path the
SAME recovery via an optional PER-OPERAND learned 1-D pre-warp
(``fe_pair_prewarp_enable=True``). Before the binary op combines the operands,
each operand is optionally transformed by a learned 1-D orthogonal-polynomial
warp ``f(x)`` fit JOINTLY across the pair by a rank-1 ALS sweep
(``hermite_fe.fit_pair_prewarp_als`` -- the shared sibling of the orthogonal-poly
path's ``warm_start_als_seed``). ``binary(prewarp(a), prewarp(b))`` is then
searched/scored by MI with the target as usual. An independent 1-D fit cannot
recover the b-side of a product target (its b-marginal is ~0); the joint ALS
recovers both factors.

The pre-warp is GATED: its winner is admitted past the joint-MI-prevalence gate
only when it beats the best NON-prewarp engineered MI by
``fe_pair_prewarp_uplift_threshold`` (default 1.20x). This keeps it DIRECTED
(fires only where the warp adds representational power) and NOISE-SAFE (on
monotone / linear / pure-noise data the warp does not beat the elementary
library, so no spurious feature is fabricated).

Falsifiable pins (all measured n=4000):
* F-POLY WITH prewarp recovers (corr ~0.97, downstream Ridge R^2 ~ true), WITHOUT
  prewarp does not (corr ~0) -- the prewarp is the lever.
* F-MONO is byte-unchanged (the monotone inner is already representable; the
  elementary library wins and the prewarp adds no uplift).
* Pure-noise -> NO engineered feature (ON == OFF). Pure-linear -> no spurious
  prewarp feature; downstream not worse than raw.
* Replay: ``transform()`` on held-out rows reproduces the engineered column
  deterministically from X alone (no y); |corr| to fit-time values ~1.0.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR

N = 4000
RAW = {"a", "b", "c", "e"}
_LEAN = dict(dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)


# ---------------------------------------------------------------------------
# Fixtures (distinct seeds; the process-wide fit cache is cleared per fit).
# ---------------------------------------------------------------------------
def _make_poly(seed: int = 202, n: int = N):
    """F-POLY fixture: y = (a**3-2a)*(b**2-b), a non-monotone polynomial inner distortion."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = (a**3 - 2 * a) * (b**2 - b)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _make_mono(seed: int = 101, n: int = N):
    """F-MONO fixture: y = exp(a)*log(b), a monotone inner distortion MI is invariant to."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.2, 2.0, n)
    b = rng.uniform(1.2, 5.0, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = np.exp(a) * np.log(b)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _make_linear(seed: int = 606, n: int = N):
    """Negative-control fixture: y = a+b, a purely linear target the elementary library already covers."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = a + b
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _make_noise(seed: int = 404, n: int = N):
    """Negative-control fixture: y is pure noise, independent of every feature."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), None


# ---------------------------------------------------------------------------
# Makers / helpers.
# ---------------------------------------------------------------------------
def _unb(prewarp: bool):
    """UNB pair path with the orthogonal-poly + hybrid paths DISABLED, so the
    only thing under test is the elementary unary/binary search with or without
    the per-operand pre-warp."""
    return MRMR(verbose=0, n_jobs=1, random_seed=0, fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False, fe_pair_prewarp_enable=prewarp, **_LEAN)


def _fit(make, df, y):
    """Fit a fresh MRMR with the process-wide fit cache cleared first, so a
    prior fit on the same (X, y) cannot alias its result onto this config."""
    MRMR.clear_fit_cache()
    fs = make()
    fs.fit(df, y)
    return fs


@cache
def _poly_data():
    """Cached ``(df, y, true)`` for the default-seeded F-POLY fixture; every
    caller uses the default seed so this is a single deterministic triple.
    """
    return _make_poly()


@cache
def _poly_unb_fit(prewarp: bool):
    """Cached fitted MRMR for ``_unb(prewarp=prewarp)`` on the (shared, cached)
    F-POLY fixture. 4 differently-named tests fit prewarp=True and 2 fit
    prewarp=False on the identical default-seeded data to check different
    assertions; this collapses each such group to a single ``MRMR.fit`` call.
    Nothing downstream mutates the fitted estimator in place.
    """
    df, y, _true = _poly_data()
    return _fit(lambda: _unb(prewarp=prewarp), df, y)


def _eng_names(fs):
    """Return the fitted MRMR's selected column names that are NOT one of the raw input columns."""
    return [nm for nm in fs.get_feature_names_out() if nm not in RAW]


def _best_engineered_corr(fs, df, true):
    """(name, |pearson|) of the engineered column most correlated with the true pre-distorted signal."""
    names = list(fs.get_feature_names_out())
    eng = [nm for nm in names if nm not in RAW]
    if not eng or true is None:
        return None, 0.0
    Xt = np.asarray(fs.transform(df))
    best = (None, 0.0)
    for i, nm in enumerate(names):
        if nm not in eng:
            continue
        col = Xt[:, i]
        if not np.isfinite(col).all() or float(np.std(col)) < 1e-12:
            continue
        r = abs(float(np.corrcoef(col, true)[0, 1]))
        if r > best[1]:
            best = (nm, r)
    return best


def _ridge_r2(X, y):
    """5-fold standardized Ridge R^2; nan for a 0-column X."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, KFold

    if np.ndim(X) == 1:
        X = np.asarray(X).reshape(-1, 1)
    if X.shape[1] == 0:
        return float("nan")
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    return float(np.mean(cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, np.asarray(y, dtype=float), cv=cv, scoring="r2")))


# ---------------------------------------------------------------------------
# UNIT: the shared pre-warp helpers (joint ALS vs independent 1-D fit).
# ---------------------------------------------------------------------------
def test_prewarp_helpers_joint_als_beats_independent_on_product_target():
    """The joint ALS pre-warp recovers BOTH operand factors of a product target;
    the independent 1-D fit recovers the a-side but NOT the b-side (its
    b-marginal is ~0). This is exactly why the pair search uses the joint ALS.
    Also pins that ``apply_operand_prewarp`` replays the fitted spec closed-form
    (corr 1.0 to the fit-time warp) from x alone."""
    from mlframe.feature_selection.filters.hermite_fe import (
        apply_operand_prewarp,
        fit_operand_prewarp,
        fit_pair_prewarp_als,
    )

    rng = np.random.default_rng(202)
    n = N
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    P = a**3 - 2 * a
    Q = b**2 - b
    y = P * Q  # centred-ish product; the inner factors are the targets to recover

    sa, sb = fit_pair_prewarp_als(a, b, y, basis="chebyshev", max_degree=4)
    assert sa is not None and sb is not None
    fa = apply_operand_prewarp(a, sa)
    fb = apply_operand_prewarp(b, sb)
    corr_a = abs(float(np.corrcoef(fa, P)[0, 1]))
    corr_b = abs(float(np.corrcoef(fb, Q)[0, 1]))
    assert corr_a >= 0.90, f"joint ALS a-side corr {corr_a:.3f} < 0.90"
    assert corr_b >= 0.90, f"joint ALS b-side corr {corr_b:.3f} < 0.90"

    # Independent 1-D fit recovers the b-side far WORSE (its b-marginal ~0;
    # measured ~0.54 vs the joint ~1.0). Pin the joint beats it by a wide margin.
    sb_indep = fit_operand_prewarp(b, y, basis="chebyshev", max_degree=4)
    fb_indep = apply_operand_prewarp(b, sb_indep)
    corr_b_indep = abs(float(np.corrcoef(fb_indep, Q)[0, 1]))
    assert corr_b >= corr_b_indep + 0.30, (
        f"joint ALS b-side ({corr_b:.3f}) did not beat the independent 1-D fit "
        f"({corr_b_indep:.3f}) by the expected margin; the joint-ALS justification "
        f"would be undermined"
    )

    # Closed-form replay parity (no y): apply twice -> identical.
    np.testing.assert_allclose(apply_operand_prewarp(a, sa), fa, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# RECOVERY: F-POLY through the unary/binary path WITH prewarp.
# ---------------------------------------------------------------------------
def test_fpoly_unary_binary_with_prewarp_recovers():
    """The per-operand pre-warp lets the elementary unary/binary path engineer a
    feature highly correlated with the true non-monotone ``P(a)*Q(b)`` signal.
    Measured |corr| ~= 0.97 (was ~0 without the prewarp). Pinned >= 0.70."""
    df, _y, true = _poly_data()
    fs = _poly_unb_fit(True)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, (
        "F-POLY/UNB+prewarp engineered nothing; the per-operand pre-warp is expected to recover the non-monotone inner via the unary/binary path"
    )
    assert "prewarp" in name, (
        f"F-POLY recovery used '{name}' which does NOT involve the prewarp pseudo-unary; the recovery should be attributable to the prewarp"
    )
    assert corr >= 0.70, f"F-POLY/UNB+prewarp best engineered |corr|={corr:.3f} < 0.70 ({name}); the per-operand pre-warp recovery regressed"


def test_fpoly_prewarp_is_the_lever_vs_no_prewarp_control():
    """MECHANISM PROOF: the per-operand prewarp is the LEVER for F-POLY recovery --
    enabling it lifts the best engineered |corr| by a large margin over the same
    unary/binary path without it. Single-knob A/B on ``fe_pair_prewarp_enable``.

    The control is framed as a MARGIN, not an absolute floor: the elementary
    unary/binary library + escalation can PARTIALLY recover the non-monotone inner
    via a relu threshold split / escalation residual (the OFF |corr| ranges from a
    relu piecewise ~0.49 up to ~0.74 once the auto-escalation path partially
    reconstructs the quadratic), and which partial feature wins is sensitive to the
    numba-warmed search order across the process (so the OFF |corr| drifts within
    that band depending on what ran before). The genuine, order-robust claim is that
    the prewarp lands a NEAR-EXACT reconstruction (|corr| ~1.0, restored after the
    2026-06-11 continuous-y ALS-target fix; was ~0.97 before the target-rebin guard)
    that beats the best the library can do WITHOUT it by a clear margin -- that
    margin is the lever proof, and it survives the control's higher partial-recovery
    ceiling."""
    df, _y, true = _poly_data()
    fs_on = _poly_unb_fit(True)
    fs_off = _poly_unb_fit(False)
    n_on, corr_on = _best_engineered_corr(fs_on, df, true)
    _n_off, corr_off = _best_engineered_corr(fs_off, df, true)
    # Prewarp ON must reach a near-exact reconstruction AND use the prewarp pseudo-
    # unary -- this is what makes it the lever (not a generic library unary).
    assert corr_on >= 0.85, f"F-POLY/UNB+prewarp best engineered |corr|={corr_on:.3f} < 0.85 ({n_on}); the per-operand prewarp recovery regressed"
    assert n_on is not None and "prewarp" in n_on, (
        f"F-POLY recovery used '{n_on}' which does NOT involve the prewarp pseudo-unary; the recovery should be attributable to the prewarp"
    )
    # The lever margin: prewarp ON beats the best the library does WITHOUT it. A
    # 0.20 margin tolerates the control's partial-recovery ceiling (OFF reaches
    # ~0.74 once escalation partially reconstructs the quadratic) while still failing
    # if the prewarp stops being the decisive lever. ON is now a near-exact ~1.0
    # reconstruction, so the lever gap stays comfortably above this bar.
    assert corr_on >= corr_off + 0.20, (
        f"prewarp ON (corr={corr_on:.3f}) did not beat prewarp OFF "
        f"(corr={corr_off:.3f}) by the expected margin; the prewarp-is-the-lever "
        f"hypothesis would be falsified"
    )


def test_fpoly_downstream_score_recovers_with_prewarp():
    """END-TO-END: the prewarp selection lifts downstream 5-fold Ridge R^2 from
    the all-raw baseline (~0.22; raw cannot linearly express P(a)*Q(b)) to near
    the true-signal fit (~1.0). Pinned within 0.10 of the true-signal R^2 AND a
    large lift over raw. The no-prewarp control stays stuck at the raw baseline."""
    df, y, true = _poly_data()
    raw_r2 = _ridge_r2(df.values, y)
    true_r2 = _ridge_r2(np.asarray(true), y)
    assert true_r2 > 0.95, f"sanity: true-signal Ridge R^2={true_r2:.3f} should be ~1.0"
    assert raw_r2 < 0.5, f"sanity: all-raw Ridge R^2={raw_r2:.3f} should be low"

    fs_on = _poly_unb_fit(True)
    sel_r2 = _ridge_r2(np.asarray(fs_on.transform(df)), y)
    assert sel_r2 >= true_r2 - 0.10, (
        f"F-POLY/UNB+prewarp downstream R^2={sel_r2:.3f} not within 0.10 of the "
        f"true-signal R^2 {true_r2:.3f}; the engineered feature did not deliver "
        f"the predictive lift"
    )
    assert sel_r2 >= raw_r2 + 0.40, f"F-POLY/UNB+prewarp downstream R^2={sel_r2:.3f} did not materially beat the all-raw baseline {raw_r2:.3f}"

    fs_off = _poly_unb_fit(False)
    sel_r2_off = _ridge_r2(np.asarray(fs_off.transform(df)), y)
    # The no-prewarp control can PARTIALLY lift R^2 above raw via a relu-threshold
    # unary (a piecewise approximation of the quadratic inner; ~0.43 depending on the
    # numba-warmed search order across the process), so the order-robust claim is the
    # MARGIN -- the prewarp lift reaches near the true-signal fit and clears the best
    # the library does without it by a wide margin -- NOT that the control is pinned
    # at raw. ``sel_r2`` (prewarp ON) is already asserted within 0.10 of true_r2 above.
    assert sel_r2 >= sel_r2_off + 0.30, (
        f"F-POLY/UNB+prewarp downstream R^2={sel_r2:.3f} did not beat the no-prewarp "
        f"control R^2={sel_r2_off:.3f} by the expected margin; the prewarp is no "
        f"longer the decisive downstream lever"
    )


# ---------------------------------------------------------------------------
# F-MONO control: prewarp does NOT disturb the already-recoverable monotone case.
# ---------------------------------------------------------------------------
def test_fmono_prewarp_does_not_disturb_monotone_recovery():
    """The monotone inner (exp(a)*log(b)) is already recoverable by the
    elementary unary/binary library, which the prewarp does not beat by the
    uplift margin -- so enabling the prewarp leaves the monotone recovery intact
    (still |corr| >= 0.80) and does not flood the selection with prewarp cols."""
    df, y, true = _make_mono()
    fs = _fit(lambda: _unb(prewarp=True), df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-MONO/UNB+prewarp engineered nothing (must still recover)"
    assert corr >= 0.80, f"F-MONO/UNB+prewarp best engineered |corr|={corr:.3f} < 0.80 ({name}); the prewarp disturbed the monotone recovery"


# ---------------------------------------------------------------------------
# NEGATIVE controls: the prewarp must not fabricate signal.
# ---------------------------------------------------------------------------
def test_noise_control_prewarp_engineers_nothing():
    """REGRESSION GUARD: on a target INDEPENDENT of (a, b), the unary/binary path
    WITH prewarp must engineer NO feature -- the prewarp fit always produces SOME
    polynomial, but its engineered MI cannot beat the elementary library by the
    uplift margin on noise, so the alternative-acceptance gate rejects it. ON and
    OFF must both be empty."""
    df, y, _ = _make_noise()
    fs_on = _fit(lambda: _unb(prewarp=True), df, y)
    fs_off = _fit(lambda: _unb(prewarp=False), df, y)
    eng_on = _eng_names(fs_on)
    eng_off = _eng_names(fs_off)
    assert not eng_on, f"noise control fabricated engineered feature(s) {eng_on} with prewarp ON; the uplift gate let a spurious prewarp feature through"
    assert not eng_off, f"noise control fabricated engineered feature(s) {eng_off} with prewarp OFF; the elementary library overfit pure noise"
    # Downstream stays at the (near-zero) noise baseline.
    raw_r2 = _ridge_r2(df.values, y)
    sel_r2 = _ridge_r2(np.asarray(fs_on.transform(df)), y)
    assert sel_r2 <= raw_r2 + 0.10, (
        f"noise control downstream R^2={sel_r2:.3f} beats the raw noise baseline {raw_r2:.3f}: a spurious prewarp feature is leaking signal"
    )


def test_linear_control_prewarp_adds_no_spurious_overfit_feature():
    """On a purely-LINEAR target y=a+b the prewarp must not add a spurious
    over-fit engineered feature beyond what the raw path recovers: enabling the
    prewarp must not engineer a *prewarp* column, and downstream Ridge R^2 must
    not be worse than the no-prewarp path."""
    df, y, _true = _make_linear()
    fs_on = _fit(lambda: _unb(prewarp=True), df, y)
    fs_off = _fit(lambda: _unb(prewarp=False), df, y)
    eng_on = _eng_names(fs_on)
    prewarp_cols = [nm for nm in eng_on if "prewarp" in nm]
    assert not prewarp_cols, (
        f"linear control fabricated prewarp feature(s) {prewarp_cols}; the prewarp over-fit a linear target the elementary library already covers"
    )
    r2_on = _ridge_r2(np.asarray(fs_on.transform(df)), y)
    r2_off = _ridge_r2(np.asarray(fs_off.transform(df)), y)
    assert r2_on >= r2_off - 0.05, (
        f"linear control downstream R^2 with prewarp ON ({r2_on:.3f}) is worse than OFF ({r2_off:.3f}); the prewarp degraded a linear target"
    )


# ---------------------------------------------------------------------------
# Replay determinism: transform() on held-out rows reproduces the column from X
# alone (no y), bit-deterministically.
# ---------------------------------------------------------------------------
def test_prewarp_recipe_replay_is_deterministic_and_leak_free():
    """The prewarp engineered column replays deterministically at transform()
    time from X alone: re-transforming the SAME held-out rows twice is identical,
    and the held-out engineered values correlate ~1.0 with the recipe applied to
    the same rows (no y is consulted at replay -- the fitted coeffs live in the
    EngineeredRecipe)."""
    _df, _y, _true = _poly_data()
    fs = _poly_unb_fit(True)
    eng = _eng_names(fs)
    assert eng, "no engineered feature to replay"

    # Held-out rows from the SAME distribution (the recipe must replay on unseen
    # rows without re-fitting / re-quantiling and without y).
    df_test, _y_test, _true_test = _make_poly(seed=909)
    names = list(fs.get_feature_names_out())
    eng_idx = [i for i, nm in enumerate(names) if nm in eng]

    Xt1 = np.asarray(fs.transform(df_test))
    Xt2 = np.asarray(fs.transform(df_test))
    for i in eng_idx:
        np.testing.assert_allclose(Xt1[:, i], Xt2[:, i], rtol=0, atol=0, err_msg=f"replay of '{names[i]}' is non-deterministic")
        col = Xt1[:, i]
        assert np.isfinite(col).all(), f"replayed '{names[i]}' has non-finite values"
        assert float(np.std(col)) > 1e-9, f"replayed '{names[i]}' is constant"

    # Direct recipe-apply parity: the stored recipe reproduces the same column.
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    recipes = {r.name: r for r in fs._engineered_recipes_}
    for i in eng_idx:
        nm = names[i]
        assert nm in recipes, f"no recipe stored for engineered column '{nm}'"
        direct = np.asarray(apply_recipe(recipes[nm], df_test)).reshape(-1)
        r = abs(float(np.corrcoef(direct, Xt1[:, i])[0, 1]))
        assert r > 0.999, f"recipe-apply replay of '{nm}' diverges from transform() output (|corr|={r:.4f}); replay is not deterministic / leak-free"
