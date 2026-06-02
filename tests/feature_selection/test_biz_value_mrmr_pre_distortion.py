"""biz_value: pre-distortion pair-FE recovery.

Pins the empirically measured answer to the question "if the true dependency
strongly distorts each input BEFORE the nonlinear pair-interaction, does
pair-Feature-Engineering still recover it?".

Three fixtures, each ``y = bin( f_a(a) , f_b(b) )`` where ``f_a`` / ``f_b`` is an
INNER per-operand distortion applied before the pair interaction:

* ``F-MONO``  -- monotone inner (``exp(a)`` * ``log(b)``). MI is invariant to
  monotone transforms, so the signal is NOT hidden: even the elementary
  unary/binary pair search recovers it. (Control / sanity.)
* ``F-POLY``  -- NON-monotone polynomial inner (``P(a)=a**3-2a``, ``Q(b)=b**2-b``,
  ``y = P(a)*Q(b)``). No single library unary equals ``a**3-2a``, so the
  unary/binary path cannot represent it; an orthogonal-poly basis CAN (a
  degree<=4 separable Chebyshev/Hermite reconstruction fits P, Q exactly). The
  HARD case -- and now the RECOVERED case at default settings (see below).
* ``F-OSC``   -- oscillatory inner (``y = sin(a**2)*b``). High-curvature.

MEASURED VERDICT (n=4000, default-style budget fe_smart_polynom_iters=15..20):

  fixture  UNB(func-search)   ORTH(smart_polynom, cma_batch DEFAULT)  hybrid-orth-pair
  F-MONO   corr 0.93  PASS    corr 0.94  PASS                         corr 0.93 PASS
  F-POLY   corr ~0    FAIL*   corr 0.97  PASS  (sel R^2 0.95)         eng=0    FAIL*
  F-OSC    corr ~0    FAIL*   corr 0.95  PASS  (degree-6 default)     corr 0.67 PARTIAL

  (* the F-POLY/UNB and F-POLY/hybrid cells, and F-OSC/UNB, are TRUE-NEGATIVE
  pins: those paths structurally cannot represent the non-monotone inner, so
  they correctly engineer nothing. Only the smart_polynom ORTH path is expected
  to recover.)

THE FIX (2026-06-02) that turned the F-POLY smart_polynom cell from the
historical FAIL (corr ~0.003, eng=0) into corr 0.97 / downstream Ridge R^2 0.95:

  1. SCALE-SATURATING coefficient penalty. The old raw ``fe_hermite_l2_penalty
     * ||c||^2`` grew without bound; the true separable Chebyshev coefficients
     of ``P(a)*Q(b)`` have joint ``||c||^2 ~ 86`` so the raw penalty ~4.3
     dwarfed the MI peak ~1.5 and the optimiser fled to a small-||c||
     atan2/div plateau. The penalty now saturates toward a constant
     ``lambda`` ceiling (``hermite_fe._l2_penalty_value``), so high-MI /
     high-coefficient solutions are no longer crushed while pure noise still
     pays ~full ``lambda``.
  2. Per-operand ALS WARM START. A rank-1 alternating-least-squares fit of
     ``y ~ f(x_a)*g(x_b)`` in the basis (``hermite_fe.warm_start_als_seed``)
     seeds the joint optimiser DIRECTLY in the true large-coefficient basin --
     three cheap ``lstsq`` solves recover both factors at corr ~1.0 on F-POLY.
     This is the lever that lets the DEFAULT ``cma_batch`` optimiser escape the
     plateau (the ``test_fpoly_recovery_is_real_and_warm_start_is_the_lever``
     control flips it back to corr ~0.003 by disabling the warm start).
  3. ``fe_max_polynom_degree`` default 8 -> 6, trimming the joint search from
     ~18 to 14 dims; degree 6 still recovers BOTH F-POLY and the higher-
     frequency F-OSC inner.

The ideal feature ``mul(cheb_3(a), cheb_2(b))`` scores plug-in MI ~1.5 and
CLEARS the ``fe_min_engineered_mi_prevalence`` gate (0.9 * max-marginal ~0.54):
capacity and gate were never the problem -- the optimiser+penalty were. The
F-MONO control, the recovered F-POLY/F-OSC smart_polynom wins, the F-OSC
hybrid-orth partial win, the true-negative pins, and a pure-noise negative
control (the saturating penalty must NOT fabricate an engineered feature on
``y`` independent of ``a, b``) are ALL pinned, every one falsifiable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR

N = 4000
RAW = {"a", "b", "c", "e"}
# Turn OFF the redundancy-control extras unrelated to the pair-FE engine under
# test (DCD / friend-graph / cluster-aggregate). Keeps each fit a few seconds.
_LEAN = dict(dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)


# ---------------------------------------------------------------------------
# Fixtures (deterministic; distinct seeds so the process-wide MRMR fit cache
# cannot alias one fixture's result onto another).
# ---------------------------------------------------------------------------
def _make_mono(seed: int = 101, n: int = N):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.2, 2.0, n)   # exp(a) monotone
    b = rng.uniform(1.2, 5.0, n)   # log(b) monotone, positive
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = np.exp(a) * np.log(b)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _make_poly(seed: int = 202, n: int = N):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = (a**3 - 2 * a) * (b**2 - b)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _make_osc(seed: int = 303, n: int = N):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = np.sin(a**2) * b
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _make_noise(seed: int = 404, n: int = N):
    """Negative control: y is pure noise, INDEPENDENT of every feature. Same
    feature layout (a, b same support as F-POLY) so the smart_polynom path
    considers the (a, b) pair, but there is no signal to engineer. Guards
    against the saturating-penalty relaxation fabricating a spurious engineered
    feature (overfitting noise)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # independent of a, b, c, e
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _best_engineered_corr(fs: MRMR, df: pd.DataFrame, true: np.ndarray):
    """(name, |pearson|) of the engineered column most correlated with the true
    pre-distorted signal, or (None, 0.0) if MRMR engineered nothing usable."""
    names = list(fs.get_feature_names_out())
    eng = [nm for nm in names if nm not in RAW]
    if not eng:
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


def _fit(make_mrmr, df, y):
    """Fit a fresh MRMR with the process-wide fit cache cleared first, so a
    prior fit on the same (X, y) cannot alias its result onto this config."""
    MRMR.clear_fit_cache()
    fs = make_mrmr()
    fs.fit(df, y)
    return fs


def _unb():
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False, **_LEAN)


def _orth_smart_polynom(basis="chebyshev"):
    # Default optimiser is cma_batch (the production default).
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=20, fe_smart_polynom_optimization_steps=400,
                fe_polynomial_basis=basis, fe_optimizer="cma_batch",
                fe_hybrid_orth_enable=False, **_LEAN)


def _orth_hybrid_pair():
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=True,
                fe_hybrid_orth_pair_enable=True, fe_hybrid_orth_degrees=(2, 3, 4),
                fe_hybrid_orth_pair_max_degree=4, fe_hybrid_orth_top_k=8, **_LEAN)


# ---------------------------------------------------------------------------
# F-MONO control: monotone inner -> BOTH paths recover (MI-invariance).
# ---------------------------------------------------------------------------
def test_fmono_control_unary_binary_recovers():
    """Monotone inner distortion hides nothing: the elementary unary/binary
    pair search engineers a feature highly correlated with exp(a)*log(b).
    Measured |corr| ~= 0.93; pinned >= 0.80 with margin."""
    df, y, true = _make_mono()
    fs = _fit(_unb, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-MONO/UNB engineered nothing (control must recover)"
    assert corr >= 0.80, f"F-MONO/UNB best engineered |corr|={corr:.3f} < 0.80 ({name})"


def test_fmono_control_orth_polynom_recovers():
    """The orthogonal-poly (smart_polynom) path also recovers the monotone
    case -- sanity that ORTH is not broken in general."""
    df, y, true = _make_mono()
    fs = _fit(_orth_smart_polynom, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-MONO/ORTH engineered nothing (control must recover)"
    assert corr >= 0.80, f"F-MONO/ORTH best engineered |corr|={corr:.3f} < 0.80 ({name})"


# ---------------------------------------------------------------------------
# F-POLY boundary: non-monotone polynomial inner -> ALL paths fail at default
# settings. This is the falsifiable pin of the negative result.
# ---------------------------------------------------------------------------
def test_fpoly_unary_binary_fails_to_recover():
    """No single library unary equals a**3-2a, so the unary/binary pair search
    cannot represent P(a) and finds no feature correlated with P(a)*Q(b).
    Measured |corr| ~= 0; pinned < 0.30 (well below the 0.80 the method hits
    when the inner distortion is representable)."""
    df, y, true = _make_poly()
    fs = _fit(_unb, df, y)
    _name, corr = _best_engineered_corr(fs, df, true)
    assert corr < 0.30, (
        f"F-POLY/UNB unexpectedly recovered |corr|={corr:.3f} ({_name}); "
        f"the unary/binary path was expected to fail on the non-monotone inner"
    )


@pytest.mark.parametrize("basis", ["chebyshev", "hermite"])
def test_fpoly_orth_smart_polynom_default_recovers(basis):
    """RECOVERY: the orthogonal-poly smart_polynom path with the PRODUCTION
    defaults (``fe_optimizer='cma_batch'``, saturating ``fe_hermite_l2_penalty``,
    per-operand ALS warm start, ``fe_max_polynom_degree=6``) now recovers the
    non-monotone F-POLY pre-distortion: it engineers a ``mul``-combined feature
    highly correlated with the true ``P(a)*Q(b)`` signal.

    Historical state (pre-fix, raw L2 penalty + no warm start): corr ~0.003,
    eng=0. Measured post-fix: |corr| ~= 0.97 (chebyshev AND hermite). Pinned
    >= 0.70 with margin -- the bar from the recovery contract.

    The win is REAL (not just "a feature exists"): it is verified against the
    true pre-distorted signal by Pearson here and by downstream Ridge R^2 in
    ``test_fpoly_downstream_score_recovers_for_smart_polynom``. The
    ``test_fpoly_recovery_is_real_and_warm_start_is_the_lever`` control proves
    disabling the ALS warm start sends it back to corr ~0."""
    df, y, true = _make_poly()
    fs = _fit(lambda: _orth_smart_polynom(basis=basis), df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, (
        f"F-POLY/ORTH-{basis} engineered nothing; the smart_polynom path is "
        f"expected to recover the non-monotone inner at default settings"
    )
    assert corr >= 0.70, (
        f"F-POLY/ORTH-{basis} (cma_batch default) best engineered |corr|="
        f"{corr:.3f} < 0.70 ({name}); the warm-start + saturating-penalty "
        f"recovery regressed"
    )


def test_fpoly_orth_hybrid_pair_default_does_not_recover():
    """TRUE-NEGATIVE pin: the hybrid orthogonal-pair path (fixed low-degree
    cross-basis cells like ``T2(a)*T1(b)``, a SEPARATE code path from the
    smart_polynom CMA/ALS optimiser) does not recover F-POLY -- its fixed
    bilinear cells cannot express the full ``cheb_3(a)*cheb_2(b)`` product. This
    path was NOT changed by the 2026-06-02 warm-start fix; pinned < 0.30 so a
    future accidental coupling that made it spuriously "recover" is caught.

    Recovery on F-POLY is the job of the smart_polynom path
    (``test_fpoly_orth_smart_polynom_default_recovers``), not this one."""
    df, y, true = _make_poly()
    fs = _fit(_orth_hybrid_pair, df, y)
    _name, corr = _best_engineered_corr(fs, df, true)
    assert corr < 0.30, (
        f"F-POLY/hybrid-orth-pair unexpectedly recovered |corr|={corr:.3f} ({_name})"
    )


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
    return float(np.mean(cross_val_score(
        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        X, np.asarray(y, dtype=float), cv=cv, scoring="r2")))


def test_fpoly_downstream_score_recovers_for_smart_polynom():
    """END-TO-END RECOVERY: the smart_polynom default selection lifts downstream
    5-fold Ridge R^2 from the all-raw baseline (~0.22, raw cannot linearly
    express ``P(a)*Q(b)``) to near the true-signal fit (~1.0). Measured sel R^2
    ~= 0.95. Pinned within 0.10 of the true-signal R^2 AND a large lift over raw
    -- the downstream half of the recovery contract (so "an engineered feature
    exists" is not enough; it must actually be USABLE for prediction)."""
    df, y, true = _make_poly()
    raw_r2 = _ridge_r2(df.values, y)
    true_r2 = _ridge_r2(np.asarray(true), y)
    assert true_r2 > 0.95, f"sanity: true-signal Ridge R^2={true_r2:.3f} should be ~1.0"
    assert raw_r2 < 0.5, f"sanity: all-raw Ridge R^2={raw_r2:.3f} should be low"

    fs = _fit(_orth_smart_polynom, df, y)
    sel_r2 = _ridge_r2(np.asarray(fs.transform(df)), y)
    assert sel_r2 >= true_r2 - 0.10, (
        f"F-POLY/ORTH-cma downstream R^2={sel_r2:.3f} is not within 0.10 of the "
        f"true-signal R^2 {true_r2:.3f}; the engineered feature did not deliver "
        f"the predictive lift"
    )
    assert sel_r2 >= raw_r2 + 0.40, (
        f"F-POLY/ORTH-cma downstream R^2={sel_r2:.3f} did not materially beat the "
        f"all-raw baseline {raw_r2:.3f}; recovery regressed"
    )


def test_fpoly_downstream_score_stuck_at_raw_baseline_for_true_negative_paths():
    """TRUE-NEGATIVE pin: the paths that structurally cannot represent the
    non-monotone inner -- unary/binary function search and the fixed-cell
    hybrid-orth pair -- leave downstream 5-fold Ridge R^2 at the all-raw
    baseline (~0.22), far below the true-signal fit (~1.0). Pins the business
    cost of those paths' representational limit (recovery is the smart_polynom
    path's job, asserted separately)."""
    df, y, true = _make_poly()
    raw_r2 = _ridge_r2(df.values, y)
    true_r2 = _ridge_r2(np.asarray(true), y)
    assert true_r2 > 0.95, f"sanity: true-signal Ridge R^2={true_r2:.3f} should be ~1.0"
    assert raw_r2 < 0.5, f"sanity: all-raw Ridge R^2={raw_r2:.3f} should be low"

    for label, maker in [("UNB", _unb), ("hybrid-orth", _orth_hybrid_pair)]:
        fs = _fit(maker, df, y)
        sel_r2 = _ridge_r2(np.asarray(fs.transform(df)), y)
        # Margin 0.10 absorbs an occasional weak-but-useless engineered column.
        assert sel_r2 <= raw_r2 + 0.10, (
            f"F-POLY/{label} downstream R^2={sel_r2:.3f} materially beats raw "
            f"baseline {raw_r2:.3f}: a true-negative path now rescues F-POLY -- "
            f"if intended, move it to the recovery assertions."
        )
        assert sel_r2 < true_r2 - 0.4, (
            f"F-POLY/{label} downstream R^2={sel_r2:.3f} approached the "
            f"true-signal R^2 {true_r2:.3f}; this path was expected to fail."
        )


def test_fpoly_recovery_is_real_and_warm_start_is_the_lever():
    """MECHANISM PROOF: the F-POLY recovery is REAL and the per-operand ALS
    warm start is the lever (NOT basis capacity, NOT the injection gate).

    (1) The ideal feature mul(cheb_3(a), cheb_2(b)) -- the exact separable
        Chebyshev reconstruction of P(a)*Q(b) -- scores a high plug-in MI and
        CLEARS the fe_min_engineered_mi_prevalence gate (0.9 * max-marginal MI),
        and its coefficients are large (joint ||c||^2 > 20). Capacity and gate
        were never the blocker.
    (2) The DEFAULT optimiser (cma_batch + saturating L2 penalty) WITH the ALS
        warm start recovers a high correlation to the true signal; turning the
        warm start OFF (canonical seeds only) on the SAME basis/degree/budget
        collapses it back toward zero. That single-knob A/B isolates the warm
        start as the cause -- the recovery is not luck and not the penalty
        alone.
    """
    pytest.importorskip("cma")
    from numpy.polynomial import chebyshev as C
    from mlframe.feature_selection.filters.hermite_fe import (
        optimise_hermite_pair, _POLY_BASES, _plugin_mi_regression_njit,
    )

    df, y, true = _make_poly()
    a = df["a"].values
    b = df["b"].values
    yf = np.ascontiguousarray(np.asarray(y, dtype=np.float64))

    bi = _POLY_BASES["chebyshev"]
    za, _pa = bi["fit"](a)
    zb, _pb = bi["fit"](b)
    za = np.ascontiguousarray(za, dtype=np.float64)
    zb = np.ascontiguousarray(zb, dtype=np.float64)

    # (1) Ideal separable Chebyshev reconstruction + its plug-in MI vs the gate.
    ca = C.chebfit(za, a**3 - 2 * a, 3)
    cb = C.chebfit(zb, b**2 - b, 2)
    ideal = C.chebval(za, ca) * C.chebval(zb, cb)
    assert abs(float(np.corrcoef(ideal, true)[0, 1])) > 0.99, (
        "ideal Chebyshev reconstruction should match P(a)*Q(b) almost exactly"
    )
    mi_ideal = float(_plugin_mi_regression_njit(np.ascontiguousarray(ideal), yf, 20))
    mi_a = float(_plugin_mi_regression_njit(np.ascontiguousarray(a.astype(np.float64)), yf, 20))
    mi_b = float(_plugin_mi_regression_njit(np.ascontiguousarray(b.astype(np.float64)), yf, 20))
    gate = 0.90 * max(mi_a, mi_b)  # MRMR fe_min_engineered_mi_prevalence (default 0.90)
    assert mi_ideal > gate, (
        f"ideal feature MI={mi_ideal:.3f} does NOT clear the injection gate "
        f"{gate:.3f}; the boundary would then be the gate, not the optimiser"
    )
    # Large true coefficients are exactly why the OLD raw L2 penalty crushed the
    # solution; pin that the solution lives in the large-coef region.
    assert (np.sum(ca**2) + np.sum(cb**2)) > 20.0, (
        "true Chebyshev coefficients should be large (drives the penalty effect)"
    )

    # (2) Single-knob A/B on the per-operand ALS warm start. Everything else is
    # the production default (cma_batch + saturating penalty). max_degree=4 is
    # the natural degree of the cubic*quadratic product and keeps the control
    # fast.
    common = dict(discrete_target=False, max_degree=4, min_degree=1,
                  basis="chebyshev", coef_range=(-10.0, 10.0),
                  optimizer="cma_batch", mi_estimator="plugin",
                  multi_fidelity=False, baseline_uplift_threshold=1.01)

    def _corr_of(res):
        if res is None:
            return 0.0
        fa = C.chebval(za, res.coef_a)
        fb = C.chebval(zb, res.coef_b)
        comb = res.bin_func(fa, fb)
        if float(np.std(comb)) < 1e-12:
            return 0.0
        return abs(float(np.corrcoef(comb, true)[0, 1]))

    res_on = optimise_hermite_pair(a, b, np.asarray(y), n_trials=400,
                                   warm_start_als=True, **common)
    res_off = optimise_hermite_pair(a, b, np.asarray(y), n_trials=400,
                                    warm_start_als=False, **common)
    corr_on = _corr_of(res_on)
    corr_off = _corr_of(res_off)
    # Default (warm start ON) recovers the true signal.
    assert corr_on >= 0.70, (
        f"default cma_batch + ALS warm start did not recover F-POLY "
        f"(corr={corr_on:.3f}); the recovery regressed"
    )
    # Disabling the warm start collapses recovery -- proves it is the lever.
    assert corr_on >= corr_off + 0.30, (
        f"ALS warm start ON (corr={corr_on:.3f}) did not beat warm start OFF "
        f"(corr={corr_off:.3f}) by the expected margin; the warm-start-is-the-"
        f"lever hypothesis would be falsified"
    )


# ---------------------------------------------------------------------------
# F-OSC boundary: oscillatory inner. UNB fails; the hybrid-orth pair path
# PARTIALLY recovers (a genuine positive worth pinning).
# ---------------------------------------------------------------------------
def test_fosc_unary_binary_fails_to_recover():
    """sin(a**2) is not a single library unary and is non-monotone -> the
    unary/binary path finds nothing. Measured |corr| ~= 0; pinned < 0.30."""
    df, y, true = _make_osc()
    fs = _fit(_unb, df, y)
    _name, corr = _best_engineered_corr(fs, df, true)
    assert corr < 0.30, f"F-OSC/UNB unexpectedly recovered |corr|={corr:.3f} ({_name})"


def test_fosc_hybrid_orth_pair_partially_recovers_and_beats_unb():
    """On the oscillatory boundary the hybrid orthogonal-poly PAIR path (the
    bilinear cross-basis cell, e.g. T2(a)*T1(b)) PARTIALLY recovers the signal
    where the unary/binary path does not. Measured |corr| ~= 0.66 for hybrid-orth
    vs ~0 for unary/binary; pinned hybrid >= 0.45 AND hybrid > unb + 0.30."""
    df, y, true = _make_osc()

    fs_unb = _fit(_unb, df, y)
    _n_unb, corr_unb = _best_engineered_corr(fs_unb, df, true)

    fs_hyb = _fit(_orth_hybrid_pair, df, y)
    name_hyb, corr_hyb = _best_engineered_corr(fs_hyb, df, true)

    assert name_hyb is not None, "F-OSC/hybrid-orth engineered nothing"
    assert corr_hyb >= 0.45, (
        f"F-OSC/hybrid-orth best engineered |corr|={corr_hyb:.3f} < 0.45 ({name_hyb})"
    )
    assert corr_hyb > corr_unb + 0.30, (
        f"F-OSC/hybrid-orth ({corr_hyb:.3f}) did not beat unary/binary "
        f"({corr_unb:.3f}) by the expected margin"
    )


def test_fosc_orth_smart_polynom_default_recovers():
    """BONUS RECOVERY: with the degree-6 default the smart_polynom path also
    recovers the oscillatory inner sin(a**2)*b -- the warm-start lands the
    higher-degree Chebyshev factors that the degree-4-only fit (and the
    hybrid-orth fixed cells) miss. Measured |corr| ~= 0.95, downstream Ridge
    R^2 ~= 0.91 (vs ~0.07 raw). Pinned >= 0.70 corr AND a downstream lift; this
    is a documented improvement over the F-OSC hybrid-orth partial (~0.67)."""
    df, y, true = _make_osc()
    raw_r2 = _ridge_r2(df.values, y)
    fs = _fit(_orth_smart_polynom, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-OSC/ORTH-smart_polynom engineered nothing"
    assert corr >= 0.70, (
        f"F-OSC/ORTH-smart_polynom best engineered |corr|={corr:.3f} < 0.70 ({name})"
    )
    sel_r2 = _ridge_r2(np.asarray(fs.transform(df)), y)
    assert sel_r2 >= raw_r2 + 0.40, (
        f"F-OSC/ORTH-smart_polynom downstream R^2={sel_r2:.3f} did not lift over "
        f"the raw baseline {raw_r2:.3f}"
    )


# ---------------------------------------------------------------------------
# Pure-noise negative control: the saturating-penalty relaxation must NOT open
# the floodgates to spurious engineered features on a target independent of the
# inputs. This is the regression guard for the penalty change.
# ---------------------------------------------------------------------------
def test_noise_control_smart_polynom_engineers_no_spurious_feature():
    """REGRESSION GUARD: on a target INDEPENDENT of (a, b), the smart_polynom
    path with the new saturating penalty + ALS warm start must engineer NO
    polynom feature. The ALS warm start always *fits* something, but the
    downstream MI-prevalence / uplift gates (the engineered MI cannot clear
    0.9 * the pair-joint MI on noise) must still reject it -- otherwise the
    penalty relaxation would be fabricating signal.

    Falsifiable: asserts zero ``_polynom_*`` columns in the selected set AND
    that the selection's downstream Ridge R^2 stays at the (near-zero) noise
    baseline."""
    df, y, _ = _make_noise()
    fs = _fit(_orth_smart_polynom, df, y)
    names = list(fs.get_feature_names_out())
    polynom_cols = [nm for nm in names if nm.startswith("_polynom_")]
    assert not polynom_cols, (
        f"noise control fabricated polynom engineered feature(s) {polynom_cols}; "
        f"the saturating-penalty relaxation opened the floodgates on a target "
        f"independent of the inputs"
    )
    # Downstream stays at the noise baseline (no engineered lift on pure noise).
    raw_r2 = _ridge_r2(df.values, y)
    sel_r2 = _ridge_r2(np.asarray(fs.transform(df)), y)
    assert sel_r2 <= raw_r2 + 0.10, (
        f"noise control downstream R^2={sel_r2:.3f} beats the raw noise baseline "
        f"{raw_r2:.3f}: a spurious feature is leaking predictive signal"
    )


def test_noise_control_optimiser_uplift_is_rejected_by_gate():
    """Unit-level companion to the MRMR-level noise control: even the raw
    ``optimise_hermite_pair`` (warm start + saturating penalty) must return
    None on a pair independent of y, because its best engineered MI cannot beat
    the baseline by ``baseline_uplift_threshold``. Pins that the warm start does
    not manufacture an uplift on noise."""
    pytest.importorskip("cma")
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair
    df, y, _ = _make_noise()
    res = optimise_hermite_pair(
        df["a"].values, df["b"].values, np.asarray(y),
        discrete_target=False, max_degree=6, min_degree=1, n_trials=400,
        basis="chebyshev", optimizer="cma_batch", mi_estimator="plugin",
        multi_fidelity=False, warm_start=True, warm_start_als=True,
        baseline_uplift_threshold=1.01,
    )
    # Either None (failed the uplift gate) or, if a marginal result slips
    # through, its uplift must be negligible (<= 1.10x) -- never a real signal.
    if res is not None:
        assert res.uplift <= 1.10, (
            f"noise pair produced uplift {res.uplift:.2f}x (> 1.10x); the warm "
            f"start is manufacturing signal on a target independent of (a, b)"
        )
