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

  (* the F-POLY/UNB and F-POLY/hybrid cells, and F-OSC/UNB, WERE TRUE-NEGATIVE
  pins: those base paths structurally cannot represent the non-monotone inner on
  their own. As of the default-on FE AUTO-ESCALATION (2026-06-10,
  ``fe_auto_escalation_enable``, ``_fe_auto_escalation.py``) they now DO recover:
  the escalation detects the unrecovered high-MI pair and engineers an
  orthogonal-poly product cell -- ``esc_poly_hermite_mul(a,b)`` at |corr| ~0.736
  for F-POLY (both UNB and hybrid), ``esc_poly_laguerre_mul(a,b)`` at |corr|
  ~0.964 for F-OSC. Those three cells are RE-FRAMED below from negatives into
  positive capability pins, each with a paired ``escalation OFF`` control that
  re-asserts the old negative -- proving the recovery is attributable to the
  escalation, not the base search or noise. Only the smart_polynom ORTH path
  reaches the highest fidelity (corr ~0.97).)

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
    # The ELEMENTARY unary/binary pair search ONLY: no smart-polynom, no hybrid
    # orth, and prewarp OFF. ``fe_pair_prewarp_enable`` defaults to True (it is a
    # learned compose-then-expand warp that CAN represent a non-monotone inner --
    # see ``test_fpoly_unary_binary_prewarp_recovers``), so it must be disabled
    # here for the boundary tests to measure the elementary library-unary search
    # in isolation, which is what their docstrings describe.
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False,
                fe_pair_prewarp_enable=False, **_LEAN)


def _unb_no_escalation():
    # Elementary unary/binary search with FE auto-escalation OFF: isolates the
    # base search's intrinsic (true-negative) behaviour on a non-representable
    # inner, so the escalation recovery is attributable, not baked into the base.
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False,
                fe_pair_prewarp_enable=False, fe_auto_escalation_enable=False,
                **_LEAN)


def _orth_hybrid_pair_no_escalation():
    # Fixed-cell hybrid orthogonal-pair path with FE auto-escalation OFF.
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=True,
                fe_hybrid_orth_pair_enable=True, fe_hybrid_orth_degrees=(2, 3, 4),
                fe_hybrid_orth_pair_max_degree=4, fe_hybrid_orth_top_k=8,
                fe_pair_prewarp_enable=False, fe_auto_escalation_enable=False,
                **_LEAN)


def _unb_prewarp():
    # Elementary unary/binary search WITH the production-default prewarp on.
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False,
                fe_pair_prewarp_enable=True, **_LEAN)


def _orth_smart_polynom(basis="chebyshev"):
    # Default optimiser is cma_batch (the production default).
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=20, fe_smart_polynom_optimization_steps=400,
                fe_polynomial_basis=basis, fe_optimizer="cma_batch",
                fe_hybrid_orth_enable=False, **_LEAN)


def _orth_hybrid_pair():
    # Prewarp OFF: this config isolates the FIXED-CELL hybrid orthogonal-pair
    # path's intrinsic behaviour. Prewarp (default on) is a separate
    # compose-then-expand mechanism that recovers F-POLY via
    # ``mul(prewarp(a),prewarp(b))`` regardless of the path it rides on (see the
    # dedicated prewarp recovery test), which would mask what THIS path does on
    # its own.
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=True,
                fe_hybrid_orth_pair_enable=True, fe_hybrid_orth_degrees=(2, 3, 4),
                fe_hybrid_orth_pair_max_degree=4, fe_hybrid_orth_top_k=8,
                fe_pair_prewarp_enable=False, **_LEAN)


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
def test_fpoly_unary_binary_recovers_via_auto_escalation():
    """RECOVERY via AUTO-ESCALATION (capability win, 2026-06-10, default ON):
    no single library unary equals ``a**3-2a``, so the ELEMENTARY unary/binary
    pair search (prewarp off, smart_polynom off) cannot itself represent
    ``P(a)`` -- and historically this cell was a TRUE-NEGATIVE pin (corr ~0).

    The default-on FE auto-escalation (``fe_auto_escalation_enable``,
    ``_fe_auto_escalation.py``) now detects the unrecovered high-MI pair and
    escalates it through an orthogonal-poly product cell, engineering
    ``esc_poly_hermite_mul(a,b)`` correlated with the true ``P(a)*Q(b)`` signal.
    Measured |corr| ~= 0.736; pinned >= 0.65 with margin.

    This converts the former negative into a positive capability pin. The
    recovery is ATTRIBUTABLE to the escalation, not the elementary search: the
    ``_unb_no_escalation`` control below (escalation OFF, same config) does NOT
    reach this level -- it only finds a weak relu-threshold artifact
    (|corr| ~0.49), well short of the escalated 0.736."""
    df, y, true = _make_poly()
    fs = _fit(_unb, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-POLY/UNB auto-escalation engineered nothing"
    assert corr >= 0.65, (
        f"F-POLY/UNB+auto-escalation best engineered |corr|={corr:.3f} < 0.65 "
        f"({name}); the auto-escalation recovery of the non-monotone inner "
        f"regressed"
    )


def test_fpoly_unary_binary_no_recovery_with_escalation_off():
    """GATED TRUE-NEGATIVE control: with FE auto-escalation OFF the elementary
    unary/binary path (prewarp off, smart_polynom off) does NOT recover the
    non-monotone ``P(a)*Q(b)`` inner -- it engineers at most a weak
    relu-threshold artifact. Measured |corr| ~= 0.49; pinned < 0.60, strictly
    below the escalated 0.736 of
    ``test_fpoly_unary_binary_recovers_via_auto_escalation``.

    This preserves the original true-negative (the elementary search structurally
    cannot represent the inner) AND proves the recovery in the companion test is
    attributable to the auto-escalation, not the base search or noise."""
    df, y, true = _make_poly()
    fs = _fit(_unb_no_escalation, df, y)
    _name, corr = _best_engineered_corr(fs, df, true)
    assert corr < 0.60, (
        f"F-POLY/UNB (escalation OFF) unexpectedly recovered |corr|={corr:.3f} "
        f"({_name}); with escalation off the elementary unary/binary path was "
        f"expected to fail on the non-monotone inner"
    )


def test_fpoly_unary_binary_prewarp_recovers():
    """RECOVERY via prewarp: the elementary unary/binary pair search AUGMENTED
    with the production-default ``fe_pair_prewarp_enable`` recovers the
    non-monotone F-POLY inner that the plain search
    (``test_fpoly_unary_binary_fails_to_recover``) cannot.

    No single library unary equals ``a**3 - 2a``, so the plain pair search finds
    nothing (corr ~0). Prewarp is a learned compose-then-expand warp of each
    operand BEFORE the binary combine, which can approximate the non-monotone
    inner; the search then engineers ``mul(prewarp(a), prewarp(b))`` highly
    correlated with the true ``P(a)*Q(b)`` signal. Measured |corr| ~= 0.97;
    pinned >= 0.70 with margin. This is the elementary-path companion to the
    smart_polynom recovery -- prewarp (on by default) closes the former
    unary/binary boundary, so the only way to OBSERVE that boundary now is to
    explicitly disable prewarp (which ``_unb`` does)."""
    df, y, true = _make_poly()
    fs = _fit(_unb_prewarp, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, (
        "F-POLY/UNB+prewarp engineered nothing; prewarp is expected to recover "
        "the non-monotone inner"
    )
    assert corr >= 0.70, (
        f"F-POLY/UNB+prewarp best engineered |corr|={corr:.3f} < 0.70 ({name}); "
        f"the prewarp recovery of the non-monotone inner regressed"
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


def test_fpoly_orth_hybrid_pair_recovers_via_auto_escalation():
    """RECOVERY via AUTO-ESCALATION (capability win, 2026-06-10, default ON):
    the hybrid orthogonal-pair path's fixed low-degree bilinear cells (e.g.
    ``T2(a)*T1(b)``) cannot themselves express the full
    ``cheb_3(a)*cheb_2(b)`` product -- historically a TRUE-NEGATIVE pin
    (corr ~0).

    The default-on FE auto-escalation now escalates the unrecovered high-MI pair
    to an orthogonal-poly product cell, engineering ``esc_poly_hermite_mul(a,b)``
    correlated with the true ``P(a)*Q(b)`` signal. Measured |corr| ~= 0.736;
    pinned >= 0.65 with margin.

    The recovery is ATTRIBUTABLE to the escalation: the
    ``_orth_hybrid_pair_no_escalation`` control below (escalation OFF, same
    config) does NOT reach this level (only the weak relu artifact, |corr| ~0.49).
    A higher-fidelity recovery is still the smart_polynom path's job
    (``test_fpoly_orth_smart_polynom_default_recovers``, corr ~0.97)."""
    df, y, true = _make_poly()
    fs = _fit(_orth_hybrid_pair, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-POLY/hybrid-orth auto-escalation engineered nothing"
    assert corr >= 0.65, (
        f"F-POLY/hybrid-orth+auto-escalation best engineered |corr|={corr:.3f} "
        f"< 0.65 ({name}); the auto-escalation recovery regressed"
    )


def test_fpoly_orth_hybrid_pair_no_recovery_with_escalation_off():
    """GATED TRUE-NEGATIVE control: with FE auto-escalation OFF the hybrid
    orthogonal-pair path's fixed bilinear cells do NOT recover the full
    ``cheb_3(a)*cheb_2(b)`` product -- only the weak relu-threshold artifact
    (|corr| ~0.49). Pinned < 0.60, strictly below the escalated 0.736.

    Preserves the original true-negative (the fixed cells structurally cannot
    express the product) and proves the companion recovery is attributable to the
    auto-escalation, not the hybrid path itself."""
    df, y, true = _make_poly()
    fs = _fit(_orth_hybrid_pair_no_escalation, df, y)
    _name, corr = _best_engineered_corr(fs, df, true)
    assert corr < 0.60, (
        f"F-POLY/hybrid-orth (escalation OFF) unexpectedly recovered "
        f"|corr|={corr:.3f} ({_name})"
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


def test_fpoly_downstream_score_recovers_for_true_negative_paths_via_escalation():
    """END-TO-END RECOVERY via AUTO-ESCALATION (capability win, 2026-06-10,
    default ON): the paths that on their own cannot represent the non-monotone
    inner -- the ELEMENTARY unary/binary search and the fixed-cell hybrid-orth
    pair (both prewarp OFF) -- historically left downstream 5-fold Ridge R^2 at
    the all-raw baseline (~0.22). The default-on FE auto-escalation now escalates
    the unrecovered pair to ``esc_poly_hermite_mul(a,b)``, lifting downstream
    R^2 to ~0.67 (measured) -- a material gain over raw, attributable to the
    escalation.

    The lift is escalation-driven: the ``escalation OFF`` control below leaves
    R^2 at ~0.43 (only the weak relu artifact), strictly below the escalated
    ~0.67 and still far below the true-signal fit (~1.0) -- the base paths'
    representational limit persists once escalation is removed. Higher-fidelity
    recovery (R^2 ~0.95) remains the smart_polynom path's job
    (``test_fpoly_downstream_score_recovers_for_smart_polynom``)."""
    df, y, true = _make_poly()
    raw_r2 = _ridge_r2(df.values, y)
    true_r2 = _ridge_r2(np.asarray(true), y)
    assert true_r2 > 0.95, f"sanity: true-signal Ridge R^2={true_r2:.3f} should be ~1.0"
    assert raw_r2 < 0.5, f"sanity: all-raw Ridge R^2={raw_r2:.3f} should be low"

    for label, maker, maker_off in [
        ("UNB", _unb, _unb_no_escalation),
        ("hybrid-orth", _orth_hybrid_pair, _orth_hybrid_pair_no_escalation),
    ]:
        fs = _fit(maker, df, y)
        sel_r2 = _ridge_r2(np.asarray(fs.transform(df)), y)
        fs_off = _fit(maker_off, df, y)
        off_r2 = _ridge_r2(np.asarray(fs_off.transform(df)), y)
        # Escalation lifts downstream materially over the all-raw baseline ...
        assert sel_r2 >= raw_r2 + 0.30, (
            f"F-POLY/{label}+escalation downstream R^2={sel_r2:.3f} did not lift "
            f"over raw baseline {raw_r2:.3f}; the escalation recovery regressed"
        )
        # ... and the lift is attributable to the escalation (beats escalation OFF).
        assert sel_r2 >= off_r2 + 0.15, (
            f"F-POLY/{label}+escalation R^2={sel_r2:.3f} did not beat the "
            f"escalation-OFF baseline {off_r2:.3f}; the lift is not attributable "
            f"to the auto-escalation"
        )
        # GATED TRUE-NEGATIVE: with escalation off these base paths stay below the
        # true-signal fit -- their representational limit persists.
        assert off_r2 < true_r2 - 0.4, (
            f"F-POLY/{label} (escalation OFF) downstream R^2={off_r2:.3f} "
            f"approached the true-signal R^2 {true_r2:.3f}; the base path was "
            f"expected to fall short without escalation."
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
def test_fosc_unary_binary_recovers_via_auto_escalation():
    """RECOVERY via AUTO-ESCALATION (capability win, 2026-06-10, default ON):
    ``sin(a**2)`` is not a single library unary and is non-monotone, so the
    elementary unary/binary path finds nothing itself -- historically a
    TRUE-NEGATIVE pin (corr ~0).

    The default-on FE auto-escalation escalates the unrecovered high-MI pair to
    an orthogonal-poly product cell, engineering ``esc_poly_laguerre_mul(a,b)``
    correlated with the true ``sin(a**2)*b`` signal. Measured |corr| ~= 0.964;
    pinned >= 0.85 with margin.

    The recovery is ATTRIBUTABLE to the escalation: the
    ``test_fosc_unary_binary_no_recovery_with_escalation_off`` control (escalation
    OFF, same config) engineers nothing (corr ~0)."""
    df, y, true = _make_osc()
    fs = _fit(_unb, df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, "F-OSC/UNB auto-escalation engineered nothing"
    assert corr >= 0.85, (
        f"F-OSC/UNB+auto-escalation best engineered |corr|={corr:.3f} < 0.85 "
        f"({name}); the auto-escalation recovery of the oscillatory inner regressed"
    )


def test_fosc_unary_binary_no_recovery_with_escalation_off():
    """GATED TRUE-NEGATIVE control: with FE auto-escalation OFF the elementary
    unary/binary path engineers nothing usable for the oscillatory
    ``sin(a**2)*b`` inner. Measured |corr| ~= 0; pinned < 0.30, far below the
    escalated 0.964.

    Preserves the original true-negative and proves the companion recovery is
    fully attributable to the auto-escalation."""
    df, y, true = _make_osc()
    fs = _fit(_unb_no_escalation, df, y)
    _name, corr = _best_engineered_corr(fs, df, true)
    assert corr < 0.30, (
        f"F-OSC/UNB (escalation OFF) unexpectedly recovered |corr|={corr:.3f} "
        f"({_name})"
    )


def test_fosc_hybrid_orth_pair_partially_recovers_and_beats_unb():
    """On the oscillatory boundary the hybrid orthogonal-poly PAIR path (the
    bilinear cross-basis cell, e.g. T2(a)*T1(b)) INTRINSICALLY PARTIALLY recovers
    the signal where the unary/binary path does not. Measured |corr| ~= 0.66 for
    hybrid-orth vs ~0 for unary/binary; pinned hybrid >= 0.45 AND hybrid >
    unb + 0.30.

    Both fits run with FE auto-escalation OFF: the default-on auto-escalation
    (2026-06-10) escalates BOTH paths' unrecovered pair to the same
    ``esc_poly_laguerre_mul(a,b)`` (|corr| ~0.96 -- see
    ``test_fosc_unary_binary_recovers_via_auto_escalation``), which would collapse
    this comparison (both paths equal). Disabling escalation isolates the hybrid
    path's INTRINSIC bilinear partial-recovery advantage over the elementary
    search, which is what this test pins."""
    df, y, true = _make_osc()

    fs_unb = _fit(_unb_no_escalation, df, y)
    _n_unb, corr_unb = _best_engineered_corr(fs_unb, df, true)

    fs_hyb = _fit(_orth_hybrid_pair_no_escalation, df, y)
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


def test_noise_floor_permutation_guard_is_the_lever():
    """MECHANISM PROOF: the permutation-null noise floor is what rejects the fabricated polynom feature on a discrete (MRMR-style)
    noise target. On the EXACT MRMR code path (discrete quantised y, coef_range (-10, 10) -- the widest-capacity overfit case) the
    optimiser can clear both the trivial baseline and ``baseline_uplift_threshold`` on pure noise across the restart sweep; the
    permutation-null guard (default ON, ``noise_floor_perm_ratio=1.50``) re-checks the winning column's MI against shuffled y and
    rejects it. Disabling the guard (``noise_floor_perm_ratio=0``) lets at least one restart fabricate a feature, isolating the
    guard as the lever -- not the uplift gate alone."""
    pytest.importorskip("cma")
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair
    from mlframe.feature_selection.filters.discretization import discretize_array
    df, y, _ = _make_noise()
    a = df["a"].values
    e = df["e"].values  # the fabricated pair in the regression was (a, e), both pure noise
    y_disc = discretize_array(
        arr=np.asarray(y, dtype=np.float64), n_bins=10, method="quantile", dtype=np.int32,
    ).reshape(-1).astype(np.int64)

    def _sweep(noise_floor_perm_ratio):
        n_pass = 0
        for so in range(20):
            res = optimise_hermite_pair(
                x_a=a, x_b=e, y=y_disc, discrete_target=True,
                max_degree=6, min_degree=1, n_trials=400, coef_range=(-10.0, 10.0),
                l2_penalty=0.05, seed=42 + so, sweep_degrees=True, basis="chebyshev",
                mi_estimator="plugin", optimizer="cma_batch", warm_start=True,
                multi_fidelity=True, noise_floor_perm_ratio=noise_floor_perm_ratio,
            )
            if res is not None:
                n_pass += 1
        return n_pass

    # Guard ON (default): every restart is rejected on pure noise.
    assert _sweep(1.50) == 0, (
        "noise-floor guard ON still let a restart fabricate a feature on a pair independent of y"
    )
    # Guard OFF: at least one restart slips a spurious feature through -- the guard is the lever.
    assert _sweep(0.0) >= 1, (
        "disabling the noise-floor guard did NOT surface the fabricated feature; the guard is not the lever the regression needs"
    )
