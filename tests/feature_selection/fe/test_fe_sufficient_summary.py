"""Unit + adversarial tests for the sufficient-summary early-stop (backlog #22).

The user's "compare-to-theoretical-max" idea via a DPI residual test:
``r = y - E_hat[y|selected]`` (cheap ridge of y on the SMALL selected set); STOP the FE
search iff ``MI(r; x_j) <= the maxT permutation null`` for EVERY raw AND the residual is
small relative to y (``Var(r)/Var(y) <= guard``). By the Data-Processing Inequality any
future engineered candidate is a function of the raws, so it cannot carry more MI with the
residual than the raws do -> a residual that is pure noise w.r.t. the observables certifies
the selection reached ``I(observables; y)`` (the theoretical max) and the remaining search
is provably pointless.

These tests drive the PURE helper ``sufficient_summary_reached`` directly (deterministic,
fast, single-process, small-n) so the positive / negative / adversarial MATRIX is pinned
WITHOUT the RNG-sensitivity of a full end-to-end FE recipe selection. The end-to-end
wall-cut + selection-byte-identity is covered in the biz_value sibling
(``test_biz_value_mrmr_sufficient_summary.py``).
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_sufficient_summary import (
    SufficientSummaryVerdict,
    sufficient_summary_reached,
)

N = 4000  # small-n per the early-stop unit-fixture budget
SEED = 0


def _bin10(v, nbins: int = 10):
    """Equi-frequency 10-bin codes (the screening resolution the selector sees)."""
    v = np.asarray(v, dtype=np.float64)
    edges = np.quantile(v, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.searchsorted(edges, v).astype(np.int64)


def _stack(*arrays):
    """Column-stack 1-D arrays into the int64 discretised ``data`` matrix."""
    return np.column_stack([_bin10(a) for a in arrays]).astype(np.int64)


def _call(data, y_cont, target_col_idx, selected_cols_idx, selected_continuous, raw_cols_idx, cols_names, **kw):
    """Helper that call."""
    nbins = np.full(data.shape[1], 10, dtype=np.int64)
    return sufficient_summary_reached(
        data=data,
        nbins=nbins,
        y_continuous=y_cont,
        target_col_idx=target_col_idx,
        selected_cols_idx=selected_cols_idx,
        selected_continuous=selected_continuous,
        raw_cols_idx=raw_cols_idx,
        cols_names=cols_names,
        quantization_nbins=10,
        random_seed=SEED,
        **kw,
    )


# =====================================================================================
# POSITIVE cases -- the early-stop SHOULD fire (residual is pure noise w.r.t. observables).
# =====================================================================================
def test_positive_a_f1_ideal_composite_stops():
    """(a) F1 ``y = a**2/b + f/5 + log(c)*sin(d)``: once the IDEAL composites are
    selected (the two engineered terms), the residual is exactly the irreducible ``f/5``
    noise -- f is UNOBSERVED, so every observed raw sits at the maxT null and the variance
    guard passes -> STOP."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(1, 5, N)
    b = rng.uniform(1, 5, N)
    c = rng.uniform(1, 5, N)
    d = rng.uniform(0, 2 * np.pi, N)
    e = rng.normal(0, 1, N)
    f = rng.normal(0, 1, N)
    term1 = a**2 / b
    term2 = np.log(c) * np.sin(d)
    y = term1 + f / 5.0 + term2
    # Selected = the two ideal engineered composites (what step 1 would recover).
    data = _stack(a, b, c, d, e, term1, term2, y)
    cols = ["a", "b", "c", "d", "e", "t1", "t2", "y"]
    v = _call(data, y, 7, [5, 6], {"t1": term1, "t2": term2}, [0, 1, 2, 3, 4], cols)
    assert v.reached, v.reason
    assert v.residual_entropy_frac < 0.25, v.residual_entropy_frac
    assert v.max_raw_mi <= v.maxt_floor + 1e-12, (v.max_raw_mi, v.maxt_floor)


def test_positive_b_stops_only_after_BOTH_a_and_b_selected():
    """(b) ``y = a + b + noise``: the stop must fire ONLY after BOTH a and b are selected,
    NOT after a alone. This is the key case that kills the 1-D-top design: the residual is
    tested against the FULL selected set."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    b = rng.normal(size=N)
    e = rng.normal(size=N)
    y = a + b + 0.05 * rng.normal(size=N)
    data = _stack(a, b, e, y)
    cols = ["a", "b", "e", "y"]

    # After a ALONE: residual still carries b -> must NOT stop.
    v_half = _call(data, y, 3, [0], {"a": a}, [0, 1, 2], cols)
    assert not v_half.reached, f"stopped prematurely after a alone: {v_half.reason}"

    # After BOTH a and b: residual is the 0.05 noise -> STOP.
    v_full = _call(data, y, 3, [0, 1], {"a": a, "b": b}, [0, 1, 2], cols)
    assert v_full.reached, v_full.reason
    assert v_full.residual_entropy_frac < 0.05, v_full.residual_entropy_frac


# =====================================================================================
# NEGATIVE cases -- the early-stop must NOT fire (a false stop would lose real signal).
# =====================================================================================
def test_negative_c_pure_noise_target_guard_blocks():
    """(c) PURE-NOISE target: every raw legitimately sits at the maxT null (no signal to
    find), so the maxT test ALONE would pass -- the H(y)-relative variance guard
    (Var(r)/Var(y) ~ 1.0, nothing explained) is what prevents the false stop."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    b = rng.normal(size=N)
    e = rng.normal(size=N)
    yn = rng.normal(size=N)  # independent of everything
    data = _stack(a, b, e, yn)
    cols = ["a", "b", "e", "y"]
    v = _call(data, yn, 3, [0], {"a": a}, [0, 1, 2], cols)
    assert not v.reached, v.reason
    assert v.residual_entropy_frac > 0.5, v.residual_entropy_frac  # ~nothing explained


def test_negative_d_second_independent_signal_not_found_blocks():
    """(d) A genuine SECOND independent signal not yet found: ``y = a + 3*g`` with g a raw
    NOT yet selected. The residual against {a} still carries g's full signal -> do NOT stop
    (the search must continue to find g). g is the DOMINANT term, so the residual is large
    relative to y and the variance guard blocks first (the cheap-first ordering -- we don't
    pay for permutations when the residual is obviously not pure noise). The weak-leftover /
    nonlinear-leftover cases (g)/(i) below exercise the maxT-blocking path where the guard
    passes but a raw still beats the null."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    g = rng.normal(size=N)
    e = rng.normal(size=N)
    y = a + 3.0 * g + 0.05 * rng.normal(size=N)
    data = _stack(a, g, e, y)
    cols = ["a", "g", "e", "y"]
    v = _call(data, y, 3, [0], {"a": a}, [0, 1, 2], cols)
    assert not v.reached, v.reason
    # g dominates -> residual carries most of Var(y); the variance guard alone blocks.
    assert v.residual_entropy_frac > 0.5, v.residual_entropy_frac


# =====================================================================================
# ADVERSARIAL cases.
# =====================================================================================
def test_adversarial_e_collinear_selected_ridge_not_fooled():
    """(e) Redundant / COLLINEAR selected features: the ridge fit must not be fooled by
    collinearity (no blow-up of the normal equations). ``y = s + noise`` with two
    near-duplicate selected columns of s -> the ridge still recovers s and STOPS."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    b = rng.normal(size=N)
    e = rng.normal(size=N)
    s = a + b
    y = s + 0.05 * rng.normal(size=N)
    s1 = s + 1e-4 * rng.normal(size=N)  # near-duplicate
    s2 = s * 1.0001 + 1e-4 * rng.normal(size=N)  # collinear
    data = _stack(a, b, e, s1, s2, y)
    cols = ["a", "b", "e", "s1", "s2", "y"]
    v = _call(data, y, 5, [3, 4], {"s1": s1, "s2": s2}, [0, 1, 2], cols)
    assert v.reached, v.reason
    assert np.isfinite(v.residual_entropy_frac) and v.residual_entropy_frac < 0.05, v.residual_entropy_frac


def test_adversarial_f_feature_looks_complete_alone_but_isnt():
    """(f) A feature that LOOKS complete alone but isn't (multi-term signal). ``y = a +
    0.4*a*b`` selected on {a}: a appears in BOTH terms so the linear fit on a explains the
    bulk of y (the guard alone would NOT block), but the residual ``0.4*a*b`` is still
    explained by b via MI -> the maxT test must catch b and BLOCK the stop. The interaction
    coefficient is tuned so the residual passes the variance guard, FORCING the maxT path to
    do the blocking (the case that pins MI-detection of a partially-captured feature)."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(-2, 2, N)
    b = rng.uniform(-2, 2, N)
    e = rng.normal(size=N)
    y = a + 0.4 * a * b + 0.05 * rng.normal(size=N)
    data = _stack(a, b, e, y)
    cols = ["a", "b", "e", "y"]
    v = _call(data, y, 3, [0], {"a": a}, [0, 1, 2], cols)
    assert not v.reached, v.reason
    # The residual is small enough to clear the variance guard, so the maxT MI test is what
    # blocks -- and it identifies an OPERAND of the unexplained ``a*b`` interaction (a or b;
    # the interaction residual carries MI with both, never the pure-noise e).
    assert v.residual_entropy_frac <= 0.25, v.residual_entropy_frac
    assert v.blocking_raw in (0, 1), (v.blocking_raw, v.per_raw_mi)
    assert v.max_raw_mi > v.maxt_floor, (v.max_raw_mi, v.maxt_floor)
    assert v.per_raw_mi.get(2, 0.0) <= v.maxt_floor + 1e-9, ("noise e flagged", v.per_raw_mi)


def test_adversarial_g_weak_third_term_not_noise_blocks():
    """(g) A small-but-NOT-noise residual (a weak 3rd term): ``y = a + b + 0.5*w``. After
    selecting {a, b} the residual is dominated by the weak ``0.5*w`` term -- small, but NOT
    noise -- so w still beats its maxT null -> must NOT stop (residual still > null)."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    b = rng.normal(size=N)
    w = rng.normal(size=N)
    e = rng.normal(size=N)
    y = a + b + 0.5 * w + 0.02 * rng.normal(size=N)
    data = _stack(a, b, w, e, y)
    cols = ["a", "b", "w", "e", "y"]
    v = _call(data, y, 4, [0, 1], {"a": a, "b": b}, [0, 1, 2, 3], cols)
    assert not v.reached, v.reason
    assert v.blocking_raw == 2, (v.blocking_raw, v.per_raw_mi)  # w is cols-index 2


def test_adversarial_h_heteroscedastic_residual_blocks_via_mi():
    """(h) Heteroscedastic residual: ``y = a + b + s(a)*eps`` where the noise SCALE depends
    on a (a is a raw, also selected for the mean). After selecting {a, b} the residual's
    MEAN is captured (variance guard passes -- the residual is SMALL relative to y), but the
    residual's *spread* still tracks a, so the equi-frequency MI test ``MI(r; a)`` detects
    that a predicts which residual BIN -> the maxT test BLOCKS the stop. This pins that the
    MI-on-residual test catches a heteroscedastic (variance-only) leftover, NOT just a
    location leftover -- it does not falsely declare sufficiency when an observable still
    carries information about the residual's distribution."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0, 1, N)
    b = rng.normal(size=N)
    e = rng.normal(size=N)
    scale = 0.08 * (1.0 + 6.0 * a)  # spread depends strongly on a; mean does not
    y = a + b + scale * rng.normal(size=N)
    data = _stack(a, b, e, y)
    cols = ["a", "b", "e", "y"]
    v = _call(data, y, 3, [0, 1], {"a": a, "b": b}, [0, 1, 2], cols)
    assert not v.reached, v.reason
    # The variance guard passes (residual mean captured), so the MI test does the blocking,
    # and it points at a (cols-index 0), whose value governs the residual SPREAD.
    assert v.residual_entropy_frac <= 0.25, v.residual_entropy_frac
    assert v.blocking_raw == 0, (v.blocking_raw, v.per_raw_mi)


def test_adversarial_i_nonlinear_leftover_linear_underfits_MI_still_catches():
    """(i) NONLINEAR leftover the linear E_hat UNDERFITS -> the MI(r; raws) test must STILL
    catch it. ``y = a + 2*sin(5*b)``: a linear fit on {a, b} CANNOT represent ``sin(5*b)``
    (corr(b, sin 5b) ~ 0), so the linear residual still carries the full sin(5b) signal.
    The variance guard would be borderline, but the KEY assertion is that b's MI with the
    residual beats its maxT null -> do NOT stop. This is the explicit verification that an
    MI-based residual test (not a linear score) catches nonlinear leftovers."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    b = rng.uniform(-np.pi, np.pi, N)
    e = rng.normal(size=N)
    # Big linear-a signal + SMALL nonlinear-b leftover: a dominates the variance, so after
    # the linear fit on {a, b} captures the a term, the residual (the small sin(5b)) is
    # below the variance guard -- yet sin(5b) is fully explained by b via MI (corr(b,sin 5b)
    # ~ 0, so the LINEAR fit never touched it). The maxT MI test must STILL catch b.
    y = 5.0 * a + 0.30 * np.sin(5.0 * b) + 0.02 * rng.normal(size=N)
    data = _stack(a, b, e, y)
    cols = ["a", "b", "e", "y"]
    v = _call(data, y, 3, [0, 1], {"a": a, "b": b}, [0, 1, 2], cols)
    assert not v.reached, v.reason
    # The variance guard PASSES (the nonlinear leftover is small), so the MI test is the
    # only thing that can block -- and it identifies b (cols-index 1) via MI even though the
    # linear E_hat underfit the sin(5b) term (never reduced the residual).
    assert v.residual_entropy_frac <= 0.25, v.residual_entropy_frac
    assert v.blocking_raw == 1, (v.blocking_raw, v.per_raw_mi)
    assert v.max_raw_mi > v.maxt_floor, (v.max_raw_mi, v.maxt_floor)


# =====================================================================================
# DEGENERATE-INPUT guards -- never skip when sufficiency cannot be proven.
# =====================================================================================
def test_degenerate_no_raw_pool_does_not_stop():
    """Degenerate no raw pool does not stop."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    y = a + 0.05 * rng.normal(size=N)
    data = _stack(a, y)
    v = _call(data, y, 1, [0], {"a": a}, [], ["a", "y"])
    assert not v.reached and "raw" in v.reason


def test_degenerate_constant_target_does_not_stop():
    """Degenerate constant target does not stop."""
    a = np.linspace(0, 1, N)
    y = np.zeros(N)
    data = _stack(a, np.arange(N))
    v = _call(data, y, 1, [0], {"a": a}, [0], ["a", "y"])
    assert not v.reached


def test_verdict_dataclass_default_is_no_stop():
    """Verdict dataclass default is no stop."""
    v = SufficientSummaryVerdict()
    assert v.reached is False
