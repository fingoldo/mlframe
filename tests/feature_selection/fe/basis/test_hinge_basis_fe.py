"""Unit + biz_value triad for the hinge / piecewise-linear change-point basis
(backlog #11, 2026-06-09).

A hinge captures a SLOPE CHANGE at a data-dependent threshold
``y = a*x + b*max(x - tau, 0)`` (pricing tiers / dose-response / saturation) --
a signal shape the catalog cannot: ``numeric_rounding`` is piecewise-CONSTANT,
the cubic B-spline rounds off a sharp kink at its FIXED quantile knots, and
orth-poly needs a high degree + rings (Gibbs) around the kink.

UNIT contracts:
* breakpoint detection recovers a planted tau (held-out validated);
* recipe replay is a pure, leak-safe function of X (bit-exact + pickle roundtrip);
* pure noise admits no hinge (held-out tau-validation rejects chance breakpoints);
* the dispatcher routes the ``hinge_basis`` kind.

BIZ_VALUE contracts (the decisive ones -- the hinge leg is MONOTONE in x, hence
MI-invariant, so its value is downstream linear usability, NOT MI -- exactly like
RankGauss/isotonic):
* SLOPE-CHANGE WIN: held-out Ridge R^2 of [x, hinge] BEATS raw x, best Chebyshev,
  and best B-spline on the slope-change fixture;
* SMOOTH COMPLEMENTARITY: on y = x^2 the hinge does NOT beat a degree-2 poly
  (proves it is not a spline-clone / universal approximator).

DEFAULT-PATH contracts (the operator is now DEFAULT-ON, 2026-06-09): the win must
manifest with a plain ``MRMR()`` (no opt-in flag) and self-limit to zero columns
on data without a slope change:
* the hinge leg is MI-invariant so the greedy screen drops it, but the support-
  finalisation HINGE-PROTECTION block RE-ADDS held-out-validated legs whose raw
  source survived -> the win is delivered on the default transform output;
* on pure noise / a purely linear column / a smooth y=x^2 the default path
  retains ZERO hinge legs (no spurious columns on neutral data);
* a cheap 3-cut SSE pre-check skips the full scan on no-kink columns so
  default-on does not bloat wide / large-p fits.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

N = 4000


def _heldout_r2(feat_cols, y):
    from sklearn.linear_model import Ridge
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in feat_cols])
    idx = np.arange(len(y))
    va = (idx % 3) == 0
    tr = ~va
    r = Ridge(alpha=1e-3)
    r.fit(A[tr], y[tr])
    pred = r.predict(A[va])
    yv = y[va]
    sse = float(np.sum((yv - pred) ** 2))
    sst = float(np.sum((yv - yv.mean()) ** 2))
    return 1.0 - sse / sst


# ---------------------------------------------------------------------------
# UNIT
# ---------------------------------------------------------------------------


def test_detect_recovers_planted_breakpoint():
    """The breakpoint scan locates a planted slope change near the true tau."""
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        _detect_hinge_breakpoints,
    )
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.1 * rng.standard_normal(N)
    taus = _detect_hinge_breakpoints(x, y, max_breakpoints=2)
    assert taus, "expected at least one breakpoint on the slope-change fixture"
    assert min(abs(t - 0.7) for t in taus) < 0.06, (
        f"detected tau(s) {taus} not near the true 0.7"
    )


def test_noise_admits_no_hinge():
    """Pure-noise x vs noise y -> held-out tau-validation rejects the chance
    breakpoint -> no hinge column."""
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        _detect_hinge_breakpoints,
    )
    admitted = 0
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        xn = rng.standard_normal(N)
        yn = rng.standard_normal(N)
        if _detect_hinge_breakpoints(xn, yn, max_breakpoints=2):
            admitted += 1
    assert admitted == 0, f"{admitted}/20 noise frames falsely admitted a hinge"


def test_recipe_replay_is_leak_safe_and_bit_exact():
    """The hinge_basis recipe replays max(x-tau,0) / max(tau-x,0) / 1[x>tau]
    bit-exactly from X alone (no y), survives pickle, and routes through the
    apply_recipe dispatcher."""
    from mlframe.feature_selection.filters.engineered_recipes import (
        apply_recipe, build_hinge_basis_recipe,
    )
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 3, N)
    X = pd.DataFrame({"a": x})
    for side, ref in (
        ("gt", np.maximum(x - 0.5, 0.0)),
        ("lt", np.maximum(0.5 - x, 0.0)),
        ("ind", (x > 0.5).astype(np.float64)),
    ):
        r = build_hinge_basis_recipe(name=f"a__{side}", src_name="a", tau=0.5, side=side)
        r2 = pickle.loads(pickle.dumps(r))  # pickle / clone roundtrip
        out = apply_recipe(r2, X)
        assert np.array_equal(out, ref), f"replay mismatch for side={side}"


def test_recipe_replay_independent_of_y():
    """Replay reads only X: a recipe built with one y replays identically on a
    frame with a totally different target (proves no y leak)."""
    from mlframe.feature_selection.filters.engineered_recipes import (
        apply_recipe, build_hinge_basis_recipe,
    )
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 1, N)
    X = pd.DataFrame({"a": x})
    r = build_hinge_basis_recipe(name="a__relu", src_name="a", tau=0.42, side="gt")
    out = apply_recipe(r, X)
    assert np.array_equal(out, np.maximum(x - 0.42, 0.0))


def test_detect_hinge_fwl_rank1_taus_bit_identical_to_lstsq_per_cut():
    """The per-cut SSE in ``_detect_hinge_breakpoints`` is scored via the Frisch-Waugh-Lovell rank-1 update (QR the fixed ``[1, x, *extra]`` block once per round, then
    ``SSE_B - (r_relu . r_y)^2 / (r_relu . r_relu)`` per cut) instead of a full ``lstsq`` per cut. That is mathematically identical to the full-design SSE, so the argmin
    ``best_tau`` (and thus the returned breakpoint list) must be BIT-IDENTICAL to the legacy lstsq-per-cut reference. This pins the optimisation so a future rewrite that
    perturbs the SSE comparison (and could flip a near-tied tau) is caught. Covers kink / linear / quadratic / noise / 2-kink columns across sizes."""
    from mlframe.feature_selection.filters import _hinge_basis_fe as hinge_mod
    from mlframe.feature_selection.filters._hinge_basis_fe import _detect_hinge_breakpoints

    def legacy(x, y, *, max_breakpoints=2, min_heldout_r2_uplift=0.02):
        x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel(); n = x.size
        if n < hinge_mod._HINGE_MIN_ROWS:
            return []
        f = np.isfinite(x) & np.isfinite(y)
        if not f.all():
            x = x[f]; y = y[f]; n = x.size
        if n < hinge_mod._HINGE_MIN_ROWS:
            return []
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return []
        if not hinge_mod._hinge_slope_change_plausible(x, y, min_sse_drop=hinge_mod._HINGE_PRECHECK_MIN_SSE_DROP):
            return []
        qs = np.linspace(hinge_mod._HINGE_CAND_Q_LO, hinge_mod._HINGE_CAND_Q_HI, hinge_mod._HINGE_N_CANDIDATES)
        cand = np.unique(np.quantile(x, qs))
        found = []; extra = []
        for _ in range(max(1, int(max_breakpoints))):
            bt = None; bs = float("inf")
            for c in cand:
                nr = int(np.count_nonzero(x > c))
                if nr < hinge_mod._HINGE_MIN_SEG_ROWS or (n - nr) < hinge_mod._HINGE_MIN_SEG_ROWS:
                    continue
                if any(abs(c - t) < 1e-9 for t in found):
                    continue
                relu = np.maximum(x - c, 0.0)
                A = np.column_stack([np.ones_like(x), x, relu, *extra])
                try:
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                except Exception:
                    continue
                resid = y - A @ coef
                sse = float(resid @ resid)
                if sse < bs:
                    bs = sse; bt = float(c)
            if bt is None:
                break
            if hinge_mod._heldout_hinge_r2_uplift(x, y, bt) < min_heldout_r2_uplift:
                break
            found.append(bt); extra.append(np.maximum(x - bt, 0.0))
        return found

    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([200, 500, 1200, 4000]))
        x = np.sort(rng.uniform(-3, 3, n))
        kind = seed % 5
        if kind == 0:
            y = np.where(x < 0, 0.5 * x, 2.0 * x) + 0.2 * rng.standard_normal(n)
        elif kind == 1:
            y = 0.7 * x + 0.3 * rng.standard_normal(n)
        elif kind == 2:
            y = x * x + 0.3 * rng.standard_normal(n)
        elif kind == 3:
            y = rng.standard_normal(n)
        else:
            y = np.where(x < -1, 1.0 * x, np.where(x < 1, 0.2 * x, 1.5 * x)) + 0.2 * rng.standard_normal(n)
        ref = legacy(x, y)
        got = _detect_hinge_breakpoints(x, y)
        assert len(ref) == len(got), f"seed={seed} kind={kind}: tau count {got} != legacy {ref}"
        for a, b in zip(ref, got):
            assert abs(a - b) <= 1e-9, f"seed={seed} kind={kind}: tau {b} != legacy {a}"


def test_bad_side_raises():
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        build_hinge_basis_recipe,
    )
    with pytest.raises(ValueError):
        build_hinge_basis_recipe(name="x", src_name="a", tau=0.0, side="nope")


# ---------------------------------------------------------------------------
# BIZ_VALUE
# ---------------------------------------------------------------------------


def test_biz_value_slope_change_beats_raw_cheby_spline():
    """Held-out Ridge R^2 of [x, hinge] BEATS raw x, best Chebyshev, and best
    B-spline on the slope-change fixture -- the hinge places the knot AT the
    data-dependent slope change, while the spline's fixed quantile knots round
    off the kink and Chebyshev rings around it."""
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        _detect_hinge_breakpoints,
    )
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        _evaluate_basis_column,
    )
    from mlframe.feature_selection.filters.engineered_recipes import (
        _fit_spline_knots, _bspline_basis_values,
    )
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.1 * rng.standard_normal(N)

    taus = _detect_hinge_breakpoints(x, y, max_breakpoints=2)
    hinge_legs = [np.maximum(x - t, 0.0) for t in taus]
    assert hinge_legs

    r2_raw = _heldout_r2([x], y)
    r2_cheby = _heldout_r2(
        [_evaluate_basis_column(x, "chebyshev", d) for d in range(1, 5)], y
    )
    knots, lo, hi = _fit_spline_knots(x, 5, degree=3)
    z = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    n_basis = len(knots) - 3 - 1
    spline_legs = [_bspline_basis_values(z, knots, i, degree=3) for i in range(n_basis)]
    spline_legs = [v for v in spline_legs if float(np.std(v)) > 1e-12]
    r2_spline = _heldout_r2(spline_legs, y)
    r2_hinge = _heldout_r2([x, *hinge_legs], y)

    assert r2_hinge > r2_raw, (r2_hinge, r2_raw)
    assert r2_hinge > r2_cheby, (r2_hinge, r2_cheby)
    assert r2_hinge > r2_spline, (r2_hinge, r2_spline)


def test_biz_value_smooth_complementarity_loses_to_poly2():
    """On a SMOOTH target y = x^2 the hinge must NOT beat a degree-2 poly --
    proves it is a slope-change operator, not a spline-clone / universal
    approximator (if it beat poly here the impl would be wrong)."""
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        _detect_hinge_breakpoints,
    )
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1, 1, N)
    ys = xs ** 2 + 0.05 * rng.standard_normal(N)
    taus = _detect_hinge_breakpoints(xs, ys, max_breakpoints=2)
    hinge_legs = [np.maximum(xs - t, 0.0) for t in taus]
    r2_poly2 = _heldout_r2([xs, xs ** 2], ys)
    r2_hinge = _heldout_r2([xs, *hinge_legs], ys)
    assert r2_hinge <= r2_poly2 + 1e-6, (
        f"hinge {r2_hinge:.4f} should NOT beat degree-2 poly {r2_poly2:.4f} "
        f"on a smooth target"
    )


# ---------------------------------------------------------------------------
# COST PRE-CHECK (the cheap dispatch that keeps default-on lean on wide data)
# ---------------------------------------------------------------------------


def test_cost_precheck_skips_no_kink_columns():
    """The 3-cut SSE pre-check returns False (skip the full 24-cut scan) on
    columns with NO slope change -- pure noise and a purely linear column --
    so default-on does not bloat wide / large-p fits scanning every column."""
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        _hinge_slope_change_plausible,
    )
    rng = np.random.default_rng(11)
    x_lin = rng.uniform(0, 1, N)
    y_lin = 2 * x_lin + 0.1 * rng.standard_normal(N)
    x_no = rng.standard_normal(N)
    y_no = rng.standard_normal(N)
    assert not _hinge_slope_change_plausible(x_lin, y_lin), "linear column should skip"
    assert not _hinge_slope_change_plausible(x_no, y_no), "noise column should skip"


def test_cost_precheck_passes_genuine_kink():
    """The pre-check returns True on a genuine slope-change column, so the full
    scan + held-out tau-validation run and recover the breakpoint (the cost gate
    never vetoes a column the full gate would admit)."""
    from mlframe.feature_selection.filters._hinge_basis_fe import (
        _hinge_slope_change_plausible, _detect_hinge_breakpoints,
    )
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.1 * rng.standard_normal(N)
    assert _hinge_slope_change_plausible(x, y), "kink column must pass the pre-check"
    taus = _detect_hinge_breakpoints(x, y, max_breakpoints=2)
    assert taus and min(abs(t - 0.7) for t in taus) < 0.06


def test_batched_detector_matches_percolumn_detector():
    """Batched cross-column precheck (detect_hinge_breakpoints_gpu_batch) must find the SAME breakpoints
    as the per-column detector across noise / linear / genuine-kink / near-zero-variance columns -- the
    generate_hinge_features dispatcher picks whichever measured faster (K=1 per-column, K>=2 batch; see
    _benchmarks/bench_hinge_batch_vs_percolumn.py), so both must agree exactly regardless of which the
    caller lands on."""
    import cupy as cp

    from mlframe.feature_selection.filters._hinge_detect_gpu_resident import (
        detect_hinge_breakpoints_gpu, hinge_gpu_enabled,
    )
    from mlframe.feature_selection.filters._hinge_detect_gpu_resident_batch import (
        detect_hinge_breakpoints_gpu_batch,
    )
    if not hinge_gpu_enabled():
        pytest.skip("GPU-strict-resident hinge path not engaged on this host")

    rng = np.random.default_rng(3)
    n = 20000
    y = rng.standard_normal(n)
    cols = []
    for i in range(20):
        if i % 4 == 0:
            tau = rng.uniform(-1, 1)
            x = rng.standard_normal(n)
            x = x + np.maximum(x - tau, 0) * 3.0 + 0.3 * y
        elif i % 4 == 1:
            x = rng.standard_normal(n)
        elif i % 4 == 2:
            x = 0.5 * y + 0.01 * rng.standard_normal(n)
        else:
            x = rng.standard_normal(n) * 1e-8
        cols.append(x)

    kw = dict(max_breakpoints=2, min_heldout_r2_uplift=0.01, precheck_qs=(0.25, 0.5, 0.75),
              precheck_min_sse_drop=0.02, cand_q_lo=0.1, cand_q_hi=0.9, n_candidates=24,
              min_rows=500, min_seg_rows=100)
    batch_out = detect_hinge_breakpoints_gpu_batch(cols, y, **kw)
    assert batch_out is not None
    cp.cuda.Stream.null.synchronize()
    for j, x in enumerate(cols):
        ref = detect_hinge_breakpoints_gpu(x, y, **kw)
        b_l = sorted(round(t, 6) for t in (batch_out[j] or []))
        r_l = sorted(round(t, 6) for t in (ref or []))
        assert b_l == r_l, f"col {j}: batch={b_l} per-column={r_l}"


# ---------------------------------------------------------------------------
# INTEGRATION (DEFAULT path: hinge default-ON, retained where it wins,
# self-limited to zero columns on data without a slope change)
# ---------------------------------------------------------------------------


def _hinge_recipes(m):
    """Hinge recipes the FE stage PRODUCED this fit (the produced-recipe ledger
    records every recipe before the MI screen / dedup drop a subset)."""
    return [
        r for r in (getattr(m, "_produced_recipes_", None) or [])
        if getattr(r, "kind", None) == "hinge_basis"
    ]


def _hinge_survivors(m):
    """Hinge recipes that SURVIVED into support_ / get_feature_names_out (the
    held-out-validated legs the HINGE-PROTECTION block re-added past the
    MI-invariant screen drop)."""
    eng = getattr(m, "_engineered_recipes_", None) or []
    if isinstance(eng, dict):
        eng = list(eng.values())
    return [r for r in eng if getattr(r, "kind", None) == "hinge_basis"]


def test_default_is_on():
    """fe_hinge_enable now defaults ON (the best variant is the default; the
    win is no longer hidden behind an opt-in flag)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    assert MRMR().fe_hinge_enable is True


def test_default_path_retains_hinge_on_slope_change():
    """On a slope-change fixture, DEFAULT MRMR() (no opt-in flag) produces the
    hinge legs AND RETAINS them in support_ -- the HINGE-PROTECTION block re-adds
    the held-out-validated legs the MI-invariant greedy screen drops. This is
    the core default-on fix: the win is delivered on the default path, not just
    when a power user flips a flag."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, N)
    b = rng.standard_normal(N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.5 * b + 0.1 * rng.standard_normal(N)
    X = pd.DataFrame({"a": x, "b": b})
    m = MRMR(max_runtime_mins=2, verbose=0, random_seed=0)  # DEFAULT path
    m.fit(X, y)
    produced = _hinge_recipes(m)
    survivors = _hinge_survivors(m)
    assert produced, "default path should PRODUCE a hinge on the slope-change fixture"
    assert survivors, "default path should RETAIN the hinge in support_ (protection)"
    out_names = set(map(str, m.get_feature_names_out()))
    assert any(r.name in out_names for r in survivors), (
        "a surviving hinge leg must appear in get_feature_names_out"
    )
    # The surviving legs replay bit-exactly through the dispatcher.
    for r in survivors:
        replayed = apply_recipe(r, X)
        tau = float(r.extra["tau"])
        if r.extra["side"] == "gt":
            ref = np.maximum(x - tau, 0.0)
        elif r.extra["side"] == "lt":
            ref = np.maximum(tau - x, 0.0)
        else:
            ref = (x > tau).astype(np.float64)
        assert np.allclose(replayed, ref, atol=1e-9), r.name


def test_default_path_biz_value_win_through_transform():
    """The decisive default-on contract: a DEFAULT MRMR().fit().transform() on
    the slope-change fixture lifts held-out Ridge R^2 over the raw [x, b] frame
    -- the hinge win is delivered through the ordinary transform output on the
    default path (not only in an isolated micro-benchmark)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, N)
    b = rng.standard_normal(N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.5 * b + 0.1 * rng.standard_normal(N)
    X = pd.DataFrame({"a": x, "b": b})
    m = MRMR(max_runtime_mins=2, verbose=0, random_seed=0)
    m.fit(X, y)
    Xt = m.transform(X)
    Xt = Xt.to_numpy() if hasattr(Xt, "to_numpy") else np.asarray(Xt)
    Xt = Xt.astype(np.float64)
    r2_transformed = _heldout_r2([Xt[:, i] for i in range(Xt.shape[1])], y)
    r2_raw = _heldout_r2([x, b], y)
    assert r2_transformed > r2_raw, (r2_transformed, r2_raw)


@pytest.mark.parametrize("kind", ["noise", "linear", "smooth"])
def test_default_path_self_limits_no_spurious_hinge(kind):
    """SELF-LIMITING: on data with NO exploitable slope change -- pure noise, a
    purely linear column, and a smooth y=x^2 -- DEFAULT MRMR() retains ZERO
    hinge legs in support_ (noise/linear are rejected at detection; the smooth
    quadratic's leg is left out because its raw source is subsumed by the poly
    child, so the self-limiting protection gate does not re-add it). Default-on
    therefore adds no spurious columns on neutral data."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(40 + len(kind))
    b = rng.standard_normal(N)
    if kind == "noise":
        x = rng.standard_normal(N)
        y = rng.standard_normal(N)
    elif kind == "linear":
        x = rng.uniform(0, 1, N)
        y = 3 * x + 0.5 * b + 0.1 * rng.standard_normal(N)
    else:  # smooth
        x = rng.uniform(-1, 1, N)
        y = x ** 2 + 0.5 * b + 0.05 * rng.standard_normal(N)
    X = pd.DataFrame({"a": x, "b": b})
    m = MRMR(max_runtime_mins=2, verbose=0, random_seed=0)
    m.fit(X, y)
    assert not _hinge_survivors(m), (
        f"{kind}: no hinge leg should survive into support_ on neutral data"
    )
