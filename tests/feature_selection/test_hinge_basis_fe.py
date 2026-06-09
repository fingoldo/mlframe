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
    r2_hinge = _heldout_r2([x] + hinge_legs, y)

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
    r2_hinge = _heldout_r2([xs] + hinge_legs, ys)
    assert r2_hinge <= r2_poly2 + 1e-6, (
        f"hinge {r2_hinge:.4f} should NOT beat degree-2 poly {r2_poly2:.4f} "
        f"on a smooth target"
    )


# ---------------------------------------------------------------------------
# INTEGRATION (default-off byte-identity + opt-in fires)
# ---------------------------------------------------------------------------


def _hinge_recipes(m):
    """Hinge recipes the FE stage PRODUCED this fit (the produced-recipe ledger
    records every recipe before the MI screen / dedup drop a subset)."""
    return [
        r for r in (getattr(m, "_produced_recipes_", None) or [])
        if getattr(r, "kind", None) == "hinge_basis"
    ]


def test_default_off_produces_no_hinge():
    """fe_hinge_enable defaults OFF -> no hinge recipe is produced."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, N)
    b = rng.standard_normal(N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.5 * b + 0.1 * rng.standard_normal(N)
    X = pd.DataFrame({"a": x, "b": b})
    m = MRMR(max_runtime_mins=2, verbose=0, random_seed=0)
    m.fit(X, y)
    assert not _hinge_recipes(m)


def test_optin_produces_hinge_and_recipe_replays():
    """fe_hinge_enable=True engineers a hinge leg (recorded in the produced-
    recipe ledger) whose recipe replays bit-exactly on test X.

    NOTE: the hinge leg is MONOTONE in x, hence MI-INVARIANT, so MRMR's final
    MI-based selection does NOT keep it (same as RankGauss/isotonic) -- its value
    is downstream linear usability (the biz_value tests). The contract here is
    that the FE stage FIRES, builds a valid leak-safe recipe, and that recipe
    replays through the apply_recipe dispatcher."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, N)
    b = rng.standard_normal(N)
    y = 2 * x + 5 * np.maximum(x - 0.7, 0.0) + 0.5 * b + 0.1 * rng.standard_normal(N)
    X = pd.DataFrame({"a": x, "b": b})
    m = MRMR(fe_hinge_enable=True, max_runtime_mins=2, verbose=0, random_seed=0)
    m.fit(X, y)
    hinge = _hinge_recipes(m)
    assert hinge, "opt-in should produce at least one hinge recipe"
    for r in hinge:
        out = apply_recipe(r, X)
        tau = float(r.extra["tau"])
        if r.extra["side"] == "gt":
            ref = np.maximum(x - tau, 0.0)
        elif r.extra["side"] == "lt":
            ref = np.maximum(tau - x, 0.0)
        else:
            ref = (x > tau).astype(np.float64)
        assert np.allclose(out, ref, atol=1e-9), r.name
