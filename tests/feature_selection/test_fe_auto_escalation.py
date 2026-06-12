"""Tests for the FE AUTO-ESCALATION to richer shipped bases (2026-06-10, backlog idea B).

Covers:
* biz_value WIN: an INNER-frequency pair signal (``y = sin(3.7*a)*b``) that the
  unary/binary search (incl. the default degree-4 chebyshev prewarp) cannot express is
  recovered by the demodulated adaptive-frequency Fourier escalation -- and the emitted
  recipe replays bit-identically at transform() time (regression for the nested
  wavelet-of-fourier replay KeyError too: pre-fix, transform() raised
  ``KeyError('x0__p2sin1')`` on this very fixture).
* NOISE control (hard gate): pure-noise pairs FORCED into escalation across seeds
  propose/admit nothing.
* Silence on EXACT captures: a pair whose library capture is complete (``y=x0*x1``,
  expressible exactly) is never escalated -- the underdelivery trigger's
  discretisation-residual control (leg 3) keeps it out.
* COMPLETION of an underdelivering capture: ``y = He3(x0)*x1`` is admitted via the
  default prewarp but the capture is measurably INCOMPLETE (held-out R^2 ~0.80 of
  ~0.99 achievable; leftover CMI ~0.40 nats beyond its 10-bin code), so the
  UNDERDELIVERY trigger escalates the pair, the proposers fit the binned-mean
  RESIDUAL given the capture, and the admitted escalated feature must ADD genuine
  held-out R^2 (and never degrade the default capture's).
* ``fourier_adaptive`` prewarp spec replay: closed-form, bit-identical fit/transform for
  both the linear and the quadratic-chirp argument.
* Target-rebin guard: the adaptive ``nbins_strategy`` (default mdlp) must never leave a
  degenerate self-referential encoding on a heavy-tailed CONTINUOUS target.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _frame(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X, pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), rng


# ---------------------------------------------------------------------------
# biz_value WIN: inner-frequency pair recovery (fixture a)
# ---------------------------------------------------------------------------


def test_biz_val_escalation_recovers_inner_frequency_pair_and_replays():
    """``y = sin(3.7*x0)*x1``: the library unaries cannot express the inner frequency
    (sin(x) has the wrong period; the degree-4 poly prewarp cannot track ~3.5 cycles),
    so the default unary/binary search admits nothing genuine for the (x0, x1) pair.
    The escalation's demodulated adaptive-Fourier proposer must recover it: an
    ``esc_fourier_mul(x0,x1)`` (or chirp) recipe whose detected frequency maps back to
    ~3.7 on the raw axis, admitted by the existing gates, and replayable on fresh rows."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, df, rng = _frame(5000, 6, 42)
    y = np.sin(3.7 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=len(X))
    sel = MRMR(verbose=0, random_seed=42)
    sel.fit(df, pd.Series(y, name="y"))

    # ``_engineered_recipes_`` is a LIST of EngineeredRecipe (selected ones only).
    recipes = {r.name: r for r in (getattr(sel, "_engineered_recipes_", None) or [])}
    esc_names = [nm for nm in recipes if nm.startswith("esc_")]
    assert esc_names, (
        f"escalation must admit a richer-basis pair feature; recipes={list(recipes)}; "
        f"info={getattr(sel, 'fe_escalation_info_', None)}"
    )
    # The detected z-space frequency must map back to ~3.7 rad on the raw x0 axis:
    # inner_freq = 2*pi*f_z / span.
    import orjson
    name = esc_names[0]
    extra = dict(recipes[name].extra)
    pp = orjson.loads(extra["prewarp_a_preprocess"])
    inner = 2.0 * np.pi * float(pp["freqs"][0]) / float(pp["span"])
    assert 3.2 <= inner <= 4.2, f"detected inner frequency {inner:.3f} should approximate 3.7"

    # transform() replay on FRESH rows must work (also the regression test for the
    # nested wavelet-of-fourier KeyError: pre-fix this raised KeyError('x0__p2sin1')).
    Xt = rng.normal(size=(400, 6))
    out = sel.transform(pd.DataFrame(Xt, columns=df.columns))
    assert name in list(out.columns)
    # Replay correctness: the engineered column must track sin(3.7*a)*b on fresh rows.
    truth = np.sin(3.7 * Xt[:, 0]) * Xt[:, 1]
    got = np.asarray(out[name], dtype=np.float64)
    corr = abs(float(np.corrcoef(got, truth)[0, 1]))
    assert corr >= 0.8, f"replayed escalated feature must track the true signal; corr={corr:.3f}"


def test_escalation_silent_when_library_capture_is_exact():
    """``y = x0*x1`` is EXACTLY expressible by the library (mul/div forms), so the
    admitted capture is COMPLETE: the underdelivery trigger's discretisation-residual
    control (leg 3 -- the joint's leftover CMI does not exceed the capture's own
    finer-binning refinement) must keep (x0, x1) OUT of escalation entirely, and no
    esc_ feature may appear anywhere in the fit."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, df, rng = _frame(5000, 6, 7)
    y = X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=len(X))
    sel = MRMR(verbose=0, random_seed=7)
    sel.fit(df, pd.Series(y, name="y"))
    recipes = {r.name: r for r in (getattr(sel, "_engineered_recipes_", None) or [])}
    assert any(("x0" in nm and "x1" in nm) for nm in recipes), (
        f"the exact library capture should be admitted+selected: {list(recipes)}"
    )
    for h in getattr(sel, "fe_escalation_history_", []) or []:
        assert ("x0", "x1") not in (h.get("eligible_pairs") or []), (
            "a COMPLETELY captured pair must never be escalated (leg-3 control)"
        )
    assert not [nm for nm in recipes if nm.startswith("esc_")]


def test_biz_val_escalation_skips_complete_he3_capture():
    """``y = He3(x0)*x1``: the per-operand pre-warp ALS captures the degree-3
    non-monotone inner ESSENTIALLY EXACTLY (held-out R^2 ~1.0 of the true signal),
    so this pair is NOT underdelivering and the escalation trigger must LEAVE IT
    ALONE -- there is no residual signal for a complement to add.

    HISTORY: before the 2026-06-11 continuous-y ALS-reconstruction-target fix the
    He3 capture was throttled to held-out R^2 ~0.80 because the prewarp ALS fit
    against the coarse 10-bin equal-frequency target codes the 2026-06-10
    target-rebin guard produces (the reconstruction is a least-squares solve and the
    binned codes cost it the degree-3 tail). That regression made the capture LOOK
    underdelivering and the escalation fired to backfill the lost ~0.20 R^2. With
    the ALS now reconstructing against the CONTINUOUS y the capture is complete, so
    the correct behaviour is the same as the He2 leg-3 control: a completely captured
    pair is never escalated, and no ``esc_`` complement is admitted (it would carry
    no genuine missing signal). The MI screen / gates still see the binned codes --
    only the ALS reconstruction target changed."""
    import numpy.linalg as la

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, df, rng = _frame(5000, 6, 42)
    y = (X[:, 0] ** 3 - 3 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=len(X))
    sel = MRMR(verbose=0, random_seed=42)
    sel.fit(df, pd.Series(y, name="y"))
    recipes = {r.name: r for r in (getattr(sel, "_engineered_recipes_", None) or [])}
    assert any("prewarp(x0)" in nm for nm in recipes), f"default prewarp should stay: {list(recipes)}"
    # A COMPLETELY captured pair must never be escalated (same contract as the He2
    # leg-3 control): the continuous-y ALS makes the He3 capture complete, so the
    # (x0, x1) pair must NOT appear in any escalation round's eligible_pairs.
    for h in getattr(sel, "fe_escalation_history_", []) or []:
        assert ("x0", "x1") not in (h.get("eligible_pairs") or []), (
            "a COMPLETELY captured He3 pair must never be escalated"
        )

    Xt = rng.normal(size=(4000, 6))
    out = sel.transform(pd.DataFrame(Xt, columns=df.columns))
    truth = (Xt[:, 0] ** 3 - 3 * Xt[:, 0]) * Xt[:, 1]

    def _r2(cols):
        if not cols:
            return 0.0
        A = np.column_stack([np.asarray(out[c], dtype=np.float64) for c in cols] + [np.ones(len(out))])
        coef, *_ = la.lstsq(A, truth, rcond=None)
        return float(1.0 - (truth - A @ coef).var() / truth.var())

    pre_cols = [c for c in out.columns if "prewarp" in c and not c.startswith("esc_")]
    r2_default = _r2(pre_cols)
    # The capture alone must reach near-exact held-out reconstruction (the recovered
    # ~1.0; was throttled to ~0.80 under the target-rebin regression).
    assert r2_default >= 0.95, (
        f"the continuous-y prewarp ALS should capture He3 essentially exactly "
        f"(held-out R^2={r2_default:.4f}); a low value means the ALS reconstruction "
        f"target regressed back to the coarse binned codes"
    )
    # No esc complement was needed for x0*x1; if one was admitted at all it must not
    # have been for the already-complete He3 pair.
    esc_cols = [c for c in out.columns if c.startswith("esc_") and ("x0" in c and "x1" in c)]
    assert not esc_cols, (
        f"no escalation complement should target the already-complete He3 pair: {esc_cols}"
    )


# ---------------------------------------------------------------------------
# NOISE control: escalation proposes, gates reject -> 0 admitted (hard gate)
# ---------------------------------------------------------------------------


def test_escalation_noise_control_zero_admissions():
    """Pure-noise pairs FORCED into escalation (as if they passed the prescreen by
    chance) must yield 0 admitted features: the held-out detector floors + maxT /
    permutation floors + S5 CMI gate are the hard noise gates."""
    from mlframe.feature_selection.filters._fe_auto_escalation import run_fe_auto_escalation
    from mlframe.feature_selection.filters.discretization import discretize_array
    from mlframe.feature_selection.filters.mrmr import MRMR

    n, p = 4000, 6
    total_admitted = 0
    total_proposed = 0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
        y_cont = rng.normal(size=n)  # independent of X
        classes_y = discretize_array(arr=y_cont, n_bins=10, method="quantile", dtype=np.int32)
        sel = MRMR(verbose=0, random_seed=seed)
        sel.feature_names_in_ = list(X.columns)
        rk = np.argsort(np.argsort(y_cont, kind="stable"), kind="stable").astype(np.float64)
        sel._fe_escalation_y_rank_ = rk / (n - 1)
        failed = [((i, j), 0.01) for i, j in combinations(range(p), 2)]
        admitted = run_fe_auto_escalation(
            sel, failed_pairs=failed, X=X, cols=list(X.columns), classes_y=classes_y,
            pair_maxt_floor=0.0, admitted_pool={}, verbose=0,
        )
        total_admitted += len(admitted)
        total_proposed += sel.fe_escalation_info_["proposed"]
    assert total_admitted == 0, (
        f"noise control violated: {total_admitted} admitted, {total_proposed} proposed"
    )


# ---------------------------------------------------------------------------
# fourier_adaptive prewarp spec: closed-form replay, fit == transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arg", ["linear", "quadratic"])
def test_fourier_adaptive_prewarp_replay_bit_identical(arg):
    from mlframe.feature_selection.filters.engineered_recipes import (
        _apply_unary_binary, build_unary_binary_recipe,
    )
    from mlframe.feature_selection.filters.hermite_fe import apply_operand_prewarp

    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    m = rng.normal(size=1000)
    if arg == "linear":
        pp = {"arg": "linear", "lo": float(x.min()), "span": float(x.max() - x.min()),
              "freqs": [3.925, 1.5]}
    else:
        z = (x - x.mean()) / x.std()
        u = np.sign(z) * z * z
        pp = {"arg": "quadratic", "mean": float(x.mean()), "std": float(x.std()),
              "lo": float(u.min()), "span": float(u.max() - u.min()), "freqs": [5.0]}
    coef = np.asarray([0.7, -0.3, 0.2, 0.1][: 2 * len(pp["freqs"])], dtype=np.float64)
    spec_w = {"basis": "fourier_adaptive", "degree": len(pp["freqs"]), "coef": coef,
              "preprocess": pp}
    warped = apply_operand_prewarp(x, spec_w)
    assert np.all(np.isfinite(warped)) and float(np.std(warped)) > 0

    # Identity mate spec (chebyshev degree-1 coef [0,1]).
    from mlframe.feature_selection.filters.hermite_fe import _POLY_BASES
    _, mp = _POLY_BASES["chebyshev"]["fit"](m)
    spec_m = {"basis": "chebyshev", "degree": 1,
              "coef": np.asarray([0.0, 1.0]), "preprocess": dict(mp)}
    fit_vals = np.nan_to_num(warped * apply_operand_prewarp(m, spec_m),
                             nan=0.0, posinf=0.0, neginf=0.0)
    recipe = build_unary_binary_recipe(
        name=f"esc_test_{arg}", src_a_name="xa", src_b_name="xb",
        unary_a_name="prewarp", unary_b_name="prewarp", binary_name="mul",
        unary_preset="medium", binary_preset="minimal",
        quantization_nbins=10, quantization_method="quantile",
        quantization_dtype=np.int32, fit_values_for_edges=fit_vals,
        prewarp_a=spec_w, prewarp_b=spec_m,
    )
    df = pd.DataFrame({"xa": x, "xb": m})
    replayed = _apply_unary_binary(recipe, df)
    np.testing.assert_array_equal(replayed, fit_vals)


# ---------------------------------------------------------------------------
# F2-shape (c,d) proposer recovery (module-level; the end-to-end F2 verdict is
# benchmark-documented, this pins the PROPOSER + GATE behaviour on clean codes)
# ---------------------------------------------------------------------------


def test_escalation_poly_proposer_recovers_smooth_product_term():
    """``t = log(2c)*sin(d/3)`` inside a mixed target: the rank-1 ALS escalation
    proposer must reconstruct the genuine (c,d) form (corr vs truth >= 0.9) and the
    gates must admit it on clean equal-frequency target codes."""
    from mlframe.feature_selection.filters._fe_auto_escalation import run_fe_auto_escalation
    from mlframe.feature_selection.filters.discretization import discretize_array
    from mlframe.feature_selection.filters.mrmr import MRMR

    n = 8000
    rng = np.random.default_rng(0)
    a = rng.normal(size=n); b = rng.normal(size=n)
    c = np.abs(rng.normal(size=n)) + 0.1
    d = rng.normal(size=n)
    f = rng.normal(size=n)
    y = 0.2 * a ** 2 / b + f / 5 + np.log(c * 2) * np.sin(d / 3)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "f": f})
    classes_y = discretize_array(arr=y, n_bins=10, method="quantile", dtype=np.int32)
    sel = MRMR(verbose=0, random_seed=42)
    sel.feature_names_in_ = list(X.columns)
    rk = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    sel._fe_escalation_y_rank_ = rk / (n - 1)
    admitted = run_fe_auto_escalation(
        sel, failed_pairs=[((0, 1), 0.5), ((2, 3), 0.4)], X=X, cols=list(X.columns),
        classes_y=classes_y, pair_maxt_floor=0.0, admitted_pool={}, verbose=0,
    )
    cd = [x for x in admitted if x["pair"] == ("c", "d")]
    assert cd, f"(c,d) must be admitted; info={sel.fe_escalation_info_}"
    truth = np.log(c * 2) * np.sin(d / 3)
    corr = abs(float(np.corrcoef(cd[0]["values"], truth)[0, 1]))
    assert corr >= 0.9, f"escalated (c,d) candidate must track the true term; corr={corr:.3f}"


# ---------------------------------------------------------------------------
# Target-rebin guard: adaptive nbins_strategy must not degrade a continuous target
# ---------------------------------------------------------------------------


def test_target_rebin_guard_fires_on_heavy_tailed_continuous_target(caplog):
    """A heavy-tailed continuous y under the default ``nbins_strategy='mdlp'`` used to
    get a degenerate self-referential encoding (37 bins, 84% of rows in ONE bin on the
    F2 fixture) that blinded every downstream MI gate. The guard must re-bin the target
    with the legacy equal-frequency quantile codes (and log that it did)."""
    import logging

    from mlframe.feature_selection.filters.mrmr import MRMR

    n = 4000
    rng = np.random.default_rng(0)
    a = rng.normal(size=n); b = rng.normal(size=n)
    e = rng.normal(size=n)
    y = 0.2 * a ** 2 / b + 0.1 * rng.normal(size=n)  # heavy-tailed via 1/b
    df = pd.DataFrame({"a": a, "b": b, "e": e})
    sel = MRMR(verbose=1, random_seed=42)
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters.mrmr"):
        sel.fit(df, pd.Series(y, name="y"))
    assert any("target-rebin guard" in r.message for r in caplog.records), (
        "the target-rebin guard must fire on a heavy-tailed continuous target under "
        "the adaptive nbins_strategy default"
    )
