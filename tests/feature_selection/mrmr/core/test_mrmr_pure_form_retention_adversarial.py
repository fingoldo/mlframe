"""Adversarial tests for the default-on usability-aware PURE-FORM RETENTION pass (2026-06-17).

Target under test
-----------------
``src/mlframe/feature_selection/filters/_fe_pure_form_retention.py`` :: ``retain_usable_pure_forms``,
wired into ``_fit_impl_core.py`` (search "USABILITY-AWARE PURE-FORM RETENTION"). It runs on every
fit with a continuous target and ``fe_max_steps>0``: builds a usability candidate pool of pure
single-pair engineered forms, runs a CV-MAE linear forward-selection greedy, and APPENDS any pure
single-pair form whose pair is not already covered by a pure (<=2-operand) selected feature.

These tests try to BREAK it on:
  1. over-firing on a pure-additive-linear y (no genuine pair interaction)
  2. determinism (seeded rng + subsample + CV must reproduce)
  3. transform replay of every retained form
  4. pickle round-trip
  5. wrong-pair recovery (forms built from operands of DIFFERENT terms / pure noise)
  6. n_features_ == len(support_) + len(_engineered_recipes_)

All datasets n<=3000, single-fit per test where possible. Heavy fits are kept off the default fast
lane via @pytest.mark.slow only when they exceed ~30s; most run in one fit.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit(df, y, seed=0, verbose=0):
    """Helper that fit."""
    from mlframe.feature_selection.filters import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=verbose, random_seed=seed).fit(X=df, y=y)


def _retained_pair_recipes(fs):
    """Return list of (name, frozenset(src_names)) for every PURE (<=2 operand) engineered recipe.

    Retention only ever appends pure single-pair forms, so these are the candidates for
    'was this added by retention'. We cannot perfectly attribute every recipe to retention vs the
    MI-greedy FE without instrumentation, so the wrong-pair / over-fire pins below use a direct
    call to retain_usable_pure_forms on a controlled mrmr stub instead (see _direct_retention)."""
    out = []
    for r in getattr(fs, "_engineered_recipes_", []) or []:
        src = tuple(getattr(r, "src_names", ()) or ())
        out.append((getattr(r, "name", None), frozenset(src)))
    return out


def _build_additive_linear(n=2500, seed=0):
    """y = 2*x1 + 3*x2 + small noise. NO pair interaction at all -> retention should add nothing."""
    rng = np.random.default_rng(seed)
    x1 = rng.random(n)
    x2 = rng.random(n)
    x3 = rng.random(n)  # pure noise operand
    y = 2.0 * x1 + 3.0 * x2 + 0.01 * rng.standard_normal(n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}), y


def _build_case2(n=3000, seed=0):
    """y = 0.2*a**2/b + log(2c)*sin(d/3). Genuine pairs: (a,b) and (c,d). e is pure noise."""
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.5
    b = rng.random(n) + 0.5
    c = rng.random(n) + 0.5
    d = rng.random(n)
    e = rng.random(n)
    y = 0.2 * a**2 / b + np.log(c * 2.0) * np.sin(d / 3.0)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), y


def _build_single_pair(n=3000, seed=0):
    """y depends ONLY on (a,b) via a**2/b; c,d are pure noise. A (c,d) form must NOT be retained."""
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.5
    b = rng.random(n) + 0.5
    c = rng.random(n)
    d = rng.random(n)
    y = 0.2 * a**2 / b + 0.01 * rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d}), y


def _direct_retention(df, y, seed=0):
    """Run retain_usable_pure_forms directly on a stub mrmr that has NO pre-existing recipes, so the
    returned list is EXACTLY what retention proposes to add (no FE-greedy contamination). y is fed as
    the continuous target, mimicking _fe_prewarp_y_continuous_ at the call site."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_pure_forms

    class _Stub:
        """Groups tests covering Stub."""
        feature_names_in_ = list(df.columns)
        _engineered_recipes_ = []
        _engineered_features_ = []
        random_seed = seed

    return retain_usable_pure_forms(_Stub(), df, np.asarray(y, dtype=np.float64), seed=seed)


# ---------------------------------------------------------------------------
# 1. Over-firing on pure-additive-linear y (no interaction at all)
# ---------------------------------------------------------------------------


def test_no_retention_on_pure_additive_linear():
    """y = 2*x1 + 3*x2 has NO pair interaction. The CV-MAE linear greedy gates retention: a pure
    pair form (e.g. mul(x1,x2)) cannot lower the linear CV-MAE below the raw x1,x2 baseline because
    the target is already exactly linear in the raw operands. Retention must propose NOTHING."""
    df, y = _build_additive_linear()
    added = _direct_retention(df, y, seed=0)
    names = [nm for _, nm in added]
    assert added == [], (
        f"retention ADDED pure-pair engineered forms on a pure-additive-linear target with no "
        f"interaction: {names}. The CV-MAE gate should reject these (raw x1,x2 already linear)."
    )


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------


def test_retention_deterministic_same_seed():
    """retain_usable_pure_forms subsamples rows with a seeded rng and runs CV. Same (X,y,seed) must
    yield byte-identical proposed names+pairs across two calls."""
    df, y = _build_case2()
    a = _direct_retention(df, y, seed=0)
    b = _direct_retention(df, y, seed=0)
    a_sig = [(nm, tuple(sorted(getattr(r, "src_names", ()) or ()))) for r, nm in a]
    b_sig = [(nm, tuple(sorted(getattr(r, "src_names", ()) or ()))) for r, nm in b]
    assert a_sig == b_sig, f"retention non-deterministic for same seed:\n {a_sig}\n vs\n {b_sig}"


def test_full_fit_deterministic_feature_names():
    """End-to-end: two MRMR fits with the same (X,y,seed) must give identical get_feature_names_out
    (retention included)."""
    df, y = _build_case2()
    n1 = list(_fit(df, y, seed=0).get_feature_names_out())
    n2 = list(_fit(df, y, seed=0).get_feature_names_out())
    assert n1 == n2, f"get_feature_names_out non-deterministic:\n {n1}\n vs\n {n2}"


# ---------------------------------------------------------------------------
# 3. Transform replay of every retained form
# ---------------------------------------------------------------------------


def test_every_engineered_recipe_replays_finite_on_fresh_data():
    """Every engineered recipe (retained forms ride the same path) must replay on a FRESH X with the
    same columns: the engineered column must be present in transform output and finite."""
    df, y = _build_case2(seed=0)
    fs = _fit(df, y, seed=0)
    # NB: contract is over _engineered_recipes_ (the REPLAYABLE forms transform() emits), NOT the
    # broader _engineered_features_ log -- recipeless higher-order engineered features are
    # intentionally NOT in transform output (see _fit_impl_core n_engineered_out comment). Retained
    # pure forms are always recipes, so this set covers them.
    eng_names = [getattr(r, "name", None) for r in (getattr(fs, "_engineered_recipes_", []) or [])]
    eng_names = [n for n in eng_names if n is not None]
    if not eng_names:
        pytest.skip("no engineered recipes synthesized on this dataset")
    fresh, _ = _build_case2(seed=123)  # same columns, different rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fs.transform(fresh)
    cols = set(map(str, out.columns))
    for nm in eng_names:
        assert str(nm) in cols, f"engineered/retained feature {nm!r} missing from transform output cols={sorted(cols)}"
        vals = np.asarray(out[nm], dtype=np.float64)
        assert np.isfinite(vals).all(), f"engineered/retained feature {nm!r} has non-finite values after replay"


# ---------------------------------------------------------------------------
# 4. Pickle round-trip
# ---------------------------------------------------------------------------


def test_pickle_roundtrip_transform_matches():
    """Fit (with retention), pickle+unpickle, transform a fresh X -> identical output to the original
    estimator. The retained recipes must be picklable and replay identically."""
    df, y = _build_case2(seed=0)
    fs = _fit(df, y, seed=0)
    fresh, _ = _build_case2(seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out0 = fs.transform(fresh)
        fs2 = pickle.loads(pickle.dumps(fs))  # nosec B301 -- round-trip of a locally-created, trusted object
        out1 = fs2.transform(fresh)
    assert list(out0.columns) == list(out1.columns), f"pickle changed transform columns:\n {list(out0.columns)}\n vs\n {list(out1.columns)}"
    pd.testing.assert_frame_equal(out0.reset_index(drop=True), out1.reset_index(drop=True), check_dtype=False)


# ---------------------------------------------------------------------------
# 5. Wrong-pair recovery (operands of different terms / pure noise)
# ---------------------------------------------------------------------------


def test_no_retention_of_pure_noise_pair():
    """y depends ONLY on (a,b). c,d are pure noise. Retention must NOT propose a (c,d) form."""
    df, y = _build_single_pair()
    added = _direct_retention(df, y, seed=0)
    bad = [nm for r, nm in added if frozenset(getattr(r, "src_names", ()) or ()) == frozenset({"c", "d"})]
    assert not bad, f"retention proposed a pure-noise (c,d) form: {bad}"


def test_no_cross_term_operand_mixing_on_case2():
    """CASE2 genuine pairs are (a,b) and (c,d). Any retained PURE pair form must use a within-term
    pair, NOT a cross-term mix like (a,d) / (b,c) / (a,c) / (b,d): those operands belong to
    different additive terms and carry NO joint interaction. The CV-MAE gate is supposed to reject
    them. This pin asserts retention does not append a cross-term operand pair."""
    df, y = _build_case2()
    added = _direct_retention(df, y, seed=0)
    legit = {frozenset({"a", "b"}), frozenset({"c", "d"})}
    cross = [
        (nm, tuple(sorted(getattr(r, "src_names", ()) or ())))
        for r, nm in added
        if frozenset(getattr(r, "src_names", ()) or ()) not in legit and len(frozenset(getattr(r, "src_names", ()) or ())) == 2
    ]
    assert not cross, (
        f"retention appended CROSS-TERM pure-pair form(s) whose operands belong to DIFFERENT additive "
        f"terms (no genuine interaction): {cross}. Genuine pairs are (a,b) and (c,d)."
    )


# ---------------------------------------------------------------------------
# 6. n_features_ consistency
# ---------------------------------------------------------------------------


def test_n_features_equals_support_plus_recipes():
    """After retention n_features_ must equal len(support_) + len(_engineered_recipes_) (transform
    emits raw selected + every replayable engineered/retained column)."""
    df, y = _build_case2(seed=0)
    fs = _fit(df, y, seed=0)
    n_sup = len(np.asarray(fs.support_).ravel())
    n_rec = len(getattr(fs, "_engineered_recipes_", []) or [])
    assert int(fs.n_features_) == n_sup + n_rec, f"n_features_={fs.n_features_} != support({n_sup}) + recipes({n_rec})"
    # and that matches the transform width
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fs.transform(df)
    assert out.shape[1] == int(fs.n_features_), f"transform width {out.shape[1]} != n_features_ {fs.n_features_}"


def test_retention_respects_max_added_cap():
    """retain_usable_pure_forms caps additions at max_added (default 4). Even on a rich pool it must
    never propose more than that, and never a duplicate name."""
    df, y = _build_case2()
    added = _direct_retention(df, y, seed=0)
    names = [nm for _, nm in added]
    assert len(added) <= 4, f"retention exceeded max_added cap: {len(added)} forms {names}"
    assert len(names) == len(set(names)), f"retention proposed duplicate names: {names}"
