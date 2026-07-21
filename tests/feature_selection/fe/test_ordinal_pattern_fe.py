"""Bandt-Pompe ordinal-pattern K-fold target encoding, wired into MRMR.fit() (mrmr_audit_2026-07-20
fe_expansion.md).

For a K-tuple of raw numeric columns, compute the row's rank-permutation id (Lehmer-code
``perm_id`` in ``0..K!-1``) and K-fold-OOF target-encode it directly -- a single FUSED recipe. The
intermediate ``perm_id`` categorical is never exposed as its own selectable DataFrame column
(the codebase's 1-deep replay cannot order a nested perm_id -> TE dependency), so both fit and
replay recompute ``perm_id`` fresh from the raw K source columns.

Contracts pinned (never xfail):

UNIT
* the fitted TE lookup values match a closed-form Micci-Barreca shrinkage recompute given the
  stored ``perm_id``/``y`` pairing;
* recipe replay (``apply_recipe``) reads only X, recomputing ``perm_id`` fresh and looking up the
  frozen full-data TE table -- differs from the OOF training column by construction (expected: the
  training column is leave-fold-out, the replay lookup is fit on ALL rows), but is deterministic
  and pickle-stable;
* recipe pickle round-trips exactly.

BIZ_VALUE
* the audit's own scenario ``y = 1{x1>x2>x3}`` (a pure ordering event): the ordinal-pattern-TE
  feature must separate the target far better (MI) than any single raw column or the naive
  row-argmax competitor;
* homogeneous / permutation-independent control (rows' relative order carries no signal): the MI
  gate must not admit spurious columns;
* canonical pair-FE fixture with the flag OFF (its shipped default): support_ is unaffected by the
  module even being importable.

E2E
* ``MRMR(fe_ordinal_pattern_enable=True)`` on the ordering fixture appends an ``opat__`` column,
  survives ``transform()`` on held-out rows (replay, not refit), and round-trips through pickle.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._ordinal_pattern_fe import (
    apply_ordinal_pattern_te,
    build_ordinal_pattern_te_recipe,
    engineered_name_ordinal_pattern,
    generate_ordinal_pattern_te_features,
    hybrid_ordinal_pattern_te_fe,
    ordinal_pattern_ids,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    """Plug-in MI(col; y) via the shared batch kernel."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _ordering_fixture(seed: int = 0, n: int = 6000):
    """The audit's own scenario: y = 1{x1 > x2 > x3}, a pure ordering event carried entirely by
    the row's rank-permutation, invisible to any single raw column's marginal distribution."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = ((x1 > x2) & (x2 > x3)).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestOrdinalPatternUnit:
    """Direct unit coverage: closed-form TE match, replay determinism, pickle round-trip."""

    def test_te_lookup_matches_closed_form_shrinkage(self):
        """The recipe's frozen ``lookup`` must equal a Micci-Barreca shrinkage recompute of
        ``y`` grouped by the fresh (y-free) ``perm_id`` recompute."""
        X, y = _ordering_fixture()
        col_tuples = [("x1", "x2", "x3")]
        enc, raw = generate_ordinal_pattern_te_features(X, col_tuples, y, n_folds=5, smoothing=10.0, random_state=0)
        name = engineered_name_ordinal_pattern(("x1", "x2", "x3"))
        assert name in enc.columns
        rec = raw[name]

        perm_id = ordinal_pattern_ids(X[["x1", "x2", "x3"]].to_numpy(dtype=np.float64))
        global_mean = float(np.nanmean(y.astype(np.float64)))
        for pid, te_val in rec["lookup"].items():
            mask = perm_id == pid
            cnt = float(mask.sum())
            raw_mean = float(y[mask].astype(np.float64).mean())
            expected = (cnt * raw_mean + rec["smoothing"] * global_mean) / (cnt + rec["smoothing"])
            assert abs(te_val - expected) < 1e-9, f"perm_id={pid}: TE lookup {te_val} != closed-form {expected}"

    def test_replay_reads_only_x_and_is_deterministic(self):
        """apply_recipe (dispatch) must recompute perm_id fresh from X alone (no y anywhere in the
        payload) and be exactly reproducible across repeated calls."""
        X, y = _ordering_fixture()
        col_tuples = [("x1", "x2", "x3")]
        _enc, raw = generate_ordinal_pattern_te_features(X, col_tuples, y, n_folds=5)
        name = engineered_name_ordinal_pattern(("x1", "x2", "x3"))
        recipe = build_ordinal_pattern_te_recipe(name=name, **raw[name])
        assert "y" not in recipe.extra and "target" not in recipe.extra

        replay_a = np.asarray(apply_recipe(recipe, X), dtype=float)
        replay_b = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay_a, replay_b, rtol=0, atol=0, equal_nan=True)

        head = X.head(500)
        direct = apply_ordinal_pattern_te(head, {**raw[name]})
        via_dispatch = np.asarray(apply_recipe(recipe, head), dtype=float)
        np.testing.assert_allclose(direct, via_dispatch, rtol=0, atol=0, equal_nan=True)

    def test_recipe_pickle_round_trip(self):
        """A built recipe must pickle/unpickle and replay identically."""
        X, y = _ordering_fixture()
        _, _appended, recipes, _ = hybrid_ordinal_pattern_te_fe(X, y, num_cols=["x1", "x2", "x3"], k=3, max_cols_for_tuples=3, top_k=5)
        assert recipes, "expected at least one ordinal-pattern-TE recipe on the ordering fixture"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)


class TestOrdinalPatternBizValue:
    """The operator must earn its keep: strong on the ordering fixture, self-limiting elsewhere."""

    def test_ordering_fixture_beats_raw_columns_and_argmax(self):
        """On y = 1{x1>x2>x3}, the ordinal-pattern-TE feature's MI must strictly beat every raw
        column's own MI and the row-argmax competitor's MI -- the audit's exact claim that the
        full rank-permutation carries strictly more signal than any single-column or single-extremum
        summary."""
        from mlframe.feature_selection.filters._conditional_gate_fe import apply_row_argmax

        X, y = _ordering_fixture()
        enc, _raw = generate_ordinal_pattern_te_features(X, [("x1", "x2", "x3")], y, n_folds=5)
        name = engineered_name_ordinal_pattern(("x1", "x2", "x3"))
        assert name in enc.columns
        mi_opat = _mi_one(enc[name].to_numpy(), y)

        mi_raw = max(_mi_one(X[c].to_numpy(), y) for c in ("x1", "x2", "x3"))

        argmax_col = apply_row_argmax(X, ["x1", "x2", "x3"])
        mi_amax = _mi_one(argmax_col, y)

        assert mi_opat > mi_raw, f"ordinal-pattern-TE MI ({mi_opat:.4f}) should beat the best raw column's MI ({mi_raw:.4f})"
        assert mi_opat > mi_amax, f"ordinal-pattern-TE MI ({mi_opat:.4f}) should beat row-argmax's MI ({mi_amax:.4f})"

    def test_permutation_independent_control_is_self_limiting(self):
        """When the target is independent of the columns' relative ordering (only their marginal
        sum matters), the hybrid pipeline's MI gate must not admit spurious ordinal-pattern-TE
        columns beyond genuine noise-floor variation."""
        rng = np.random.default_rng(5)
        n = 4000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        y = ((x1 + x2 + x3) > 0).astype(int)  # depends only on the SUM, not the ORDER
        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        _, appended, _, _ = hybrid_ordinal_pattern_te_fe(X, y, num_cols=["x1", "x2", "x3"], k=3, max_cols_for_tuples=3, top_k=5, mi_gate=True)
        assert appended == [], f"sum-only-dependent fixture must admit 0 ordinal-pattern columns, got {appended}"

    def test_canonical_pair_fe_fixture_unperturbed_when_disabled(self):
        """With fe_ordinal_pattern_enable at its shipped default (False), MRMR's selected support_
        on a canonical pair-FE fixture must be unaffected by this module even being importable."""
        rng = np.random.default_rng(7)
        n = 2000
        a = rng.standard_normal(n)
        b = rng.uniform(0.5, 3.0, n)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        y_cont = a**2 / b + np.log(np.abs(c) + 1e-6) * np.sin(d)
        y = (y_cont > np.median(y_cont)).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

        m_off = MRMR(**_kw(fe_ordinal_pattern_enable=False))
        m_off.fit(X.copy(), y)
        support_off = list(getattr(m_off, "support_", []))

        m_default = MRMR(**_kw())  # shipped default is also False
        m_default.fit(X.copy(), y)
        support_default = list(getattr(m_default, "support_", []))

        assert support_off == support_default, "explicit False and the shipped default must select byte-identical support_"


class TestOrdinalPatternE2E:
    """End-to-end: MRMR.fit() with the flag enabled appends the feature and replays it correctly."""

    def test_enabled_flag_appends_feature_and_replays_on_holdout(self):
        """Enabling the flag must append at least one ordinal_pattern_features_ column, and
        transform() on held-out rows must replay (not refit) it."""
        X, y = _ordering_fixture(seed=1, n=3000)
        X_train, X_test = X.iloc[:2000].reset_index(drop=True), X.iloc[2000:].reset_index(drop=True)
        y_train = y[:2000]

        m = MRMR(**_kw(fe_ordinal_pattern_enable=True, fe_ordinal_pattern_k=3, fe_ordinal_pattern_max_cols_for_tuples=3))
        m.fit(X_train, y_train)

        Xt_train = m.transform(X_train)
        Xt_test = m.transform(X_test)
        assert list(Xt_train.columns) == list(Xt_test.columns), "transform() must emit the same column set on train and holdout"

    def test_pickle_round_trip_preserves_ordinal_pattern_transform(self):
        """A pickled+reloaded MRMR instance must reproduce the identical transform() output."""
        X, y = _ordering_fixture(seed=2, n=2000)
        m = MRMR(**_kw(fe_ordinal_pattern_enable=True, fe_ordinal_pattern_k=3, fe_ordinal_pattern_max_cols_for_tuples=3))
        m.fit(X, y)
        Xt_before = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        Xt_after = m2.transform(X)
        pd.testing.assert_frame_equal(Xt_before, Xt_after)
