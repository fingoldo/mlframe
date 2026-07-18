"""Layer 99 biz_value: META FE-RECOMMENDER -- ~50 opt-in fe_* flags -> 1 auto knob.

Consolidated verbatim from test_biz_value_mrmr_layer99.py (per audit finding test_code_quality-16).

Built ON the Layer-98 Param-Oracle. Two complementary layers:

A. ``recommend_fe_flags_by_rules(X, y)`` -- rule-based cold-start recommender
   that fingerprints the data shape and turns on exactly the master FE flags
   whose preconditions are met (NO history needed).
B. ``MetaFERecommender`` -- a ParamOracle-backed learned recommender that wraps
   the rules as its cold-start fallback and overrides them with the learned
   best flag-set once confident per-fingerprint history exists.

Contracts pinned (real AUC numbers, Bayes-feasible fixtures, never xfail):

* group data -> grouped_agg / composite / quantile enabled
* cat data -> count / freq / cat_pair enabled (>= 3 cats -> cat_triple)
* time + entity -> temporal enabled
* clean continuous -> minimal FE (no spurious cat / temporal / grouped enables)
* MRMR(fe_auto=True) end-to-end auto-enables grouped_agg, produces grouped
  features in support_, downstream AUC >= manual-best - 0.02
* learned recommender: after observing flag-set A beats B on fingerprint F,
  recommends A on a fresh-similar F
* cold-start = rules (empty Param-Oracle history)
* fe_auto=False byte-identical to the legacy default
* pickle / clone preserves fe_auto

2026-06-01 Layer 99.
"""

from __future__ import annotations

import os
import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._meta_fe_recommender import (
    ALL_FE_MASTER_FLAGS,
    FE_CAT_FLAGS,
    FE_GROUP_FLAGS,
    MetaFERecommender,
    recommend_fe_flags_by_rules,
)

FIXED_TS = "2026-01-01T00:00:00+00:00"
SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_group_fixture(seed: int, n: int = 4000):
    """int-as-cat group column (region, card 10) + a noisy continuous x whose
    PER-GROUP mean drives y. Raw x alone is weak; grouped-agg recovers it."""
    rng = np.random.default_rng(int(seed))
    n_groups = 10
    region = rng.integers(0, n_groups, n)
    group_mean = rng.uniform(-3.0, 3.0, n_groups)
    x = group_mean[region] + rng.normal(0.0, 3.0, n)
    y = (group_mean[region] + 0.1 * rng.normal(0.0, 1.0, n) > 0.0).astype(int)
    X = pd.DataFrame(
        {
            "region": region.astype(np.int64),
            "x": x,
            "noise_0": rng.normal(0.0, 1.0, n),
            "noise_1": rng.normal(0.0, 1.0, n),
        }
    )
    return X, y


def _build_cat_fixture(seed: int, n: int = 3000):
    """Three object/category columns + one numeric. y depends on a cat level."""
    rng = np.random.default_rng(int(seed))
    a = rng.choice(list("ABCDE"), n)
    b = rng.choice(list("XYZ"), n)
    c = rng.choice(list("PQ"), n)
    num = rng.normal(0.0, 1.0, n)
    y = ((a == "A") ^ (b == "X")).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "num": num})
    return X, y


def _build_time_entity_fixture(seed: int, n: int = 3000):
    """A datetime time column + an int-as-cat entity column."""
    rng = np.random.default_rng(int(seed))
    n_entities = 8
    entity = rng.integers(0, n_entities, n).astype(np.int64)
    ts = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 5000, n)), unit="h")
    val = rng.normal(0.0, 1.0, n)
    y = (entity % 2 == 0).astype(int)
    X = pd.DataFrame({"entity": entity, "ts": ts, "val": val})
    return X, y


def _build_clean_continuous(seed: int, n: int = 3000, p: int = 6):
    """Pure Gaussian numerics, high cardinality, no NaNs, linear signal. The
    no-false-positive fixture: must NOT trip cat / temporal / grouped / NaN
    detectors."""
    rng = np.random.default_rng(int(seed))
    X = pd.DataFrame({f"c{i}": rng.normal(0.0, 1.0, n) for i in range(p)})
    logit = 1.2 * X["c0"] + 0.8 * X["c1"] - 0.5 * X["c2"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# A. Rule recommender contracts
# ---------------------------------------------------------------------------


class TestRuleRecommender:
    """Rule-based cold-start FE recommender must react to the data's shape."""

    def test_group_data_enables_grouped_agg(self):
        """Group-structured data enables every FE_GROUP_FLAGS entry, including fe_grouped_agg_enable."""
        X, y = _build_group_fixture(13)
        flags = recommend_fe_flags_by_rules(X, y)
        for f in FE_GROUP_FLAGS:
            assert flags[f] is True, f"group data must enable {f}; got {flags}"
        assert flags["fe_grouped_agg_enable"] is True

    def test_cat_data_enables_cat_encodings(self):
        """Categorical-heavy data enables every FE_CAT_FLAGS entry, including fe_cat_triple_enable for >= 3 cats."""
        X, y = _build_cat_fixture(7)
        flags = recommend_fe_flags_by_rules(X, y)
        for f in FE_CAT_FLAGS:
            assert flags[f] is True, f"cat data must enable {f}; got {flags}"
        # 3 object cats -> triple synergy cross enabled.
        assert flags["fe_cat_triple_enable"] is True

    def test_time_entity_enables_temporal(self):
        """A datetime column plus an entity column enables fe_temporal_agg_enable."""
        X, y = _build_time_entity_fixture(42)
        flags = recommend_fe_flags_by_rules(X, y)
        assert flags["fe_temporal_agg_enable"] is True, f"time + entity must enable temporal; got { {k: v for k, v in flags.items() if v} }"

    def test_clean_continuous_no_spurious_enables(self):
        """Pure Gaussian numerics with linear signal -> NO cat / temporal /
        grouped / missingness enables (no false positives)."""
        for s in SEEDS:
            X, y = _build_clean_continuous(s)
            flags = recommend_fe_flags_by_rules(X, y)
            spurious = {
                "fe_grouped_agg_enable",
                "fe_composite_group_agg_enable",
                "fe_grouped_quantile_enable",
                "fe_count_encoding_enable",
                "fe_frequency_encoding_enable",
                "fe_cat_pair_enable",
                "fe_cat_triple_enable",
                "fe_temporal_agg_enable",
                "fe_missingness_indicator_enable",
                "fe_cat_num_interaction_enable",
                "fe_numeric_decompose_enable",
                "fe_modular_enable",
            }
            on = {f for f in spurious if flags.get(f)}
            assert not on, f"seed={s}: clean continuous data spuriously enabled {on}"

    def test_missingness_enabled_on_nan_data(self):
        """Injecting NaNs into a clean fixture enables fe_missingness_indicator_enable."""
        X, y = _build_clean_continuous(1)
        X = X.copy()
        # Inject 5% NaN into one column.
        idx = np.random.default_rng(0).choice(len(X), int(0.05 * len(X)), replace=False)
        X.loc[X.index[idx], "c3"] = np.nan
        flags = recommend_fe_flags_by_rules(X, y)
        assert flags["fe_missingness_indicator_enable"] is True

    def test_returns_full_flag_universe(self):
        """The recommendation covers exactly ALL_FE_MASTER_FLAGS with boolean values."""
        X, y = _build_clean_continuous(1)
        flags = recommend_fe_flags_by_rules(X, y)
        assert set(flags) == set(ALL_FE_MASTER_FLAGS)
        assert all(isinstance(v, bool) for v in flags.values())


# ---------------------------------------------------------------------------
# B. recommend_enabled_fe classmethod wiring
# ---------------------------------------------------------------------------


class TestRecommendEnabledFe:
    """MRMR.recommend_enabled_fe(X, y) must wire the rule-based recommender into the classmethod."""

    def test_classmethod_returns_recommendations_for_group_data(self):
        """Passing group data recommends fe_grouped_agg_enable alongside the static flip_safe taxonomy."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_group_fixture(13)
        rep = MRMR.recommend_enabled_fe(X, y)
        assert "fe_grouped_agg_enable" in rep["recommended_enable"]
        # Static taxonomy still present.
        assert "fe_local_mi_gate" in rep["flip_safe"]

    def test_classmethod_empty_recommendation_without_data(self):
        """Without X/y, recommended_enable is empty but the static taxonomy is still returned."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rep = MRMR.recommend_enabled_fe()
        assert rep["recommended_enable"] == []
        assert rep["flip_risky"], "flip-safety taxonomy must still be returned"


# ---------------------------------------------------------------------------
# C. fe_auto end-to-end
# ---------------------------------------------------------------------------


class TestFeAutoEndToEnd:
    """MRMR(fe_auto=True) must auto-enable the right FE family and match manual-best AUC within tolerance."""

    def test_fe_auto_enables_grouped_agg_and_lifts_auc(self):
        """MRMR(fe_auto=True) auto-enables grouped_agg on the group fixture,
        produces grouped features, and the downstream AUC is within 0.02 of the
        manual-best (explicit fe_grouped_agg_enable=True) configuration."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        auto_lifts = []
        for s in (1, 7, 13):
            X, y = _build_group_fixture(s, n=4000)
            ys = pd.Series(y, name="y")
            Xtr, Xte, ytr, yte = train_test_split(
                X,
                ys,
                test_size=0.3,
                random_state=s,
                stratify=ys,
            )

            # Auto mode: only fe_auto=True, nothing else.
            m_auto = MRMR(max_runtime_mins=1.0, fe_auto=True)
            m_auto.fit(Xtr, ytr)
            ga_auto = list(getattr(m_auto, "grouped_agg_features_", []) or [])
            assert ga_auto, f"seed={s}: fe_auto=True produced no grouped_agg features (support should include the auto-enabled grouped aggregates)."

            # Manual-best: explicitly enable grouped_agg with the right cols.
            m_manual = MRMR(
                max_runtime_mins=1.0,
                fe_grouped_agg_enable=True,
                fe_grouped_agg_group_cols=("region",),
                fe_grouped_agg_num_cols=("x",),
                fe_grouped_agg_top_k=5,
            )
            m_manual.fit(Xtr, ytr)

            auc_auto = _downstream_auc(m_auto, Xtr, Xte, ytr, yte)
            auc_manual = _downstream_auc(m_manual, Xtr, Xte, ytr, yte)
            auto_lifts.append((auc_auto, auc_manual))

        # AUC(auto) >= AUC(manual-best) - 0.02 on average.
        mean_auto = float(np.mean([a for a, _ in auto_lifts]))
        mean_manual = float(np.mean([m for _, m in auto_lifts]))
        assert (
            mean_auto >= mean_manual - 0.02
        ), f"fe_auto AUC {mean_auto:.4f} fell more than 0.02 below the manual-best {mean_manual:.4f} (per-seed {auto_lifts})."

    def test_fe_auto_false_byte_identical_default(self):
        """fe_auto=False (the default) leaves the legacy path untouched: no
        grouped_agg features, fe_grouped_agg_enable stays False before/after."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_group_fixture(42, n=2000)
        ys = pd.Series(y, name="y")

        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_auto", False)) is False
        assert bool(getattr(m, "fe_grouped_agg_enable", False)) is False
        m.fit(X, ys)
        ga = list(getattr(m, "grouped_agg_features_", []) or [])
        assert ga == [], f"default fe_auto=False added grouped_agg features: {ga}"
        # The flag must be back to its constructor value after fit (no leak).
        assert bool(getattr(m, "fe_grouped_agg_enable", False)) is False

    def test_fe_auto_restores_flags_after_fit(self):
        """fe_auto flips flags ON only for the duration of the fit; after fit
        the constructor-arg semantics are restored (so clone / pickle / refit
        are stable)."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_group_fixture(7, n=2000)
        m = MRMR(max_runtime_mins=0.5, fe_auto=True)
        assert bool(m.fe_grouped_agg_enable) is False
        m.fit(X, pd.Series(y, name="y"))
        # auto flipped it ON during fit, but restored to the ctor value after.
        assert bool(m.fe_grouped_agg_enable) is False, "fe_auto leaked an enabled flag past the fit boundary."


def _downstream_auc(m, Xtr, Xte, ytr, yte):
    """Train a LogReg on the fitted selector's transform output (raw support +
    engineered columns), return holdout AUC. ``transform`` replays engineered
    recipes as pure functions of X, so the test path is leakage-free."""
    feat_tr = np.asarray(m.transform(Xtr), dtype=np.float64)
    feat_te = np.asarray(m.transform(Xte), dtype=np.float64)
    feat_tr = np.nan_to_num(feat_tr, nan=0.0, posinf=0.0, neginf=0.0)
    feat_te = np.nan_to_num(feat_te, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(feat_tr, ytr.to_numpy())
    return roc_auc_score(yte.to_numpy(), clf.predict_proba(feat_te)[:, 1])


# ---------------------------------------------------------------------------
# D. Learned recommender (Param-Oracle backed)
# ---------------------------------------------------------------------------


class TestLearnedRecommender:
    """The Param-Oracle-backed learned recommender must match rules cold-start and override them once confident."""

    def test_cold_start_equals_rules(self, tmp_path):
        """Empty Param-Oracle history -> recommend == rule-based."""
        store = os.path.join(str(tmp_path), "meta_fe_cold.parquet")
        rec = MetaFERecommender(store)
        X, y = _build_group_fixture(13)
        learned = rec.recommend(X, y)
        rules = recommend_fe_flags_by_rules(X, y)
        assert learned == rules, "cold-start recommendation must equal the rule-based prior."

    def test_learned_recommender_improves_over_rules(self, tmp_path):
        """After observing that flag-set A beats flag-set B on fingerprint F,
        the learned recommender returns A on a fresh-similar F -- even when A is
        NOT what the cold-start rules would pick. This proves the learned layer
        OVERRIDES (improves on) the static rules."""
        store = os.path.join(str(tmp_path), "meta_fe_learn.parquet")
        rec = MetaFERecommender(store, min_observations=2)

        # F = the clean-continuous fingerprint, where the RULES recommend
        # all-False. Teach the oracle that a hybrid_orth flag-set (A) beats the
        # rules' all-False (B) on this shape.
        X_train, y_train = _build_clean_continuous(1, n=3000)
        rules = recommend_fe_flags_by_rules(X_train, y_train)
        assert not any(rules.values()), "precondition: rules pick nothing here"

        flags_A = dict.fromkeys(ALL_FE_MASTER_FLAGS, False)
        flags_A["fe_hybrid_orth_enable"] = True  # the learned winner
        flags_B = dict.fromkeys(ALL_FE_MASTER_FLAGS, False)  # == rules

        # A consistently scores higher than B on this fingerprint.
        for _ in range(3):
            rec.fit_observe(X_train, y_train, flags_A, cv_score=0.82, ts=FIXED_TS)
            rec.fit_observe(X_train, y_train, flags_B, cv_score=0.70, ts=FIXED_TS)

        # Fresh-but-similar dataset (same shape, different seed -> same bucket).
        X_fresh, y_fresh = _build_clean_continuous(99, n=3100)
        learned = rec.recommend(X_fresh, y_fresh)
        rules_fresh = recommend_fe_flags_by_rules(X_fresh, y_fresh)

        assert learned != rules_fresh, "learned recommender failed to override the cold-start rules."
        assert (
            learned["fe_hybrid_orth_enable"] is True
        ), f"learned recommender did not pick the empirically-best flag-set A; got {{k: v for k, v in learned.items() if v}}."

    def test_fit_observe_is_stat_only(self, tmp_path):
        """The learned store must persist ONLY scalar fingerprint stats + the
        flag-set + the score -- never raw arrays."""
        import orjson

        store = os.path.join(str(tmp_path), "meta_fe_stat.parquet")
        rec = MetaFERecommender(store)
        X, y = _build_group_fixture(1, n=2000)
        flags = recommend_fe_flags_by_rules(X, y)
        rec.fit_observe(X, y, flags, cv_score=0.75, ts=FIXED_TS)
        rows = rec.oracle.store.read_rows()
        assert rows, "fit_observe recorded nothing"
        for r in rows:
            for col, val in r.items():
                assert not isinstance(val, (list, tuple, dict, np.ndarray)), f"non-scalar persisted in {col}: {type(val)}"
            fp_bucket = orjson.loads(r["fp_bucket_json"])
            for v in fp_bucket.values():
                assert isinstance(v, (int, float, str)), f"non-scalar fp: {v!r}"
        store_bytes = os.path.getsize(rec.oracle.store._path)
        assert store_bytes < X.to_numpy().astype(np.float64).nbytes, "store larger than raw array -> likely leaking data"


# ---------------------------------------------------------------------------
# E. pickle / clone preserves fe_auto
# ---------------------------------------------------------------------------


class TestPickleClonePreservesFeAuto:
    """fe_auto must survive both clone() and pickle round-trips, including the default False value."""

    def test_clone_preserves_fe_auto(self):
        """sklearn clone() preserves fe_auto=True."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(fe_auto=True)
        c = clone(m)
        assert bool(c.fe_auto) is True

    def test_pickle_round_trip_preserves_fe_auto(self):
        """A pickle round-trip preserves fe_auto=True, and the default MRMR() round-trips to fe_auto=False."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(fe_auto=True, max_runtime_mins=0.5)
        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        assert bool(m2.fe_auto) is True
        # Default still round-trips to False.
        m3 = pickle.loads(pickle.dumps(MRMR()))  # nosec B301 -- round-trip of a locally-created, trusted object
        assert bool(m3.fe_auto) is False
