"""Per-strategy unit tests for the LTR plumbing.

Covers:
- ``supports_native_ranking`` flag is True for CB/XGB/LGB, False for HGB/Linear
- ``get_ranker_objective_kwargs`` returns library-correct kwargs
- XGB ``rank:map`` auto-fallback to ``rank:ndcg`` when y.max()>1 (graded)
- Pre-fit input prep: CB requires contiguous group rows, LGB needs per-query
  sizes, XGB takes per-row qid
- ``fit_ranker`` + ``predict_ranker_scores`` end-to-end on synthetic data
- HGB / Linear / unknown strategy raises NotImplementedError
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import LearningToRankConfig
from mlframe.training.ranking import (
    fit_ranker,
    predict_ranker_scores,
    prepare_cb_inputs,
    prepare_lgb_inputs,
    prepare_xgb_inputs,
    qid_to_group_sizes,
)
from mlframe.training.strategies import (
    CatBoostStrategy,
    HGBStrategy,
    TreeModelStrategy,  # LGB
    XGBoostStrategy,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def synthetic_ranking_data():
    """1000-row synthetic LTR set: 100 queries x 10 docs, graded relevance 0..3.

    First feature is informative; rest are noise. Strong learning signal so
    NDCG@10 >> random for any working ranker.
    """
    rng = np.random.default_rng(42)
    n_queries = 100
    n_per_query = 10
    n_rows = n_queries * n_per_query
    qid = np.repeat(np.arange(n_queries), n_per_query)
    X = rng.standard_normal((n_rows, 6)).astype(np.float32)
    true_score = 2.0 * X[:, 0] - 0.7 * X[:, 1]
    noise = 0.5 * rng.standard_normal(n_rows)
    y = np.clip(np.round(true_score + noise + 1.5), 0, 3).astype(int)

    train_q = np.arange(0, 70)
    val_q = np.arange(70, 85)
    test_q = np.arange(85, 100)
    tm = np.isin(qid, train_q)
    vm = np.isin(qid, val_q)
    tem = np.isin(qid, test_q)

    cols = [f"f{i}" for i in range(6)]
    return {
        "X_train": pd.DataFrame(X[tm], columns=cols),
        "y_train": y[tm],
        "g_train": qid[tm],
        "X_val": pd.DataFrame(X[vm], columns=cols),
        "y_val": y[vm],
        "g_val": qid[vm],
        "X_test": pd.DataFrame(X[tem], columns=cols),
        "y_test": y[tem],
        "g_test": qid[tem],
    }


# ----------------------------------------------------------------------------
# supports_native_ranking flag
# ----------------------------------------------------------------------------


class TestNativeRankingFlag:
    """``supports_native_ranking`` lights up only for CB/XGB/LGB."""

    def test_cb_supports_ranking(self):
        assert CatBoostStrategy().supports_native_ranking is True

    def test_xgb_supports_ranking(self):
        assert XGBoostStrategy().supports_native_ranking is True

    def test_lgb_via_tree_model_supports_ranking(self):
        # LightGBM uses TreeModelStrategy as its base (no separate LGB class).
        assert TreeModelStrategy().supports_native_ranking is True

    def test_hgb_does_not_support_ranking(self):
        assert HGBStrategy().supports_native_ranking is False


# ----------------------------------------------------------------------------
# Objective kwargs dispatch + auto-fallback
# ----------------------------------------------------------------------------


class TestRankerObjectiveKwargs:
    def test_cb_default_loss_is_yetirank_pairwise(self):
        cfg = LearningToRankConfig()
        out = CatBoostStrategy().get_ranker_objective_kwargs(cfg)
        assert out["loss_function"] == "YetiRankPairwise"
        assert out["eval_metric"] == "NDCG"

    def test_cb_loss_overridable_via_config(self):
        cfg = LearningToRankConfig(cb_loss_fn="QuerySoftMax")
        out = CatBoostStrategy().get_ranker_objective_kwargs(cfg)
        assert out["loss_function"] == "QuerySoftMax"

    def test_xgb_default_objective_is_rank_ndcg(self):
        cfg = LearningToRankConfig()
        out = XGBoostStrategy().get_ranker_objective_kwargs(cfg, y_max=1)
        assert out["objective"] == "rank:ndcg"

    def test_xgb_rank_map_with_binary_labels_passes(self):
        """rank:map is valid when y is binary (y.max() <= 1)."""
        cfg = LearningToRankConfig(xgb_objective="rank:map")
        out = XGBoostStrategy().get_ranker_objective_kwargs(cfg, y_max=1)
        assert out["objective"] == "rank:map"

    def test_xgb_rank_map_with_graded_labels_autofallback(self, caplog):
        """Auto-fall-back to rank:ndcg when graded labels detected (y.max>1)."""
        cfg = LearningToRankConfig(xgb_objective="rank:map")
        with caplog.at_level("WARNING"):
            out = XGBoostStrategy().get_ranker_objective_kwargs(cfg, y_max=4)
        assert out["objective"] == "rank:ndcg"
        assert any("rank:map requires binary" in r.message for r in caplog.records)

    def test_xgb_autofallback_disabled_keeps_user_pin(self):
        """``autodetect_label_format=False`` keeps the user's pin."""
        cfg = LearningToRankConfig(xgb_objective="rank:map", autodetect_label_format=False)
        out = XGBoostStrategy().get_ranker_objective_kwargs(cfg, y_max=4)
        assert out["objective"] == "rank:map"

    def test_lgb_default_objective_is_lambdarank(self):
        cfg = LearningToRankConfig()
        out = TreeModelStrategy().get_ranker_objective_kwargs(cfg)
        assert out["objective"] == "lambdarank"

    def test_lgb_xendcg_overridable(self):
        cfg = LearningToRankConfig(lgb_objective="rank_xendcg")
        out = TreeModelStrategy().get_ranker_objective_kwargs(cfg)
        assert out["objective"] == "rank_xendcg"

    def test_no_config_uses_strategy_defaults(self):
        """Passing ranking_config=None should not crash; defaults applied."""
        for s in (CatBoostStrategy(), XGBoostStrategy(), TreeModelStrategy()):
            out = s.get_ranker_objective_kwargs(None)
            assert isinstance(out, dict) and len(out) >= 1


# ----------------------------------------------------------------------------
# Pre-fit input prep
# ----------------------------------------------------------------------------


class TestPrepareInputs:
    """CB needs sort-by-group; XGB takes per-row qid; LGB needs group_sizes."""

    def test_qid_to_group_sizes_basic(self):
        gids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        out = qid_to_group_sizes(gids)
        np.testing.assert_array_equal(out, np.array([3, 2, 4]))

    def test_qid_to_group_sizes_empty(self):
        assert qid_to_group_sizes(np.array([], dtype=int)).tolist() == []

    def test_qid_to_group_sizes_single_query(self):
        out = qid_to_group_sizes(np.array([0, 0, 0]))
        np.testing.assert_array_equal(out, np.array([3]))

    def test_cb_prep_already_sorted_no_copy(self):
        """Already-sorted input: returned X is the same object (no sort)."""
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 2, 3])
        gids = np.array([0, 0, 1, 1])
        X_out, y_out, g_out, sort_idx = prepare_cb_inputs(X, y, gids)
        # sort_idx is identity when already sorted.
        np.testing.assert_array_equal(sort_idx, np.arange(4))

    def test_cb_prep_unsorted_gets_sorted(self):
        """Unsorted input: rows reordered so groups are contiguous."""
        X = pd.DataFrame({"f": [10.0, 20.0, 30.0, 40.0]})
        y = np.array([5, 6, 7, 8])
        # Interleaved: should sort to [0, 0, 1, 1]
        gids = np.array([0, 1, 0, 1])
        X_out, y_out, g_out, sort_idx = prepare_cb_inputs(X, y, gids)
        # Groups now contiguous
        assert np.all(g_out[1:] >= g_out[:-1])
        np.testing.assert_array_equal(g_out, np.array([0, 0, 1, 1]))
        # y matches the reordered X
        np.testing.assert_array_equal(y_out, y[sort_idx])

    def test_xgb_prep_no_sort_required(self):
        """XGB accepts arbitrary qid order."""
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 2, 3])
        gids = np.array([0, 1, 0, 1])  # interleaved, not sorted
        X_out, y_out, qid_out = prepare_xgb_inputs(X, y, gids)
        # Same shape, no sort done
        np.testing.assert_array_equal(qid_out, gids)
        np.testing.assert_array_equal(y_out, y)

    def test_lgb_prep_returns_per_query_sizes(self):
        """LGB expects per-query sizes (sum=N), not per-row qid."""
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = np.array([0, 1, 2, 3, 4])
        gids = np.array([0, 0, 0, 1, 1])
        X_out, y_out, group_sizes, sort_idx = prepare_lgb_inputs(X, y, gids)
        np.testing.assert_array_equal(group_sizes, np.array([3, 2]))
        assert int(group_sizes.sum()) == len(X)

    def test_lgb_prep_unsorted_gets_sorted_then_sized(self):
        X = pd.DataFrame({"f": np.arange(6, dtype=float)})
        y = np.arange(6)
        gids = np.array([2, 0, 1, 0, 2, 1])
        X_out, y_out, group_sizes, sort_idx = prepare_lgb_inputs(X, y, gids)
        # After sort by gid: [0, 0, 1, 1, 2, 2] -> sizes [2, 2, 2]
        np.testing.assert_array_equal(group_sizes, np.array([2, 2, 2]))

    def test_prep_length_mismatch_raises(self):
        X = pd.DataFrame({"f": [1.0, 2.0]})
        y = np.array([0, 1, 2])  # mismatched
        gids = np.array([0, 0])
        with pytest.raises(ValueError, match="length mismatch"):
            prepare_cb_inputs(X, y, gids)


# ----------------------------------------------------------------------------
# fit_ranker / predict_ranker_scores end-to-end
# ----------------------------------------------------------------------------


class TestFitPredictPerStrategy:
    """End-to-end fit + predict on the same synthetic data for CB/XGB/LGB."""

    @pytest.mark.parametrize(
        "flavor,strategy_cls",
        [
            ("cb", CatBoostStrategy),
            ("xgb", XGBoostStrategy),
            ("lgb", TreeModelStrategy),
        ],
    )
    def test_fit_predict_returns_per_row_scores(self, synthetic_ranking_data, flavor, strategy_cls):
        d = synthetic_ranking_data
        cfg = LearningToRankConfig()
        iter_kw = "iterations" if flavor == "cb" else "n_estimators"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fit_ranker(
                strategy_cls(),
                d["X_train"],
                d["y_train"],
                d["g_train"],
                X_val=d["X_val"],
                y_val=d["y_val"],
                group_ids_val=d["g_val"],
                ranking_config=cfg,
                model_kwargs={iter_kw: 50, "learning_rate": 0.1},
                early_stopping_rounds=10,
            )
        assert fitted["flavor"] == flavor.replace("lgb", "lightgbm").replace("cb", "catboost").replace("xgb", "xgboost")
        scores = predict_ranker_scores(fitted, d["X_test"])
        assert scores.shape == (len(d["X_test"]),)
        assert scores.dtype.kind == "f"

    def test_hgb_strategy_raises_not_implemented(self, synthetic_ranking_data):
        """HGBStrategy has no native ranker -- fit_ranker rejects it."""
        d = synthetic_ranking_data
        with pytest.raises(NotImplementedError, match="does not support native ranking"):
            fit_ranker(
                HGBStrategy(),
                d["X_train"],
                d["y_train"],
                d["g_train"],
                model_kwargs={"max_iter": 10},
            )
