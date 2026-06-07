"""Tests for MLP × LearningToRank target type.

Phase MLP-C coverage:
- ``ranknet_pairwise_loss`` correctness on hand-crafted pairs (1 query)
- ``listnet_top1_loss`` correctness (softmax KL)
- ``GroupBatchSampler`` skips singleton + single-class queries
- ``MLPRanker.fit/predict`` returns 1-D per-row scores
- ``NeuralNetStrategy.get_ranker_objective_kwargs`` returns ``loss_fn=ranknet`` default
- ``ranking.fit_ranker(NeuralNetStrategy)`` dispatches to MLPRanker
- ``ranker_suite._filter_models_for_ranking`` keeps mlp (not dropped)
- End-to-end ``train_mlframe_models_suite(target_type=LEARNING_TO_RANK)`` with
  4-model ensemble (cb+xgb+lgb+mlp) -- mlp learns the planted signal
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightning")
pytest.importorskip("torch")

import torch

from mlframe.training import (
    LearningToRankConfig, TargetTypes, train_mlframe_models_suite,
)
from mlframe.training.extractors import FeaturesAndTargetsExtractor
from mlframe.training.neural.ranker import (
    GroupBatchSampler, MLPRanker, listnet_top1_loss, ranknet_pairwise_loss,
)
from mlframe.training.ranking import fit_ranker, predict_ranker_scores
from mlframe.training.ranking.ranker_suite import _filter_models_for_ranking
from mlframe.training.strategies import NeuralNetStrategy


# ----------------------------------------------------------------------------
# Loss functions: hand-computed correctness
# ----------------------------------------------------------------------------


class TestRankNetPairwiseLoss:
    def test_perfect_ordering_yields_low_loss(self):
        # Scores aligned with relevance -> all pair diffs positive ->
        # sigmoid(positive) close to 1 -> -log(close to 1) small.
        # Score gaps [5,3,1,0] give BCE ~0.107 (smaller for bigger gaps);
        # threshold at 0.15 is comfortably above floor and well below
        # an inverse-ordering value (>1.0).
        scores = torch.tensor([5.0, 3.0, 1.0, 0.0])
        rel = torch.tensor([3, 2, 1, 0])
        loss = ranknet_pairwise_loss(scores, rel).item()
        assert loss < 0.15, f"perfect ordering loss={loss:.4f} too high"

    def test_inverse_ordering_yields_high_loss(self):
        scores = torch.tensor([0.0, 1.0, 3.0, 5.0])
        rel = torch.tensor([3, 2, 1, 0])
        loss = ranknet_pairwise_loss(scores, rel).item()
        assert loss > 1.0, f"inverse ordering loss={loss:.4f} too low"

    def test_all_equal_relevance_returns_zero(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        rel = torch.tensor([1, 1, 1])  # no informative pairs
        assert ranknet_pairwise_loss(scores, rel).item() == 0.0

    def test_single_doc_query_returns_zero(self):
        scores = torch.tensor([1.0])
        rel = torch.tensor([1])
        assert ranknet_pairwise_loss(scores, rel).item() == 0.0


class TestListNetTop1Loss:
    def test_perfect_ordering_yields_low_loss(self):
        scores = torch.tensor([5.0, 3.0, 1.0, 0.0])
        rel = torch.tensor([3.0, 2.0, 1.0, 0.0])
        # Same softmax => low cross-entropy.
        loss = listnet_top1_loss(scores, rel).item()
        # Can't compare to 0 (softmax of [3,2,1,0] != softmax of [5,3,1,0]
        # but high overlap on the top-1 mass).
        assert loss < 1.5

    def test_all_equal_relevance_returns_zero(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        rel = torch.tensor([1.0, 1.0, 1.0])
        assert listnet_top1_loss(scores, rel).item() == 0.0


# ----------------------------------------------------------------------------
# GroupBatchSampler
# ----------------------------------------------------------------------------


class TestGroupBatchSampler:
    def test_yields_one_query_per_batch(self):
        gids = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        rel = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        sampler = GroupBatchSampler(gids, rel, shuffle=False)
        batches = list(sampler)
        # 3 queries, all pass the >=2 docs + multi-class checks.
        assert len(batches) == 3

    def test_skips_singleton_queries(self):
        gids = np.array([0, 0, 1, 2, 2])
        rel = np.array([1, 0, 1, 1, 0])
        sampler = GroupBatchSampler(gids, rel, shuffle=False)
        batches = list(sampler)
        # Query 1 has only 1 doc -> dropped.
        assert len(batches) == 2
        for b in batches:
            assert len(b) >= 2

    def test_skips_single_class_queries(self):
        gids = np.array([0, 0, 0, 1, 1])
        rel = np.array([1, 1, 1, 1, 0])  # query 0 all-rel=1
        sampler = GroupBatchSampler(gids, rel, shuffle=False)
        batches = list(sampler)
        # Query 0 single-class -> dropped; query 1 has both 0 and 1.
        assert len(batches) == 1


# ----------------------------------------------------------------------------
# MLPRanker fit / predict
# ----------------------------------------------------------------------------


@pytest.fixture
def synthetic_ltr_data():
    """100 queries × 8 docs, graded relevance 0..3, signal in feature 0."""
    rng = np.random.default_rng(42)
    n_q = 100
    n_per = 8
    n = n_q * n_per
    qid = np.repeat(np.arange(n_q), n_per)
    X = rng.standard_normal((n, 5)).astype(np.float32)
    score = 1.5 * X[:, 0] - 0.5 * X[:, 1]
    y = np.clip(np.round(score + 0.4 * rng.standard_normal(n) + 1.5), 0, 3).astype(int)
    cols = [f"f{i}" for i in range(5)]
    return {
        "X_train": pd.DataFrame(X[:600], columns=cols),
        "y_train": y[:600],
        "g_train": qid[:600],
        "X_val": pd.DataFrame(X[600:700], columns=cols),
        "y_val": y[600:700],
        "g_val": qid[600:700],
        "X_test": pd.DataFrame(X[700:], columns=cols),
        "y_test": y[700:],
        "g_test": qid[700:],
    }


class TestMLPRankerFitPredict:
    def test_fit_predict_returns_1d_scores(self, synthetic_ltr_data):
        d = synthetic_ltr_data
        model = MLPRanker(n_estimators=10, learning_rate=0.01, verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                d["X_train"], d["y_train"], d["g_train"],
                X_val=d["X_val"], y_val=d["y_val"], group_ids_val=d["g_val"],
            )
        scores = model.predict(d["X_test"])
        assert scores.shape == (len(d["X_test"]),)
        assert scores.dtype.kind == "f"

    def test_listnet_loss_runs_too(self, synthetic_ltr_data):
        d = synthetic_ltr_data
        model = MLPRanker(loss_fn="listnet", n_estimators=5, learning_rate=0.01, verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(d["X_train"], d["y_train"], d["g_train"])
        scores = model.predict(d["X_test"])
        assert scores.shape == (len(d["X_test"]),)


# ----------------------------------------------------------------------------
# Strategy + dispatch wiring
# ----------------------------------------------------------------------------


class TestMLPRankerStrategyDispatch:
    def test_get_ranker_objective_kwargs_default_ranknet(self):
        out = NeuralNetStrategy().get_ranker_objective_kwargs(LearningToRankConfig())
        assert out["loss_fn"] == "ranknet"

    def test_get_ranker_objective_kwargs_listnet_via_config(self):
        cfg = LearningToRankConfig(mlp_loss_fn="listnet")
        out = NeuralNetStrategy().get_ranker_objective_kwargs(cfg)
        assert out["loss_fn"] == "listnet"

    def test_fit_ranker_dispatches_to_mlp(self, synthetic_ltr_data):
        d = synthetic_ltr_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fit_ranker(
                NeuralNetStrategy(),
                d["X_train"], d["y_train"], d["g_train"],
                X_val=d["X_val"], y_val=d["y_val"], group_ids_val=d["g_val"],
                ranking_config=LearningToRankConfig(),
                model_kwargs={"n_estimators": 10, "learning_rate": 0.01, "verbose": 0},
                early_stopping_rounds=5,
            )
        assert fitted["flavor"] == "mlp"
        assert isinstance(fitted["model"], MLPRanker)
        scores = predict_ranker_scores(fitted, d["X_test"])
        assert scores.shape == (len(d["X_test"]),)

    def test_filter_models_for_ranking_keeps_mlp(self):
        kept = _filter_models_for_ranking(["cb", "xgb", "lgb", "mlp"])
        assert "mlp" in kept

    def test_filter_drops_unsupported_keeps_mlp(self):
        kept = _filter_models_for_ranking(["mlp", "linear", "hgb"])
        assert "mlp" in kept
        assert "linear" not in kept
        assert "hgb" not in kept


# ----------------------------------------------------------------------------
# End-to-end suite (4-model ensemble inc. MLP)
# ----------------------------------------------------------------------------


class _RankFTE(FeaturesAndTargetsExtractor):
    def __init__(self):
        super().__init__(group_field="qid")

    def build_targets(self, df):
        rel = df["relevance"]
        if hasattr(rel, "to_numpy"):
            rel = rel.to_numpy()
        return {TargetTypes.LEARNING_TO_RANK: {"relevance": np.asarray(rel)}}


class TestMLPRankerInSuite:
    def test_mlp_only_via_suite(self):
        rng = np.random.default_rng(42)
        n_q, n_per = 200, 8
        n = n_q * n_per
        qid = np.repeat(np.arange(n_q), n_per)
        X = rng.standard_normal((n, 6)).astype(np.float32)
        true = 1.5 * X[:, 0] - 0.6 * X[:, 1]
        y = np.clip(np.round(true + 0.5 * rng.standard_normal(n) + 1.5), 0, 3).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["qid"] = qid
        df["relevance"] = y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=df, target_name="relevance", model_name="mlp_ltr",
                features_and_targets_extractor=_RankFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False, verbose=0,
            )
        assert "mlp" in models
        assert models["mlp"]["test_metrics"]["ndcg@10"] >= 0.55, (
            f"MLP ranker NDCG@10={models['mlp']['test_metrics']['ndcg@10']:.4f} "
            "below 0.55 baseline -- learning broken?"
        )

    def test_4_model_ensemble_includes_mlp(self):
        """4-model ensemble (cb+xgb+lgb+mlp) -- assert MLP scores
        contribute to the RRF ensemble."""
        rng = np.random.default_rng(42)
        n_q, n_per = 200, 8
        n = n_q * n_per
        qid = np.repeat(np.arange(n_q), n_per)
        X = rng.standard_normal((n, 6)).astype(np.float32)
        true = 1.5 * X[:, 0] - 0.6 * X[:, 1]
        y = np.clip(np.round(true + 0.5 * rng.standard_normal(n) + 1.5), 0, 3).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["qid"] = qid
        df["relevance"] = y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, meta = train_mlframe_models_suite(
                df=df, target_name="relevance", model_name="mlp_ltr_ens",
                features_and_targets_extractor=_RankFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
                mlframe_models=["cb", "xgb", "lgb", "mlp"],
                use_mlframe_ensembles=True, verbose=0,
            )
        # All 4 models trained
        for f in ("cb", "xgb", "lgb", "mlp"):
            assert f in models
        # Ensemble exists and includes mlp as a member
        assert "ensemble" in models
        assert "mlp" in models["ensemble"]["members"]
