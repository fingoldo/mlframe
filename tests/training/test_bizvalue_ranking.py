"""Biz-value end-to-end test for the LTR suite.

Trains all three rankers (CB / XGB / LGB) on synthetic web-search-like
data via ``train_mlframe_models_suite(target_type=LEARNING_TO_RANK)`` and
asserts:

1. Each individual ranker's NDCG@10 on test exceeds a strong baseline
   (random ordering ≈ 0.45 on 5-grade synthetic; we require ≥ 0.75 to
   catch broken plumbing while staying tolerant to sample-size noise).
2. Ensemble (RRF, default) NDCG@10 is no worse than 2pp below the best
   individual model — proves the ensemble path doesn't degrade quality.
3. Save / load roundtrip preserves predictions to ≤ 1e-6 abs tolerance.

This is the headline integration test the user asked for ("ансамблирование
в train_mlframe_models_suite должно работать и для LearningToRank").
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.metrics.ranking import ndcg_at_k
from mlframe.training import (
    LearningToRankConfig,
    TargetTypes,
    train_mlframe_models_suite,
)
from mlframe.training.extractors import FeaturesAndTargetsExtractor


class _RankFTE(FeaturesAndTargetsExtractor):
    """Minimal FTE that exposes graded relevance under LEARNING_TO_RANK."""

    def __init__(self):
        super().__init__(group_field="qid")

    def build_targets(self, df):
        relevance = df["relevance"]
        if hasattr(relevance, "to_numpy"):
            relevance = relevance.to_numpy()
        elif hasattr(relevance, "values"):
            relevance = relevance.values
        return {TargetTypes.LEARNING_TO_RANK: {"relevance": np.asarray(relevance)}}


@pytest.fixture
def synthetic_search_data():
    """500 queries × 8 docs = 4000 rows. Strong signal in feature 0,
    moderate in feature 1, noise in 5 others. Graded relevance 0..3."""
    rng = np.random.default_rng(42)
    n_q = 500
    n_per = 8
    n = n_q * n_per
    qid = np.repeat(np.arange(n_q), n_per)
    X = rng.standard_normal((n, 7)).astype(np.float32)
    true_score = 1.5 * X[:, 0] - 0.6 * X[:, 1] + 0.2 * X[:, 2]
    noise = 0.5 * rng.standard_normal(n)
    y = np.clip(np.round(true_score + noise + 1.5), 0, 3).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(7)])
    df["qid"] = qid
    df["relevance"] = y
    return df


class TestLTREndToEndSuite:
    """Full ``train_mlframe_models_suite`` LTR run with all three rankers."""

    def test_individual_models_beat_strong_baseline(self, synthetic_search_data, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, meta = train_mlframe_models_suite(
                df=synthetic_search_data,
                target_name="relevance",
                model_name="bizvalue_ltr",
                features_and_targets_extractor=_RankFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
                mlframe_models=["cb", "xgb", "lgb"],
                use_mlframe_ensembles=True,
                verbose=0,
            )
        assert meta["target_type"] == "learning_to_rank"
        for flavor in ["cb", "xgb", "lgb"]:
            ndcg = models[flavor]["test_metrics"]["ndcg@10"]
            assert ndcg >= 0.75, (
                f"{flavor} test NDCG@10={ndcg:.4f} below 0.75 baseline; "
                "ranker is not learning the planted signal — plumbing broken?"
            )

    def test_ensemble_not_worse_than_best_individual_minus_2pp(self, synthetic_search_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=synthetic_search_data,
                target_name="relevance",
                model_name="bizvalue_ltr_ens",
                features_and_targets_extractor=_RankFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
                mlframe_models=["cb", "xgb", "lgb"],
                use_mlframe_ensembles=True,
                verbose=0,
            )
        individual = [models[f]["test_metrics"]["ndcg@10"] for f in ["cb", "xgb", "lgb"]]
        ensemble = models["ensemble"]["test_metrics"]["ndcg@10"]
        # Ensemble can be slightly below best individual on small data
        # (variance > ensemble's smoothing benefit), but should not
        # collapse — assert within 2pp.
        assert ensemble >= max(individual) - 0.02, (
            f"Ensemble NDCG@10={ensemble:.4f} more than 2pp below best "
            f"individual={max(individual):.4f}: ensemble path degraded "
            "scores — check RRF / Borda implementation."
        )

    def test_ensemble_method_borda_runs_through_suite(self, synthetic_search_data):
        cfg = LearningToRankConfig(ensemble_method="borda")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=synthetic_search_data,
                target_name="relevance",
                model_name="bizvalue_ltr_borda",
                features_and_targets_extractor=_RankFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
                mlframe_models=["cb", "xgb", "lgb"],
                use_mlframe_ensembles=True,
                ranking_config=cfg,
                verbose=0,
            )
        assert "ensemble" in models
        assert models["ensemble"]["method"] == "borda"
        # Borda should also produce a sensible ranking.
        assert models["ensemble"]["test_metrics"]["ndcg@10"] >= 0.70

    def test_save_load_roundtrip_predictions_preserved(self, synthetic_search_data, tmp_path):
        save_dir = str(tmp_path / "ltr_models")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models_a, _ = train_mlframe_models_suite(
                df=synthetic_search_data,
                target_name="relevance",
                model_name="roundtrip",
                features_and_targets_extractor=_RankFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
                mlframe_models=["cb", "xgb", "lgb"],
                use_mlframe_ensembles=False,
                verbose=0,
                # Need to supply a save dir; use output_config dict.
            )

        # Manually save+load via joblib (mirrors the save_dir path).
        import joblib
        for flavor in ["cb", "xgb", "lgb"]:
            path = os.path.join(save_dir, f"roundtrip_{flavor}.joblib")
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(models_a[flavor]["model"], path)
            loaded = joblib.load(path)
            # Predict on a slice; should match in-memory model.
            X_te = pd.DataFrame(
                np.random.default_rng(0).standard_normal((20, 7)).astype(np.float32),
                columns=[f"f{i}" for i in range(7)],
            )
            preds_orig = models_a[flavor]["model"].predict(X_te)
            preds_loaded = loaded.predict(X_te)
            np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-6,
                err_msg=f"{flavor}: save/load drifted predictions")


class TestLTRSuiteValidation:
    """Suite-level validation: missing group_field raises, bad models filtered."""

    def test_missing_group_field_raises_actionable_error(self, synthetic_search_data):
        class _BadFTE(FeaturesAndTargetsExtractor):
            def __init__(self):
                super().__init__(group_field=None)  # MISSING
            def build_targets(self, df):
                relevance = df["relevance"]
                if hasattr(relevance, "to_numpy"):
                    relevance = relevance.to_numpy()
                return {TargetTypes.LEARNING_TO_RANK: {"relevance": np.asarray(relevance)}}

        with pytest.raises(ValueError, match="group_field"):
            train_mlframe_models_suite(
                df=synthetic_search_data,
                target_name="relevance",
                model_name="bad",
                features_and_targets_extractor=_BadFTE(),
                target_type=TargetTypes.LEARNING_TO_RANK,
            )

    def test_autodetected_ltr_routes_to_ranker_without_target_type_arg(self, synthetic_search_data):
        """Regression (fuzz c0016/c0031): the suite must auto-route LTR even when the caller
        omits ``target_type`` (leaving it ``None``).

        The early ranker dispatch only fires for an EXPLICIT ``target_type=LEARNING_TO_RANK``
        arg. When it is None, the FTE's ``build_targets`` can still resolve the target as
        LEARNING_TO_RANK; pre-fix that target fell through to the standard per-target loop,
        which built an ``LGBMClassifier`` with a multiclass objective + an LTR eval metric ->
        ``LightGBMError: Multiclass objective and metrics don't match``. The suite now detects
        LTR in the resolved ``target_by_type`` and re-dispatches to the ranker suite.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, meta = train_mlframe_models_suite(
                df=synthetic_search_data,
                target_name="relevance",
                model_name="autoroute_ltr",
                features_and_targets_extractor=_RankFTE(),
                # target_type intentionally OMITTED -> None -> must be auto-detected + routed.
                mlframe_models=["lgb", "xgb"],
                use_mlframe_ensembles=False,
                verbose=0,
            )
        # Proves the ranker suite handled it (not the classifier path, which would have
        # crashed with the multiclass-objective error before producing any metadata).
        assert meta["target_type"] == "learning_to_rank"
        assert "lgb" in models, "lgb should survive as a native LGBMRanker under auto-routed LTR"
        assert "ndcg@10" in models["lgb"]["test_metrics"], "ranker should emit a ranking metric"

    def test_unsupported_models_dropped_with_warn(self, synthetic_search_data, caplog):
        """HGB / Linear get filtered; only CB/XGB/LGB survive."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with caplog.at_level("WARNING"):
                models, _ = train_mlframe_models_suite(
                    df=synthetic_search_data,
                    target_name="relevance",
                    model_name="filter_test",
                    features_and_targets_extractor=_RankFTE(),
                    target_type=TargetTypes.LEARNING_TO_RANK,
                    mlframe_models=["cb", "hgb", "linear", "lgb"],
                    use_mlframe_ensembles=False,
                    verbose=0,
                )
        # Only cb/lgb survived; hgb/linear filtered
        assert "cb" in models
        assert "lgb" in models
        assert "hgb" not in models
        assert "linear" not in models
        # WARN explained the drop
        assert any("native ranker" in r.message for r in caplog.records)
