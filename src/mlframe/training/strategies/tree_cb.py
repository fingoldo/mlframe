"""``TreeModelStrategy`` + ``CatBoostStrategy`` pipeline strategies."""
from __future__ import annotations

from typing import Any, List, Optional

from sklearn.pipeline import Pipeline

from .base import ModelPipelineStrategy


class TreeModelStrategy(ModelPipelineStrategy):
    """
    Strategy for tree-based models (CatBoost, LightGBM, XGBoost).

    These models:
    - Handle NaN values natively
    - Don't require feature scaling
    - CatBoost handles categorical features natively
    - LightGBM/XGBoost can handle categoricals with proper setup
    """

    cache_key = "tree"
    requires_scaling = False
    requires_encoding = False
    requires_imputation = False
    # All tree models (CB/LGB/XGB) support multiclass natively via library
    # objective kwargs. Multilabel native is CB-only -- overridden in
    # CatBoostStrategy. LGB has no native multilabel (issue #524 since 2017),
    # XGB 3.x experimental but unstable.
    supports_native_multiclass = True
    # LGB has native LGBMRanker; CB/XGB override below with their own
    # objective dispatch. Setting True at TreeModelStrategy level means
    # the default (LGB) path is correctly enabled.
    supports_native_ranking = True

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """LGBMRanker objective. ``lambdarank`` (default) handles both
        binary and graded relevance. ``rank_xendcg`` is an alternative.
        """
        objective = "lambdarank"
        if ranking_config is not None:
            objective = getattr(ranking_config, "lgb_objective", None) or objective
        return {
            "objective": objective,
            # eval_metric defaults to ndcg for ranker; expose explicitly.
            "metric": "ndcg",
        }

    def build_pipeline(
        self,
        base_pipeline: Optional[Pipeline],
        cat_features: List[str],
        category_encoder: Optional[Any] = None,
        imputer: Optional[Any] = None,
        scaler: Optional[Any] = None,
        embedding_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
    ) -> Optional[Pipeline]:
        """Tree models just use the base pipeline (feature selection) if any. ``embedding_features`` / ``text_features``
        are accepted for signature parity but ignored here -- tree libraries consume those columns natively (CatBoost)
        or via the feature-handling layer, not through a sklearn pre-pipeline step."""
        return base_pipeline


class CatBoostStrategy(TreeModelStrategy):
    """
    Strategy for CatBoost models.

    Inherits tree model behavior and additionally supports native Polars DataFrames,
    allowing training without pandas conversion (CatBoost >= 1.2.7).
    Also supports text_features and embedding_features natively.
    """

    supports_polars = True
    supports_text_features = True
    supports_embedding_features = True
    # 2026-04-24: native multi-output support via loss_function='MultiClass'
    # for K>2 single-label and 'MultiLogloss' for K independent binary
    # outputs. The dispatch wires these via
    # ModelPipelineStrategy.get_classif_objective_kwargs +
    # _maybe_wrap_multilabel (which short-circuits the wrapper for
    # supports_native_multilabel=True strategies).
    supports_native_multiclass = True
    supports_native_multilabel = True
    supports_native_ranking = True
    # 2026-05-08 QR: CatBoost MultiQuantile loss handles K alphas in one
    # fit; predict returns (N, K) directly.
    supports_native_quantile = True
    # F-34 (2026-05-31): CatBoost MultiRMSE loss handles K continuous
    # targets in one fit; predict returns (N, K) directly.
    supports_native_multi_target = True
    # Inherits cache_key = "tree" from TreeModelStrategy so CB/LGB/XGB share
    # transformed-DF cache (they have identical preprocessing requirements).

    def get_multi_target_objective_kwargs(self) -> dict:
        """CatBoost ``MultiRMSE`` loss_function for K-target regression.

        Single ensemble outputs (N, K) directly. Use
        ``MultiRMSEWithMissingValues`` if any target column has NaN.
        """
        return {"loss_function": "MultiRMSE"}

    def get_quantile_objective_kwargs(self, qr_config) -> dict:
        """CatBoost ``MultiQuantile`` loss_function with comma-joined alphas.

        Format: ``"MultiQuantile:alpha=0.1,0.5,0.9"`` (no brackets, no
        spaces). predict() then returns shape (N, K).
        """
        alphas_str = ",".join(str(a) for a in qr_config.alphas)
        return {"loss_function": f"MultiQuantile:alpha={alphas_str}"}

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """CatBoostRanker loss_function + sensible eval_metric.

        Defaults to ``YetiRankPairwise`` (listwise pairwise -- robust on
        both graded and binary labels). Override via
        ``LearningToRankConfig.cb_loss_fn``.

        ``y_max`` is unused by CB (its ranker losses accept both binary
        and graded labels uniformly).
        """
        loss_fn = "YetiRankPairwise"
        if ranking_config is not None:
            loss_fn = getattr(ranking_config, "cb_loss_fn", None) or loss_fn
        return {
            "loss_function": loss_fn,
            # CB ranker exposes NDCG / MAP / MRR via PFound-family eval
            # metrics; use NDCG as the default for early-stopping. Users
            # can override via hyperparams.
            "eval_metric": "NDCG",
        }
