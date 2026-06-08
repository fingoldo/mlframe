"""``NeuralNetStrategy`` / ``LinearModelStrategy`` / ``RecurrentModelStrategy`` pipeline strategies."""
from __future__ import annotations

from .base import ModelPipelineStrategy


class NeuralNetStrategy(ModelPipelineStrategy):
    """
    Strategy for neural network models (MLP, NGBoost).

    These models:
    - Cannot handle NaN values - need imputation
    - Benefit significantly from feature scaling
    - Require category encoding

    Multi-output dispatch (2026-05-07):
    - **multiclass**: native via ``F.cross_entropy`` (default loss_fn) +
      softmax in ``MLPTorchModel.predict_step`` for K>1 outputs. Already
      works at the model level; the flag below makes the dispatch
      consistent across strategies.
    - **multilabel**: native via per-label ``F.binary_cross_entropy_with_logits``
      + sigmoid output (separate path; see ``get_classif_objective_kwargs``).
    - **learning_to_rank**: native via RankNet/ListNet pairwise loss in
      ``mlframe.training.neural.ranker.MLPRanker``.
    """

    cache_key = "neural"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True
    supports_native_multiclass = True
    supports_native_multilabel = True
    supports_native_ranking = True
    # F-34 (2026-05-31): PytorchLightningRegressor auto-detects (N, K>=2)
    # float y at fit-time and routes ``num_classes = K`` through
    # generate_mlp -> K output heads sharing the trunk + MSE between
    # (N, K) preds and (N, K) labels (F-24 commit 2d300944).
    supports_native_multi_target = True

    def get_classif_objective_kwargs(self, target_type, n_classes: int,
                                      multilabel_config=None) -> dict:
        """Per-target loss_fn dispatch for the MLP estimator.

        Returned dict is consumed by ``_configure_mlp_params`` (trainer.py)
        which threads it into ``mlp_kwargs.model_params.loss_fn`` +
        ``mlp_kwargs.datamodule_params.labels_dtype``. Returns the empty
        dict for binary (default ``F.cross_entropy`` already correct).
        """
        from ..configs import TargetTypes

        # Lazy import torch so a non-MLP run doesn't pay for PL/torch import.
        import torch
        import torch.nn.functional as F

        if target_type is None or target_type == TargetTypes.BINARY_CLASSIFICATION:
            return {}  # default cross_entropy is correct for binary
        if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
            # Default ``F.cross_entropy`` + ``int64`` labels already
            # handle K>2 -- explicit return for symmetry with other strategies.
            return {"loss_fn": F.cross_entropy, "labels_dtype": torch.int64}
        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
            # Per-label sigmoid: BCE with logits is numerically stable
            # and accepts (N, K) float32 labels.
            return {
                "loss_fn": F.binary_cross_entropy_with_logits,
                "labels_dtype": torch.float32,
                # Predict-time sigmoid signal so MLPTorchModel.predict_step
                # uses sigmoid (not softmax) for K>1 outputs.
                "task_type": "multilabel",
            }
        return {}

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """MLPRanker loss_fn dispatch. Default ``ranknet`` (pairwise BCE
        on score differences); alternative ``listnet`` (listwise softmax
        cross-entropy). Both accept binary or graded relevance.

        ``y_max`` unused -- both losses handle the full label range.
        ``ranking_config.lgb_objective`` doesn't apply to MLP; MLPRanker
        consumes loss_fn directly via the ``loss_fn`` key.
        """
        loss_fn = "ranknet"
        if ranking_config is not None:
            # Optional override via a dedicated MLP key. Keeps the per-
            # library config clean (cb_loss_fn / xgb_objective / lgb_objective
            # for those three; mlp_loss_fn for MLP).
            loss_fn = getattr(ranking_config, "mlp_loss_fn", None) or loss_fn
        return {"loss_fn": loss_fn}

    def _extra_pre_encoding_steps(self, embedding_features, text_features):
        """Make embedding-vector + free-text columns numeric for the MLP, which has no native embedding/text layers.

        Embedding ``List`` columns are expanded to their float components and text columns are turned into dense
        HuggingFace transformer embeddings (see ``neural.feature_prep.NeuralEmbeddingTextEncoder``). Runs before
        cat-encoding/imputation/scaling, so the result is numeric for every target type (regression / binary /
        multiclass / multilabel / learning-to-rank). No-op when no embedding/text columns are present.
        """
        if not embedding_features and not text_features:
            return []
        from ..neural.feature_prep import NeuralEmbeddingTextEncoder
        return [(
            "neural_emb_text",
            NeuralEmbeddingTextEncoder(
                embedding_features=list(embedding_features or []),
                text_features=list(text_features or []),
            ),
        )]


class LinearModelStrategy(ModelPipelineStrategy):
    """
    Strategy for linear models (Linear, Ridge, Lasso, ElasticNet, etc.).

    These models:
    - Cannot handle NaN values - need imputation
    - Require feature scaling for proper regularization
    - Require category encoding

    Multi-output dispatch:

    - **multiclass**: ``LogisticRegression`` auto-detects K since
      sklearn 1.5; ``multi_class`` kwarg removed in 1.8 (defaults to
      multinomial when liblinear isn't the solver). ``RidgeClassifier``
      / ``SGDClassifier`` use OvR by default. Strategy-level flag = True.
    - **multilabel**: known sklearn quirk that ``RidgeClassifier`` /
      ``RidgeClassifierCV`` accept 2-D y natively (treats as multi-output
      ridge regression + threshold; ``predict`` returns ``(N, K)``).
      However, the metric-reporter pipeline assumes per-class probability
      output (N, K) AND breaks on RidgeClassifier's lack of
      ``predict_proba``. Until the eval path is generalised, all linear
      multilabel goes through ``MultiOutputClassifier`` wrap (correct
      but suboptimal -- one extra fit per label). Tracked as known
      limitation; wrapper path is correct.
    """

    cache_key = "linear"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True
    # sklearn LogisticRegression supports multiclass natively (auto since
    # 1.5; ``multi_class`` kwarg removed in 1.8).
    supports_native_multiclass = True
    # F-34 (2026-05-31): sklearn linear regressors (LinearRegression,
    # Ridge, Lasso, ElasticNet) handle (N, K) y natively — the closed-form
    # solution is a single matrix Y = X B + e with B of shape (D, K).
    # MultiTaskLasso / MultiTaskElasticNet add joint L1 across targets;
    # plain Lasso/ElasticNet do K independent column fits.
    supports_native_multi_target = True


class RecurrentModelStrategy(ModelPipelineStrategy):
    """
    Strategy for recurrent models (LSTM, GRU, RNN, Transformer).

    These models:
    - Process sequences internally (handled by RecurrentDataModule)
    - In HYBRID mode, tabular features require preprocessing
    - Need imputation and scaling for tabular features
    - Require category encoding for tabular features

    Multi-output dispatch (2026-05-07):
    - **multiclass**: native via ``num_classes>1`` + CrossEntropyLoss
      + softmax in ``predict_step``. Already wired at the model level
      (RecurrentLightningModule); the flag below makes the dispatch
      consistent across strategies.
    - **multilabel**: native via ``task_type='multilabel'`` ->
      BCEWithLogitsLoss + sigmoid output. Output layer stays at K units,
      activation switches at predict time.
    - **learning_to_rank**: NOT native -- group-aware sequence batching
      (one query's docs per batch, where each doc has its own sequence)
      is non-trivial for recurrent architectures. Deferred; suite
      filters out 'recurrent' models when target_type=LEARNING_TO_RANK.
    """

    cache_key = "recurrent"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True
    supports_native_multiclass = True
    supports_native_multilabel = True
    # supports_native_ranking stays False -- group-batching for sequences
    # would require a custom sampler that yields one query's sequences
    # per batch; non-trivial integration with RecurrentDataModule.

    def get_classif_objective_kwargs(self, target_type, n_classes: int,
                                      multilabel_config=None) -> dict:
        """Per-target task_type for ``RecurrentLightningModule``.

        Returns a dict with the ``task_type`` kwarg consumed by the
        Lightning module to switch loss + activation. For multiclass
        the default (None / 'multiclass') already uses CrossEntropy +
        softmax -- empty return suffices.
        """
        from ..configs import TargetTypes

        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
            return {"task_type": "multilabel"}
        # binary / multiclass / None -> defaults are correct
        return {}
