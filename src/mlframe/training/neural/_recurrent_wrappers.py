"""Sklearn-compatible RecurrentClassifierWrapper / RecurrentRegressorWrapper, carved out of
``recurrent_dataset_helpers.py`` (X_EFFICIENCY_ARCHITECTURE-1 fix, mrmr_audit_2026-07-22) to clear the
repo's enforced hard 1000-LOC CI gate (that file was 1049 lines). Behaviour preserved bit-for-bit; the
parent re-exports both classes so existing ``from mlframe.training.neural.recurrent import
RecurrentClassifierWrapper`` callers keep working unchanged.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import lightning as L
import torch
from sklearn.base import ClassifierMixin, RegressorMixin

from ._recurrent_config import RecurrentConfig
from ._recurrent_torch_model import RecurrentTorchModel
from .recurrent_dataset_helpers import _DEFAULT_SEQ_INPUT_SIZE, _RecurrentWrapperBase

logger = logging.getLogger("mlframe.training.neural.recurrent_dataset_helpers")  # matches the pre-carve logger name; preserves log-filter/caplog compatibility for existing callers/tests


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn Wrappers - Classifier
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentClassifierWrapper(_RecurrentWrapperBase, ClassifierMixin):
    """
    Sklearn-compatible wrapper for RecurrentTorchModel (classification).

    Provides fit/predict/predict_proba interface.
    """

    _estimator_type = "classifier"
    _is_regression = False

    def fit(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        eval_sample_weight: np.ndarray | None = None,
        class_weight: dict[int, float] | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
        cat_features: list[str] | None = None,
    ) -> RecurrentClassifierWrapper:
        """
        Train the model.

        Args:
            features: (n_samples, n_features) tabular features
            labels: (n_samples,) labels
            sample_weight: (n_samples,) per-sample weights
            sequences: List of (seq_len, n_features) arrays
            eval_set: Validation data tuple
            eval_sample_weight: (n_val_samples,) per-validation-row weights, threaded into val_loss /
                early-stopping / checkpoint-selection so weighted validation is actually honoured.
            class_weight: Class weights dict
            plot: Whether to enable logging
            plot_file: Path for logs (unused, for compatibility)
            cat_features: Tabular categorical column names to factorize + learn entity embeddings for (HYBRID / FEATURES_ONLY). No-op in
                SEQUENCE_ONLY (no tabular block) or when ``use_learnable_cat_embeddings`` is False.

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        # Resolve a fresh fit-local config copy: every mutation below lands here, leaving the verbatim ``self.config`` (clone source) untouched.
        self._cfg_resolved = copy.deepcopy(self.config) if self.config is not None else RecurrentConfig()

        # Record the label set so predict/predict_proba map argmax positions back to the ORIGINAL labels (sklearn ClassifierMixin contract).
        # Single-label targets are also ENCODED to contiguous 0..k-1 positions before training: the CrossEntropy head indexes by position, so a
        # raw label like 9 with a 3-output head would be out-of-bounds. predict() inverts the mapping via ``classes_``.
        _labels_arr = np.asarray(labels)
        _is_single_label = not (hasattr(labels, "ndim") and _labels_arr.ndim == 2 and _labels_arr.shape[1] >= 2)
        if _is_single_label:
            self.classes_ = np.unique(_labels_arr)
            labels = np.searchsorted(self.classes_, _labels_arr).astype(np.int64)
            # Size the output head to the observed class count so the CrossEntropy positions are always in range.
            self._cfg.num_classes = int(self.classes_.shape[0])

        # Factorize tabular cat columns to int codes (reordered leading) BEFORE the scaler fit + dataset build, so the learnable aux
        # CategoricalEmbedding can index them and the scaler skips the code columns. Scopes to ``features`` only; sequences are untouched.
        features = self._factorize_cats_fit(features, cat_features)

        # Detect multilabel from 2-D y: switches model to BCEWithLogitsLoss + sigmoid output.
        # Multilabel torchmetrics are skipped here (metrics come from the suite's downstream evaluation pipeline).
        # ``shape[1] >= 2`` lower bound: a single-label target delivered as
        # a 1-column 2-D array is still single-label - same gotcha as the
        # flat MLPClassifier in base.py (see commit + comment there); the
        # consequence here would be a misconfigured BCEWithLogitsLoss with
        # a num_classes=1 output head for what is actually multi-class
        # classification.
        self._is_multilabel = bool(hasattr(labels, "ndim") and labels.ndim == 2 and np.asarray(labels).shape[1] >= 2)
        if self._is_multilabel:
            self._n_labels = int(np.asarray(labels).shape[1])
            # Override config.num_classes to match label count so the MLP head builds the right number of output units.
            self._cfg.num_classes = self._n_labels

        self._validate_inputs(features, sequences)
        if self._aux_features_emptied_by_selection(features):
            logger.warning(
                "Skipping RecurrentClassifierWrapper fit: aux feature frame has 0 features (feature selection / column dropping removed every "
                "column). Nothing to fit -- the model is left unfitted (predict raises NotFittedError) so the suite skips it instead of crashing "
                "StandardScaler on an empty array."
            )
            self.model = None
            return self
        self._clear_cache()

        if self._cfg.scale_features and features is not None:
            # Numeric-only scaler fit: skips the leading cat-code columns (scaling embedding indices would corrupt them).
            self._scaler_fit_numeric_only(features)

        L.seed_everything(self.random_state, workers=True)

        train_dataset = self._create_dataset(sequences, features, labels, sample_weight)
        # Encode the eval_set labels through the same ``classes_`` mapping so the val CrossEntropy head sees in-range 0..k-1 positions too.
        if eval_set is not None and _is_single_label:
            eval_set = (*eval_set[:-1], np.searchsorted(self.classes_, np.asarray(eval_set[-1])).astype(np.int64))
        val_dataset = self._create_eval_dataset(eval_set, eval_sample_weight) if eval_set else None

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None

        self._aux_input_size = features.shape[1] if features is not None else 0
        self._seq_input_size = sequences[0].shape[1] if sequences is not None and len(sequences) > 0 else _DEFAULT_SEQ_INPUT_SIZE

        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=class_weight,
        )

        self.trainer, checkpoint_callback = self._create_trainer(val_loader is not None, plot)
        _prior_cudnn_benchmark = self._maybe_enable_cudnn_rnn_autotune()
        from ._base_logging import suppress_lightning_workers_warning
        try:
            with suppress_lightning_workers_warning():
                self.trainer.fit(self.model, train_loader, val_loader)
        finally:
            if _prior_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = _prior_cudnn_benchmark

        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            try:
                self.model = RecurrentTorchModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    config=self._cfg,
                    seq_input_size=self._seq_input_size,
                    aux_input_size=self._aux_input_size,
                    is_regression=False,
                    aux_categorical_cardinalities=list(self._cat_cardinalities_) if self._cat_cardinalities_ else None,
                    aux_categorical_embed_dim=self.categorical_embed_dim,
                    weights_only=False,
                )
            except Exception:
                # Wave 41 (2026-05-20): checkpoint-fallback to final-epoch model is a
                # quality regression source; preserve traceback for triage.
                logger.warning("Failed to load checkpoint, using final model", exc_info=True)

        return self

    def predict_proba(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, n_features) arrays
            batch_size: Override batch size for prediction

        Returns:
            (n_samples, num_classes) array of probabilities
        """
        if self.model is None:
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("Model not trained. Call fit() first.")

        self._validate_inputs(features, sequences)

        # Replay the fit-time cat factorization + coerce to float32 ndarray BEFORE the cache key (the key reads ``.dtype``, a DataFrame lacks it)
        # and BEFORE dataset construction. No-op when no cats were factorized; ``None`` (SEQUENCE_ONLY) passes through.
        features = self._prepare_predict_features(features)

        cache_key = self._compute_cache_key(features, sequences)
        _cached = self._cache_get(cache_key)
        if _cached is not None:
            return _cached

        n_samples = len(sequences) if sequences is not None else len(features)

        dataset = self._create_dataset(
            sequences,
            features,
            labels=np.zeros(n_samples, dtype=np.int64),
        )
        loader = self._create_dataloader(dataset, shuffle=False, batch_size=batch_size)

        # Same CUDA-broken-host guard _create_trainer applies at fit time (see that method's comment): on a
        # host where fit already downgraded to CPU because the CUDA probe failed, a bare
        # accelerator=self._cfg.accelerator here would still try CUDA and crash with "CUDA error: an
        # illegal memory access". _probe_cuda_is_usable() is memoised process-wide so this costs nothing
        # after the first call.
        from ._base_tensor_helpers import safe_accelerator
        predict_trainer = L.Trainer(
            accelerator=safe_accelerator(self._cfg.accelerator),
            precision=cast(Any, self._auto_precision()),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        from ._base_logging import suppress_lightning_workers_warning
        with suppress_lightning_workers_warning():
            predictions = predict_trainer.predict(self.model, loader)

        result = np.asarray(torch.cat(cast(list, predictions), dim=0).float().cpu().numpy().astype(np.float32))

        self._cache_put(cache_key, result)
        return result

    def predict(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, n_features) arrays

        Returns:
            (n_samples,) array of predictions
        """
        proba = self.predict_proba(features, sequences)
        positions = proba.argmax(axis=1)
        # Map argmax POSITIONS back through ``classes_`` so the returned labels are the original ones (sklearn ClassifierMixin contract); a raw
        # argmax would return 0..(k-1) positional indices, mislabelling any non-0..k-1 label set and breaking cross_val_predict / CalibratedClassifierCV / Stacking.
        classes_ = getattr(self, "classes_", None)
        if classes_ is None:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("Model not trained. Call fit() first.")
        return np.asarray(np.asarray(classes_)[positions])


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn Wrappers - Regressor
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentRegressorWrapper(_RecurrentWrapperBase, RegressorMixin):
    """
    Sklearn-compatible wrapper for RecurrentTorchModel (regression).

    Provides fit/predict interface.
    """

    _estimator_type = "regressor"
    _is_regression = True

    def fit(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        eval_sample_weight: np.ndarray | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
        cat_features: list[str] | None = None,
    ) -> RecurrentRegressorWrapper:
        """
        Train the model.

        Args:
            features: (n_samples, n_features) tabular features
            labels: (n_samples,) continuous target values
            sample_weight: (n_samples,) per-sample weights
            sequences: List of (seq_len, n_features) arrays
            eval_set: Validation data tuple
            eval_sample_weight: (n_val_samples,) per-validation-row weights, threaded into val_loss /
                early-stopping / checkpoint-selection so weighted validation is actually honoured.
            plot: Whether to enable logging
            plot_file: Path for logs (unused, for compatibility)
            cat_features: Tabular categorical column names to factorize + learn entity embeddings for (HYBRID / FEATURES_ONLY). No-op in
                SEQUENCE_ONLY (no tabular block) or when ``use_learnable_cat_embeddings`` is False.

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        # Resolve a fresh fit-local config copy: mutations below land here, leaving the verbatim ``self.config`` (clone source) untouched.
        self._cfg_resolved = copy.deepcopy(self.config) if self.config is not None else RecurrentConfig()
        # Regression has no AUPRC; redirect the classification default monitor to val_loss on the fit-local copy (never the verbatim config).
        if self._cfg.early_stopping_monitor == "val_auprc":
            self._cfg.early_stopping_monitor = "val_loss"

        # Factorize tabular cat columns to int codes (reordered leading) BEFORE the scaler fit + dataset build (sequences untouched).
        features = self._factorize_cats_fit(features, cat_features)

        self._validate_inputs(features, sequences)
        if self._aux_features_emptied_by_selection(features):
            logger.warning(
                "Skipping RecurrentRegressorWrapper fit: aux feature frame has 0 features (feature selection / column dropping removed every "
                "column). Nothing to fit -- the model is left unfitted (predict raises NotFittedError) so the suite skips it instead of crashing "
                "StandardScaler on an empty array."
            )
            self.model = None
            return self
        self._clear_cache()

        if self._cfg.scale_features and features is not None:
            self._scaler_fit_numeric_only(features)

        L.seed_everything(self.random_state, workers=True)

        train_dataset = self._create_dataset(sequences, features, labels, sample_weight)
        val_dataset = self._create_eval_dataset(eval_set, eval_sample_weight) if eval_set else None

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None

        self._aux_input_size = features.shape[1] if features is not None else 0
        self._seq_input_size = sequences[0].shape[1] if sequences is not None and len(sequences) > 0 else _DEFAULT_SEQ_INPUT_SIZE

        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=None,  # No class weights for regression
        )

        self.trainer, checkpoint_callback = self._create_trainer(val_loader is not None, plot)
        _prior_cudnn_benchmark = self._maybe_enable_cudnn_rnn_autotune()
        from ._base_logging import suppress_lightning_workers_warning
        try:
            with suppress_lightning_workers_warning():
                self.trainer.fit(self.model, train_loader, val_loader)
        finally:
            if _prior_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = _prior_cudnn_benchmark

        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            try:
                self.model = RecurrentTorchModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    config=self._cfg,
                    seq_input_size=self._seq_input_size,
                    aux_input_size=self._aux_input_size,
                    is_regression=True,
                    aux_categorical_cardinalities=list(self._cat_cardinalities_) if self._cat_cardinalities_ else None,
                    aux_categorical_embed_dim=self.categorical_embed_dim,
                    weights_only=False,
                )
            except Exception:
                # Wave 41 (2026-05-20): checkpoint-fallback to final-epoch model is a
                # quality regression source; preserve traceback for triage.
                logger.warning("Failed to load checkpoint, using final model", exc_info=True)

        return self

    def predict(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Predict continuous values.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, n_features) arrays
            batch_size: Override batch size for prediction

        Returns:
            (n_samples,) array of predictions
        """
        if self.model is None:
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("Model not trained. Call fit() first.")

        self._validate_inputs(features, sequences)

        # Replay the fit-time cat factorization + coerce to float32 ndarray BEFORE the cache key (the key reads ``.dtype``, a DataFrame lacks it)
        # and BEFORE dataset construction. No-op when no cats were factorized; ``None`` (SEQUENCE_ONLY) passes through.
        features = self._prepare_predict_features(features)

        cache_key = self._compute_cache_key(features, sequences)
        _cached = self._cache_get(cache_key)
        if _cached is not None:
            return _cached

        n_samples = len(sequences) if sequences is not None else len(features)

        dataset = self._create_dataset(
            sequences,
            features,
            labels=np.zeros(n_samples, dtype=np.float32),
        )
        loader = self._create_dataloader(dataset, shuffle=False, batch_size=batch_size)

        # Same CUDA-broken-host guard _create_trainer applies at fit time (see that method's comment): on a
        # host where fit already downgraded to CPU because the CUDA probe failed, a bare
        # accelerator=self._cfg.accelerator here would still try CUDA and crash with "CUDA error: an
        # illegal memory access". _probe_cuda_is_usable() is memoised process-wide so this costs nothing
        # after the first call.
        from ._base_tensor_helpers import safe_accelerator
        predict_trainer = L.Trainer(
            accelerator=safe_accelerator(self._cfg.accelerator),
            precision=cast(Any, self._auto_precision()),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        from ._base_logging import suppress_lightning_workers_warning
        with suppress_lightning_workers_warning():
            predictions = predict_trainer.predict(self.model, loader)

        result = np.asarray(torch.cat(cast(list, predictions), dim=0).float().cpu().numpy().astype(np.float32))

        self._cache_put(cache_key, result)
        return result
