"""Fit mixin carved out of ``neural.base``.

``_FitMixin`` holds the single cohesive training run (``_fit_common``) plus
the ``fit`` / ``partial_fit`` wrappers. It operates purely on ``self`` so the
estimator mixes it in unchanged. Live trainer/accelerator objects set here are
dropped on pickle by the estimator's ``__getstate__`` (see ``base``).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import lightning as L
from sklearn.base import ClassifierMixin

from .._base_logging import suppress_lightning_workers_warning
from .._base_tensor_helpers import safe_accelerator
from ._base_fit_prep import _FitPrepMixin

logger = __import__("logging").getLogger("mlframe.training.neural.base")


from ._base_fit_helpers import _FitCommonHelpersMixin
from ._base_fit_steps import _FitStepsMixin


class _FitMixin(_FitStepsMixin, _FitCommonHelpersMixin, _FitPrepMixin):
    """Common fit / partial_fit training run for the estimator.

    The fit-time categorical / embedding-text feature-prep methods
    (``_encode_emb_text_fit`` / ``_factorize_cats_fit`` / ``_apply_cat_codes``)
    live in :class:`._base_fit_prep._FitPrepMixin`, inherited here.
    """

    def _fit_common(
        self,
        X,
        y,
        eval_set: tuple = (None, None),
        is_partial_fit: bool = False,
        classes: Optional[np.ndarray] = None,
        fit_params: Optional[dict] = None,
        sample_weight=None,
    ):
        """Common logic for fit and partial_fit."""
        if fit_params is None:
            fit_params = {}

        # Make embedding-vector + free-text columns numeric BEFORE validation + input-dim computation (the MLP has no
        # native embedding/text layers). Stashes the fitted encoder on self for predict(). No-op when none are named.
        X, eval_set = self._encode_emb_text_fit(X, eval_set, fit_params)

        # Factorize raw categorical columns to integer codes (reordered leading) BEFORE validation, so the learnable ``CategoricalEmbedding``
        # can index them and ``_validate_no_nan_inf`` sees a pure-numeric frame. No-op when no ``cat_features`` are named or the knob is off.
        X, eval_set = self._factorize_cats_fit(X, eval_set, fit_params, is_partial_fit=is_partial_fit)

        self._fit_step_seed_and_validate(X, y, eval_set)

        eval_set = self._fit_step_normalize_eval_set(eval_set)
        has_validation = eval_set[0] is not None

        eval_sample_weight = fit_params.get("eval_sample_weight")

        _is_multilabel_target, _local_dm_params = self._fit_step_dm_params(y)

        # Single-label classifier label encoding. sklearn convention is that
        # ``y`` can be any hashable (strings, non-dense ints, booleans);
        # CrossEntropyLoss + ``labels_dtype=int64`` require ``{0..K-1}``
        # integer indices. Without this encoding, ``fit`` crashed with
        # ``IndexError: Target N is out of bounds`` for any y whose value set
        # is not exactly ``{0..K-1}`` (e.g. ``{10, 20}`` or ``{"low","high"}``;
        # F-19 in the mlp audit). Build the bidirectional encoder
        # once and stash on ``self`` so ``predict`` can ``inverse_transform``
        # at inference time (F-01).
        _classifier_single_label = isinstance(self, ClassifierMixin) and not _is_multilabel_target
        eval_set, y = self._fit_common_inference_time(_classifier_single_label, is_partial_fit, classes, y, eval_set, _local_dm_params)

        sample_weight = self._fit_common_multilabel_non_classifier_paths(_classifier_single_label, y, sample_weight)

        dm = self.datamodule_class(
            train_features=X,
            train_labels=y,
            train_sample_weight=sample_weight,
            val_features=eval_set[0],
            val_labels=eval_set[1],
            val_sample_weight=eval_sample_weight,
            **_local_dm_params,
        )
        # Stash for predict-time reuse so we don't re-instantiate (and trigger the
        # "No datamodule found from training. Creating temporary datamodule for
        # prediction." misleading warning at every predict call).
        self.prediction_datamodule = dm

        num_classes = self._fit_step_num_classes(y, is_partial_fit, classes, _is_multilabel_target)

        metrics, metric_name = self._fit_step_network_and_metrics(X, y, num_classes, is_partial_fit)

        # When no validation data, monitor train_loss instead of train metrics (which may not be logged)
        if has_validation:
            monitor_metric = f"val_{metric_name}"
        else:
            monitor_metric = "train_loss"

        checkpointing, _ckpt_root = self._fit_step_checkpointing(monitor_metric)

        trainer_params, _requested = self._fit_step_trainer_params(has_validation, _ckpt_root, safe_accelerator)

        callbacks = self._fit_step_callbacks(checkpointing, trainer_params, has_validation, metric_name)

        trainer = L.Trainer(**trainer_params, callbacks=callbacks)

        self._fit_step_build_model(trainer, metrics, X)

        self._fit_common_self_tune_params_partial(is_partial_fit, trainer, dm)

        from ._cuda_fallback import run_with_cuda_cpu_fallback

        # Use the ORIGINALLY-REQUESTED accelerator (``_requested``, captured above BEFORE the
        # ``safe_accelerator`` downgrade), not ``trainer_params["accelerator"]`` (already overwritten
        # with the resolved value at "trainer_params["accelerator"] = _resolved" above). On a host
        # where the CUDA probe legitimately failed, ``_resolved`` is "cpu" -- reading THAT here made
        # ``is_cuda_runtime_error`` see accelerator="cpu" and refuse to recognize a CUDA-fingerprinted
        # failure as fallback-worthy, so it re-raised instead of retrying: exactly on the broken/absent-
        # CUDA hosts this fallback exists to protect. ``_requested`` still reflects what the caller
        # actually asked for (e.g. a test simulating a CUDA failure via ``trainer_params={"accelerator":
        # "cuda"}"``), which is what the gate is meant to test against.
        _fit_accelerator = str(_requested)
        # suppress_lightning_workers_warning() was defined + documented as "wrap the trainer.fit()/
        # predict() invocations" but never actually called anywhere.
        with suppress_lightning_workers_warning():
            _, trainer = run_with_cuda_cpu_fallback(
                action="fit",
                primary_trainer=trainer,
                model=self.model,
                accelerator=_fit_accelerator,
                run_fn=lambda t: t.fit(model=self.model, datamodule=dm),
                build_cpu_trainer=lambda: L.Trainer(**{**trainer_params, "accelerator": "cpu", "devices": 1}, callbacks=callbacks),
            )

        # Expose per-epoch train/val history (booster ``evals_result_`` shape) + the best epoch so the
        # reporting layer's training-curve chart picks it up with no neural-specific code (it already
        # consumes ``evals_result_``/``best_iteration_`` for lgb/xgb/cb).
        self._fit_common_consumes_evals_result_best(trainer)
        self._fit_common_callback_trainer_callbacks_type(trainer)

        # Extract best epoch from model (set by checkpoint callback, DDP-safe). Prefer model.best_epoch over callback.best_epoch
        # for distributed training compatibility.
        self._fit_common_distributed_training_compatibility(trainer)

        # Clean up to avoid pickle issues and free memory
        self.trainer = None

        # Free the train/val tensors held by the cached datamodule
        # WITHOUT dropping the datamodule shell itself. The full
        # train+val feature / label / sample_weight tensors were the
        # actual save() bloat (1788 MB on disk for a 4M x 323 float32
        # frame, per a TVT regression log) -- the shell (~few KB
        # of config + class refs) is fine to pickle. Keeping the shell
        # lets predict() reuse the configured pre-pipeline /
        # batch_size / dataloader_params without rebuilding the
        # datamodule from scratch, AND silences the spurious "No
        # datamodule found from training" WARNING that fired on every
        # predict-after-fit when we used to NULL the whole reference.
        # Opt out via env MLFRAME_KEEP_PREDICTION_DATAMODULE=1 for
        # operators relying on the prior whole-stash behaviour.
        self._fit_common_operators_relying_prior_whole()

        return self

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None, **fit_params: Any) -> Any:
        """Fit the model to the data.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional per-sample weights for training
            **fit_params: Additional parameters including:
                - eval_set: Tuple of (X_val, y_val) for validation
                - eval_sample_weight: Optional validation sample weights

        Returns:
            The result of ``_fit_common``, i.e. the fitted estimator (``self``).
        """
        eval_set = fit_params.get("eval_set", (None, None))
        # Support sample_weight both as parameter and in fit_params
        if sample_weight is None:
            sample_weight = fit_params.get("sample_weight")
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=False, fit_params=fit_params, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes: Optional[np.ndarray] = None, sample_weight=None, **fit_params):
        """Incremental training for online learning."""
        eval_set = fit_params.get("eval_set", (None, None))
        if sample_weight is None:
            sample_weight = fit_params.get("sample_weight")
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=True, classes=classes, fit_params=fit_params, sample_weight=sample_weight)
