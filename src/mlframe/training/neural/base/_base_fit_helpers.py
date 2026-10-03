"""Helper methods of ``_FitMixin._fit_common``, carved out to keep ``_base_fit`` under its size budget."""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping as EarlyStoppingCallback
from sklearn.base import ClassifierMixin, RegressorMixin

from ._base_losses import _make_binary_focal_loss, _validate_no_nan_inf
from .._base_callbacks import BestEpochModelCheckpoint, ValLossDivergenceCallback, MonotonicDeclineStopCallback
from .._history_recorder import TrainingHistoryRecorder

logger = __import__("logging").getLogger("mlframe.training.neural.base")


class _FitCommonHelpersMixin:
    """The step helpers ``_fit_common`` delegates to."""

    # Constructor params, mirrored onto self by store_params_in_object() in the composed
    # estimator's __init__; declared here so mypy can type-check this mixin's reads of them.
    random_state: Optional[int]
    class_weight: Any
    float32_matmul_precision: Optional[str]
    datamodule_class: Any
    datamodule_params: dict
    network_params: dict
    trainer_params: dict
    use_swa: bool
    use_ema: bool
    _is_multilabel: bool
    swa_params: Optional[dict]
    ema_params: Optional[dict]
    model_params: dict
    model_class: Any
    early_stopping_rounds: int
    focal_loss_gamma: Any
    focal_loss_alpha: Any
    tune_params: Any
    tune_batch_size: Any
    # Runtime state (not constructor params) shared with _PredictMixin -- declared identically
    # there and here so mypy sees one consistent type across the composed estimator's MRO.
    model: Any
    trainer: Any
    _label_encoder: Any
    classes_: Any

    def _fit_common_idempotent_re_seeding_before(self):
        """Block of _fit_common starting at ``if self.random_state is not None:``."""
        if self.random_state is not None:
            # ``verbose`` was added to ``L.seed_everything`` in lightning >=2.x;
            # older installs (the TVT-regression test box) raise TypeError on
            # the kwarg. Try the quiet form first, fall back to the legacy
            # signature so the same code base stays portable across Lightning
            # versions. (When we fall through, Lightning prints the seed line
            # at INFO; the ``_LightningRankZeroNoiseFilter`` further down still
            # suppresses noisy rank-zero chatter, so the on-disk log stays
            # essentially identical.)
            try:
                L.seed_everything(int(self.random_state), workers=True, verbose=False)
            except TypeError:
                L.seed_everything(int(self.random_state), workers=True)

    @staticmethod
    def _fit_common_reject_those_further_down(eval_set):
        """Block of _fit_common starting at ``if eval_set is not None and not (isinstance(eval_set, tuple) and eval_``."""
        if eval_set is not None and not (isinstance(eval_set, tuple) and eval_set[0] is None):
            # eval_set may be a 2-tuple ``(X_val, y_val)`` or a list-of-tuples
            # (LightGBM convention) -- normalise to peek at the val frame.
            _ev = eval_set[0] if isinstance(eval_set, list) and eval_set else eval_set
            if isinstance(_ev, tuple) and _ev[0] is not None:
                _validate_no_nan_inf("X_val", _ev[0])
                _validate_no_nan_inf("y_val", _ev[1], allow_object_dtype=True)

    def _fit_common_enable_tf32_float32_matmul(self):
        """Block of _fit_common starting at ``if self.float32_matmul_precision and torch.cuda.is_available():``."""
        if self.float32_matmul_precision and torch.cuda.is_available():
            _allowed_matmul = ("highest", "high", "medium")
            if self.float32_matmul_precision not in _allowed_matmul:
                raise ValueError(f"float32_matmul_precision must be one of {_allowed_matmul}, " f"got {self.float32_matmul_precision!r}")
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(self.float32_matmul_precision)
                logger.info("Enabled float32_matmul_precision=%s", self.float32_matmul_precision)

    def _fit_common_inference_time(self, _classifier_single_label, is_partial_fit, classes, y, eval_set, _local_dm_params):
        """Block of _fit_common starting at ``if _classifier_single_label:``."""
        if _classifier_single_label:
            from sklearn.preprocessing import LabelEncoder as _LabelEncoder
            if is_partial_fit and classes is not None:
                # ``classes`` is the caller's full universe of labels even if
                # this partial_fit batch only sees a subset. Fit encoder to it
                # so the index space stays stable across partial_fit calls.
                self._label_encoder = _LabelEncoder().fit(np.asarray(classes))
                self.classes_ = self._label_encoder.classes_
            elif not hasattr(self, "_label_encoder") or self._label_encoder is None:
                _y_for_le = y.values if isinstance(y, pd.Series) else np.asarray(y)
                if _y_for_le.ndim == 2 and _y_for_le.shape[1] == 1:
                    _y_for_le = _y_for_le.ravel()
                self._label_encoder = _LabelEncoder().fit(_y_for_le)
                self.classes_ = self._label_encoder.classes_
            # else: partial_fit continuation with encoder already built; reuse.

            # Encode training y to integer indices for the loss function.
            _y_arr_train = y.values if isinstance(y, pd.Series) else np.asarray(y)
            if _y_arr_train.ndim == 2 and _y_arr_train.shape[1] == 1:
                _y_arr_train = _y_arr_train.ravel()
            y = self._label_encoder.transform(_y_arr_train)

            # Encode validation labels with the SAME encoder so val_loss /
            # val_MSE share the index space the model trains on.
            if eval_set[1] is not None:
                _y_arr_val = eval_set[1].values if isinstance(eval_set[1], pd.Series) else np.asarray(eval_set[1])
                if _y_arr_val.ndim == 2 and _y_arr_val.shape[1] == 1:
                    _y_arr_val = _y_arr_val.ravel()
                eval_set = (eval_set[0], self._label_encoder.transform(_y_arr_val))

            # F-05 (2026-05-30): binary classification uses 1-output
            # sigmoid + BCEWithLogitsLoss instead of 2-output softmax +
            # CrossEntropyLoss. The two-output softmax head is
            # overparameterised (softmax is shift-invariant in z0-z1)
            # and inconsistent with the multilabel BCE path. Switching
            # halves the output-layer params and aligns binary with the
            # K=1 case of multilabel. predict_proba keeps returning the
            # sklearn-canonical (N, 2) shape by stacking [1-p, p] in the
            # classifier wrapper. Detection happens here (before dm
            # construction) so labels_dtype can be set to float32 in
            # time for BCEWithLogitsLoss.
            self._binary_sigmoid_head = bool(len(self.classes_) == 2)
            if self._binary_sigmoid_head:
                _local_dm_params["labels_dtype"] = torch.float32
        else:
            # Multilabel or non-classifier paths: never binary.
            self._binary_sigmoid_head = False
        return eval_set, y

    def _fit_common_multilabel_non_classifier_paths(self, _classifier_single_label, y, sample_weight):
        """Block of _fit_common starting at ``if _classifier_single_label:``."""
        if _classifier_single_label:
            # F-13 (2026-05-30): sklearn-canonical ``class_weight`` support.
            # ``class_weight="balanced"`` -> per-sample weights = n / (K * count(class))
            # ``class_weight={cls: w, ...}`` -> per-sample weights = w[cls]
            # ``class_weight=None`` -> no per-class weighting
            # The resulting per-sample weights are multiplied INTO any
            # caller-supplied ``sample_weight`` (sklearn convention) so
            # both knobs compose: a caller can weight rare events AND
            # rebalance classes simultaneously.
            if self.class_weight is not None:
                from sklearn.utils.class_weight import (
                    compute_sample_weight as _compute_sample_weight,
                )
                # compute_sample_weight expects ORIGINAL (un-encoded)
                # class labels; pass the train y BEFORE the encoder
                # transformed it. We reconstruct the original via
                # inverse_transform from the already-encoded ``y``.
                _y_for_cw = self._label_encoder.inverse_transform(y)
                _cw_weights = _compute_sample_weight(
                    class_weight=self.class_weight, y=_y_for_cw,
                ).astype(np.float32)
                if sample_weight is None:
                    sample_weight = _cw_weights
                else:
                    # Multiplicative composition with caller's weights.
                    _sw_arr = np.asarray(sample_weight, dtype=np.float32).ravel()
                    if _sw_arr.shape != _cw_weights.shape:
                        raise ValueError(
                            f"class_weight-derived weights shape " f"{_cw_weights.shape} != sample_weight shape " f"{_sw_arr.shape}; cannot multiply."
                        )
                    sample_weight = _sw_arr * _cw_weights
                logger.info(
                    "Applied class_weight=%r -> per-sample weights " "with mean=%.4g, min=%.4g, max=%.4g",
                    self.class_weight,
                    float(np.mean(sample_weight)),
                    float(np.min(sample_weight)),
                    float(np.max(sample_weight)),
                )
        return sample_weight

    def _fit_common_explicit_user_set_scale(self, _out_act, num_classes, _scale_set, _center_set, y, _net_params):
        """Block of _fit_common starting at ``if _out_act == "tanh_train_range" and num_classes == 1 and not getattr``."""
        if _out_act == "tanh_train_range" and num_classes == 1 and not getattr(self, "_is_multi_target_regression", False) and not (_scale_set and _center_set):
            try:
                _y_arr = np.asarray(
                    y.values if isinstance(y, pd.Series) else y,
                    dtype=np.float64,
                ).reshape(-1)
                # Single-pass numba kernel: min + max + mean + std over
                # finite entries in ONE traversal of the buffer (Welford's
                # online variance is numerically stable on high-range y;
                # the naive ``y_finite.min() / max() / std()`` triple did
                # three independent passes after materialising an
                # ``isfinite`` mask). Saves ~3x memory bandwidth on a
                # multi-million-row regression target and stays bit-exact
                # vs numpy ddof=0 to ~1e-15.
                from mlframe.training.neural._neural_numba_kernels import finite_min_max_std as _fmms
                _n_finite, _ymin, _ymax, _ymean, _ystd = _fmms(_y_arr)
                if _n_finite > 1:
                    # scale = (max-min)/2 + 3*std; ~6-sigma half-window
                    # around the train midpoint. center = (min+max)/2.
                    # Fill ONLY the None slots so an explicit user-set
                    # value (scale OR center) is preserved. Asymmetric-
                    # partial input (scale=2.0, center=None) is the case
                    # the pre-fix AND-condition skipped, then
                    # ``generate_mlp`` raised on the missing field.
                    if _net_params.get("output_activation_scale") is None:
                        _net_params["output_activation_scale"] = (_ymax - _ymin) / 2.0 + 3.0 * _ystd
                    if _net_params.get("output_activation_center") is None:
                        _net_params["output_activation_center"] = (_ymin + _ymax) / 2.0
                    logger.info(
                        "MLP output_activation='tanh_train_range' "
                        "auto-derived from y_train: scale=%.4g, center=%.4g "
                        "(y_min=%.4g, y_max=%.4g, y_std=%.4g).",
                        _net_params["output_activation_scale"],
                        _net_params["output_activation_center"],
                        _ymin, _ymax, _ystd,
                    )
                else:
                    logger.warning(
                        "MLP output_activation='tanh_train_range' requested " "but y_train has <=1 finite value; falling back to " "'linear' for this fit.",
                    )
                    _net_params["output_activation"] = "linear"
            except Exception as _oa_err:
                logger.warning(
                    "MLP output_activation='tanh_train_range' y_train " "derivation failed (%s); falling back to 'linear'.",
                    _oa_err,
                )
                _net_params["output_activation"] = "linear"

    @staticmethod
    def _fit_common_unaffected_because_bf16_mixed(trainer_params, _resolved):
        """Block of _fit_common starting at ``if "precision" not in trainer_params and _resolved in ("cuda", "gpu"):``."""
        if "precision" not in trainer_params and _resolved in ("cuda", "gpu"):
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    _cc_major, _ = torch.cuda.get_device_capability(0)
                    if _cc_major >= 8:
                        trainer_params["precision"] = "bf16-mixed"
                        logger.info(
                            "F-27: auto-enabled precision='bf16-mixed' on "
                            "Ampere+ GPU (compute capability %d.x). Set "
                            "trainer_params['precision'] explicitly to "
                            "override (e.g. '32-true' or '16-mixed').",
                            _cc_major,
                        )
            except Exception as _cc_err:
                logger.debug(
                    "F-27 bf16 auto-enable probe failed (%s); leaving " "precision at Lightning default.",
                    _cc_err,
                )

    @staticmethod
    def _fit_common_chrome_trace_export_mlframe(trainer_params, _ckpt_root):
        """Block of _fit_common starting at ``if "profiler" not in trainer_params:``."""
        if "profiler" not in trainer_params:
            if os.environ.get("MLFRAME_TORCH_PROFILE", "0") == "1":
                try:
                    from lightning.pytorch.profilers import PyTorchProfiler
                    _activities = [torch.profiler.ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        _activities.append(torch.profiler.ProfilerActivity.CUDA)
                    _prof_dir = os.environ.get(
                        "MLFRAME_TORCH_PROFILE_DIR",
                        os.path.join(_ckpt_root, "torch_traces"),
                    )
                    os.makedirs(_prof_dir, exist_ok=True)
                    # group_by_input_shapes helps recurrent models where
                    # variable seq-lens would otherwise collapse into a
                    # single bucket; harmless for fixed-shape MLP.
                    trainer_params["profiler"] = PyTorchProfiler(
                        dirpath=_prof_dir,
                        filename=f"mlp_{os.getpid()}",
                        export_to_chrome=True,
                        record_module_names=True,
                        activities=_activities,
                        schedule=torch.profiler.schedule(
                            wait=1, warmup=1, active=3, repeat=1,
                        ),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False,
                        with_flops=True,
                        group_by_input_shapes=True,
                    )
                    logger.info(
                        "F-36: MLFRAME_TORCH_PROFILE=1 active; chrome traces " "land in %s. Open via chrome://tracing or Perfetto.",
                        _prof_dir,
                    )
                except Exception as _prof_err:
                    logger.warning(
                        "MLFRAME_TORCH_PROFILE=1 but profiler setup failed " "(%s); fit continues without profiling.",
                        _prof_err,
                    )

    def _fit_common_self_use_ema(self, callbacks):
        """Block of _fit_common starting at ``if self.use_ema:``."""
        if self.use_ema:
            # F-28 (2026-05-31): exponential moving average of weights via
            # Lightning's WeightAveraging callback + torch's EMA averaging
            # function. Lightning auto-swaps the averaged weights into the
            # live model on on_train_end, so downstream predict() uses the
            # EMA copy transparently — zero changes to save/load needed.
            # Cross-cited in two 2026-05-31 research agents
            # (Lightning-plugins + activations/optimizers): +0.04-0.66% on
            # tabular MLPs, cheaper than SWA (no LR warm-restart phase).
            # Falls back to a SWA-as-EMA shim when WeightAveraging is not
            # in the installed Lightning (added in Lightning ~2.5).
            try:
                from lightning.pytorch.callbacks import WeightAveraging
                _ema_has_native = True
            except ImportError:
                _ema_has_native = False
            from torch.optim.swa_utils import get_ema_avg_fn
            _ema_params = dict(self.ema_params or {})
            # ``decay`` is exposed at the mlframe level for ergonomics;
            # plumb it into get_ema_avg_fn. Default 0.999 mirrors the
            # torch.optim.swa_utils default.
            _decay = float(_ema_params.pop("decay", 0.999))
            _ema_params.setdefault("avg_fn", get_ema_avg_fn(decay=_decay))
            if _ema_has_native:
                from lightning.pytorch.callbacks import WeightAveraging
                callbacks.append(WeightAveraging(**_ema_params))
            else:
                # SWA-as-EMA fallback: SWA accepts ``avg_fn`` (passes to
                # torch's AveragedModel under the hood). Default
                # ``swa_lrs`` to the user's learning_rate so SWA does NOT
                # trigger a LR-restart phase — that would defeat the EMA
                # semantic by tuning a separate "averaged" model with a
                # different LR. ``swa_epoch_start=0.5`` starts averaging
                # halfway through training (standard SWA default).
                _ema_params.setdefault(
                    "swa_lrs",
                    float(self.model_params.get("learning_rate", 1e-3)),
                )
                _ema_params.setdefault("swa_epoch_start", 0.5)
                callbacks.append(StochasticWeightAveraging(**_ema_params))
                logger.info(
                    "use_ema=True: lightning.pytorch.callbacks.WeightAveraging "
                    "is unavailable (Lightning < 2.5?); falling back to "
                    "StochasticWeightAveraging with EMA avg_fn + constant "
                    "swa_lrs=learning_rate so no LR-restart phase. Upgrade "
                    "Lightning to >=2.5 for the dedicated EMA path."
                )

    def _fit_common_has_validation(self, has_validation, callbacks, metric_name):
        """Block of _fit_common starting at ``if has_validation:``."""
        if has_validation:
            logger.info("Using early_stopping_rounds=%d", self.early_stopping_rounds)
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=f"val_{metric_name}",
                    min_delta=0.001,
                    patience=self.early_stopping_rounds,
                    mode="min",
                    # verbose=False: BestEpochModelCheckpoint already emits
                    # "New best model at epoch X with metric=..." via mlframe's
                    # logger (neural/base.py:771). Lightning's verbose=True
                    # would duplicate that as both a logger.info and a print()
                    # for every improvement -- 3 lines per best-epoch event.
                    verbose=False,
                )
            )
            # 2026-05-23 audit-followup #6: divergence detector. Warns
            # when val_loss climbs >=100x its baseline within training
            # so operators catch Identity-MLP-style collapses before
            # paying the full training budget. No automatic stop --
            # ES already covers the no-improvement case.
            callbacks.append(
                ValLossDivergenceCallback(
                    monitor=f"val_{metric_name}",
                    divergence_factor=100.0,
                )
            )
            # Monotonic strict-decline overfitting stop, COMPLEMENTARY to EarlyStoppingCallback above:
            # stops once the monitored val metric strictly worsens for ``monotonic_decline_patience``
            # consecutive epochs since the best (a confident-overfitting signal that fires faster than
            # patience). Default-on; the monitored val_<metric> is min-direction (RMSE / ICE). The
            # BestEpochModelCheckpoint still restores the global-best epoch, so an early stop keeps the
            # right weights. ``monotonic_decline_patience=None`` on the estimator disables it.
            _mono_patience = getattr(self, "monotonic_decline_patience", 7)
            if _mono_patience is not None:
                callbacks.append(
                    MonotonicDeclineStopCallback(
                        monitor=f"val_{metric_name}",
                        patience=_mono_patience,
                        mode="min",
                    )
                )
            # Record per-epoch train/val history in the booster ``evals_result_`` shape so the per-model
            # training-curve chart (reporting._render_training_curves, default-ON) auto-emits for neural
            # models exactly as it does for lgb/xgb/cb -- with the early-stop vline + wasted-post-ES shading.
            callbacks.append(TrainingHistoryRecorder(monitor=f"val_{metric_name}", mode="min"))
            # Per-epoch full-metric-suite capture for meta-learning / HPO-from-early-observation. Default-ON for
            # neural: val predictions are already concatenated each validation epoch, so the only marginal cost is
            # the cheap metric kernel. ``capture_iteration_metrics=False`` on the estimator opts out.
            _cap_iter = getattr(self, "capture_iteration_metrics", None)
            if _cap_iter is None:
                _cap_iter = True  # neural family default
            if _cap_iter:
                from mlframe.training.neural._history_recorder import IterationMetricsRecorder
                if self._is_multilabel:
                    _tt, _ncls = "multilabel_classification", None
                elif isinstance(self, ClassifierMixin):
                    _classes = getattr(self, "classes_", None)
                    _ncls = len(_classes) if _classes is not None else 2
                    _tt = "binary_classification" if _ncls <= 2 else "multiclass_classification"
                else:
                    _tt, _ncls = "regression", None
                callbacks.append(IterationMetricsRecorder(target_type=_tt, n_classes=_ncls or None))

    def _fit_common_bleed_multilabel_config_into(self, _local_model_params):
        """Block of _fit_common starting at ``if self._is_multilabel:``."""
        if self._is_multilabel:
            import torch.nn.functional as _F
            _local_model_params["loss_fn"] = _F.binary_cross_entropy_with_logits
            _local_model_params["task_type"] = "multilabel"
        elif self._binary_sigmoid_head:
            # F-05: binary sigmoid head -> BCEWithLogitsLoss + task_type
            # marker so predict_step / compute_metrics emit sigmoid probs
            # and the classifier wrapper stacks (N, 2).
            # F-29 (2026-05-31): optional focal loss for binary. When
            # ``focal_loss_gamma`` is set, replace BCE with the sigmoid
            # focal loss formulation (Lin et al. 2017): heavier penalty
            # on hard examples, mitigates class imbalance even WITHOUT
            # explicit class_weight. Default off — focal loss degrades
            # the model's probability calibration (Cattan 2024) so it's
            # opt-in for users who care more about F1 / recall on
            # severely imbalanced binary targets than about calibrated
            # probabilities. focal_loss_alpha is the class-1 weight
            # (default 0.25 per the original paper).
            if self.focal_loss_gamma is not None:
                _local_model_params["loss_fn"] = _make_binary_focal_loss(
                    gamma=float(self.focal_loss_gamma),
                    alpha=float(self.focal_loss_alpha),
                )
            else:
                _local_model_params["loss_fn"] = torch.nn.BCEWithLogitsLoss()
            _local_model_params["task_type"] = "binary"
        elif isinstance(self, RegressorMixin):
            # F-24 (2026-05-31): tag regressors so predict_step returns
            # raw values for ALL shapes including (N, K>=2) multi-target.
            # Without this tag, predict_step's existing
            # ``logits.shape[1] > 1`` branch would mistakenly apply
            # softmax to (N, K) regression outputs.
            _local_model_params["task_type"] = "regression"
        elif isinstance(self, ClassifierMixin) and not self._is_multilabel and not self._binary_sigmoid_head and self.label_smoothing > 0.0:
            # F-30 (2026-05-31): label smoothing for MULTICLASS only.
            # Replaces the caller's CrossEntropyLoss with one carrying
            # label_smoothing=epsilon. Per RealMLP-TD NeurIPS 2024:
            # +1.8% multiclass accuracy on the ablation. Skipped for
            # binary (Cattan 2024 shows calibration regression on
            # imbalanced binary; focal_loss_gamma is the analogue knob).
            _local_model_params["loss_fn"] = torch.nn.CrossEntropyLoss(
                label_smoothing=float(self.label_smoothing),
            )

    def _fit_common_self_tune_params_partial(self, is_partial_fit, trainer, dm):
        """Block of _fit_common starting at ``if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned"``."""
        if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned")):
            tuner = Tuner(trainer)

            if self.tune_batch_size:
                tuner.scale_batch_size(model=self.model, datamodule=dm, mode="binsearch", init_val=self.datamodule_params.get("batch_size", 32))

            lr_finder = tuner.lr_find(self.model, datamodule=dm, num_training=300)
            if lr_finder is not None:
                new_lr = lr_finder.suggestion()
                logger.info("Using suggested LR=%s", new_lr)
                self.model.hparams.learning_rate = new_lr
            else:
                logger.warning("tuner.lr_find returned no suggestion; keeping the configured learning_rate.")

            if is_partial_fit:
                self._tuned = True

    def _fit_common_consumes_evals_result_best(self, trainer):
        """Block of _fit_common starting at ``for callback in trainer.callbacks: # type: ignore[attr-defined] # Trai``."""
        for callback in trainer.callbacks:  # Trainer.callbacks is a real runtime attr, just not in the stub's public surface
            if isinstance(callback, TrainingHistoryRecorder):
                if callback.evals_result_:
                    self.evals_result_ = callback.evals_result_
                    if callback.best_iteration_ is not None:
                        self.best_iteration_ = callback.best_iteration_
                break

    def _fit_common_callback_trainer_callbacks_type(self, trainer):
        """Block of _fit_common starting at ``for callback in trainer.callbacks: # type: ignore[attr-defined] # Trai``."""
        from mlframe.training.neural._history_recorder import IterationMetricsRecorder

        for callback in trainer.callbacks:  # Trainer.callbacks is a real runtime attr, just not in the stub's public surface
            if isinstance(callback, IterationMetricsRecorder):
                if callback.iteration_metrics_:
                    self.iteration_metrics_ = callback.iteration_metrics_
                break

    def _fit_common_distributed_training_compatibility(self, trainer):
        """Block of _fit_common starting at ``if hasattr(self.model, "best_epoch") and self.model.best_epoch is not ``."""
        if hasattr(self.model, "best_epoch") and self.model.best_epoch is not None:
            self.best_epoch = self.model.best_epoch
            logger.info("Best epoch recorded: %s", self.best_epoch)
        else:
            # Fallback to callback for backward compatibility
            for callback in trainer.callbacks:  # Trainer.callbacks is a real runtime attr, just not in the stub's public surface
                if isinstance(callback, BestEpochModelCheckpoint):
                    self.best_epoch = callback.best_epoch
                    if self.best_epoch is not None:
                        logger.info("Best epoch recorded from callback: %s", self.best_epoch)
                    break

    def _fit_common_operators_relying_prior_whole(self):
        """Block of _fit_common starting at ``if not _os_drop_dm.environ.get("MLFRAME_KEEP_PREDICTION_DATAMODULE"):``."""
        import os as _os_drop_dm

        if not _os_drop_dm.environ.get("MLFRAME_KEEP_PREDICTION_DATAMODULE"):
            _dm = getattr(self, "prediction_datamodule", None)
            if _dm is not None:
                for _attr in (
                    "train_features", "train_labels", "train_sample_weight",
                    "val_features", "val_labels", "val_sample_weight",
                ):
                    if hasattr(_dm, _attr):
                        setattr(_dm, _attr, None)
                # ``_train_dataset`` / ``_val_dataset`` -- if the
                # datamodule materialised PyTorch ``Dataset`` wrappers
                # (which hold the same tensors via the Dataset's own
                # attributes), null those too. Predict-path setup
                # rebuilds them from the predict-side X / y.
                for _attr in ("_train_dataset", "_val_dataset", "train_dataset", "val_dataset"):
                    if hasattr(_dm, _attr):
                        setattr(_dm, _attr, None)
            self._datamodule_tensors_dropped = True
