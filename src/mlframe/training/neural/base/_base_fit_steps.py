"""Step methods of ``_FitMixin._fit_common``, carved out to keep that method under the function-length gate."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from sklearn.base import ClassifierMixin

from mlframe.metrics.core import compute_probabilistic_multiclass_error
from .._base_callbacks import BestEpochModelCheckpoint
from .._base_logging import MetricSpec, _rmse_metric
from ._base_losses import _validate_no_nan_inf
from .._base_tensor_helpers import to_tensor_any

logger = __import__("logging").getLogger("mlframe.training.neural.base")


class _FitStepsMixin:
    """Coherent blocks of the training run, called in order by ``_FitMixin._fit_common``."""

    if TYPE_CHECKING:
        # Attributes and sibling-mixin methods the owning estimator provides at runtime; declared for mypy only.
        datamodule_params: Any
        _binary_sigmoid_head: Any
        model_class: Any
        model_params: Any
        network_params: Any
        example_input_array: Any
        swa_params: Any
        trainer_params: Any
        use_ema: Any
        use_swa: Any
        _fit_common_bleed_multilabel_config_into: Any
        _fit_common_chrome_trace_export_mlframe: Any
        _fit_common_enable_tf32_float32_matmul: Any
        _fit_common_explicit_user_set_scale: Any
        _fit_common_has_validation: Any
        _fit_common_idempotent_re_seeding_before: Any
        _fit_common_reject_those_further_down: Any
        _fit_common_self_use_ema: Any
        _fit_common_unaffected_because_bf16_mixed: Any

    def _fit_step_dm_params(self, y: Any) -> "tuple[bool, dict]":
        """Detect a multilabel target and build the per-fit datamodule kwargs; returns ``(is_multilabel_target, dm_params)``."""
        from . import _PREDICT_ONLY_DM_PARAM_KEYS

        # Multilabel detection must precede datamodule construction: the per-fit
        # ``labels_dtype`` override (int64 -> float32 for BCEWithLogitsLoss) is
        # applied at datamodule time, so it MUST be decided before the dm is
        # built. The earlier code did the check after dm construction, which
        # silently fed int64 labels into a CE-loss model, producing the (3) vs
        # (65536) shape mismatch observed in fuzz combo c0030.
        _is_multilabel_target = False
        if isinstance(self, ClassifierMixin):
            _y_check = y.values if isinstance(y, pd.Series) else y
            _y_check = np.asarray(_y_check) if not isinstance(_y_check, np.ndarray) else _y_check
            _is_multilabel_target = bool(_y_check.ndim == 2 and _y_check.shape[1] >= 2)

        # ``predict_batch_size`` is a predict-time-only knob the suite plumbs
        # into ``datamodule_params`` (see _helpers_training_configs.py:733); it
        # is consumed at predict() directly off ``self.datamodule_params`` and
        # is NOT a constructor parameter of TorchDataModule / RecurrentDataModule.
        # Strip it (and any future predict-only keys) before splatting into the
        # datamodule constructor, otherwise the fit-time build raises
        # ``TorchDataModule.__init__() got an unexpected keyword argument
        # 'predict_batch_size'`` the moment a caller sets mlp_predict_batch_size.
        _local_dm_params = {k: v for k, v in self.datamodule_params.items() if k not in _PREDICT_ONLY_DM_PARAM_KEYS}
        if _is_multilabel_target:
            # BCEWithLogitsLoss requires float labels; CrossEntropyLoss (the
            # classifier default in helpers.py) requires Long. The estimator
            # owns the dispatch — datamodule just delivers the dtype the loss
            # expects.
            _local_dm_params["labels_dtype"] = torch.float32
        return _is_multilabel_target, _local_dm_params

    def _fit_step_num_classes(self, y: Any, is_partial_fit: bool, classes: Optional[np.ndarray], _is_multilabel_target: bool) -> int:
        """Set the classifier / regressor target-shape state on ``self`` and return the network ``num_classes``."""
        num_classes: Any = None
        if isinstance(self, ClassifierMixin):
            # Multilabel was already detected upstream (``_is_multilabel_target``)
            # so the datamodule could swap labels_dtype to float32 in time. The
            # K >= 2 lower bound matters: a single-column 1-D-ish 2-D target
            # (N, 1) is still SINGLE-LABEL classification (the upstream just
            # delivered it as a 1-column frame instead of a 1-D array).
            # Treating it as multilabel sets num_classes=1, so MLP gets
            # output_dim=1, predictions squeeze to (N,), labels also squeeze
            # to (N,), then CrossEntropyLoss interprets predictions.shape ==
            # labels.shape as the class-probabilities input mode and rejects
            # Long labels with ``Expected floating point type for target with
            # class probabilities, got Long``. Observed on S: in
            # fuzz_3way combo cb_lgb_mlp_xgb-pl_nullable-n1000 binary
            # classification.
            self._is_multilabel = _is_multilabel_target

            if self._is_multilabel:
                _y_check = y.values if isinstance(y, pd.Series) else y
                _y_check = np.asarray(_y_check) if not isinstance(_y_check, np.ndarray) else _y_check
                self.n_labels_ = int(_y_check.shape[1])
                self.classes_ = None  # sentinel; predict_proba returns per-label sigmoid probs
                num_classes = self.n_labels_
            else:
                if is_partial_fit and classes is not None:
                    # self.classes_ was already correctly set above (line ~209) from
                    # self._label_encoder.classes_ -- guaranteed SORTED, matching the index space
                    # _label_encoder.transform()/inverse_transform() use. Re-assigning it here to the
                    # caller's raw (possibly-unsorted) `classes` array would desync self.classes_[i] from
                    # predict_proba's column i.
                    pass
                elif not hasattr(self, "classes_"):
                    # Must be ndarray (not list) for numpy fancy indexing in evaluation.py::report_probabilistic_model_perf
                    # (line ``preds = model.classes_[preds]`` fails on list + ndarray index). Sklearn convention is classes_ ndarray.
                    _y_arr = y.unique() if isinstance(y, pd.Series) else np.unique(y)
                    # object-dtype y (mixed-type label set
                    # incl. None / np.nan + str) would TypeError on Python sorted();
                    # use np.sort for ndarrays and str-key fallback for object dtype.
                    if hasattr(_y_arr, "dtype") and _y_arr.dtype != object:
                        self.classes_ = np.sort(_y_arr)
                    else:
                        self.classes_ = np.asarray(sorted(_y_arr, key=lambda v: (v is None, str(v))))
                num_classes = len(self.classes_)
        else:
            # F-24: native multi-target regression. When y has
            # shape (N, K>=2) for a regressor (not multilabel), train K
            # output heads sharing the trunk. MSE between (N, K) preds and
            # (N, K) labels works without any loss-shape gymnastics.
            # Single-target (N,) or (N, 1) y keeps num_classes=1.
            _y_check_reg = y.values if isinstance(y, pd.Series) else y
            _y_check_reg = np.asarray(_y_check_reg) if not isinstance(_y_check_reg, np.ndarray) else _y_check_reg
            if _y_check_reg.ndim == 2 and _y_check_reg.shape[1] >= 2:
                num_classes = int(_y_check_reg.shape[1])
                self._is_multi_target_regression = True
            else:
                num_classes = 1
                self._is_multi_target_regression = False
            self._is_multilabel = False
        return int(num_classes)

    def _fit_step_network_and_metrics(self, X: Any, y: Any, num_classes: int, is_partial_fit: bool) -> "tuple[list[MetricSpec], str]":
        """Reset / build the network (fit resets, partial_fit continues) and return ``(metrics, metric_name)``."""
        from ..flat import generate_mlp

        # F-05: binary uses 1-output sigmoid + BCE instead of
        # 2-output softmax + CE -- see the matching block above the dm
        # construction. ``_binary_sigmoid_head`` flag was set there; here
        # we just resolve the network output dim for the network reset
        # below.
        _network_output_dim = 1 if self._binary_sigmoid_head else num_classes

        # Reset network on fit() to match sklearn convention (fit resets, partial_fit continues). Each fit() call must create a
        # fresh network with correct input dimensions; critical when feature counts change between training iterations.
        if not is_partial_fit:
            self.network = None
            self.model = None  # also reset the LightningModule wrapper

        # Compute output_activation_scale / center from the y the MLP sees
        # at fit-time (Fix 1). When the wrapping TTR z-scores y,
        # the MLP sees scaled y and the tanh window lives in scaled space;
        # TTR.inverse_transform unwinds it correctly. Only applied for
        # regression (num_classes==1) with output_activation set; left
        # untouched for classification and the linear default.
        _net_params = dict(self.network_params)
        _out_act = _net_params.get("output_activation", "linear")
        # condition uses OR (any None) so a partially-set
        # ``scale OR center`` triggers the auto-fill instead of falling
        # through to ``generate_mlp`` which raises on either one being
        # None. Pre-fix the AND condition would skip the derivation for
        # the (scale=2.0, center=None) shape, then ``generate_mlp`` at
        # the scale/center check in flat.generate_mlp would error on the missing center. The auto-fill
        # body below only overwrites the missing field via ``setdefault``
        # so an explicit user-set scale or center is preserved.
        _scale_set = _net_params.get("output_activation_scale") is not None
        _center_set = _net_params.get("output_activation_center") is not None
        self._fit_common_explicit_user_set_scale(_out_act, num_classes, _scale_set, _center_set, y, _net_params)

        # Thread the fit-time categorical cardinalities into the network params so ``generate_mlp`` prepends a ``CategoricalEmbedding`` whose
        # tables match the factorizer's per-cat code counts. The first ``_n_cat_features_`` columns of X are the (reordered-leading) cat codes;
        # the rest are numeric. No-op when no cats were factorized (``_cat_cardinalities_`` is None).
        _cat_cards = getattr(self, "_cat_cardinalities_", None)
        if _cat_cards:
            _net_params["categorical_cardinalities"] = list(_cat_cards)
            _net_params.setdefault("categorical_embed_dim", getattr(self, "categorical_embed_dim", None))

        # getattr handles freshly cloned models that don't have network attribute yet
        if getattr(self, "network", None) is None:
            self.network = generate_mlp(num_features=X.shape[1], num_classes=_network_output_dim, **_net_params)

        if num_classes > 1:
            metric_name = "ICE"
            metrics = [MetricSpec(name=metric_name, fcn=compute_probabilistic_multiclass_error, requires_probs=True)]
        else:
            # F-02 (mlp audit): the metric function is sklearn's
            # ``root_mean_squared_error`` (RMSE), so the label MUST be "RMSE"
            # too. Pre-fix the label was "MSE" -- monitor keys ("val_MSE"),
            # checkpoint filenames (``model-val_MSE=0.7555.ckpt``), and
            # CSV-logger columns all carried the wrong scale label. The
            # metric_direction_dispatcher / metric_name_higher_is_better
            # registry already knew both keys as min-direction, so the
            # rename does not break direction-dependent code paths.
            metric_name = "RMSE"
            metrics = [MetricSpec(name=metric_name, fcn=_rmse_metric)]
        return metrics, metric_name

    def _fit_step_checkpointing(self, monitor_metric: str) -> "tuple[BestEpochModelCheckpoint, str]":
        """Resolve the per-fit checkpoint root and build the best-epoch checkpoint callback; returns ``(checkpointing, ckpt_root)``."""
        # Nest checkpoints + lightning_logs under a unique per-fit subdir so concurrent / sequential fits don't dump into a shared
        # project-root ``logs/`` folder and resolve different runs only by the (unsafe) ``model-val_MSE=0.7555.ckpt`` filename
        # collision via Lightning's version counter.
        #
        # Path resolution (in order of preference):
        #   1. ``self.checkpoint_dir_override`` - public attribute the suite sets to a target-nested path (eg
        #      ``data/models/{target}/{exp}/regression/{tgt}/{model_file_basename}/``). Honoured verbatim.
        #   2. Auto-derived ``{default_root_dir}/_run_{id(self)}_{ts}`` - unique sub-dir under the root; fully isolates concurrent
        #      runs even when no caller plumbing.
        # ``CSVLogger`` save_dir resolved the same way - mirror nesting so the on-disk layout stays uniform per fit.
        _ckpt_root = getattr(self, "checkpoint_dir_override", None)
        if _ckpt_root is None:
            import time as _time
            # trainer_params["default_root_dir"] is caller-controlled
            # per the standard Lightning Trainer contract. Caller is responsible for any
            # trusted-root validation upstream; this join is intentionally permissive and
            # matches Lightning's documented behaviour for default_root_dir.
            _cfg_root = self.trainer_params.get("default_root_dir")
            _default_root = "logs" if _cfg_root is None or _cfg_root == "" else str(_cfg_root)
            _ckpt_root = os.path.join(_default_root, f"_run_{id(self)}_{int(_time.time())}")
        os.makedirs(_ckpt_root, exist_ok=True)

        checkpointing = BestEpochModelCheckpoint(
            monitor=monitor_metric,
            dirpath=_ckpt_root,
            # Filename no longer needs the ``model-`` prefix - the enclosing dir already identifies the model uniquely.
            filename=f"{{{monitor_metric}:.4f}}",
            enable_version_counter=True,
            save_last=False,
            save_top_k=1,
            mode="min",
            # F-25 (cProfile finding): checkpoint writes were
            # 9.59s out of 15.6s total fit wall (61%) on a 10k x 50 / 10-epoch
            # baseline. Lightning's default ModelCheckpoint includes the
            # optimizer state + LR scheduler state + RNG state in every
            # snapshot -- but on_train_end only reads checkpoint["state_dict"]
            # (see _flat_torch_module.py:530-533), so the optimizer / scheduler
            # / RNG bytes are written then discarded at load time. Switching
            # to save_weights_only=True drops them at write time: ~6x smaller
            # snapshot, ~6x faster per-write. Net fit-wall speedup is
            # proportional to checkpoint-write share -- larger networks +
            # longer fits see the most benefit.
            save_weights_only=True,
        )
        return checkpointing, _ckpt_root

    def _fit_step_trainer_params(self, has_validation: bool, _ckpt_root: str, safe_accelerator: Callable[[Any], Any]) -> "tuple[dict, Any]":
        """Build the Lightning trainer kwargs (accelerator probe via ``safe_accelerator``, precision, logger, profiler); returns ``(trainer_params, requested_accelerator)``."""
        trainer_params = self.trainer_params.copy()
        if not has_validation:
            logger.info("No validation data - training without validation")
            trainer_params.update({"num_sanity_val_steps": 0, "limit_val_batches": 0})

        # CUDA-broken-host guard: when the caller leaves the accelerator at
        # ``auto`` (or asks for ``cuda``/``gpu`` outright), probe a 1-element
        # allocation BEFORE Lightning builds the strategy. On hosts with CUDA
        # libs but a broken driver / no device / a context the calling proc
        # can't open, ``Trainer`` would otherwise die deep inside
        # ``model_to_device`` with ``CUDA error: an illegal memory access``;
        # the probe lets us fall back to CPU cleanly so the fit completes.
        # When the operator explicitly forces ``accelerator='cuda'`` and CUDA
        # is unusable, surface that as a log warning + still downgrade
        # (silently failing the fit on a 100-call suite is worse than
        # ignoring a single forced flag).
        _requested = trainer_params.get("accelerator", "auto")
        _resolved = safe_accelerator(_requested)
        if _resolved != _requested and _requested in ("cuda", "gpu"):
            logger.warning(
                "Requested accelerator=%r but CUDA probe failed; " "downgrading to CPU so fit can complete.",
                _requested,
            )
        trainer_params["accelerator"] = _resolved

        # F-27: auto-enable bf16-mixed precision on Ampere+
        # GPUs. bf16 has the same dynamic range as fp32 (no GradScaler,
        # no NaN risk -- unlike '16-mixed' / fp16). Measured 1.2-1.8x
        # forward+backward speedup on Ampere+ for GEMM-bound workloads,
        # ~30-40% activation-memory reduction.
        #
        # Gating:
        #   * Only when caller didn't set ``precision`` in trainer_params
        #     (explicit > default).
        #   * Only when resolved accelerator is cuda/gpu (CPU bf16 is
        #     slow / unsupported).
        #   * Only when the device's compute capability is >= 8 (Ampere
        #     A100, RTX 30/40 series, H100, etc.). Pre-Ampere bf16 falls
        #     back to fp32 with no speedup but adds autocast overhead.
        # The predict path already accepts precision (base.py:840-957)
        # so inference parity is automatic; fp32 checkpoint load is
        # unaffected because bf16-mixed stores fp32 master weights.
        self._fit_common_unaffected_because_bf16_mixed(trainer_params, _resolved)

        # Default logger for LearningRateMonitor compatibility. CSV logs land in the SAME per-fit subdir as the checkpoint so the
        # entire run's artifacts (ckpt + metrics + LR-monitor csvs) are co-located under one path; trivially diffable / archivable.
        if "logger" not in trainer_params:
            trainer_params["logger"] = CSVLogger(save_dir=_ckpt_root, name="")

        # F-36: opt-in torch.profiler integration via
        # MLFRAME_TORCH_PROFILE=1. Per the PyTorch optimization
        # audit (Agent B profiler research), shallow tabular MLPs are
        # typically kernel-launch-bound rather than compute-bound — the
        # 20-40% wall typically spent in inter-kernel gaps is invisible
        # to cProfile (a pure CPU profiler) but immediately visible in
        # torch.profiler's CUDA trace. Lightning's PyTorchProfiler wraps
        # torch.profiler with per-hook record_function ranges already
        # present in the LightningModule call graph, so the trace shows
        # training_step / backward / optimizer_step bounds for free.
        # Defaults: 5-step rolling window (wait=1, warmup=1, active=3) +
        # Chrome trace export to MLFRAME_TORCH_PROFILE_DIR (or ./torch_traces).
        self._fit_common_chrome_trace_export_mlframe(trainer_params, _ckpt_root)
        return trainer_params, _requested

    def _fit_step_callbacks(self, checkpointing: BestEpochModelCheckpoint, trainer_params: dict, has_validation: bool, metric_name: str) -> list[Callback]:
        """Assemble the Lightning callback list for the fit."""
        callbacks: list[Callback] = [checkpointing]
        # Lightning raises ``MisconfigurationException`` when both
        # ``enable_progress_bar=False`` is in trainer_params AND a
        # ``TQDMProgressBar`` is registered in callbacks. Only attach the
        # progress-bar callback when the caller hasn't explicitly disabled it.
        if trainer_params.get("enable_progress_bar", True):
            callbacks.append(TQDMProgressBar(refresh_rate=10))

        # Only add LearningRateMonitor if logger is enabled
        if trainer_params.get("logger") is not False:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        if self.use_swa and self.use_ema:
            raise ValueError(
                "use_swa and use_ema are mutually exclusive — both rewrite "
                "the live model weights at train end (last-write-wins). "
                "Pick one: SWA (broad LR cycle averaging) or EMA "
                "(per-step exponential moving average)."
            )
        if self.use_swa:
            swa_params = self.swa_params or {}
            callbacks.append(StochasticWeightAveraging(**swa_params))
        self._fit_common_self_use_ema(callbacks)

        self._fit_common_has_validation(has_validation, callbacks, metric_name)
        return callbacks

    def _fit_step_seed_and_validate(self, X: Any, y: Any, eval_set: Any) -> None:
        """Seed the RNGs, reject NaN / inf inputs and enable TF32 matmul before any random op fires."""
        # F-06: sklearn-canonical reproducibility seed. When
        # ``random_state`` is an int, seed torch + numpy + Python random +
        # the Lightning DataLoader worker seed BEFORE any random op fires
        # (network init at line 357, dataloader shuffle, dropout mask
        # sampling). Same data + same random_state -> bit-identical
        # predictions. ``None`` leaves the prior non-deterministic
        # behaviour intact; callers managing their own seed are not
        # overridden. partial_fit honours the same seed on every batch
        # (idempotent — re-seeding before each call is fine).
        self._fit_common_idempotent_re_seeding_before()

        # F-23: reject NaN / inf in features or labels at fit
        # entry. Pre-fix any NaN propagated through the first Linear ->
        # all-NaN activations -> all-NaN gradients -> all-NaN weights after
        # one step -> all-NaN predictions; the suite saw a flat val curve
        # with no log signal. Now: explicit ValueError with a remediation
        # hint. Skip the check on string / object dtypes (LabelEncoder will
        # reject those further down with its own clear error).
        _validate_no_nan_inf("X", X)
        _validate_no_nan_inf("y", y, allow_object_dtype=True)
        self._fit_common_reject_those_further_down(eval_set)

        # Enable TF32 for float32 matmul if on GPU.
        self._fit_common_enable_tf32_float32_matmul()

    def _fit_step_normalize_eval_set(self, eval_set: Any) -> Any:
        """Collapse the list-of-tuples eval_set convention to the bare ``(X_val, y_val)`` tuple."""
        # Accept both eval_set conventions:
        #   - bare 2-tuple ``(X_val, y_val)`` (this estimator's native form)
        #   - list-of-tuples ``[(X_val, y_val), ...]`` (LightGBM / XGBoost form,
        #     which ``_maybe_pass_sample_weight`` in composite_ensemble.py emits
        #     uniformly so the same fit-call works across boosters and MLP).
        # Without this normalisation, the OOF refit path indexes ``eval_set[1]``
        # below and raises IndexError on the 1-element list -> MLP component
        # silently dropped from CT_ENSEMBLE for every target (observed in prod).
        if isinstance(eval_set, list) and eval_set and isinstance(eval_set[0], tuple):
            eval_set = eval_set[0]
        return eval_set

    def _fit_step_build_model(self, trainer: Any, metrics: list[MetricSpec], X: Any) -> None:
        """Instantiate the LightningModule on the trainer's init context and attach its ``example_input_array``."""
        # Per-fit model_params override for multilabel: swap CE loss -> BCE,
        # tag task_type so predict_step uses sigmoid not softmax. We DON'T
        # mutate self.model_params (would break sklearn clone + introspection
        # and bleed multilabel config into subsequent fits on different y).
        _local_model_params = dict(self.model_params)
        self._fit_common_bleed_multilabel_config_into(_local_model_params)

        with trainer.init_module():
            self.model = self.model_class(network=self.network, metrics=metrics, **_local_model_params)

            features_dtype = self.datamodule_params.get("features_dtype", torch.float32)
            data_slice = X.iloc[0:2, :].values if isinstance(X, pd.DataFrame) else X[0:2, :]

            try:
                _fit_model: Any = self.model
                _fit_model.example_input_array = to_tensor_any(data_slice, dtype=features_dtype, safe=True)
            except Exception:
                logger.warning("Failed to prepare example_input_array", exc_info=True)
