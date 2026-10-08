"""Fit mixin carved out of ``neural.base``.

``_FitMixin`` holds the single cohesive training run (``_fit_common``) plus
the ``fit`` / ``partial_fit`` wrappers. It operates purely on ``self`` so the
estimator mixes it in unchanged. Live trainer/accelerator objects set here are
dropped on pickle by the estimator's ``__getstate__`` (see ``base``).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from sklearn.base import ClassifierMixin

from mlframe.metrics.core import compute_probabilistic_multiclass_error
from .._base_logging import MetricSpec, _rmse_metric, suppress_lightning_workers_warning
from .._base_tensor_helpers import to_tensor_any, safe_accelerator
from ._base_losses import _validate_no_nan_inf
from .._base_callbacks import BestEpochModelCheckpoint
from ._base_fit_prep import _FitPrepMixin

logger = __import__("logging").getLogger("mlframe.training.neural.base")


from ._base_fit_helpers import _FitCommonHelpersMixin


class _FitMixin(_FitCommonHelpersMixin, _FitPrepMixin):
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
        # Lazy imports to avoid circular dependency (the parent imports this
        # mixin at class-definition time).
        num_classes: Any = None
        metric_name: Any = None
        from ..flat import generate_mlp
        from . import _PREDICT_ONLY_DM_PARAM_KEYS

        if fit_params is None:
            fit_params = {}

        # Make embedding-vector + free-text columns numeric BEFORE validation + input-dim computation (the MLP has no
        # native embedding/text layers). Stashes the fitted encoder on self for predict(). No-op when none are named.
        X, eval_set = self._encode_emb_text_fit(X, eval_set, fit_params)

        # Factorize raw categorical columns to integer codes (reordered leading) BEFORE validation, so the learnable ``CategoricalEmbedding``
        # can index them and ``_validate_no_nan_inf`` sees a pure-numeric frame. No-op when no ``cat_features`` are named or the knob is off.
        X, eval_set = self._factorize_cats_fit(X, eval_set, fit_params, is_partial_fit=is_partial_fit)

        # F-06 (2026-05-30): sklearn-canonical reproducibility seed. When
        # ``random_state`` is an int, seed torch + numpy + Python random +
        # the Lightning DataLoader worker seed BEFORE any random op fires
        # (network init at line 357, dataloader shuffle, dropout mask
        # sampling). Same data + same random_state -> bit-identical
        # predictions. ``None`` leaves the prior non-deterministic
        # behaviour intact; callers managing their own seed are not
        # overridden. partial_fit honours the same seed on every batch
        # (idempotent — re-seeding before each call is fine).
        self._fit_common_idempotent_re_seeding_before()

        # F-23 (2026-05-30): reject NaN / inf in features or labels at fit()
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
        has_validation = eval_set[0] is not None

        eval_sample_weight = fit_params.get("eval_sample_weight")

        # Multilabel detection must precede datamodule construction: the per-fit
        # ``labels_dtype`` override (int64 -> float32 for BCEWithLogitsLoss) is
        # applied at datamodule time, so it MUST be decided before the dm is
        # built. The earlier code did the check after dm construction, which
        # silently fed int64 labels into a CE-loss model, producing the (3) vs
        # (65536) shape mismatch observed in fuzz combo c0030 (2026-05-20).
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

        # Single-label classifier label encoding. sklearn convention is that
        # ``y`` can be any hashable (strings, non-dense ints, booleans);
        # CrossEntropyLoss + ``labels_dtype=int64`` require ``{0..K-1}``
        # integer indices. Without this encoding, ``fit`` crashed with
        # ``IndexError: Target N is out of bounds`` for any y whose value set
        # is not exactly ``{0..K-1}`` (e.g. ``{10, 20}`` or ``{"low","high"}``;
        # F-19 in the 2026-05-30 mlp audit). Build the bidirectional encoder
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
            # class probabilities, got Long``. Observed 2026-05-20 on S: in
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
                    # Wave 61 (2026-05-20): object-dtype y (mixed-type label set
                    # incl. None / np.nan + str) would TypeError on Python sorted();
                    # use np.sort for ndarrays and str-key fallback for object dtype.
                    if hasattr(_y_arr, "dtype") and _y_arr.dtype != object:
                        self.classes_ = np.sort(_y_arr)
                    else:
                        self.classes_ = np.asarray(sorted(_y_arr, key=lambda v: (v is None, str(v))))
                num_classes = len(self.classes_)
        else:
            # F-24 (2026-05-31): native multi-target regression. When y has
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

        # F-05 (2026-05-30): binary uses 1-output sigmoid + BCE instead of
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
        # at fit-time (Fix 1, 2026-05-26). When the wrapping TTR z-scores y,
        # the MLP sees scaled y and the tanh window lives in scaled space;
        # TTR.inverse_transform unwinds it correctly. Only applied for
        # regression (num_classes==1) with output_activation set; left
        # untouched for classification and the linear default.
        _net_params = dict(self.network_params)
        _out_act = _net_params.get("output_activation", "linear")
        # 2026-06-01: condition uses OR (any None) so a partially-set
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
            # F-02 (2026-05-30 mlp audit): the metric function is sklearn's
            # ``root_mean_squared_error`` (RMSE), so the label MUST be "RMSE"
            # too. Pre-fix the label was "MSE" -- monitor keys ("val_MSE"),
            # checkpoint filenames (``model-val_MSE=0.7555.ckpt``), and
            # CSV-logger columns all carried the wrong scale label. The
            # metric_direction_dispatcher / metric_name_higher_is_better
            # registry already knew both keys as min-direction, so the
            # rename does not break direction-dependent code paths.
            metric_name = "RMSE"
            metrics = [MetricSpec(name=metric_name, fcn=_rmse_metric)]

        # When no validation data, monitor train_loss instead of train metrics (which may not be logged)
        if has_validation:
            monitor_metric = f"val_{metric_name}"
        else:
            monitor_metric = "train_loss"

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
            # Wave 46 (2026-05-20): trainer_params["default_root_dir"] is caller-controlled
            # per the standard Lightning Trainer contract. Caller is responsible for any
            # trusted-root validation upstream; this join is intentionally permissive and
            # matches Lightning's documented behaviour for default_root_dir.
            _default_root = self.trainer_params.get("default_root_dir") or "logs"
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
            # F-25 (2026-05-31 cProfile finding): checkpoint writes were
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

        # F-27 (2026-05-31): auto-enable bf16-mixed precision on Ampere+
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

        # F-36 (2026-05-31): opt-in torch.profiler integration via
        # MLFRAME_TORCH_PROFILE=1. Per the 2026-05-31 PyTorch optimization
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

        trainer = L.Trainer(**trainer_params, callbacks=callbacks)

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
                self.model.example_input_array = to_tensor_any(data_slice, dtype=features_dtype, safe=True)
            except Exception:
                logger.warning("Failed to prepare example_input_array", exc_info=True)

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
