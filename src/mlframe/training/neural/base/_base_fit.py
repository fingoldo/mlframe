"""Fit mixin carved out of ``neural.base``.

``_FitMixin`` holds the single cohesive training run (``_fit_common``) plus
the ``fit`` / ``partial_fit`` wrappers. It operates purely on ``self`` so the
estimator mixes it in unchanged. Live trainer/accelerator objects set here are
dropped on pickle by the estimator's ``__getstate__`` (see ``base``).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping as EarlyStoppingCallback
from lightning.pytorch.loggers import CSVLogger
from sklearn.base import ClassifierMixin, RegressorMixin

from mlframe.metrics.core import compute_probabilistic_multiclass_error
from .._base_logging import MetricSpec, _rmse_metric
from .._base_tensor_helpers import to_tensor_any, safe_accelerator
from ._base_losses import _make_binary_focal_loss, _validate_no_nan_inf
from .._base_callbacks import BestEpochModelCheckpoint, ValLossDivergenceCallback, MonotonicDeclineStopCallback
from .._history_recorder import TrainingHistoryRecorder
from ._base_fit_prep import _FitPrepMixin

logger = __import__("logging").getLogger("mlframe.training.neural.base")


class _FitMixin(_FitPrepMixin):
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
        fit_params: dict = None,
        sample_weight=None,
    ):
        """Common logic for fit and partial_fit."""
        # Lazy imports to avoid circular dependency (the parent imports this
        # mixin at class-definition time).
        from ..flat import generate_mlp
        from . import _PREDICT_ONLY_DM_PARAM_KEYS

        if fit_params is None:
            fit_params = {}

        # Make embedding-vector + free-text columns numeric BEFORE validation + input-dim computation (the MLP has no
        # native embedding/text layers). Stashes the fitted encoder on self for predict(). No-op when none are named.
        X, eval_set = self._encode_emb_text_fit(X, eval_set, fit_params)

        # Factorize raw categorical columns to integer codes (reordered leading) BEFORE validation, so the learnable ``CategoricalEmbedding``
        # can index them and ``_validate_no_nan_inf`` sees a pure-numeric frame. No-op when no ``cat_features`` are named or the knob is off.
        X, eval_set = self._factorize_cats_fit(X, eval_set, fit_params)

        # F-06 (2026-05-30): sklearn-canonical reproducibility seed. When
        # ``random_state`` is an int, seed torch + numpy + Python random +
        # the Lightning DataLoader worker seed BEFORE any random op fires
        # (network init at line 357, dataloader shuffle, dropout mask
        # sampling). Same data + same random_state -> bit-identical
        # predictions. ``None`` leaves the prior non-deterministic
        # behaviour intact; callers managing their own seed are not
        # overridden. partial_fit honours the same seed on every batch
        # (idempotent — re-seeding before each call is fine).
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

        # F-23 (2026-05-30): reject NaN / inf in features or labels at fit()
        # entry. Pre-fix any NaN propagated through the first Linear ->
        # all-NaN activations -> all-NaN gradients -> all-NaN weights after
        # one step -> all-NaN predictions; the suite saw a flat val curve
        # with no log signal. Now: explicit ValueError with a remediation
        # hint. Skip the check on string / object dtypes (LabelEncoder will
        # reject those further down with its own clear error).
        _validate_no_nan_inf("X", X)
        _validate_no_nan_inf("y", y, allow_object_dtype=True)
        if eval_set is not None and not (isinstance(eval_set, tuple) and eval_set[0] is None):
            # eval_set may be a 2-tuple ``(X_val, y_val)`` or a list-of-tuples
            # (LightGBM convention) -- normalise to peek at the val frame.
            _ev = eval_set[0] if isinstance(eval_set, list) and eval_set else eval_set
            if isinstance(_ev, tuple) and _ev[0] is not None:
                _validate_no_nan_inf("X_val", _ev[0])
                _validate_no_nan_inf("y_val", _ev[1], allow_object_dtype=True)

        # Enable TF32 for float32 matmul if on GPU.
        if self.float32_matmul_precision and torch.cuda.is_available():
            _allowed_matmul = ("highest", "high", "medium")
            if self.float32_matmul_precision not in _allowed_matmul:
                raise ValueError(
                    f"float32_matmul_precision must be one of {_allowed_matmul}, "
                    f"got {self.float32_matmul_precision!r}"
                )
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(self.float32_matmul_precision)
                logger.info("Enabled float32_matmul_precision=%s", self.float32_matmul_precision)

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
        _local_dm_params = {
            k: v for k, v in self.datamodule_params.items()
            if k not in _PREDICT_ONLY_DM_PARAM_KEYS
        }
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
        _classifier_single_label = (
            isinstance(self, ClassifierMixin) and not _is_multilabel_target
        )
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
                _y_arr_val = (
                    eval_set[1].values if isinstance(eval_set[1], pd.Series)
                    else np.asarray(eval_set[1])
                )
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
                            f"class_weight-derived weights shape "
                            f"{_cw_weights.shape} != sample_weight shape "
                            f"{_sw_arr.shape}; cannot multiply."
                        )
                    sample_weight = _sw_arr * _cw_weights
                logger.info(
                    "Applied class_weight=%r -> per-sample weights "
                    "with mean=%.4g, min=%.4g, max=%.4g",
                    self.class_weight,
                    float(np.mean(sample_weight)),
                    float(np.min(sample_weight)),
                    float(np.max(sample_weight)),
                )

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
                    self.classes_ = np.asarray(classes)
                elif not hasattr(self, "classes_"):
                    # Must be ndarray (not list) for numpy fancy indexing in evaluation.py::report_probabilistic_model_perf
                    # (line ``preds = model.classes_[preds]`` fails on list + ndarray index). Sklearn convention is classes_ ndarray.
                    _y_arr = (y.unique() if isinstance(y, pd.Series) else np.unique(y))
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
        # flat.py:537 would error on the missing center. The auto-fill
        # body below only overwrites the missing field via ``setdefault``
        # so an explicit user-set scale or center is preserved.
        _scale_set = _net_params.get("output_activation_scale") is not None
        _center_set = _net_params.get("output_activation_center") is not None
        if (
            _out_act == "tanh_train_range"
            and num_classes == 1
            and not getattr(self, "_is_multi_target_regression", False)
            and not (_scale_set and _center_set)
        ):
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
                from .._neural_numba_kernels import finite_min_max_std as _fmms
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
                        "MLP output_activation='tanh_train_range' requested "
                        "but y_train has <=1 finite value; falling back to "
                        "'linear' for this fit.",
                    )
                    _net_params["output_activation"] = "linear"
            except Exception as _oa_err:
                logger.warning(
                    "MLP output_activation='tanh_train_range' y_train "
                    "derivation failed (%s); falling back to 'linear'.",
                    _oa_err,
                )
                _net_params["output_activation"] = "linear"

        # Thread the fit-time categorical cardinalities into the network params so ``generate_mlp`` prepends a ``CategoricalEmbedding`` whose
        # tables match the factorizer's per-cat code counts. The first ``_n_cat_features_`` columns of X are the (reordered-leading) cat codes;
        # the rest are numeric. No-op when no cats were factorized (``_cat_cardinalities_`` is None).
        _cat_cards = getattr(self, "_cat_cardinalities_", None)
        if _cat_cards:
            _net_params["categorical_cardinalities"] = list(_cat_cards)
            _net_params.setdefault("categorical_embed_dim", getattr(self, "categorical_embed_dim", None))

        # getattr handles freshly cloned models that don't have network attribute yet
        if getattr(self, 'network', None) is None:
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
                "Requested accelerator=%r but CUDA probe failed; "
                "downgrading to CPU so fit can complete.",
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
                    "F-27 bf16 auto-enable probe failed (%s); leaving "
                    "precision at Lightning default.", _cc_err,
                )

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
                        "F-36: MLFRAME_TORCH_PROFILE=1 active; chrome traces "
                        "land in %s. Open via chrome://tracing or Perfetto.",
                        _prof_dir,
                    )
                except Exception as _prof_err:
                    logger.warning(
                        "MLFRAME_TORCH_PROFILE=1 but profiler setup failed "
                        "(%s); fit continues without profiling.",
                        _prof_err,
                    )

        callbacks = [checkpointing]
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
                from lightning.pytorch.callbacks import WeightAveraging  # noqa: F401
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
            callbacks.append(
                TrainingHistoryRecorder(monitor=f"val_{metric_name}", mode="min")
            )
            # Per-epoch full-metric-suite capture for meta-learning / HPO-from-early-observation. Default-ON for
            # neural: val predictions are already concatenated each validation epoch, so the only marginal cost is
            # the cheap metric kernel. ``capture_iteration_metrics=False`` on the estimator opts out.
            _cap_iter = getattr(self, "capture_iteration_metrics", None)
            if _cap_iter is None:
                _cap_iter = True  # neural family default
            if _cap_iter:
                from .._history_recorder import IterationMetricsRecorder
                if self._is_multilabel:
                    _tt, _ncls = "multilabel_classification", None
                elif isinstance(self, ClassifierMixin):
                    _classes = getattr(self, "classes_", None)
                    _ncls = int(len(_classes)) if _classes is not None else 2
                    _tt = "binary_classification" if _ncls <= 2 else "multiclass_classification"
                else:
                    _tt, _ncls = "regression", None
                callbacks.append(IterationMetricsRecorder(target_type=_tt, n_classes=_ncls or None))

        trainer = L.Trainer(**trainer_params, callbacks=callbacks)

        # Per-fit model_params override for multilabel: swap CE loss -> BCE,
        # tag task_type so predict_step uses sigmoid not softmax. We DON'T
        # mutate self.model_params (would break sklearn clone + introspection
        # and bleed multilabel config into subsequent fits on different y).
        _local_model_params = dict(self.model_params)
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
        elif (
            isinstance(self, ClassifierMixin)
            and not self._is_multilabel
            and not self._binary_sigmoid_head
            and self.label_smoothing > 0.0
        ):
            # F-30 (2026-05-31): label smoothing for MULTICLASS only.
            # Replaces the caller's CrossEntropyLoss with one carrying
            # label_smoothing=epsilon. Per RealMLP-TD NeurIPS 2024:
            # +1.8% multiclass accuracy on the ablation. Skipped for
            # binary (Cattan 2024 shows calibration regression on
            # imbalanced binary; focal_loss_gamma is the analogue knob).
            _local_model_params["loss_fn"] = torch.nn.CrossEntropyLoss(
                label_smoothing=float(self.label_smoothing),
            )

        with trainer.init_module():
            self.model = self.model_class(network=self.network, metrics=metrics, **_local_model_params)

            features_dtype = self.datamodule_params.get("features_dtype", torch.float32)
            data_slice = X.iloc[0:2, :].values if isinstance(X, pd.DataFrame) else X[0:2, :]

            try:
                self.model.example_input_array = to_tensor_any(data_slice, dtype=features_dtype, safe=True)
            except Exception:
                logger.warning("Failed to prepare example_input_array", exc_info=True)

        if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned")):
            tuner = Tuner(trainer)

            if self.tune_batch_size:
                tuner.scale_batch_size(model=self.model, datamodule=dm, mode="binsearch", init_val=self.datamodule_params.get("batch_size", 32))

            lr_finder = tuner.lr_find(self.model, datamodule=dm, num_training=300)
            new_lr = lr_finder.suggestion()
            logger.info("Using suggested LR=%s", new_lr)
            self.model.hparams.learning_rate = new_lr

            if is_partial_fit:
                self._tuned = True

        trainer.fit(model=self.model, datamodule=dm)

        # Expose per-epoch train/val history (booster ``evals_result_`` shape) + the best epoch so the
        # reporting layer's training-curve chart picks it up with no neural-specific code (it already
        # consumes ``evals_result_``/``best_iteration_`` for lgb/xgb/cb).
        for callback in trainer.callbacks:
            if isinstance(callback, TrainingHistoryRecorder):
                if callback.evals_result_:
                    self.evals_result_ = callback.evals_result_
                    if callback.best_iteration_ is not None:
                        self.best_iteration_ = callback.best_iteration_
                break
        from .._history_recorder import IterationMetricsRecorder
        for callback in trainer.callbacks:
            if isinstance(callback, IterationMetricsRecorder):
                if callback.iteration_metrics_:
                    self.iteration_metrics_ = callback.iteration_metrics_
                break

        # Extract best epoch from model (set by checkpoint callback, DDP-safe). Prefer model.best_epoch over callback.best_epoch
        # for distributed training compatibility.
        if hasattr(self.model, "best_epoch") and self.model.best_epoch is not None:
            self.best_epoch = self.model.best_epoch
            logger.info("Best epoch recorded: %s", self.best_epoch)
        else:
            # Fallback to callback for backward compatibility
            for callback in trainer.callbacks:
                if isinstance(callback, BestEpochModelCheckpoint):
                    self.best_epoch = callback.best_epoch
                    if self.best_epoch is not None:
                        logger.info("Best epoch recorded from callback: %s", self.best_epoch)
                    break

        # Clean up to avoid pickle issues and free memory
        self.trainer = None

        # Free the train/val tensors held by the cached datamodule
        # WITHOUT dropping the datamodule shell itself. The full
        # train+val feature / label / sample_weight tensors were the
        # actual save() bloat (1788 MB on disk for a 4M x 323 float32
        # frame, 2026-05-27 TVT regression log) -- the shell (~few KB
        # of config + class refs) is fine to pickle. Keeping the shell
        # lets predict() reuse the configured pre-pipeline /
        # batch_size / dataloader_params without rebuilding the
        # datamodule from scratch, AND silences the spurious "No
        # datamodule found from training" WARNING that fired on every
        # predict-after-fit when we used to NULL the whole reference.
        # Opt out via env MLFRAME_KEEP_PREDICTION_DATAMODULE=1 for
        # operators relying on the prior whole-stash behaviour.
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
                for _attr in ("_train_dataset", "_val_dataset",
                              "train_dataset", "val_dataset"):
                    if hasattr(_dm, _attr):
                        setattr(_dm, _attr, None)
            self._datamodule_tensors_dropped = True

        return self

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to the data.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional per-sample weights for training
            **fit_params: Additional parameters including:
                - eval_set: Tuple of (X_val, y_val) for validation
                - eval_sample_weight: Optional validation sample weights
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
