"""Model factory functions extracted from ``trainer.py``.

LightGBM monkey-patches, XGB/LGB classifier/regressor factories,
FLAML zero-shot integration, and neural-component (PyTorch Lightning)
lazy-loading.  All factories return un-fitted sklearn-compatible
estimator classes or instances.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _patch_lgb_feature_names_in_setter() -> None:
    """Install a no-op setter for ``LGBMModel.feature_names_in_``.

    Fix 4 defense-in-depth (2026-04-21). LightGBM >=4.6.0 exposes
    ``feature_names_in_`` as a read-only property. sklearn >=1.8's
    ``validate_data`` path (triggered whenever ``fit()`` receives a
    non-pandas input such as a Polars DataFrame or numpy array) calls
    ``self.feature_names_in_ = X.columns``, which raises
    ``AttributeError: property 'feature_names_in_' of 'LGBMClassifier'
    object has no setter`` -- aborting the run 5 seconds in.

    The primary fix is Fix 1 (ensure LGB receives pandas -> sklearn path
    skipped at ``lightgbm/sklearn.py:948``). This setter patch is a
    belt-and-braces guard for cases where a future code path slips a
    non-pandas input past the lazy-conversion hook. Storing the value in
    a private attribute makes it recoverable if anything downstream
    introspects it.

    Idempotent: safe to call multiple times (module re-import).
    """
    if LGBMClassifier is None:
        return
    import lightgbm.sklearn as _lgbm_sk

    _model_cls = _lgbm_sk.LGBMModel
    prop = _model_cls.__dict__.get("feature_names_in_")
    # Only patch if the property exists and has no setter. Avoids clobbering
    # a future upstream fix that might add one.
    if prop is None or not isinstance(prop, property) or prop.fset is not None:
        return
    if getattr(_model_cls, "_mlframe_feature_names_setter_installed", False):
        return

    def _set_feature_names_in(self, value):
        object.__setattr__(self, "_mlframe_feature_names_in_override", value)

    patched = property(
        fget=prop.fget,
        fset=_set_feature_names_in,
        fdel=prop.fdel,
        doc=prop.__doc__,
    )
    _model_cls.feature_names_in_ = patched
    _model_cls._mlframe_feature_names_setter_installed = True


_patch_lgb_feature_names_in_setter()


def _patch_dataset_constructors_with_logging() -> None:
    """Wrap ``catboost.Pool.__init__`` / ``xgboost.DMatrix.__init__`` /
    ``lightgbm.Dataset.__init__`` so every construction emits one INFO
    log line with shape + duration + callsite. Fix 9.4.1 (2026-04-21).

    Purpose: make rebuild-vs-reuse visible in the log. Without this the
    sklearn-wrapper rebuilds silently inside ``fit()`` and the operator
    has no way to tell whether the inner weight/target loop is paying
    N-times the construction cost.

    Idempotent: marker attr ``_mlframe_build_logger_installed`` on each
    wrapped class; subsequent calls are no-ops.
    """
    import time as _time
    import sys as _sys

    def _infer_shape(args, kwargs):
        # First positional or the ``data`` kwarg is the payload; try shape.
        payload = kwargs.get("data")
        if payload is None and args:
            payload = args[0]
        if payload is None:
            return None
        try:
            shp = getattr(payload, "shape", None)
            if shp is not None and len(shp) >= 1:
                rows = int(shp[0])
                cols = int(shp[1]) if len(shp) > 1 else -1
                return (rows, cols)
        except Exception:
            pass
        try:
            return (len(payload), -1)
        except Exception:
            return None

    def _infer_callsite() -> str:
        # Walk up to find the first frame outside the library internals.
        try:
            frame = _sys._getframe(2)
            for _ in range(8):
                if frame is None:
                    break
                mod = frame.f_globals.get("__name__", "?")
                if not (mod.startswith("catboost.") or mod.startswith("xgboost.") or mod.startswith("lightgbm.")):
                    return f"{mod}:{frame.f_lineno}"
                frame = frame.f_back
            return f"{frame.f_globals.get('__name__', '?')}:{frame.f_lineno}" if frame else "?"
        except Exception:
            return "?"

    def _wrap_init(cls, label: str):
        if cls is None:
            return
        # Check the marker on ``cls.__dict__`` specifically -- a subclass
        # (e.g. ``xgboost.QuantileDMatrix`` extending ``DMatrix``) inherits
        # its parent's marker via attribute lookup, which would otherwise
        # cause us to skip wrapping the subclass and log only the parent's
        # build events. Checking the own-dict guarantees each concrete
        # class gets its own wrapper.
        if cls.__dict__.get("_mlframe_build_logger_installed", False):
            return
        orig_init = cls.__init__

        def _logged_init(self, *args, **kwargs):
            t0 = _time.perf_counter()
            try:
                orig_init(self, *args, **kwargs)
            finally:
                elapsed = _time.perf_counter() - t0
                shape = _infer_shape(args, kwargs)
                callsite = _infer_callsite()
                if shape and shape[1] >= 0:
                    shape_str = f"{shape[0]}x{shape[1]}"
                elif shape:
                    shape_str = f"{shape[0]}x?"
                else:
                    shape_str = "?x?"
                # I3 fix (2026-05-11): demote composite-screening builds (typically tiny CV folds, < 50K rows) to DEBUG so 50+ log lines per discovery pass don't drown out the actually-useful build events on production-size datasets. Heuristic: callsite originates in the composite module OR row count below 50K.
                _is_screening = "composite" in (callsite or "") or (shape and shape[0] is not None and shape[0] < 50_000)
                _level = logging.DEBUG if _is_screening else logging.INFO
                logger.log(
                    _level,
                    "[dataset-build] %s shape=%s took=%.3fs site=%s",
                    label,
                    shape_str,
                    elapsed,
                    callsite,
                )

        _logged_init.__wrapped__ = orig_init  # type: ignore[attr-defined]
        cls.__init__ = _logged_init
        cls._mlframe_build_logger_installed = True

    # CatBoost Pool
    try:
        import catboost as _cb

        _wrap_init(getattr(_cb, "Pool", None), "catboost.Pool")
    except ImportError:
        pass

    # XGBoost DMatrix family. QuantileDMatrix inherits from DMatrix in
    # recent XGB; wrap each concrete class separately so subclass-level
    # __init__ overrides still see logging.
    try:
        import xgboost as _xgb

        for _name in ("DMatrix", "QuantileDMatrix", "DeviceQuantileDMatrix"):
            _wrap_init(getattr(_xgb, _name, None), f"xgboost.{_name}")
    except ImportError:
        pass

    # LightGBM Dataset
    try:
        import lightgbm as _lgb

        _wrap_init(getattr(_lgb, "Dataset", None), "lightgbm.Dataset")
    except ImportError:
        pass


_patch_dataset_constructors_with_logging()

try:
    from xgboost import XGBClassifier, XGBRegressor
    from xgboost.callback import TrainingCallback as XGBTrainingCallback
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment]
    XGBTrainingCallback = object  # type: ignore[assignment]

# DMatrix-reuse shim (2026-04-24). Subclasses XGBClassifier / XGBRegressor
# to cache QuantileDMatrix across consecutive ``.fit()`` calls on the same
# feature matrix -- saves ~100 s per repeated fit on multi-GB train frames.
# Toggle via ``USE_XGB_DMATRIX_REUSE_SHIM`` below.
try:
    from mlframe.training.xgb_shim import (
        XGBClassifierWithDMatrixReuse,
        XGBRegressorWithDMatrixReuse,
    )

    _XGB_SHIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBClassifierWithDMatrixReuse = XGBRegressorWithDMatrixReuse = None  # type: ignore[assignment]
    _XGB_SHIM_AVAILABLE = False

# Dataset-reuse shim (2026-05-08). Mirror of the XGB shim above for
# LightGBM. Subclasses LGBMClassifier / LGBMRegressor to cache the
# binned ``lightgbm.Dataset`` across consecutive ``.fit()`` calls on
# the same feature matrix -- mirrors the same weight-schema-loop saving
# the XGB shim eliminates. Toggle via ``USE_LGB_DATASET_REUSE_SHIM`` below.
try:
    from mlframe.training.lgb_shim import (
        LGBMClassifierWithDatasetReuse,
        LGBMRegressorWithDatasetReuse,
    )

    _LGB_SHIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LGBMClassifierWithDatasetReuse = LGBMRegressorWithDatasetReuse = None  # type: ignore[assignment]
    _LGB_SHIM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Feature flag: which XGBoost class do we instantiate in
# ``_configure_xgboost_params``?
#
#   True  -> use the DMatrix-reuse shim. Reuses QuantileDMatrix across
#           weight-schema iterations and target swaps on the same feature
#           matrix (the 2026-04-24 prod log saving target -- ~100 s per
#           rebuild eliminated).
#   False -> fall back to vanilla ``XGBClassifier`` / ``XGBRegressor``.
#           Use this if the shim regresses behaviour or once XGBoost
#           upstream lands the equivalent fix natively.
#
# To **revert** to vanilla XGBoost (e.g. when the upstream PR ships):
#   1. Set ``USE_XGB_DMATRIX_REUSE_SHIM = False`` here, or
#   2. Delete the import block above + ``_xgb_classifier_cls`` /
#      ``_xgb_regressor_cls`` factories below + this constant, and
#      replace ``_xgb_classifier_cls(use_flaml_zeroshot)`` calls in
#      ``_configure_xgboost_params`` with the original inline expression
#      ``flaml_zeroshot.XGBClassifier if use_flaml_zeroshot else
#      XGBClassifier``.
#   3. Delete ``mlframe/training/xgb_shim.py`` and its test counterpart.
#
# Either path is intentionally a small, localized change.
USE_XGB_DMATRIX_REUSE_SHIM: bool = _XGB_SHIM_AVAILABLE


def _xgb_classifier_cls(use_flaml_zeroshot: bool):
    """Return the XGBClassifier class to instantiate.

    Single dispatch point for the shim toggle -- see
    ``USE_XGB_DMATRIX_REUSE_SHIM`` above for revert instructions.
    """
    if use_flaml_zeroshot:
        _fz = _get_flaml_zeroshot()
        if _fz is None:
            raise ImportError("use_flaml_zeroshot=True but flaml is not installed")
        return _fz.XGBClassifier
    if USE_XGB_DMATRIX_REUSE_SHIM and XGBClassifierWithDMatrixReuse is not None:
        return XGBClassifierWithDMatrixReuse
    return XGBClassifier


def _xgb_regressor_cls(use_flaml_zeroshot: bool):
    """Return the XGBRegressor class to instantiate. Mirror of
    ``_xgb_classifier_cls``."""
    if use_flaml_zeroshot:
        _fz = _get_flaml_zeroshot()
        if _fz is None:
            raise ImportError("use_flaml_zeroshot=True but flaml is not installed")
        return _fz.XGBRegressor
    if USE_XGB_DMATRIX_REUSE_SHIM and XGBRegressorWithDMatrixReuse is not None:
        return XGBRegressorWithDMatrixReuse
    return XGBRegressor


# ---------------------------------------------------------------------------
# Feature flag: which LightGBM class do we instantiate in
# ``_configure_lightgbm_params``? Mirror of ``USE_XGB_DMATRIX_REUSE_SHIM``.
#
#   True  -> use the Dataset-reuse shim. Reuses ``lightgbm.Dataset`` across
#           weight-schema iterations and target swaps on the same feature
#           matrix.
#   False -> fall back to vanilla ``LGBMClassifier`` / ``LGBMRegressor``.
#           Use this if the shim regresses behaviour or once LightGBM
#           upstream lands the equivalent fix natively (PR pending).
#
# To **revert** to vanilla LightGBM (e.g. when the upstream PR ships):
#   1. Set ``USE_LGB_DATASET_REUSE_SHIM = False`` here, or
#   2. Delete the import block above + ``_lgb_classifier_cls`` /
#      ``_lgb_regressor_cls`` factories below + this constant, and
#      replace ``_lgb_classifier_cls(use_flaml_zeroshot)`` calls in
#      ``_configure_lightgbm_params`` with the original inline expression
#      ``flaml_zeroshot.LGBMClassifier if use_flaml_zeroshot else
#      LGBMClassifier``.
#   3. Delete ``mlframe/training/lgb_shim.py`` and its test counterpart.
USE_LGB_DATASET_REUSE_SHIM: bool = _LGB_SHIM_AVAILABLE


def _lgb_classifier_cls(use_flaml_zeroshot: bool):
    """Return the LGBMClassifier class to instantiate.

    Single dispatch point for the shim toggle -- see
    ``USE_LGB_DATASET_REUSE_SHIM`` above for revert instructions.
    """
    if use_flaml_zeroshot:
        _fz = _get_flaml_zeroshot()
        if _fz is None:
            raise ImportError("use_flaml_zeroshot=True but flaml is not installed")
        return _fz.LGBMClassifier
    if USE_LGB_DATASET_REUSE_SHIM and LGBMClassifierWithDatasetReuse is not None:
        return LGBMClassifierWithDatasetReuse
    return LGBMClassifier


def _lgb_regressor_cls(use_flaml_zeroshot: bool):
    """Return the LGBMRegressor class to instantiate. Mirror of
    ``_lgb_classifier_cls``."""
    if use_flaml_zeroshot:
        _fz = _get_flaml_zeroshot()
        if _fz is None:
            raise ImportError("use_flaml_zeroshot=True but flaml is not installed")
        return _fz.LGBMRegressor
    if USE_LGB_DATASET_REUSE_SHIM and LGBMRegressorWithDatasetReuse is not None:
        return LGBMRegressorWithDatasetReuse
    return LGBMRegressor


try:
    from ngboost import NGBClassifier, NGBRegressor
except ImportError:  # pragma: no cover
    NGBClassifier = NGBRegressor = None  # type: ignore[assignment]
# flaml.default is eagerly loaded by ``import flaml.default`` (it pulls in
# flaml.tune.searcher.suggestion -> optuna -> scipy.stats.qmc), and that
# import chain takes 30-180 s cold on Windows, blowing past per-test
# timeouts on the FIRST test of any pytest run that touches the trainer.
# Defer the import to first-actual-use via ``_get_flaml_zeroshot()`` so
# typical users / fuzz tests don't pay the cost. Set to ``None`` here as
# a sentinel; the getter populates it lazily.
flaml_zeroshot = None  # type: ignore[assignment]


def _get_flaml_zeroshot():
    """Lazy-load ``flaml.default`` on first use.

    Caches the result on the module-level ``flaml_zeroshot`` so subsequent
    calls are free. Returns ``None`` if flaml is not installed (matching
    the historical ``except ImportError`` behaviour).
    """
    global flaml_zeroshot
    if flaml_zeroshot is None:
        try:
            import flaml.default as _flaml_default

            flaml_zeroshot = _flaml_default
        except ImportError:  # pragma: no cover
            return None
    return flaml_zeroshot


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
# mlframe.training.neural eagerly imports lightning + torchmetrics at
# module-load (see neural/base.py:32-45 for the chain). On Windows that
# takes 30-180 s cold and consistently overshoots the per-test timeout
# of the FIRST test in any pytest run that touches the trainer (fuzz
# c0000 timeout, observed 2026-04-27). Defer the import to first MLP
# fit via ``_get_neural_components()`` so typical users / fuzz tests
# don't pay the cost. Sentinel ``None`` here; the getter populates the
# tuple lazily on first call and caches.
MLPNeuronsByLayerArchitecture = None  # type: ignore[assignment]
PytorchLightningRegressor = PytorchLightningClassifier = None  # type: ignore[assignment]


def _get_neural_components():
    """Lazy-load ``MLPNeuronsByLayerArchitecture`` /
    ``PytorchLightningRegressor`` / ``PytorchLightningClassifier`` on
    first MLP fit. Returns the 3-tuple, or ``(None, None, None)`` if
    the optional ``mlframe.training.neural`` extras are not installed.
    Caches into the module-level globals so subsequent calls are free.
    """
    global MLPNeuronsByLayerArchitecture, PytorchLightningRegressor, PytorchLightningClassifier
    if MLPNeuronsByLayerArchitecture is None:
        try:
            from mlframe.training.neural import (
                MLPNeuronsByLayerArchitecture as _arch,
                PytorchLightningRegressor as _reg,
                PytorchLightningClassifier as _cls,
            )

            MLPNeuronsByLayerArchitecture = _arch
            PytorchLightningRegressor = _reg
            PytorchLightningClassifier = _cls
        except ImportError:  # pragma: no cover
            return None, None, None
    return MLPNeuronsByLayerArchitecture, PytorchLightningRegressor, PytorchLightningClassifier


from pyutilz.system import clean_ram, ensure_dir_exists, compute_total_gpus_ram, get_gpuinfo_gpu_info


from mlframe.training.utils import maybe_clean_ram_adaptive as _maybe_clean_ram
from mlframe.training.phases import phase
from pyutilz.strings import slugify
from pyutilz.pandaslib import get_df_memory_consumption
from pyutilz.pythonlib import prefix_dict_elems, get_human_readable_set_size

from mlframe.helpers import get_model_best_iter, ensure_no_infinity
from mlframe.config import (
    TABNET_MODEL_TYPES,
    XGBOOST_MODEL_TYPES,
    CATBOOST_MODEL_TYPES,
    LGBM_MODEL_TYPES,
)

from numba.cuda import is_available as is_cuda_available

CUDA_IS_AVAILABLE = is_cuda_available()
MODELS_SUBDIR = "models"
GPU_VRAM_SAFE_SATURATION_LIMIT: float = 0.9
GPU_VRAM_SAFE_FREE_LIMIT_GB: float = 0.1
from mlframe.metrics import (
    compute_probabilistic_multiclass_error,
    fast_calibration_report,
    fast_roc_auc,
)

# Import helper functions from helpers module
from .helpers import (
    get_training_configs,
    parse_catboost_devices,
    LightGBMCallback,
    CatBoostCallback,
    XGBoostCallback,
    compute_cb_text_processing,
    CB_DEFAULT_OCCURRENCE_LOWER_BOUND,
)

# Fairness and feature importance functions from their respective modules
from mlframe.metrics import create_fairness_subgroups, create_fairness_subgroups_indices, compute_fairness_metrics
from mlframe.feature_importance import plot_feature_importance
from mlframe.metrics import ICE
from mlframe.feature_selection.wrappers import RFECV

from .configs import (
    DataConfig,
    TrainingControlConfig,
    MetricsConfig,
    ReportingConfig,
    FeatureImportanceConfig,
    OutputConfig,
    NamingConfig,
    ConfidenceAnalysisConfig,
    PredictionsContainer,
    LinearModelConfig,
    MultilabelDispatchConfig,
)
from .utils import log_ram_usage, get_categorical_columns, get_numeric_columns, filter_existing

# 2026-05-13 refactor: extracted modules
from ._predict_guards import _CB_VAL_POOL_CACHE  # noqa: E402,F401
from ._pipeline_helpers import (  # noqa: E402,F401
    _apply_pre_pipeline_transforms,
    _extract_feature_selector,
    _is_fitted,
    _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform,
    _prepare_test_split,
)
from ._cb_pool import (  # noqa: E402,F401
from ._eval_helpers import (  # noqa: E402,F401
from ._training_loop import (  # noqa: E402,F401
    _SigmoidAdapter,
    _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel,
    _PerClassIsotonicCalibrator,
    _maybe_apply_posthoc_calibration,
    _train_model_with_fallback,
    _handle_oom_error,
    _setup_eval_set,
    _setup_early_stopping_callback,
)
    _align_xgb_cat_categories,
    _append_split_rate_suffix,
    _compute_split_metrics,
    _decategorise_float_cat_columns,
    _filter_categorical_features,
    run_confidence_analysis,
)
    _cached_gpu_info,
    _maybe_get_or_build_cb_pool,
    _maybe_rewrite_eval_set_as_cb_pool,
    _polars_fill_null_in_categorical,
    _polars_schema_diagnostic,
)


