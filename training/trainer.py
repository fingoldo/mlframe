"""
Core training and evaluation functions.

This module contains:
- train_and_evaluate_model: Main training function
- train_and_evaluate_model_v2: Config-based wrapper
- configure_training_params: Training parameter configuration
- _build_configs_from_params: Config object builder
- Helper functions for training
"""

import copy
import inspect
import logging
import re
import pickle
from timeit import default_timer as timer
from functools import partial
import os
from os import sep as os_sep
from os.path import join, exists
from types import SimpleNamespace
from typing import Optional, Tuple, Union, Callable, Sequence, List, Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import joblib

# Heavy optional deps: defer failures to first actual use so `import mlframe.training`
# stays cheap and does not crash when a given backend is not installed. Mirrors the
# lazy-loading style in `mlframe.training.__init__.__getattr__`.
try:
    import matplotlib.pyplot as plt  # only used in a handful of plotting branches
except ImportError:  # pragma: no cover -- optional backend
    plt = None  # type: ignore[assignment]

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    max_error,
    r2_score,
    root_mean_squared_error,
    make_scorer,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Optional model backends -- lazy/tolerant of missing deps, matching __init__.py style.
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostRegressor = CatBoostClassifier = None  # type: ignore[assignment]
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[assignment]


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


def _validate_trusted_path(path: str, trusted_root: Optional[str]) -> None:
    """Raise ValueError if ``path`` is not inside ``trusted_root`` (absolute commonpath check).

    Matches the convention used in ``mlframe.inference.read_trained_models``. Callers that
    want to disable the check must pass ``trusted_root=None`` explicitly; that is only
    appropriate for internally-produced cache files (the default posture refuses silently
    loading untrusted pickles).
    """
    import os as _os

    if trusted_root is None:
        raise ValueError(
            "trusted_root is required for joblib.load() of cached model files. "
            "Pass an absolute directory under which cached artifacts are stored, "
            "or set it to the containing directory of the file being loaded."
        )
    abs_root = _os.path.abspath(trusted_root)
    abs_path = _os.path.abspath(path)
    try:
        common = _os.path.commonpath([abs_root, abs_path])
    except ValueError:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")
    if common != abs_root:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")


from .models import create_linear_model, LINEAR_MODEL_TYPES

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_function_param_names(func: Callable) -> List[str]:
    """Get parameter names from a function signature.

    Parameters
    ----------
    func : Callable
        Function to inspect.

    Returns
    -------
    list of str
        List of parameter names.
    """
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def _extract_target_subset(
    target: Optional[Union[pd.Series, pl.Series, np.ndarray]],
    idx: Optional[np.ndarray],
) -> Optional[Union[pd.Series, pl.Series, np.ndarray]]:
    """Extract target subset handling pandas Series, polars Series, and numpy arrays.

    Parameters
    ----------
    target : pd.Series, pl.Series, np.ndarray, or None
        Target values to subset.
    idx : np.ndarray or None
        Indices to select. If None, returns full target.

    Returns
    -------
    pd.Series, pl.Series, np.ndarray, or None
        Subsetted target values.
    """
    if idx is None:
        return target
    if isinstance(target, pd.Series):
        # 2026-05-12 Wave 32: ``.values[idx]`` is 9× faster than
        # ``.iloc[idx]`` for numeric targets (bench: 0.036s vs 0.324s
        # per 100k-row subset on 1M-row Series × 100 iterations).
        # Target is always numeric (float for regression, int for
        # classification/rank), so .values is safe.
        return target.values[idx]
    elif isinstance(target, pl.Series):
        return target.gather(idx)
    # numpy: ``target[idx]`` is already fast — 0.033s vs np.take 0.049s
    return target[idx]


def _subset_dataframe(
    df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    idx: Optional[np.ndarray],
    drop_columns: Optional[List[str]] = None,
) -> Optional[Union[pd.DataFrame, pl.DataFrame]]:
    """Subset DataFrame with optional column dropping, handling pandas and polars.

    Parameters
    ----------
    df : pd.DataFrame, pl.DataFrame, or None
        Input DataFrame to subset.
    idx : np.ndarray or None
        Indices to select. If None, returns full DataFrame.
    drop_columns : list of str, optional
        Columns to drop from the result.

    Returns
    -------
    pd.DataFrame, pl.DataFrame, or None
        Subsetted DataFrame with specified columns dropped.
    """
    if df is None:
        return df
    if idx is None:
        result = df
    elif isinstance(df, pd.DataFrame):
        result = df.iloc[idx]
    elif isinstance(df, pl.DataFrame):
        result = df[idx]
    else:
        return df

    if drop_columns:
        # Validate drop_columns is a list-like, not a string
        if isinstance(drop_columns, str):
            logger.warning(f"drop_columns should be a list, got string '{drop_columns}'. Converting to list.")
            drop_columns = [drop_columns]
        if isinstance(result, pd.DataFrame):
            return result.drop(columns=filter_existing(result, drop_columns))
        elif isinstance(result, pl.DataFrame):
            cols_to_drop = filter_existing(result, drop_columns)
            return result.drop(cols_to_drop) if cols_to_drop else result
    return result


def _prepare_df_for_model(df, model_type_name):
    """Convert DataFrame to numpy if required by model type (e.g., TabNet)."""
    if df is None:
        return None
    if model_type_name in TABNET_MODEL_TYPES and hasattr(df, "values"):
        return df.values
    return df


def _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params):
    """Configure sample weights in fit_params if supported by model."""
    if sample_weight is None:
        return
    if "sample_weight" not in get_function_param_names(model_obj.fit):
        return

    if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
        if train_idx is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_idx].values
        else:
            fit_params["sample_weight"] = sample_weight.values
    else:
        if train_idx is not None:
            fit_params["sample_weight"] = sample_weight[train_idx]
        else:
            fit_params["sample_weight"] = sample_weight


def _initialize_mutable_defaults(drop_columns, default_drop_columns, fi_kwargs, confidence_model_kwargs):
    """Initialize mutable default arguments."""
    if drop_columns is None:
        drop_columns = []
    if default_drop_columns is None:
        default_drop_columns = []
    if fi_kwargs is None:
        fi_kwargs = {}
    if confidence_model_kwargs is None:
        confidence_model_kwargs = {}
    return drop_columns, default_drop_columns, fi_kwargs, confidence_model_kwargs


def _validate_target_values(target, subset_name="train", is_classification=None):
    """Check target for NaN / infinity values and (for classification)
    single-class collapse before training.

    Single-class detection: when ``is_classification=True`` and the
    target carries fewer than 2 unique values, raise a ValueError
    BEFORE the per-backend fit. CatBoost otherwise crashes with
    ``target_converter.cpp:404: Target contains only one unique value``,
    XGBoost with ``num_class is 1, expected at least 2``, etc. -- all
    opaque C++ errors. The proximate cause in fuzz is upstream filter
    aggression (outlier_detection + trainset_aging_limit + rare imbalance
    class) eliminating the minority class entirely from train. The
    early raise gives operators a clear diagnostic instead of a deep
    backend crash (fuzz seed=99 c0016 / 2026-04-27).

    is_classification=None preserves the historical behaviour: only
    the NaN/inf check runs, so callers that haven't been migrated to
    pass the flag explicitly are unaffected.
    """
    arr = target.values if isinstance(target, pd.Series) else target
    try:
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
    except TypeError:
        nan_count = inf_count = 0  # non-numeric target (e.g., categorical)
    if nan_count > 0 or inf_count > 0:
        parts = []
        if nan_count > 0:
            parts.append(f"{nan_count:_} NaN")
        if inf_count > 0:
            parts.append(f"{inf_count:_} infinity")
        raise ValueError(f"{subset_name} target contains {' and '.join(parts)} value(s). " f"Clean the target before training.")
    if is_classification:
        try:
            arr_np = np.asarray(target)
            # Object dtype with nested arrays (polars ``pl.List`` roundtrip
            # for multilabel) presents as 1-D but each cell is itself an
            # array. Stack to a true 2-D shape so the per-label degenerate
            # check below works without ``np.unique`` choking on cell-array
            # comparison ("truth value ambiguous", surfaced 3-way fuzz
            # c0000).
            if arr_np.dtype == object and arr_np.ndim == 1 and arr_np.shape[0] > 0:
                _first = arr_np[0]
                if hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))):
                    try:
                        arr_np = np.stack([np.asarray(c) for c in arr_np], axis=0)
                    except Exception:
                        pass
            if arr_np.ndim > 1:
                # Multilabel / multiclass-prob: per-column unique check.
                # If ANY column has only one unique value, the model
                # for that label is degenerate but the others may
                # train. Report at WARNING and let the per-backend
                # path decide.
                degenerate_cols = []
                for _i in range(arr_np.shape[1]):
                    if len(np.unique(arr_np[:, _i])) < 2:
                        degenerate_cols.append(_i)
                if degenerate_cols:
                    logger.warning(
                        "%s target has %d label column(s) with a single unique "
                        "value: %s. The corresponding per-label model(s) will "
                        "fail; the rest may train normally if the multilabel "
                        "strategy supports it.",
                        subset_name,
                        len(degenerate_cols),
                        degenerate_cols,
                    )
            else:
                if len(np.unique(arr_np)) < 2:
                    raise ValueError(
                        f"{subset_name} target has only one unique value "
                        f"({arr_np.flat[0]!r}); classification needs at least "
                        f"2 classes. Most likely cause: upstream filtering "
                        f"(outlier_detection + trainset_aging_limit + rare "
                        f"imbalance) eliminated the minority class entirely. "
                        f"Investigate the filter pipeline OR loosen the "
                        f"contamination / aging knobs."
                    )
        except ValueError:
            raise
        except Exception:
            # np.unique/asarray edge cases on object dtype etc -- let the
            # downstream backend surface its own error.
            pass


def _validate_infinity_and_columns(df, train_df, skip_infinity_checks, drop_columns):
    """Validate DataFrames for infinity values and compute real drop columns."""
    if not skip_infinity_checks:
        if df is not None:
            ensure_no_infinity(df)
        else:
            if train_df is not None:
                ensure_no_infinity(train_df)

    if df is not None:
        real_drop_columns = filter_existing(df, drop_columns)
    elif train_df is not None:
        real_drop_columns = filter_existing(train_df, drop_columns)
    else:
        real_drop_columns = []

    return real_drop_columns


def _strip_internal_model_suffixes(name: str) -> str:
    """Remove implementation-detail suffixes from a model class name
    so user-facing chart titles / log lines show the canonical model
    type (``XGBClassifier`` not ``XGBClassifierWithDMatrixReuse``).

    These suffixes come from internal mixin subclasses we wrap around
    upstream sklearn-API classes -- valuable for code clarity, noise
    in user-visible output:

      XGBClassifierWithDMatrixReuse  -> XGBClassifier
      XGBRegressorWithDMatrixReuse   -> XGBRegressor
      LGBMClassifierWithDatasetReuse -> LGBMClassifier
      LGBMRegressorWithDatasetReuse  -> LGBMRegressor
      *WithFastpath                  -> *  (legacy)
    """
    for suffix in ("WithDMatrixReuse", "WithDatasetReuse", "WithFastpath"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _setup_model_info_and_paths(model, model_name, model_name_prefix, plot_file, data_dir, models_subdir):
    """Extract model object info and construct naming/path information."""
    if type(model).__name__ == "Pipeline":
        model_obj = model.named_steps["est"]
    else:
        model_obj = model

    if model_obj is not None:
        if isinstance(model_obj, TransformedTargetRegressor):
            model_obj = model_obj.regressor
    model_type_name = type(model_obj).__name__ if model_obj is not None else ""
    # 2026-05-09: strip internal mixin suffixes so chart titles show
    # the canonical class name (XGBClassifier, not the
    # XGBClassifierWithDMatrixReuse internal subclass). Implementation
    # detail not relevant to end users.
    model_type_name = _strip_internal_model_suffixes(model_type_name)

    if plot_file:
        if not plot_file.endswith(os_sep):
            plot_file = plot_file + "_"
        if model_name_prefix:
            plot_file = plot_file + slugify(model_name_prefix) + " "
        if model_type_name:
            plot_file = plot_file + slugify(model_type_name) + " "
        plot_file = plot_file.strip()

    if model_name_prefix:
        model_name = model_name_prefix + model_name
    if model_type_name not in model_name:
        model_name = model_type_name + " " + model_name

    # Falsy guard: avoid creating a relative `./models/` leak when data_dir="".
    # See also `_setup_model_directories` in core.py.
    if data_dir and models_subdir:
        ensure_dir_exists(join(data_dir, models_subdir))
        model_file_name = join(data_dir, models_subdir, f"{model_name}.dump")
    else:
        model_file_name = ""

    return model_obj, model_type_name, model_name, plot_file, model_file_name


def _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj):
    """Disable XGBoost early stopping when no validation data is available."""
    if model_type_name in XGBOOST_MODEL_TYPES and model_obj is not None:
        es_rounds = getattr(model_obj, "early_stopping_rounds", None)
        if es_rounds is not None:
            logger.warning(f"No validation data available - disabling early stopping for {model_type_name}")
            model_obj.set_params(early_stopping_rounds=None)


def _normalize_multilabel_target(target):
    """Stack a 1-D object array of per-row label arrays (the polars
    ``pl.List(pl.Int8)`` -> pandas object roundtrip) into a true 2-D
    ndarray ``(N, K)``. Returns ``target`` unchanged for any other shape.

    Performed once per split so every downstream consumer (sklearn
    estimators, MultiOutputClassifier, drift_report, evaluation,
    metrics, CB Pool, XGB) sees a canonical 2-D multilabel target
    instead of an object-of-arrays trap. Surfaced 3-way fuzz c0008
    where sklearn ``check_X_y`` -> ``_object_dtype_isnan(y)`` did
    ``y != y`` on the cell-array column and raised ``truth value of
    array ambiguous``.
    """
    if target is None or not isinstance(target, np.ndarray):
        return target
    if target.dtype != object or target.ndim != 1 or target.shape[0] == 0:
        return target
    _first = target[0]
    if not (hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))):
        return target
    try:
        return np.stack([np.asarray(c) for c in target], axis=0)
    except Exception:
        return target


def _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target):
    """Extract train/val/test targets from main target using indices."""
    if target is not None:
        if train_target is None and (train_idx is not None):
            train_target = _extract_target_subset(target, train_idx)
        if val_target is None and (val_idx is not None):
            val_target = _extract_target_subset(target, val_idx)
        if test_target is None and (test_idx is not None):
            test_target = _extract_target_subset(target, test_idx)
    train_target = _normalize_multilabel_target(train_target) if isinstance(train_target, np.ndarray) else train_target
    val_target = _normalize_multilabel_target(val_target) if isinstance(val_target, np.ndarray) else val_target
    test_target = _normalize_multilabel_target(test_target) if isinstance(test_target, np.ndarray) else test_target
    return train_target, val_target, test_target


def _prepare_train_df_for_fitting(train_df, model, model_type_name, fit_params):
    """Prepare training DataFrame and fit_params for model fitting."""
    if model_type_name in TABNET_MODEL_TYPES:
        train_df = train_df.values

    if fit_params and type(model).__name__ == "Pipeline":
        fit_params = prefix_dict_elems(fit_params, "est__")

    return train_df, fit_params


def _update_model_name_after_training(model_name, train_df_len, train_details, best_iter):
    """Update model name with training details and early stopping info."""
    model_name = model_name + "\n" + " ".join([f" trained on {get_human_readable_set_size(train_df_len)} rows", train_details])

    if best_iter:
        logger.info(f"es_best_iter: {best_iter:_}")
        model_name = model_name + f" @iter={best_iter:_}"

    return model_name


def _setup_eval_set(
    model_type_name: str,
    fit_params: Dict[str, Any],
    val_df: Union[pd.DataFrame, np.ndarray],
    val_target: Union[pd.Series, np.ndarray],
    callback_params: Optional[Dict[str, Any]] = None,
    model_obj: Optional[Any] = None,
    model_category: Optional[str] = None,
) -> None:
    """Configure eval_set/validation data for different model types.

    Modifies fit_params in-place to add the appropriate eval_set configuration
    based on the model type (LightGBM, CatBoost, XGBoost, etc.).

    Parameters
    ----------
    model_type_name : str
        Name of the model type (e.g., 'LGBMClassifier').
    fit_params : dict
        Dictionary to populate with eval_set configuration.
    val_df : pd.DataFrame or np.ndarray
        Validation features.
    val_target : pd.Series or np.ndarray
        Validation target values.
    callback_params : dict, optional
        Parameters for early stopping callback.
    model_obj : Any, optional
        Model object for XGBoost callback setup.
    model_category : str, optional
        Short model type name (cb, xgb, lgb, etc.). If provided, used directly
        instead of deriving from model_type_name for reliable matching.
    """
    eval_set_configs = {
        "lgb": ("eval_set", "tuple"),
        "hgb": ("X_val", "separate"),
        "ngb": ("X_val", "separate_Y"),
        "cb": ("eval_set", "list_of_tuples"),
        "xgb": ("eval_set", "list_of_tuples"),
        "tabnet": ("eval_set", "list_of_tuples_values"),
        "mlp": ("eval_set", "tuple"),
    }

    # Use provided model_category if available, otherwise derive from model_type_name
    if model_category is None:
        model_type_lower = model_type_name.lower()
        for key in eval_set_configs:
            if key in model_type_lower:
                model_category = key
                break

    if model_category is None or model_category not in eval_set_configs:
        return

    # Historical 0-row val skip in _setup_eval_set removed 2026-04-28
    # (batch 4). The original empty-val window came from outlier
    # detection rejecting almost every val row; that's now guarded at
    # the source in ``core._apply_outlier_detection_global`` (val-side
    # min_keep floor + class-balance pre-check). If a 0-row val still
    # arrives here it's an upstream bug -- let CB raise its own
    # "Labels variable is empty" so the bug surfaces immediately
    # instead of silently training without early-stopping val.

    # 2026-04-24 Session 6: when the model is wrapped in MultiOutputClassifier
    # (multilabel path), eval_set / X_val / y_val keyword args propagate
    # verbatim to each per-label inner estimator. y_val stays 2-D and crashes
    # the inner fit ("y should be a 1d array, got an array of shape (n,K)").
    # Skip the eval_set injection for wrapped models -- inner estimators must
    # rely on their own internal early-stopping (HGB validation_fraction,
    # or no early stopping for LGB/XGB/Linear).
    if model_type_name in ("MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain"):
        return

    param_name, value_format = eval_set_configs[model_category]

    if value_format == "tuple":
        fit_params[param_name] = (val_df, val_target)
    elif value_format == "list_of_tuples":
        fit_params[param_name] = [(val_df, val_target)]
    elif value_format == "list_of_tuples_values":
        fit_params[param_name] = [(val_df.values, val_target.values if hasattr(val_target, "values") else val_target)]
    elif value_format == "separate":
        fit_params["X_val"] = val_df
        fit_params["y_val"] = val_target
    elif value_format == "separate_Y":
        fit_params["X_val"] = val_df
        fit_params["Y_val"] = val_target

    if callback_params:
        _setup_early_stopping_callback(model_category, fit_params, callback_params, model_obj)


def _setup_early_stopping_callback(model_category, fit_params, callback_params, model_obj=None):
    """Set up early stopping callback for the given model category."""
    no_callback_list_models = {"xgb", "hgb", "ngb"}

    if model_category not in no_callback_list_models:
        if "callbacks" not in fit_params:
            fit_params["callbacks"] = []

    if model_category == "lgb":
        es_callback = LightGBMCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
    elif model_category == "cb":
        es_callback = CatBoostCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
    elif model_category == "xgb" and model_obj is not None:
        es_callback = XGBoostCallback(**callback_params)
        existing_callbacks = model_obj.get_params().get("callbacks", []) or []
        # Keep only valid TrainingCallback instances, excluding stale XGBoostCallback instances.
        # This also filters out any legacy callbacks (e.g. from xgb_kwargs in XGB_GENERAL_PARAMS)
        # that do not inherit from xgboost.callback.TrainingCallback, which would cause a
        # TypeError in XGBoost >= 2.x where CallbackContainer validates isinstance strictly.
        callbacks = [cb for cb in existing_callbacks if isinstance(cb, XGBTrainingCallback) and not isinstance(cb, XGBoostCallback)]
        callbacks.append(es_callback)
        model_obj.set_params(callbacks=callbacks)


def _handle_oom_error(model_obj, model_type_name):
    """Handle GPU out-of-memory error by switching model to CPU."""
    if model_type_name in XGBOOST_MODEL_TYPES:
        if model_obj.get_params().get("device") in ("gpu", "cuda"):
            model_obj.set_params(device="cpu")
            logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
            return True
    elif model_type_name in CATBOOST_MODEL_TYPES:
        if model_obj.get_params().get("task_type") == "GPU":
            model_obj.set_params(task_type="CPU")
            logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
            return True
    elif model_type_name in LGBM_MODEL_TYPES:
        if model_obj.get_params().get("device_type") in ("gpu", "cuda"):
            model_obj.set_params(device_type="cpu")
            logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
            return True
    return False


def _build_configs_from_params(
    # Data params
    df=None,
    train_df=None,
    val_df=None,
    test_df=None,
    target=None,
    train_target=None,
    val_target=None,
    test_target=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    group_ids=None,
    sample_weight=None,
    drop_columns=None,
    default_drop_columns=None,
    target_label_encoder=None,
    skip_infinity_checks=False,
    n_features=None,
    target_type=None,  # 2026-05-10: thread through for downstream chart dispatch gate
    # Control params
    verbose=False,
    # 2026-04-27: use_cache default flipped False -> True for consistency
    # with TrainingControlConfig.use_cache=True and the de-facto behavior
    # of train_eval.py:664's .get("use_cache", True). Cache loading is
    # almost always faster than retraining; force retrain via explicit False.
    use_cache=True,
    just_evaluate=False,
    compute_trainset_metrics=False,
    compute_valset_metrics=True,
    compute_testset_metrics=True,
    pre_pipeline=None,
    skip_pre_pipeline_transform=False,
    skip_preprocessing=False,
    fit_params=None,
    callback_params=None,
    model_category=None,
    # Metrics params
    nbins=10,
    custom_ice_metric=None,
    custom_rice_metric=None,
    subgroups=None,
    train_details="",
    val_details="",
    test_details="",
    # Reporting / display params (the pre-2026-04-27 display config is now
    # ReportingConfig - filesystem paths moved to OutputConfig; per-metric
    # title toggles collapsed into the ordered string template
    # `title_metrics_template`; histogram subplot toggles added; old fi_kwargs
    # dict replaced by typed FeatureImportanceConfig).
    figsize=(15, 5),
    print_report=True,
    show_perf_chart=True,
    show_fi=True,
    feature_importance_config=None,
    plot_file="",
    data_dir="",
    models_subdir=MODELS_SUBDIR,
    display_sample_size=0,
    show_feature_names=False,
    show_prob_histogram=True,
    prob_histogram_yscale="auto",
    show_inline_population_labels=True,
    title_metrics_template="ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC",
    plot_outputs="plotly[html,png]",
    plot_dpi=None,
    multiclass_panels="CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
    multilabel_panels="PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST",
    ltr_panels="NDCG_K NDCG_DIST LIFT MRR_DIST SCORE_BY_REL",
    quantile_panels="RELIABILITY PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST",
    # Naming params
    model_name="",
    model_name_prefix="",
    # Confidence params
    include_confidence_analysis=False,
    confidence_analysis_use_shap=True,
    confidence_analysis_max_features=6,
    confidence_analysis_cmap="bwr",
    confidence_analysis_alpha=0.9,
    confidence_analysis_ylabel="Feature value",
    confidence_analysis_title="Confidence of correct Test set predictions",
    confidence_model_kwargs=None,
    # Predictions params
    train_preds=None,
    train_probs=None,
    val_preds=None,
    val_probs=None,
    test_preds=None,
    test_probs=None,
):
    """Build config objects from old-style parameters."""
    merged_drop_columns = list(drop_columns or []) + list(default_drop_columns or [])

    data_config = DataConfig(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        group_ids=group_ids,
        sample_weight=sample_weight,
        drop_columns=merged_drop_columns,
        target_label_encoder=target_label_encoder,
        skip_infinity_checks=skip_infinity_checks,
        n_features=n_features,
        target_type=str(target_type) if target_type is not None else None,
    )

    control_config = TrainingControlConfig(
        verbose=verbose,
        use_cache=use_cache,
        just_evaluate=just_evaluate,
        compute_trainset_metrics=compute_trainset_metrics,
        compute_valset_metrics=compute_valset_metrics,
        compute_testset_metrics=compute_testset_metrics,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        fit_params=fit_params,
        callback_params=callback_params,
        model_category=model_category,
    )

    metrics_config = MetricsConfig(
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
    )

    if feature_importance_config is None:
        fi_cfg = FeatureImportanceConfig()
    elif isinstance(feature_importance_config, dict):
        fi_cfg = FeatureImportanceConfig(**feature_importance_config)
    else:
        fi_cfg = feature_importance_config

    reporting_config = ReportingConfig(
        figsize=figsize,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
        feature_importance_config=fi_cfg,
        display_sample_size=display_sample_size,
        show_feature_names=show_feature_names,
        show_prob_histogram=show_prob_histogram,
        prob_histogram_yscale=prob_histogram_yscale,
        show_inline_population_labels=show_inline_population_labels,
        title_metrics_template=title_metrics_template,
        plot_outputs=plot_outputs,
        plot_dpi=plot_dpi,
        multiclass_panels=multiclass_panels,
        multilabel_panels=multilabel_panels,
        ltr_panels=ltr_panels,
        quantile_panels=quantile_panels,
    )

    output_config = OutputConfig(
        plot_file=plot_file or "",
        data_dir=data_dir or "",
        models_dir=models_subdir or "models",
    )

    naming_config = NamingConfig(
        model_name=model_name,
        model_name_prefix=model_name_prefix,
    )

    confidence_config = ConfidenceAnalysisConfig(
        include=include_confidence_analysis,
        use_shap=confidence_analysis_use_shap,
        max_features=confidence_analysis_max_features,
        cmap=confidence_analysis_cmap,
        alpha=confidence_analysis_alpha,
        ylabel=confidence_analysis_ylabel,
        title=confidence_analysis_title,
        model_kwargs=confidence_model_kwargs or {},
    )

    predictions_container = PredictionsContainer(
        train_preds=train_preds,
        train_probs=train_probs,
        val_preds=val_preds,
        val_probs=val_probs,
        test_preds=test_preds,
        test_probs=test_probs,
    )

    return data_config, control_config, metrics_config, reporting_config, naming_config, confidence_config, predictions_container, output_config


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main Training Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def train_and_evaluate_model(
    model: object,
    data: DataConfig,
    control: TrainingControlConfig,
    metrics: MetricsConfig,
    reporting: ReportingConfig,
    naming: NamingConfig,
    output: Optional[OutputConfig] = None,
    confidence: Optional[ConfidenceAnalysisConfig] = None,
    predictions: Optional[PredictionsContainer] = None,
    train_od_idx: Optional[np.ndarray] = None,
    val_od_idx: Optional[np.ndarray] = None,
    trainset_features_stats: Optional[dict] = None,
    trusted_root: Optional[str] = None,
):
    """Train and evaluate a machine learning model with comprehensive metrics and optional caching.

    Parameters
    ----------
    model : object
        The model to train (sklearn estimator, Pipeline, etc.).
    data : DataConfig
        Input data configuration (DataFrames, targets, indices).
    control : TrainingControlConfig
        Training control flags (verbose, cache, metrics computation).
    metrics : MetricsConfig
        Metrics configuration (nbins, custom metrics, subgroups).
    reporting : ReportingConfig
        Reporting / display configuration (figsize, plot settings, title-metrics
        template, histogram subplot, feature-importance config). Was
        the pre-2026-04-27 display config (now renamed and slimmed).
    naming : NamingConfig
        Model naming configuration.
    confidence : ConfidenceAnalysisConfig, optional
        Confidence analysis configuration.
    predictions : PredictionsContainer, optional
        Pre-computed predictions (for just_evaluate mode).
    train_od_idx : np.ndarray, optional
        Training outlier detection indices.
    val_od_idx : np.ndarray, optional
        Validation outlier detection indices.
    trainset_features_stats : dict, optional
        Pre-computed feature statistics from training set.

    Returns
    -------
    tuple
        (result_namespace, train_df, val_df, test_df) where result_namespace contains
        model, predictions, metrics, and other training artifacts.
    """
    from IPython.display import display as ipython_display

    # Initialize optional configs with defaults
    if confidence is None:
        confidence = ConfidenceAnalysisConfig()
    if predictions is None:
        predictions = PredictionsContainer()

    # ---------------------------------------------------------------------------
    # Unpack data config
    # ---------------------------------------------------------------------------
    df = data.df
    train_df = data.train_df
    val_df = data.val_df
    test_df = data.test_df
    target = data.target
    train_target = data.train_target
    val_target = data.val_target
    test_target = data.test_target
    train_idx = data.train_idx
    val_idx = data.val_idx
    test_idx = data.test_idx
    group_ids = data.group_ids
    sample_weight = data.sample_weight
    drop_columns = list(data.drop_columns) if data.drop_columns else []
    target_label_encoder = data.target_label_encoder
    skip_infinity_checks = data.skip_infinity_checks
    n_features = data.n_features

    # ---------------------------------------------------------------------------
    # Unpack control config
    # ---------------------------------------------------------------------------
    verbose = control.verbose
    use_cache = control.use_cache
    just_evaluate = control.just_evaluate
    compute_trainset_metrics = control.compute_trainset_metrics
    compute_valset_metrics = control.compute_valset_metrics
    compute_testset_metrics = control.compute_testset_metrics
    pre_pipeline = control.pre_pipeline
    skip_pre_pipeline_transform = control.skip_pre_pipeline_transform
    skip_preprocessing = control.skip_preprocessing
    fit_params = control.fit_params
    callback_params = control.callback_params
    model_category = control.model_category

    # ---------------------------------------------------------------------------
    # Unpack metrics config
    # ---------------------------------------------------------------------------
    nbins = metrics.nbins
    custom_ice_metric = metrics.custom_ice_metric
    custom_rice_metric = metrics.custom_rice_metric
    subgroups = metrics.subgroups
    train_details = metrics.train_details
    val_details = metrics.val_details
    test_details = metrics.test_details

    # ---------------------------------------------------------------------------
    # Unpack reporting config (the pre-2026-04-27 display config). Filesystem paths read from
    # control.* / direct trainer state now (data_dir/models_subdir/plot_file no
    # longer live on the reporting config). FI plotting reads from a typed
    # FeatureImportanceConfig instead of a stringly-typed dict.
    # ---------------------------------------------------------------------------
    figsize = reporting.figsize
    print_report = reporting.print_report
    show_perf_chart = reporting.show_perf_chart
    show_fi = reporting.show_fi
    fi_config = reporting.feature_importance_config or FeatureImportanceConfig()
    fi_kwargs = dict(
        figsize=fi_config.figsize,
        num_factors=fi_config.num_factors,
        positive_fi_only=fi_config.positive_fi_only,
        show_plots=fi_config.show_plots,
        max_zero_fi_to_plot=getattr(fi_config, "max_zero_fi_to_plot", 4),
    )
    display_sample_size = reporting.display_sample_size
    show_feature_names = reporting.show_feature_names
    show_prob_histogram = reporting.show_prob_histogram
    prob_histogram_yscale = reporting.prob_histogram_yscale
    show_inline_population_labels = reporting.show_inline_population_labels
    title_metrics_tokens = reporting.title_metrics_tokens
    plot_outputs = reporting.plot_outputs
    plot_dpi = reporting.plot_dpi
    multiclass_panels = reporting.multiclass_panels
    multilabel_panels = reporting.multilabel_panels
    ltr_panels = reporting.ltr_panels
    quantile_panels = reporting.quantile_panels
    # ``quantile_alphas`` arrives via fit_params (per-fit context),
    # not via ReportingConfig -- it depends on which alphas the model
    # was trained on, not on display preference. Resolved at the
    # _compute_split_metrics call site.
    quantile_alphas = None
    if hasattr(model, "_mlframe_quantile_alphas"):
        quantile_alphas = getattr(model, "_mlframe_quantile_alphas", None)

    # ---------------------------------------------------------------------------
    # Unpack output config (was bundled into the display config pre-refactor). Default-construct
    # if the caller didn't pass one - keeps train_eval.py:_run_v2_path callers
    # that haven't migrated yet working with empty paths.
    # ---------------------------------------------------------------------------
    if output is None:
        output = OutputConfig()
    plot_file = output.plot_file
    data_dir = output.data_dir
    models_subdir = output.models_dir

    # ---------------------------------------------------------------------------
    # Unpack naming config
    # ---------------------------------------------------------------------------
    model_name = naming.model_name
    model_name_prefix = naming.model_name_prefix

    # ---------------------------------------------------------------------------
    # Unpack confidence config
    # ---------------------------------------------------------------------------
    include_confidence_analysis = confidence.include
    confidence_analysis_use_shap = confidence.use_shap
    confidence_analysis_max_features = confidence.max_features
    confidence_analysis_cmap = confidence.cmap
    confidence_analysis_alpha = confidence.alpha
    confidence_analysis_ylabel = confidence.ylabel
    confidence_analysis_title = confidence.title
    confidence_model_kwargs = dict(confidence.model_kwargs) if confidence.model_kwargs else {}

    # ---------------------------------------------------------------------------
    # Unpack predictions container
    # ---------------------------------------------------------------------------
    train_preds = predictions.train_preds
    train_probs = predictions.train_probs
    val_preds = predictions.val_preds
    val_probs = predictions.val_probs
    test_preds = predictions.test_preds
    test_probs = predictions.test_probs

    # ---------------------------------------------------------------------------
    # Begin original function logic
    # ---------------------------------------------------------------------------
    _maybe_clean_ram()

    columns = []
    best_iter = None

    _orig_train_df = train_df
    _orig_val_df = val_df
    _orig_test_df = test_df

    real_drop_columns = _validate_infinity_and_columns(
        df=df,
        train_df=train_df,
        skip_infinity_checks=skip_infinity_checks,
        drop_columns=drop_columns,
    )

    if not custom_ice_metric:
        custom_ice_metric = partial(compute_probabilistic_multiclass_error, nbins=nbins)

    model_obj, model_type_name, model_name, plot_file, model_file_name = _setup_model_info_and_paths(
        model=model,
        model_name=model_name,
        model_name_prefix=model_name_prefix,
        plot_file=plot_file,
        data_dir=data_dir,
        models_subdir=models_subdir,
    )

    if use_cache and exists(model_file_name):
        logger.info("Loading model from file %s", model_file_name)
        # Security: only load pickles from an explicitly trusted directory root.
        # Default `trusted_root` to the model file's parent dir when not provided,
        # preserving backward compat for in-process trained-then-loaded flows.
        _root = trusted_root if trusted_root is not None else os.path.dirname(os.path.abspath(model_file_name))
        _validate_trusted_path(model_file_name, _root)
        try:
            model, *_, pre_pipeline = joblib.load(model_file_name)
        except (EOFError, OSError, ModuleNotFoundError, pickle.UnpicklingError, AttributeError) as e:
            logger.warning(f"Failed to load cached model from {model_file_name}: {e}. Will retrain instead.")
            # Continue to training - model remains as originally passed

    if fit_params is None:
        fit_params = {}
    else:
        fit_params = copy.copy(fit_params)

    train_target, val_target, test_target = _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target)

    if (df is not None) or (train_df is not None):
        if train_df is None:
            train_df = _subset_dataframe(df, train_idx, real_drop_columns)
        if val_df is None and val_idx is not None:
            val_df = _subset_dataframe(df, val_idx, real_drop_columns)

    # Decategorise float-typed pandas categorical columns BEFORE the
    # pre_pipeline runs (RFECV inner CB / XGB inside the pre_pipeline
    # would otherwise reject them -- fuzz c0102, see helper docstring).
    train_df, val_df, test_df = _decategorise_float_cat_columns(
        train_df,
        val_df=val_df,
        test_df=test_df,
    )

    train_df, val_df = _apply_pre_pipeline_transforms(
        model=model,
        pre_pipeline=pre_pipeline,
        train_df=train_df,
        val_df=val_df,
        train_target=train_target,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        use_cache=use_cache,
        model_file_name=model_file_name,
        verbose=verbose,
        selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
    )

    # Check if feature selection removed all features
    if train_df is not None and train_df.shape[1] == 0:
        logger.warning(
            f"Feature selection removed all features for {model_name} - skipping training. "
            "This typically means no features had predictive power for the target."
        )
        return (
            SimpleNamespace(
                model=None,
                test_preds=None,
                test_probs=None,
                test_target=None,
                val_preds=None,
                val_probs=None,
                val_target=None,
                train_preds=None,
                train_probs=None,
                train_target=None,
                metrics={"train": {}, "val": {}, "test": {}, "best_iter": None},
                columns=[],
                pre_pipeline=pre_pipeline,
                train_od_idx=train_od_idx,
                val_od_idx=val_od_idx,
                trainset_features_stats=trainset_features_stats,
            ),
            None,
            None,
            None,
        )

    if model is not None and pre_pipeline and not skip_pre_pipeline_transform:
        _orig_train_df = train_df
        if val_df is not None:
            _orig_val_df = val_df

    if val_df is not None:
        if isinstance(val_target, pl.Series):
            val_target = val_target.to_numpy()

        _setup_eval_set(model_type_name, fit_params, val_df, val_target, callback_params, model_obj, model_category)
        _maybe_clean_ram()
    else:
        _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj)

    if model is not None and fit_params:
        _filter_categorical_features(fit_params, train_df, val_df=val_df, test_df=test_df)

    if model is not None:
        if (not use_cache) or (not exists(model_file_name)):
            _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params)
            if verbose:
                logger.info("training dataset shape: %s", train_df.shape)

            if display_sample_size:
                ipython_display(train_df.head(display_sample_size).style.set_caption(f"{model_name} features head"))
                ipython_display(train_df.tail(display_sample_size).style.set_caption(f"{model_name} features tail"))

            if train_df is not None:
                report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s)"
                if show_feature_names:
                    report_title += ": " + ", ".join(list(train_df.columns))
                report_title += f", {len(train_df):_} records"

            train_df, fit_params = _prepare_train_df_for_fitting(train_df, model, model_type_name, fit_params)

            _maybe_clean_ram()
            if verbose:
                logger.info("Training the model...")

            if isinstance(train_target, pl.Series):
                train_target = train_target.to_numpy()

            # Detect classification vs regression from the model type
            # name suffix (covers all four GBM backends + sklearn linear
            # + MultiOutputClassifier + ClassifierChain). Used by
            # ``_validate_target_values`` to flag single-class collapse
            # before the per-backend C++ crash.
            _is_clf = "Classifier" in model_type_name or model_type_name in ("ClassifierChain", "_ChainEnsemble")
            _validate_target_values(train_target, "train", is_classification=_is_clf)
            if val_target is not None:
                _validate_target_values(val_target, "val", is_classification=_is_clf)

            # XGB cat-category alignment (no-op for non-XGB models): align
            # the ``categories`` list across train / val / test so
            # val/test rows whose category wasn't seen in train don't
            # trip XGBoost's ``Found a category not in the training set``
            # rejection at predict time (fuzz seed=2024 c0060). Done AFTER
            # pre_pipeline so the alignment uses the actual cat layout
            # the model.fit + model.predict will see (pre_pipeline can
            # rename / re-cast cat columns; aligning before that would
            # be undone).
            train_df, val_df, test_df = _align_xgb_cat_categories(
                model_type_name,
                train_df,
                val_df=val_df,
                test_df=test_df,
            )

            if not just_evaluate:
                # 2026-05-13 (user request): nest Lightning checkpoints +
                # CSV logger output under the per-model directory
                # (``{dirname(model_file_name)}/{basename_no_ext}/``) so
                # different (target, model, schema_hash) combos don't
                # collide in a shared project-root ``logs/`` folder.
                # Only applies to TTR-wrapped Lightning regressors --
                # tree models ignore this attribute. Set on the inner
                # regressor (under TTR's ``.regressor``) when present,
                # falling back to the model itself for direct Lightning
                # regressors.
                try:
                    if model_file_name:
                        _ckpt_dir = os.path.splitext(model_file_name)[0]
                        _inner = getattr(model, "regressor", model)
                        if hasattr(_inner, "trainer_params"):
                            _inner.checkpoint_dir_override = _ckpt_dir
                except Exception:
                    pass
                model, best_iter = _train_model_with_fallback(
                    model=model,
                    model_obj=model_obj,
                    model_type_name=model_type_name,
                    train_df=train_df,
                    train_target=train_target,
                    fit_params=fit_params,
                    verbose=verbose,
                )

                # Handle failed model training (e.g., dtype incompatibility)
                if model is None:
                    logger.warning(f"Model {model_type_name} training failed - skipping evaluation")
                    return (
                        SimpleNamespace(
                            model=None,
                            test_preds=None,
                            test_probs=None,
                            test_target=None,
                            val_preds=None,
                            val_probs=None,
                            val_target=None,
                            train_preds=None,
                            train_probs=None,
                            train_target=None,
                            metrics={"train": {}, "val": {}, "test": {}, "best_iter": None},
                            columns=[],
                            pre_pipeline=pre_pipeline,
                            train_od_idx=train_od_idx,
                            val_od_idx=val_od_idx,
                            trainset_features_stats=trainset_features_stats,
                        ),
                        None,
                        None,
                        None,
                    )

            model_name = _update_model_name_after_training(model_name, len(train_df), train_details, best_iter)

    metrics = {"train": {}, "val": {}, "test": {}, "best_iter": best_iter}

    if compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics:
        t0_metrics = timer()
        if verbose:
            logger.info("Computing model's performance...")

        common_metrics_params = dict(
            model=model,
            model_type_name=model_type_name,
            model_name=model_name,
            group_ids=group_ids,
            target_label_encoder=target_label_encoder,
            figsize=figsize,
            nbins=nbins,
            print_report=print_report,
            plot_file=plot_file,
            show_perf_chart=show_perf_chart,
            show_fi=show_fi,
            fi_kwargs=fi_kwargs,
            subgroups=subgroups,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            n_features=n_features,
            show_prob_histogram=show_prob_histogram,
            prob_histogram_yscale=prob_histogram_yscale,
            show_inline_population_labels=show_inline_population_labels,
            title_metrics_tokens=title_metrics_tokens,
            plot_outputs=plot_outputs,
            plot_dpi=plot_dpi,
            multiclass_panels=multiclass_panels,
            multilabel_panels=multilabel_panels,
            ltr_panels=ltr_panels,
            quantile_panels=quantile_panels,
            quantile_alphas=quantile_alphas,
            # Authoritative target_type — gates auto_dispatch's
            # render_multi_target_panels so regression+group_ids doesn't
            # incorrectly render LTR/multilabel/multiclass panels.
            target_type=getattr(data, "target_type", None),
        )

        has_val = (val_idx is not None and len(val_idx) > 0) or val_df is not None
        has_test = (test_idx is not None and len(test_idx) > 0) or test_df is not None

        splits_config = [
            (
                "train",
                train_df,
                train_target,
                train_idx,
                train_preds,
                train_probs,
                train_details,
                compute_trainset_metrics and (train_idx is not None or train_df is not None),
            ),
            (
                "val",
                val_df,
                val_target,
                val_idx,
                val_preds,
                val_probs,
                val_details,
                compute_valset_metrics and ((val_idx is not None and len(val_idx) > 0) or val_df is not None),
            ),
        ]

        # Train runs sequentially (may feed into val/test setup); val+test parallelize later.
        for split_name, split_df, split_target, split_idx, split_preds, split_probs, split_details, should_compute in splits_config:
            if should_compute and split_name == "train":
                preds_result, probs_result, columns = _compute_split_metrics(
                    split_name=split_name,
                    df=split_df,
                    target=split_target,
                    idx=split_idx,
                    metrics_dict=metrics[split_name],
                    preds=split_preds,
                    probs=split_probs,
                    details=split_details,
                    has_other_splits=has_val or has_test,
                    **common_metrics_params,
                )
                train_preds, train_probs = preds_result, probs_result

        _val_cfg = next((c for c in splits_config if c[0] == "val" and c[-1]), None)
        _run_test = compute_testset_metrics and ((test_idx is not None and len(test_idx) > 0) or test_df is not None)

        if _run_test and ((df is not None) or (test_df is not None)):
            try:
                if train_df is not None:
                    del train_df
            except NameError:
                pass
            _maybe_clean_ram()

        if _run_test:
            test_df, test_target, columns = _prepare_test_split(
                df=df,
                test_df=test_df,
                test_idx=test_idx,
                test_target=test_target,
                target=target,
                real_drop_columns=real_drop_columns,
                model=model,
                pre_pipeline=pre_pipeline,
                skip_pre_pipeline_transform=skip_pre_pipeline_transform,
                skip_preprocessing=skip_preprocessing,
                selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
            )
            if test_df is not None:
                _orig_test_df = test_df

        # Parallelize val and test metric computation -- numba kernels release GIL,
        # Agg matplotlib is thread-safe. Pure-Python parts still block, but the
        # heavy cumtime (binning, AUC, calibration plot save) runs concurrently.
        def _run_val():
            if _val_cfg is None:
                return None
            _, sdf, starg, sidx, spreds, sprobs, sdet, _sc = _val_cfg
            return _compute_split_metrics(
                split_name="val",
                df=sdf,
                target=starg,
                idx=sidx,
                metrics_dict=metrics["val"],
                preds=spreds,
                probs=sprobs,
                details=sdet,
                has_other_splits=has_test,
                **common_metrics_params,
            )

        def _run_test_metrics():
            if not _run_test:
                return None
            return _compute_split_metrics(
                split_name="test",
                df=test_df,
                target=test_target,
                idx=test_idx,
                metrics_dict=metrics["test"],
                preds=test_preds,
                probs=test_probs,
                details=test_details,
                has_other_splits=False,
                **common_metrics_params,
            )

        # Note: concurrent ThreadPoolExecutor was tried but matplotlib figure creation
        # from concurrent threads races on pyplot's shared state even with Agg backend,
        # producing "Argument must be an image or collection" errors in calibration plots.
        # Sequential path is correct; the earlier _prepare_test_split refactor still stands.
        with phase("compute_split_metrics", split="val"):
            val_res = _run_val()
        with phase("compute_split_metrics", split="test"):
            test_res = _run_test_metrics()

        if val_res is not None:
            val_preds, val_probs, columns = val_res
        if test_res is not None:
            test_preds, test_probs, columns = test_res

        if _run_test:
            if include_confidence_analysis and test_df is not None:
                run_confidence_analysis(
                    test_df=test_df,
                    test_target=test_target,
                    test_probs=test_probs,
                    cat_features=fit_params.get("cat_features") if fit_params else None,
                    text_features=fit_params.get("text_features") if fit_params else None,
                    embedding_features=fit_params.get("embedding_features") if fit_params else None,
                    confidence_model_kwargs=confidence_model_kwargs,
                    fit_params=fit_params if model_type_name == "CatBoostRegressor" else None,
                    use_shap=confidence_analysis_use_shap,
                    max_features=confidence_analysis_max_features,
                    cmap=confidence_analysis_cmap,
                    alpha=confidence_analysis_alpha,
                    title=confidence_analysis_title,
                    ylabel=confidence_analysis_ylabel,
                    figsize=figsize,
                    verbose=verbose,
                )

    if (compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics) and verbose:
        logger.info("  Metrics computation done -- %.1fs", timer() - t0_metrics)

    _maybe_clean_ram()

    return (
        SimpleNamespace(
            model=model,
            test_preds=test_preds,
            test_probs=test_probs,
            test_target=test_target,
            val_preds=val_preds,
            val_probs=val_probs,
            val_target=val_target,
            train_preds=train_preds,
            train_probs=train_probs,
            train_target=train_target,
            metrics=metrics,
            columns=columns,
            pre_pipeline=pre_pipeline,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            trainset_features_stats=trainset_features_stats,
        ),
        _orig_train_df,
        _orig_val_df,
        _orig_test_df,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Configure Training Params Helper Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def _configure_xgboost_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_xgboost: bool,
    prefer_calibrated_classifiers: bool,
    use_flaml_zeroshot: bool,
    xgboost_verbose,
    metamodel_func,
):
    """Configure XGBoost model parameters.

    Goes through the ``_xgb_classifier_cls`` / ``_xgb_regressor_cls``
    factories so the DMatrix-reuse shim toggle (``USE_XGB_DMATRIX_REUSE_SHIM``,
    declared at module level) is the single switching point. To revert to
    vanilla XGBoost, see the docstring of ``USE_XGB_DMATRIX_REUSE_SHIM``.
    """
    xgb_configs = cpu_configs if prefer_cpu_for_xgboost else configs

    if use_regression:
        model_cls = _xgb_regressor_cls(use_flaml_zeroshot)
        model = metamodel_func(model_cls(**xgb_configs.XGB_GENERAL_PARAMS))
    else:
        model_cls = _xgb_classifier_cls(use_flaml_zeroshot)
        xgb_classif_params = xgb_configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else xgb_configs.XGB_GENERAL_CLASSIF
        model = model_cls(**xgb_classif_params)

    return dict(model=model, fit_params=dict(verbose=xgboost_verbose))


def _configure_lightgbm_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_lightgbm: bool,
    prefer_calibrated_classifiers: bool,
    use_flaml_zeroshot: bool,
    metamodel_func,
):
    """Configure LightGBM model parameters.

    Goes through the ``_lgb_classifier_cls`` / ``_lgb_regressor_cls``
    factories so the Dataset-reuse shim toggle (``USE_LGB_DATASET_REUSE_SHIM``,
    declared at module level) is the single switching point. To revert to
    vanilla LightGBM, see the docstring of ``USE_LGB_DATASET_REUSE_SHIM``.
    """
    lgb_configs = cpu_configs if prefer_cpu_for_lightgbm else configs

    if use_regression:
        model_cls = _lgb_regressor_cls(use_flaml_zeroshot)
        model = metamodel_func(model_cls(**lgb_configs.LGB_GENERAL_PARAMS))
        fit_params = {}
    else:
        model_cls = _lgb_classifier_cls(use_flaml_zeroshot)
        model = model_cls(**lgb_configs.LGB_GENERAL_PARAMS)
        fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    return dict(model=model, fit_params=fit_params)


def _configure_mlp_params(
    configs,
    config_params: dict,
    use_regression: bool,
    metamodel_func: callable,
    target_type=None,
) -> dict:
    """Configure MLP (PyTorch Lightning) model parameters.

    2026-05-07: when ``target_type`` is supplied (multiclass / multilabel),
    consult ``NeuralNetStrategy.get_classif_objective_kwargs`` for the
    correct loss_fn + labels_dtype + task_type. Falls back to legacy
    ``use_regression`` boolean for back-compat.
    """
    mlp_kwargs = config_params.get("mlp_kwargs", {})

    _arch_cls, _reg_cls, _cls_cls = _get_neural_components()
    if _arch_cls is None:
        raise ImportError(
            "MLP model requires the optional 'neural' extras "
            "(lightning + torchmetrics). Install via "
            "``pip install mlframe[neural]`` or omit ``mlp`` from mlframe_models."
        )
    # 2026-05-11 (user feedback): default architecture trimmed. Previous (nlayers=20, ratio=1.5) generated a 14-layer monster like 100->66->44->29->19->13->8->5->3->2->1->1->1 -- absurd funnel that collapses representational capacity to 1 neuron by mid-network. New defaults: nlayers=4 + ratio=2.0 -> 128->64->32->16->1, a classic shallow tabular MLP. Caller can still override via mlp_kwargs["network_params"] when a different topology is needed.
    #
    # 2026-05-13 (TVT-failure root cause): defaults switched to ZERO dropout +
    # NO batchnorm. The previous defaults (``dropout_prob=0.15`` +
    # ``inputs_dropout_prob=0.002`` + ``use_batchnorm=True``) catastrophically
    # killed the MLP on near-linear targets like TVT (y ~= 0.95 * TVT_prev +
    # tiny residual), where linear regression is the MLE estimator and the
    # MLP's job is to match it. Four hidden layers of dropout=0.15 means
    # ~52% of the signal (0.85^4) is destroyed on every forward pass; the
    # network simply cannot find the strong linear mapping. Production
    # symptom: 2-hour MLP run on 4M-row TVT collapsed predictions to a
    # narrow band [11k, 11.7k] around the mean, R2=0.33 vs linear R2=0.85.
    #
    # New defaults: dropout=0 + batchnorm=False. Tabular regression with
    # strong linear / additive signal does NOT benefit from dropout (none
    # of the big tabular libs -- CB / XGB / LGB -- use it either). Users
    # whose dataset is truly noise-dominated can opt in via
    # ``mlp_kwargs["network_params"]["dropout_prob"]=0.15``.
    mlp_network_params = dict(
        nlayers=4,
        first_layer_num_neurons=128,
        min_layer_neurons=16,
        neurons_by_layer_arch=_arch_cls.Declining,
        consec_layers_neurons_ratio=2.0,
        activation_function=torch.nn.LeakyReLU,
        weights_init_fcn=partial(nn.init.kaiming_normal_, nonlinearity="leaky_relu", a=0.01),
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_batchnorm=False,
    )
    if mlp_kwargs:
        mlp_network_params.update(mlp_kwargs.get("network_params", {}))

    mlp_general_params = configs.MLP_GENERAL_PARAMS.copy()
    if use_regression:
        mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
        mlp_general_params["model_params"]["loss_fn"] = F.mse_loss
        mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
        mlp_general_params["datamodule_params"]["labels_dtype"] = torch.float32
        mlp_model = _reg_cls(network_params=mlp_network_params, **mlp_general_params)
        # F1 fix (2026-05-11): auto-standardise the regression target for MLP. A kaiming-init network outputs ~0 at init; on a target with mean=11500 the network takes many epochs to learn just the constant offset.
        # F7 fix (2026-05-11): the initial F1 fix used sklearn's stock ``TransformedTargetRegressor`` which standardises ONLY the ``y`` arg of fit(), leaving ``eval_set=(X_val, y_val)`` unchanged. PyTorch-Lightning consumes ``eval_set`` for its val_dataloader and computes ``val_loss`` against RAW y_val while the model predicts on STANDARDISED scale -- train_loss=0.16 (std units) vs val_loss=1.3e+8 (raw units) gap observed in the 2026-05-12 run. New subclass intercepts ``eval_set`` in fit_kwargs and transforms its y component too, keeping train + val on the SAME scale so early-stop / val_MSE callbacks see meaningful numbers.
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler

        class _TTRWithEvalSetScaling(TransformedTargetRegressor):
            """``TransformedTargetRegressor`` extension that ALSO standardises any ``eval_set`` / ``X_val`` + ``y_val`` arrays in fit_params. Required for inner estimators (PyTorch-Lightning MLP, LightGBM, etc.) that consume eval_set for their own val-loss / early-stopping. Without this, train sees standardised y and val sees raw y, making the early-stop metric nonsensical."""

            def fit(self, X, y, **fit_params):
                # Fit the transformer FIRST on y so we have the same scale to apply to eval_set's y_val. Mirrors what ``TransformedTargetRegressor.fit`` does internally (line 167 in sklearn 1.5+).
                import numpy as _np
                from sklearn.base import clone as _clone

                y_arr = _np.asarray(y, dtype=_np.float64)
                if y_arr.ndim == 1:
                    y_arr_2d = y_arr.reshape(-1, 1)
                else:
                    y_arr_2d = y_arr
                self.transformer_ = _clone(self.transformer) if self.transformer is not None else None
                if self.transformer_ is not None:
                    self.transformer_.fit(y_arr_2d)
                    # Intercept + transform eval_set's y_val before delegating.
                    if "eval_set" in fit_params and fit_params["eval_set"] is not None:
                        es = fit_params["eval_set"]
                        # eval_set comes as ``(X_val, y_val)`` for MLP / LGB. For XGB / CB it's ``[(X_val, y_val), ...]``.
                        if isinstance(es, tuple) and len(es) == 2:
                            X_val, y_val = es
                            y_val_arr = _np.asarray(y_val, dtype=_np.float64)
                            y_val_2d = y_val_arr.reshape(-1, 1) if y_val_arr.ndim == 1 else y_val_arr
                            y_val_scaled = self.transformer_.transform(y_val_2d).reshape(y_val_arr.shape)
                            fit_params["eval_set"] = (X_val, y_val_scaled)
                        elif isinstance(es, list):
                            new_es = []
                            for entry in es:
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    X_v, y_v = entry
                                    y_v_arr = _np.asarray(y_v, dtype=_np.float64)
                                    y_v_2d = y_v_arr.reshape(-1, 1) if y_v_arr.ndim == 1 else y_v_arr
                                    y_v_scaled = self.transformer_.transform(y_v_2d).reshape(y_v_arr.shape)
                                    new_es.append((X_v, y_v_scaled))
                                else:
                                    new_es.append(entry)
                            fit_params["eval_set"] = new_es
                # Defer the actual fit to the parent (which will refit transformer + call regressor.fit).
                return super().fit(X, y, **fit_params)

        mlp_model = _TTRWithEvalSetScaling(regressor=mlp_model, transformer=StandardScaler())
    else:
        # 2026-05-07: target-type-aware loss / dtype / task_type for multi-*
        # classification. Strategy method returns the dispatch dict;
        # empty for binary (defaults already correct).
        if target_type is not None:
            from .strategies import NeuralNetStrategy
            from .configs import TargetTypes as _TT

            n_cls = 0  # not used by NeuralNetStrategy.get_classif_objective_kwargs
            mlp_obj = NeuralNetStrategy().get_classif_objective_kwargs(target_type, n_cls)
            if mlp_obj:
                mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
                if "loss_fn" in mlp_obj:
                    mlp_general_params["model_params"]["loss_fn"] = mlp_obj["loss_fn"]
                # task_type lands inside model_params and is consumed by
                # MLPTorchModel.__init__ to switch predict_step activation.
                if "task_type" in mlp_obj:
                    mlp_general_params["model_params"]["task_type"] = mlp_obj["task_type"]
                if "labels_dtype" in mlp_obj:
                    mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
                    mlp_general_params["datamodule_params"]["labels_dtype"] = mlp_obj["labels_dtype"]
        mlp_model = _cls_cls(network_params=mlp_network_params, **mlp_general_params)

    return dict(model=metamodel_func(mlp_model))


def _configure_recurrent_params(
    recurrent_models: List[str],
    recurrent_config: Optional[Any],
    sequences_train: Optional[List[np.ndarray]],
    features_train: Optional[Union[pd.DataFrame, np.ndarray]],
    use_regression: bool,
    metamodel_func: callable = None,
) -> Dict[str, dict]:
    """Configure recurrent model (LSTM, GRU, RNN, Transformer) parameters.

    Parameters
    ----------
    recurrent_models : list of str
        List of recurrent model types to configure (e.g., ["lstm", "gru"]).
    recurrent_config : RecurrentConfig or None
        Configuration for recurrent models. If None, uses defaults.
    sequences_train : list of np.ndarray or None
        Training sequences (variable length).
    features_train : DataFrame or np.ndarray or None
        Tabular features for HYBRID mode.
    use_regression : bool
        Whether to use regression (MSELoss) or classification (CrossEntropyLoss).
    metamodel_func : callable, optional
        Function to wrap the model (e.g., for calibration).

    Returns
    -------
    dict
        Dictionary mapping model names to their configurations.
    """
    from mlframe.training.neural import (
        RNNType,
        InputMode,
        RecurrentConfig,
        RecurrentClassifierWrapper,
        RecurrentRegressorWrapper,
    )

    if metamodel_func is None:

        def metamodel_func(x):
            return x

    # Determine input mode based on available data
    has_sequences = sequences_train is not None and len(sequences_train) > 0
    has_features = features_train is not None
    if hasattr(features_train, "shape"):
        has_features = has_features and features_train.shape[1] > 0

    if has_sequences and has_features:
        input_mode = InputMode.HYBRID
    elif has_sequences:
        input_mode = InputMode.SEQUENCE_ONLY
    else:
        input_mode = InputMode.FEATURES_ONLY

    # Use provided config or create default
    if recurrent_config is None:
        recurrent_config = RecurrentConfig()

    # Infer dimensions from data
    if has_sequences:
        seq_input_dim = sequences_train[0].shape[-1] if sequences_train[0].ndim > 1 else 1
    else:
        seq_input_dim = 0

    if has_features:
        if hasattr(features_train, "shape"):
            features_dim = features_train.shape[1]
        else:
            features_dim = len(features_train.columns)
    else:
        features_dim = 0

    result = {}

    for model_name in recurrent_models:
        model_name_lower = model_name.lower()

        # Map model name to RNNType
        rnn_type_map = {
            "lstm": RNNType.LSTM,
            "gru": RNNType.GRU,
            "rnn": RNNType.RNN,
            "transformer": RNNType.TRANSFORMER,
        }
        if model_name_lower not in rnn_type_map:
            raise ValueError(f"Unknown recurrent model type: {model_name}. " f"Supported: {list(rnn_type_map.keys())}")

        rnn_type = rnn_type_map[model_name_lower]

        # Create model-specific config.
        # 2026-04-24 (test_recurrent_lstm_smoke surfaced 4 bugs):
        #   * ``seq_input_dim`` / ``features_dim`` were passed as
        #     RecurrentConfig kwargs but the dataclass has no such
        #     fields. The wrapper computes both internally during
        #     ``fit`` from input shapes (see ``_RecurrentWrapperBase``
        #     in neural/recurrent.py:1041-1042: ``_aux_input_size``
        #     and ``_seq_input_size`` populated at fit-time). The
        #     dimensions captured at lines 3274-3284 above are now
        #     unused; left in place because future config-time
        #     validation may want them.
        #   * ``num_heads`` was a typo for ``n_heads`` (RecurrentConfig
        #     declares ``n_heads`` at neural/recurrent.py:170).
        #   * ``mlp_hidden_dims`` was a typo for ``mlp_hidden_sizes``
        #     (declared at neural/recurrent.py:174).
        # Without this fix every recurrent model (LSTM/GRU/RNN/
        # Transformer) crashes immediately with TypeError /
        # AttributeError on construction.
        config = RecurrentConfig(
            input_mode=input_mode,
            rnn_type=rnn_type,
            hidden_size=recurrent_config.hidden_size,
            num_layers=recurrent_config.num_layers,
            dropout=recurrent_config.dropout,
            bidirectional=recurrent_config.bidirectional,
            n_heads=recurrent_config.n_heads,
            use_attention=recurrent_config.use_attention,
            mlp_hidden_sizes=recurrent_config.mlp_hidden_sizes,
            num_classes=recurrent_config.num_classes,
            learning_rate=recurrent_config.learning_rate,
            weight_decay=recurrent_config.weight_decay,
            max_epochs=recurrent_config.max_epochs,
            batch_size=recurrent_config.batch_size,
            early_stopping_patience=recurrent_config.early_stopping_patience,
        )

        # Select wrapper class based on task type
        WrapperClass = RecurrentRegressorWrapper if use_regression else RecurrentClassifierWrapper
        wrapper = WrapperClass(config=config)

        result[model_name_lower] = dict(model=metamodel_func(wrapper))

    return result


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Configure Training Parameters
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def configure_training_params(
    df: pd.DataFrame = None,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    target: pd.Series = None,
    target_label_encoder: object = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    cat_features: list = None,
    text_features: list = None,
    embedding_features: list = None,
    fairness_features: Sequence = None,
    cont_nbins: int = 6,
    fairness_min_pop_cat_thresh: Union[float, int] = 1000,
    use_robust_eval_metric: bool = False,
    sample_weight: np.ndarray = None,
    prefer_gpu_configs: bool = True,
    nbins: int = 10,
    use_regression: bool = False,
    verbose: bool = True,
    rfecv_model_verbose: bool = True,
    prefer_cpu_for_lightgbm: bool = True,
    prefer_cpu_for_xgboost: bool = False,
    xgboost_verbose: Union[int, bool] = False,
    cb_fit_params: dict = None,
    prefer_calibrated_classifiers: bool = True,
    default_regression_scoring: dict = None,
    default_classification_scoring: dict = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: np.ndarray = None,
    model_name: str = "",
    common_params: dict = None,
    config_params: dict = None,
    metamodel_func: callable = None,
    use_flaml_zeroshot: bool = False,
    _precomputed_fairness_subgroups: dict = None,
    mlframe_models: list = None,
    linear_model_config: "LinearModelConfig" = None,
    callback_params: dict = None,
    train_df_size_bytes: Optional[float] = None,
    val_df_size_bytes: Optional[float] = None,
    target_type: Optional["TargetTypes"] = None,
    n_classes: Optional[int] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
):
    """Configure training parameters for all model types.

    Parameters
    ----------
    mlframe_models : list, optional
        List of model types to create. If None, all models are created.
        Used for lazy model creation to save memory.
    linear_model_config : LinearModelConfig, optional
        Configuration for linear models. If provided, applies shared settings
        to all linear model types.
    train_df_size_bytes : float, optional
        Precomputed RAM usage of train_df in bytes (e.g. from Polars
        ``.estimated_size()`` taken BEFORE pandas conversion). When
        provided, skips the pandas ``memory_usage`` call entirely. The
        value only feeds GPU-RAM-fit heuristics; Polars estimated_size
        is accurate enough and O(cols).
    val_df_size_bytes : float, optional
        Same as ``train_df_size_bytes`` for the validation split.
    """

    def _identity(x):
        return x

    # Helper for lazy model creation
    models_set = set(mlframe_models) if mlframe_models else None

    def _should_create_model(model_name: str) -> bool:
        """Check if a model should be created based on mlframe_models filter."""
        return models_set is None or model_name in models_set

    if metamodel_func is None:
        metamodel_func = _identity

    if default_regression_scoring is None:
        default_regression_scoring = dict(score_func=mean_absolute_error, response_method="predict", greater_is_better=False)

    if default_classification_scoring is None:
        default_classification_scoring = dict(score_func=fast_roc_auc, response_method="predict_proba", greater_is_better=True)

    if common_params is None:
        common_params = {}
    if config_params is None:
        config_params = {}
    if fairness_features is None:
        fairness_features = []
    if cb_fit_params is None:
        cb_fit_params = {}

    # ---- multilabel + post-hoc calibration safety gate ----
    # ``CalibratedClassifierCV`` is single-output only; combining it with a
    # MULTILABEL target silently fails inside the wrapper (label-list shape
    # mismatch deep in sklearn). Honour ``MultilabelDispatchConfig.
    # allow_uncalibrated_multi``: when False (default -- strict), refuse the
    # combo loudly so the misconfiguration is visible at config time; when
    # True, drop the calibration request with a warning and continue. No-op
    # when target is not multilabel or no MultilabelDispatchConfig was
    # supplied (legacy call path stays unchanged).
    if target_type is not None and prefer_calibrated_classifiers and multilabel_dispatch_config is not None:
        from .configs import TargetTypes as _TT

        if target_type == _TT.MULTILABEL_CLASSIFICATION:
            if not multilabel_dispatch_config.allow_uncalibrated_multi:
                raise NotImplementedError(
                    "prefer_calibrated_classifiers=True is incompatible with "
                    "MULTILABEL_CLASSIFICATION (CalibratedClassifierCV is "
                    "single-output only). Set MultilabelDispatchConfig."
                    "allow_uncalibrated_multi=True to drop calibration with a "
                    "warning instead of raising."
                )
            logger.warning(
                "Multilabel target + prefer_calibrated_classifiers=True; "
                "dropping calibration (MultilabelDispatchConfig."
                "allow_uncalibrated_multi=True). Trained models will be "
                "uncalibrated."
            )
            prefer_calibrated_classifiers = False

    # 2026-04-24 Session 6: route target_type/n_classes into get_training_configs
    # so the per-strategy classification dispatch (CB MultiLogloss, XGB
    # multi:softprob+num_class, LGB multiclass+num_class) gets injected.
    # Without this, multilabel targets reach CB without loss_function set,
    # and CB's _get_loss_function_for_train tries len(set(label)) on the 2-D
    # ndarray and crashes with TypeError: unhashable type: 'numpy.ndarray'.
    if target_type is not None and "target_type" not in config_params:
        config_params["target_type"] = target_type
    if n_classes is not None and "n_classes" not in config_params:
        config_params["n_classes"] = n_classes
    # 2026-05-08 perf: thread mlframe_models -> get_training_configs so
    # the MLP config block (and its ~14s pytorch / lightning import on
    # first call) is skipped when no neural model is requested.
    if mlframe_models is not None and "enabled_models" not in config_params:
        config_params["enabled_models"] = list(mlframe_models)

    if not use_regression:
        if "catboost_custom_classif_metrics" not in config_params:
            # Multi-output safe label count: 2-D multilabel uses n_columns;
            # 1-D binary/multiclass uses unique value count.
            target_arr = np.asarray(target) if target is not None else None
            # Multilabel detection: explicit 2-D, OR 1-D object dtype where
            # each cell is itself an array (the polars ``pl.List(pl.Int8)``
            # roundtrip lands here). Without the second clause,
            # ``np.unique(target_arr)`` raised ``truth value of array
            # ambiguous`` on the per-cell-array comparison surfaced fuzz
            # 3-way c0000 (cb / pandas / multilabel target).
            _is_object_of_arrays = False
            if target_arr is not None and target_arr.dtype == object and target_arr.ndim == 1 and target_arr.shape[0] > 0:
                _first = target_arr[0]
                _is_object_of_arrays = hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
            if target_arr is not None and target_arr.ndim == 2:
                nlabels = target_arr.shape[1] + 1  # treat as ">2" -> multiclass-style metrics
            elif _is_object_of_arrays:
                try:
                    _first = target_arr[0]
                    nlabels = (len(_first) if hasattr(_first, "__len__") else int(np.asarray(_first).size)) + 1
                except Exception:
                    nlabels = 3
            elif target_arr is not None:
                nlabels = len(np.unique(target_arr))
            else:
                nlabels = 2
            # When multilabel: AUC is incompatible with MultiLogloss (CB rejects
            # it at fit time). Skip the AUC/PRAUC defaults and let the per-strategy
            # multilabel dispatch in helpers.py pick a compatible eval_metric.
            if target_type is not None and getattr(target_type, "name", None) == "MULTILABEL_CLASSIFICATION":
                catboost_custom_classif_metrics = []
            elif nlabels > 2:
                catboost_custom_classif_metrics = ["AUC", "PRAUC:hints=skip_train~true"]
            else:
                catboost_custom_classif_metrics = ["AUC", "PRAUC:hints=skip_train~true", "BrierScore"]
            config_params["catboost_custom_classif_metrics"] = catboost_custom_classif_metrics

    subgroups = _precomputed_fairness_subgroups
    if subgroups is None and fairness_features:
        for next_df in (df, train_df):
            if next_df is not None:
                subgroups = create_fairness_subgroups(
                    next_df,
                    features=fairness_features,
                    cont_nbins=cont_nbins,
                    min_pop_cat_thresh=fairness_min_pop_cat_thresh,
                )
                break

    if use_robust_eval_metric and subgroups is not None:
        indexed_subgroups = create_fairness_subgroups_indices(
            subgroups=subgroups, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, group_weights={}, cont_nbins=cont_nbins
        )
    else:
        indexed_subgroups = None

    # 2026-04-27 Session 7 batch 6: per-section timers in
    # configure_training_params. Surfaced when a 9M-row prod run showed
    # ``select_target done in 18.2s`` vs ~0.5s on smaller earlier
    # runs. Three candidate hot-spots: get_training_configs (called
    # twice — CPU + GPU), get_df_memory_consumption(deep=False), and
    # the GPU probe (cached nvidia-smi subprocess). The timers below
    # localise the spend so the operator can see the breakdown
    # without instrumenting by hand.
    _t0_cfg = timer()
    cpu_configs = get_training_configs(has_gpu=False, subgroups=indexed_subgroups, **config_params)
    _t_cpu_cfg = timer() - _t0_cfg
    _t0_cfg = timer()
    gpu_configs = get_training_configs(has_gpu=None, subgroups=indexed_subgroups, **config_params)
    _t_gpu_cfg = timer() - _t0_cfg

    # Prefer caller-supplied size (typically computed on the Polars frame
    # BEFORE pandas conversion via .estimated_size() -- O(cols), microseconds).
    # Fall back to get_df_memory_consumption with deep=False -- O(cols) for
    # pandas too. Explicit deep=False avoids the O(rows) deep scan that used
    # to block this site for 3 minutes on frames with millions of unique
    # object-column strings. pyutilz default stays deep=True (back-compat);
    # mlframe opts out at this specific heuristic-only call site.
    _t0_mem = timer()
    if train_df_size_bytes is not None:
        train_df_size = float(train_df_size_bytes)
    else:
        train_df_size = get_df_memory_consumption(train_df, deep=False)
    if val_df_size_bytes is not None:
        val_df_size = float(val_df_size_bytes)
    elif val_df is not None:
        val_df_size = get_df_memory_consumption(val_df, deep=False)
    else:
        val_df_size = 0
    data_size_gb = (train_df_size + val_df_size) / (1024**3)
    _t_mem = timer() - _t0_mem

    # Skip expensive GPU probe (nvidia-smi subprocess ~0.5s) when GPU configs
    # are disabled or CatBoost is explicitly on CPU -- the result is unused.
    _t0_gpu = timer()
    cb_task_type = config_params.get("cb_kwargs", {}).get("task_type")
    cb_devices = config_params.get("cb_kwargs", {}).get("devices")
    if not prefer_gpu_configs or cb_task_type == "CPU":
        all_gpus = {}
        data_fits_gpu_ram = False
        data_fits_cb_gpu_ram = False
    else:
        all_gpus = _cached_gpu_info()
        single_gpu_limits = compute_total_gpus_ram(all_gpus)
        data_fits_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < single_gpu_limits.get("gpu_max_ram_total", 0)
        if cb_devices:
            multi_gpu_limits = compute_total_gpus_ram(parse_catboost_devices(cb_devices, all_gpus=all_gpus))
            data_fits_cb_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < multi_gpu_limits.get("gpus_ram_total", 0)
        else:
            data_fits_cb_gpu_ram = data_fits_gpu_ram
    _t_gpu = timer() - _t0_gpu

    logger.info("data_fits_gpu_ram=%s, data_fits_cb_gpu_ram=%s, cb_devices=%s", data_fits_gpu_ram, data_fits_cb_gpu_ram, cb_devices)
    if (_t_cpu_cfg + _t_gpu_cfg + _t_mem + _t_gpu) > 0.5:
        logger.info(
            "configure_training_params timing breakdown: " "cpu_configs=%.2fs, gpu_configs=%.2fs, mem_probe=%.2fs, gpu_probe=%.2fs (total %.2fs)",
            _t_cpu_cfg,
            _t_gpu_cfg,
            _t_mem,
            _t_gpu,
            _t_cpu_cfg + _t_gpu_cfg + _t_mem + _t_gpu,
        )

    configs = gpu_configs if (prefer_gpu_configs and data_fits_gpu_ram) else cpu_configs
    cb_configs = gpu_configs if (prefer_gpu_configs and data_fits_cb_gpu_ram) else cpu_configs

    common_params_result = dict(
        nbins=nbins,
        subgroups=subgroups,
        sample_weight=sample_weight,
        df=df,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        target=target,
        train_target=train_target,
        test_target=test_target,
        val_target=val_target,
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        custom_ice_metric=configs.integral_calibration_error,
        custom_rice_metric=configs.final_integral_calibration_error,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        group_ids=group_ids,
        model_name=model_name,
        callback_params=callback_params,
        # 2026-05-10: thread target_type through so the ensemble path
        # (score_ensemble -> _process_single_ensemble_method ->
        # _build_configs_from_params) can gate render_multi_target_panels
        # via DataConfig.target_type. Without this the ensemble report
        # block goes through report_model_perf with target_type=None and
        # auto_dispatch falls back to "fire LTR/multilabel/multiclass
        # panels for any target with group_ids set" — wrong on regression.
        target_type=str(target_type) if target_type is not None else None,
    )
    if common_params:
        common_params_result.update(common_params)
    common_params = common_params_result

    # Lazy model creation - only create models that are in mlframe_models (or all if None)
    cb_params = None
    if _should_create_model("cb"):
        if use_regression:
            _cb_model = metamodel_func(CatBoostRegressor(**cb_configs.CB_REGR))
        else:
            _cb_classif_params = cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF
            _cb_model = CatBoostClassifier(**_cb_classif_params)
        # Defensively pre-set the polars-fastpath sticky flag (2026-04-24).
        # Background: ``_predict_with_fallback`` lazily flips this attribute
        # to True after the FIRST polars-fastpath dispatch miss, so the
        # short-circuit fires only on the SECOND predict call onward.
        # That's fine for re-using a single fitted model (VAL -> TEST), but
        # in a suite each weight-schema iteration calls ``sklearn.clone()``
        # on this base ``_cb_model`` -- and clone strips non-param attrs,
        # giving every fresh CB instance a blank flag. The 2026-04-24 prod
        # log captured the symptom: CB uniform AND CB recency BOTH paid
        # the polars-miss + 2-3 s pandas-conversion roundtrip on their
        # first TEST predict, with WARN noise on each.
        # We know empirically (every prod run since 2026-04-19) that CB
        # 1.2.x's ``_set_features_order_data_polars_categorical_column``
        # has dispatch gaps on our nullable-Categorical / Enum schema, so
        # opting CB into pandas at predict time is bestthing -- bypasses
        # the doomed retry on success, costs nothing on failure. Set on
        # the base instance so ``clone()`` carries the param-equivalent
        # state forward (sklearn.clone preserves ``get_params()`` keys;
        # for the attr to survive clone we re-assert it inside
        # ``train_eval.py:process_model``'s clone call too -- but writing
        # it here is the ergonomic source of truth).
        try:
            _cb_model._mlframe_polars_fastpath_broken = True
        except Exception:
            # CB Python class is permissive about attributes; slot-only
            # forks could refuse -- degrade to "pay first-call retry".
            pass
        cb_params = dict(
            model=_cb_model,
            fit_params=dict(
                plot=verbose,
                cat_features=cat_features,
                **({"text_features": text_features} if text_features else {}),
                **({"embedding_features": embedding_features} if embedding_features else {}),
                **cb_fit_params,
            ),
        )

    # 2026-04-24 Session 6: per-strategy multilabel-wrap helper. Strategies
    # without native (N, K) target support (HGB, XGB-via-MultiOutputClassifier,
    # LGB, Linear) need MultiOutputClassifier when target is multilabel.
    # Inner-estimator early_stopping that depends on eval_set must be disabled
    # because the outer wrapper doesn't slice eval_set per label -- without
    # an eval_set the inner fit would crash ("at least one dataset and eval
    # metric is required for evaluation").
    def _wrap_for_multilabel_if_needed(estimator, strategy_cls):
        if use_regression or target_type is None or not hasattr(target_type, "name") or target_type.name != "MULTILABEL_CLASSIFICATION":
            return estimator
        # Disable eval_set-dependent early stopping on the inner estimator.
        try:
            params = estimator.get_params()
        except Exception:
            params = {}
        _patch = {}
        if "early_stopping_rounds" in params and params.get("early_stopping_rounds") is not None:
            _patch["early_stopping_rounds"] = None
        # XGB sklearn >=2 uses callbacks for early stopping too; strip them.
        if "callbacks" in params and params.get("callbacks"):
            _patch["callbacks"] = None
        if _patch:
            try:
                estimator.set_params(**_patch)
            except Exception:
                pass
        return strategy_cls().wrap_multilabel(
            estimator,
            target_type,
            multilabel_config=multilabel_dispatch_config,
            n_labels=n_classes,
        )

    hgb_params = None
    if _should_create_model("hgb"):
        from .strategies import HGBStrategy as _HGBS

        _hgb_est = (
            HistGradientBoostingRegressor(**configs.HGB_GENERAL_PARAMS)
            if use_regression
            else _wrap_for_multilabel_if_needed(
                HistGradientBoostingClassifier(**configs.HGB_GENERAL_PARAMS),
                _HGBS,
            )
        )
        hgb_params = dict(model=metamodel_func(_hgb_est))

    xgb_params = None
    if _should_create_model("xgb"):
        xgb_params = _configure_xgboost_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=use_regression,
            prefer_cpu_for_xgboost=prefer_cpu_for_xgboost,
            prefer_calibrated_classifiers=prefer_calibrated_classifiers,
            use_flaml_zeroshot=use_flaml_zeroshot,
            xgboost_verbose=xgboost_verbose,
            metamodel_func=metamodel_func,
        )
        # XGB sklearn wrapper rejects 2-D y unless we use multi_strategy='multi_output_tree'
        # (WIP in 3.x). Default to MultiOutputClassifier per Session-6 design.
        from .strategies import XGBoostStrategy as _XGBS

        xgb_params["model"] = _wrap_for_multilabel_if_needed(xgb_params["model"], _XGBS)

    lgb_params = None
    if _should_create_model("lgb"):
        lgb_params = _configure_lightgbm_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=use_regression,
            prefer_cpu_for_lightgbm=prefer_cpu_for_lightgbm,
            prefer_calibrated_classifiers=prefer_calibrated_classifiers,
            use_flaml_zeroshot=use_flaml_zeroshot,
            metamodel_func=metamodel_func,
        )
        # LGB has no native multilabel -- wrap with MultiOutputClassifier.
        from .strategies import TreeModelStrategy as _LGBS

        lgb_params["model"] = _wrap_for_multilabel_if_needed(lgb_params["model"], _LGBS)

    mlp_params = None
    if _should_create_model("mlp"):
        mlp_params = _configure_mlp_params(
            configs=configs,
            config_params=config_params,
            use_regression=use_regression,
            metamodel_func=metamodel_func,
            target_type=target_type,
        )

    ngb_params = None
    if _should_create_model("ngb"):
        # 2026-05-07: target-type-aware Dist for NGBClassifier.
        # Default ``Dist=Bernoulli`` (binary only) crashes on K>2 with
        # ``IndexError: index out of bounds``. For multiclass we need
        # ``Dist=k_categorical(K)``. NGBoost has no native multilabel /
        # ranker, so those target types fall through to the default
        # (likely with a downstream error if reached -- they should be
        # filtered earlier when the suite checks per-strategy multilabel
        # / ranking flags).
        ngb_init_kwargs = dict(configs.NGB_GENERAL_PARAMS)
        from .configs import TargetTypes as _TT

        if not use_regression and target_type == _TT.MULTICLASS_CLASSIFICATION:
            try:
                from ngboost.distns import k_categorical

                # n_classes pulled from the actual y -- NGB needs the
                # exact K to size the categorical Dist's internal
                # parameter array. Fall back to inspecting train_target
                # via config_params (where train_target lives at this
                # call layer).
                _train_target = config_params.get("train_target")
                if _train_target is not None:
                    _y = np.asarray(_train_target).ravel()
                    _K = int(_y.max()) + 1 if len(_y) else 2
                else:
                    _K = max(2, int(config_params.get("n_classes", 2)))
                ngb_init_kwargs["Dist"] = k_categorical(_K)
            except ImportError:
                pass  # ngboost.distns missing -> default Dist crashes loudly downstream

        ngb_params = dict(
            model=(
                metamodel_func(
                    (NGBRegressor(**ngb_init_kwargs) if use_regression else NGBClassifier(**ngb_init_kwargs)),
                )
            ),
            fit_params=({} if config_params.get("early_stopping_rounds") is None else dict(early_stopping_rounds=config_params.get("early_stopping_rounds"))),
        )

    # Linear models - only create variants that are needed
    linear_model_params = {}
    linear_models_needed = LINEAR_MODEL_TYPES & models_set if models_set else LINEAR_MODEL_TYPES
    # Keys that have incompatible meanings between tree and linear models
    # (e.g., learning_rate is float for trees but string schedule for linear SGD)
    linear_config_excluded_keys = {"learning_rate"}
    for model_type in linear_models_needed:
        # Build config by merging: config_params -> linear_model_config -> model_type
        # This allows config_params_override["iterations"] to work for linear models
        linear_config_kwargs = {"model_type": model_type}
        # Apply config_params first (includes iterations from config_params_override)
        if config_params:
            # Only include keys that LinearModelConfig recognizes
            linear_config_fields = set(LinearModelConfig.model_fields.keys()) - linear_config_excluded_keys
            # Also include 'iterations' which gets mapped to max_iter by the validator
            linear_config_fields.add("iterations")
            for key, value in config_params.items():
                if key in linear_config_fields:
                    linear_config_kwargs[key] = value
        # Override with explicit linear_model_config if provided
        if linear_model_config:
            linear_config_kwargs.update(linear_model_config.model_dump(exclude={"model_type"}))
        config = LinearModelConfig(**linear_config_kwargs)
        _linear_est = create_linear_model(model_type, config, use_regression=use_regression)
        # Linear classifiers reject 2-D y -> MultiOutputClassifier wrapper for multilabel.
        from .strategies import LinearModelStrategy as _LMS

        _linear_est = _wrap_for_multilabel_if_needed(_linear_est, _LMS)
        linear_model_params[model_type] = dict(model=metamodel_func(_linear_est))

    # Get individual params (may be None if not in mlframe_models)
    linear_params = linear_model_params.get("linear")
    ridge_params = linear_model_params.get("ridge")
    lasso_params = linear_model_params.get("lasso")
    elasticnet_params = linear_model_params.get("elasticnet")
    huber_params = linear_model_params.get("huber")
    ransac_params = linear_model_params.get("ransac")
    sgd_params = linear_model_params.get("sgd")

    # RFECV setup
    rfecv_params = configs.COMMON_RFECV_PARAMS.copy()
    cb_rfecv_params = cb_configs.COMMON_RFECV_PARAMS.copy()

    if not common_params.get("show_perf_chart", True):
        rfecv_params["optimizer_plotting"] = "No"
        cb_rfecv_params["optimizer_plotting"] = "No"

    if "rfecv_params" in common_params:
        custom_rfecv_params = common_params.pop("rfecv_params")
        rfecv_params.update(custom_rfecv_params)
        cb_rfecv_params.update(custom_rfecv_params)

    if use_regression:
        rfecv_scoring = make_scorer(**default_regression_scoring)
    else:
        if prefer_calibrated_classifiers:

            def fs_and_hpt_integral_calibration_error(*args, **kwargs):
                return configs.fs_and_hpt_integral_calibration_error(*args, **kwargs, verbose=rfecv_model_verbose)

            rfecv_scoring = make_scorer(
                score_func=fs_and_hpt_integral_calibration_error,
                response_method="predict_proba",
                greater_is_better=False,
            )
        else:
            rfecv_scoring = make_scorer(**default_classification_scoring)

    params = (cb_configs.CB_REGR if use_regression else (cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF)).copy()

    cb_rfecv = RFECV(
        estimator=(metamodel_func(CatBoostRegressor(**params)) if use_regression else CatBoostClassifier(**params)),
        fit_params=dict(plot=rfecv_model_verbose > 1),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **cb_rfecv_params,
    )

    lgb_fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    lgb_rfecv = RFECV(
        estimator=(
            metamodel_func((flaml_zeroshot.LGBMRegressor if use_flaml_zeroshot else LGBMRegressor)(**configs.LGB_GENERAL_PARAMS))
            if use_regression
            else (flaml_zeroshot.LGBMClassifier if use_flaml_zeroshot else LGBMClassifier)(**configs.LGB_GENERAL_PARAMS)
        ),
        fit_params=lgb_fit_params,
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    xgb_rfecv = RFECV(
        estimator=(
            metamodel_func((flaml_zeroshot.XGBRegressor if use_flaml_zeroshot else XGBRegressor)(**configs.XGB_GENERAL_PARAMS))
            if use_regression
            else (flaml_zeroshot.XGBClassifier if use_flaml_zeroshot else XGBClassifier)(
                **(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF)
            )
        ),
        fit_params=dict(verbose=False),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    # Build models_params dict, only including models that were created
    models_params = {}
    if cb_params is not None:
        models_params["cb"] = cb_params
    if lgb_params is not None:
        models_params["lgb"] = lgb_params
    if xgb_params is not None:
        models_params["xgb"] = xgb_params
    if hgb_params is not None:
        models_params["hgb"] = hgb_params
    if mlp_params is not None:
        models_params["mlp"] = mlp_params
    if ngb_params is not None:
        models_params["ngb"] = ngb_params
    # Add linear models (already filtered to only needed ones)
    models_params.update(linear_model_params)

    return (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    )


__all__ = [
    "train_and_evaluate_model",
    "configure_training_params",
    "_build_configs_from_params",
    "run_confidence_analysis",
]
