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
except ImportError:  # pragma: no cover — optional backend
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

# Optional model backends — lazy/tolerant of missing deps, matching __init__.py style.
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
    object has no setter`` — aborting the run 5 seconds in.

    The primary fix is Fix 1 (ensure LGB receives pandas → sklearn path
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
                if not (
                    mod.startswith("catboost.")
                    or mod.startswith("xgboost.")
                    or mod.startswith("lightgbm.")
                ):
                    return f"{mod}:{frame.f_lineno}"
                frame = frame.f_back
            return (
                f"{frame.f_globals.get('__name__', '?')}:{frame.f_lineno}"
                if frame
                else "?"
            )
        except Exception:
            return "?"

    def _wrap_init(cls, label: str):
        if cls is None:
            return
        # Check the marker on ``cls.__dict__`` specifically — a subclass
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
                logger.info(
                    f"[dataset-build] {label} shape={shape_str} took={elapsed:.3f}s site={callsite}"
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
try:
    from ngboost import NGBClassifier, NGBRegressor
except ImportError:  # pragma: no cover
    NGBClassifier = NGBRegressor = None  # type: ignore[assignment]
try:
    import flaml.default as flaml_zeroshot
except ImportError:  # pragma: no cover
    flaml_zeroshot = None  # type: ignore[assignment]
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
try:
    from mlframe.training.neural import MLPNeuronsByLayerArchitecture
    from mlframe.training.neural import PytorchLightningRegressor, PytorchLightningClassifier
except ImportError:  # pragma: no cover
    MLPNeuronsByLayerArchitecture = None  # type: ignore[assignment]
    PytorchLightningRegressor = PytorchLightningClassifier = None  # type: ignore[assignment]

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
    DisplayConfig,
    NamingConfig,
    ConfidenceAnalysisConfig,
    PredictionsContainer,
    LinearModelConfig,
)
from .utils import log_ram_usage, get_categorical_columns, get_numeric_columns, filter_existing


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
        return target.iloc[idx]
    elif isinstance(target, pl.Series):
        return target.gather(idx)
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


def _validate_target_values(target, subset_name="train"):
    """Check target for NaN and infinity values before training."""
    arr = target.values if isinstance(target, pd.Series) else target
    try:
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
    except TypeError:
        return  # non-numeric target (e.g., categorical), skip check
    if nan_count > 0 or inf_count > 0:
        parts = []
        if nan_count > 0:
            parts.append(f"{nan_count:_} NaN")
        if inf_count > 0:
            parts.append(f"{inf_count:_} infinity")
        raise ValueError(
            f"{subset_name} target contains {' and '.join(parts)} value(s). "
            f"Clean the target before training."
        )


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


def _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target):
    """Extract train/val/test targets from main target using indices."""
    if target is not None:
        if train_target is None and (train_idx is not None):
            train_target = _extract_target_subset(target, train_idx)
        if val_target is None and (val_idx is not None):
            val_target = _extract_target_subset(target, val_idx)
        if test_target is None and (test_idx is not None):
            test_target = _extract_target_subset(target, test_idx)
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
        print(f"es_best_iter: {best_iter:_}")
        model_name = model_name + f" @iter={best_iter:_}"

    return model_name


def _prepare_test_split(
    df, test_df, test_idx, test_target, target, real_drop_columns, model, pre_pipeline, skip_pre_pipeline_transform, skip_preprocessing=False,
    selector_passthrough_cols=None,
):
    """Prepare test DataFrame and target for evaluation."""
    if (df is not None) or (test_df is not None):
        if test_df is None:
            test_df = _subset_dataframe(df, test_idx, real_drop_columns)

        if test_target is None:
            test_target = _extract_target_subset(target, test_idx)

        if model is not None and pre_pipeline and not skip_pre_pipeline_transform:
            if skip_preprocessing:
                feature_selector = _extract_feature_selector(pre_pipeline)
                if feature_selector is not None:
                    test_df = _passthrough_cols_fit_transform(
                        feature_selector.transform, test_df, passthrough_cols=selector_passthrough_cols,
                    )
            else:
                test_df = _passthrough_cols_fit_transform(
                    pre_pipeline.transform, test_df, passthrough_cols=selector_passthrough_cols,
                )
        columns = list(test_df.columns) if hasattr(test_df, "columns") else []
    else:
        columns = []
        test_df = None

    return test_df, test_target, columns


def _extract_feature_selector(pre_pipeline):
    """Extract the feature selector ('pre' step) from a sklearn Pipeline.

    Feature selectors are added as the 'pre' step in pipelines built by
    ModelPipelineStrategy.build_pipeline() in strategies.py.

    Args:
        pre_pipeline: The preprocessing pipeline (sklearn Pipeline or transformer)

    Returns:
        The feature selector if found, otherwise None
    """
    if pre_pipeline is None:
        return None
    # If it's a Pipeline with named steps, look for the 'pre' step
    if hasattr(pre_pipeline, "named_steps") and "pre" in pre_pipeline.named_steps:
        return pre_pipeline.named_steps["pre"]
    # If it's not a Pipeline, it might be the feature selector itself (e.g., MRMR, RFECV)
    if not isinstance(pre_pipeline, Pipeline):
        return pre_pipeline
    return None


def _is_fitted(estimator):
    """Check if an sklearn estimator is already fitted.

    Uses sklearn's check_is_fitted() to determine if the estimator has been
    fitted. This is useful for determining whether to call fit_transform()
    or just transform() on a pipeline/feature selector that may have been
    loaded from cache.

    Args:
        estimator: An sklearn-compatible estimator (Pipeline, RFECV, etc.)

    Returns:
        bool: True if the estimator is fitted, False otherwise
    """
    if estimator is None:
        return False
    # For a sklearn Pipeline, ``check_is_fitted`` passes as long as ANY step
    # has fitted state — even if later steps are still unfitted. That bit
    # us on 2026-04-22 (fuzz c0031): LinearModelStrategy's pre_pipeline had
    # a fitted MRMR step (reused from a prior CB iteration) but un-fitted
    # encoder/imputer/scaler. _is_fitted returned True → code took the
    # ".transform only" branch → imputer.transform raised ValueError 'The
    # feature names should match those that were passed during fit'.
    # Require every non-trivial step to be fitted.
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(estimator, Pipeline):
            for _name, step in estimator.steps:
                if step is None or step == "passthrough":
                    continue
                try:
                    check_is_fitted(step)
                except NotFittedError:
                    return False
            return True
    except Exception:
        pass
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False


def _passthrough_cols_fit_transform(fn, df, *args, passthrough_cols=None, fit=False, target=None):
    """Run a selector fit/transform on df with passthrough_cols hidden, then re-attach them.

    Feature selectors (MRMR, RFECV) can't encode text or list-of-float embedding columns;
    catboost needs them back intact for fit. Hide → run → re-attach preserves both.

    Numpy-output fallback (2026-04-22): if the inner ``fn`` is a default sklearn
    Pipeline (no ``set_output(transform="pandas")``), ``out`` comes back as a numpy
    array. The original code detected this via ``hasattr(out, "columns")`` and
    silently returned numpy — dropping ``passthrough_cols`` and, worse, collapsing
    pd.Categorical dtypes in the selected columns to numpy object strings, which
    crashes LGB's Dataset construction on the ``'HOURLY'`` path. We now rebuild a
    pd.DataFrame from the reduced-input column names so passthrough_cols re-attach
    and downstream models take the native-pandas fastpath.
    """
    if not passthrough_cols or df is None or not hasattr(df, "columns"):
        return fn(df, target) if fit else fn(df)
    present = [c for c in passthrough_cols if c in df.columns]
    if not present:
        return fn(df, target) if fit else fn(df)
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        held = df.select(present)
        reduced = df.drop(present)
    else:
        held = df[present]
        reduced = df.drop(columns=present)
    out = fn(reduced, target) if fit else fn(reduced)
    if hasattr(out, "columns"):
        if isinstance(out, pl.DataFrame):
            out = out.with_columns([held[c] for c in present])
        else:
            held_pd = held.to_pandas() if is_polars else held
            for c in present:
                out[c] = held_pd[c].values if hasattr(held_pd[c], "values") else held_pd[c]
    elif isinstance(out, np.ndarray) and out.ndim == 2:
        # Reconstruct a DataFrame using the reduced-input column names when the
        # transformer preserved the column count. If the shape differs (e.g. a
        # feature selector dropped columns), we can't safely name them — fall
        # back to positional names and warn via debug log.
        reduced_cols = list(reduced.columns)
        if out.shape[1] == len(reduced_cols):
            col_names = reduced_cols
        else:
            col_names = [f"f{i}" for i in range(out.shape[1])]
        out = pd.DataFrame(out, columns=col_names, index=getattr(reduced, "index", None))
        held_pd = held.to_pandas() if is_polars else held
        for c in present:
            out[c] = held_pd[c].values if hasattr(held_pd[c], "values") else held_pd[c]
    return out


def _apply_pre_pipeline_transforms(
    model, pre_pipeline, train_df, val_df, train_target, skip_pre_pipeline_transform, skip_preprocessing, use_cache, model_file_name, verbose,
    selector_passthrough_cols=None,
):
    """Apply pre-pipeline transformations to train and validation DataFrames.

    Args:
        model: The model being trained
        pre_pipeline: Preprocessing pipeline (may include feature selector + preprocessing steps)
        train_df: Training DataFrame
        val_df: Validation DataFrame (or None)
        train_target: Training target values
        skip_pre_pipeline_transform: If True, skip entire pipeline (for cached DFs)
        skip_preprocessing: If True, skip only preprocessing steps but run feature selectors
        use_cache: Whether to use cached pipeline
        model_file_name: Model file path for cache checking
        verbose: Verbosity level
    """
    if model is not None and pre_pipeline:
        t0_pre = timer()
        with phase("pre_pipeline_fit_transform"):
            if skip_pre_pipeline_transform:
                if verbose:
                    logger.info("Skipping pre_pipeline fit/transform (using cached DFs)")
            elif skip_preprocessing:
                # Only run feature selector, skip preprocessing steps (scaler/imputer/encoder)
                # This is used when polars-ds pipeline already applied scaling/imputation
                feature_selector = _extract_feature_selector(pre_pipeline)
                if feature_selector is not None:
                    if _is_fitted(feature_selector):
                        if verbose:
                            logger.info(f"Using pre-fitted feature selector (transform only): {feature_selector}")
                        train_df = _passthrough_cols_fit_transform(
                            feature_selector.transform, train_df,
                            passthrough_cols=selector_passthrough_cols,
                        )
                    else:
                        if verbose:
                            logger.info(f"Fitting feature selector: {feature_selector}")
                        train_df = _passthrough_cols_fit_transform(
                            feature_selector.fit_transform, train_df,
                            passthrough_cols=selector_passthrough_cols, fit=True, target=train_target,
                        )
                    if verbose:
                        log_ram_usage()
                    if val_df is not None:
                        if verbose:
                            logger.info(f"Transforming val_df via feature selector...")
                        val_df = _passthrough_cols_fit_transform(
                            feature_selector.transform, val_df,
                            passthrough_cols=selector_passthrough_cols,
                        )
                        if verbose:
                            log_ram_usage()
                elif verbose:
                    logger.info("No feature selector found in pipeline, skipping all transforms")
            elif _is_fitted(pre_pipeline):
                if verbose:
                    try:
                        logger.info(f"Using pre-fitted pipeline (transform only): {pre_pipeline}")
                    except (ValueError, TypeError):
                        pass
                train_df = _passthrough_cols_fit_transform(
                    pre_pipeline.transform, train_df, passthrough_cols=selector_passthrough_cols,
                )
                if verbose:
                    log_ram_usage()
                if val_df is not None:
                    if verbose:
                        logger.info(f"Transforming val_df via pre_pipeline...")
                    val_df = _passthrough_cols_fit_transform(
                        pre_pipeline.transform, val_df, passthrough_cols=selector_passthrough_cols,
                    )
                    if verbose:
                        log_ram_usage()
            else:
                if verbose:
                    logger.info(f"Fitting & transforming train_df via pre_pipeline {pre_pipeline}...")
                train_df = _passthrough_cols_fit_transform(
                    pre_pipeline.fit_transform, train_df,
                    passthrough_cols=selector_passthrough_cols, fit=True, target=train_target,
                )
                if verbose:
                    log_ram_usage()
                if val_df is not None:
                    if verbose:
                        logger.info(f"Transforming val_df via pre_pipeline {pre_pipeline}...")
                    val_df = _passthrough_cols_fit_transform(
                        pre_pipeline.transform, val_df, passthrough_cols=selector_passthrough_cols,
                    )
                    if verbose:
                        log_ram_usage()
            _maybe_clean_ram()
            if verbose:
                shape_str = f"{train_df.shape[0]:_}×{train_df.shape[1]}" if hasattr(train_df, "shape") else ""
                logger.info(f"  pre_pipeline done — train: {shape_str}, {timer() - t0_pre:.1f}s")

    return train_df, val_df


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


class _SigmoidAdapter:
    """Thin adapter giving a fitted LogisticRegression an IsotonicRegression-
    style .predict() API that returns positive-class probabilities."""

    def __init__(self, lr):
        self.lr = lr

    def predict(self, x):
        import numpy as _np
        return self.lr.predict_proba(_np.asarray(x).reshape(-1, 1))[:, 1]


class _PostHocCalibratedModel:
    """Transparent wrapper that applies isotonic post-hoc calibration to
    predict_proba outputs of a fitted binary classifier.

    Added 2026-04-15 to make ``prefer_calibrated_classifiers=True`` actually
    calibrate tree classifiers. Prior behavior only swapped the early-stopping
    eval_metric, which was a no-op when early stopping did not trigger — so
    calibrated and uncalibrated runs produced bit-identical probabilities.

    The wrapper delegates every attribute to the underlying ``base`` model
    except ``predict_proba``, which runs the base classifier and then maps
    the positive-class probability through a fitted IsotonicRegression.
    """

    def __init__(self, base, calibrator):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_calibrator", calibrator)

    def __getattr__(self, name):  # delegate unknown attrs to base
        # During unpickling __getattr__ may fire before __dict__ is populated.
        # Guard against that to avoid infinite recursion / KeyError.
        if name in ("base", "_calibrator", "__setstate__", "__getstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "__dict__")["base"]
        except KeyError:
            raise AttributeError(name)
        return getattr(base, name)

    def __getstate__(self):
        return {"base": self.base, "_calibrator": self._calibrator}

    def __setstate__(self, state):
        object.__setattr__(self, "base", state["base"])
        object.__setattr__(self, "_calibrator", state["_calibrator"])

    def predict_proba(self, X):
        import numpy as _np
        raw = self.base.predict_proba(X)
        raw = _np.asarray(raw)
        if raw.ndim == 2 and raw.shape[1] == 2:
            p1 = self._calibrator.predict(raw[:, 1])
            p1 = _np.clip(p1, 0.0, 1.0)
            out = _np.column_stack([1.0 - p1, p1])
            return out
        return raw

    def predict(self, X):
        import numpy as _np
        probs = self.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            classes = getattr(self.base, "classes_", _np.array([0, 1]))
            return classes[(probs[:, 1] >= 0.5).astype(int)]
        return self.base.predict(X)


def _maybe_apply_posthoc_calibration(model, fit_params, model_type_name, verbose=False):
    """If the fitted estimator was tagged for post-hoc calibration and an
    eval_set is available, fit an IsotonicRegression on (val_preds, val_y)
    and return a wrapped model. Otherwise return the model unchanged.
    """
    try:
        inner = model.steps[-1][1] if hasattr(model, "steps") else model
    except Exception:
        inner = model

    want_calib = getattr(inner, "_mlframe_posthoc_calibrate", False) or getattr(model, "_mlframe_posthoc_calibrate", False)
    if not want_calib:
        return model
    # Post-hoc calibration hook is now a no-op. Calibration is handled
    # pre-fit by wrapping classifiers in CalibratedClassifierCV (see
    # _configure_*_params). This avoids the val-set overfitting problem.
    return model


def _polars_schema_diagnostic(
    df: "pl.DataFrame",
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    max_cols_logged: int = 30,
) -> str:
    """Render a per-column diagnostic of a Polars DataFrame for CatBoost
    fastpath incidents.

    CatBoost 1.2.x's `_set_features_order_data_polars_categorical_column`
    is a Cython fused cpdef with a finite set of compiled dispatch
    overloads. When a column's dtype variant isn't in the table, it
    raises the opaque ``TypeError: No matching signature found`` with no
    indication of which column is at fault. This helper dumps every
    column's dtype, its role (cat / text / other), and the specific
    sub-variant that matters for CB's dispatcher:

      - ``pl.Categorical`` with a validity bitmap (``null_count > 0``)
        — verified 2026-04-19 culprit: CB 1.2.10 has no dispatch overload
        for nullable Categorical. `_polars_nullable_categorical_cols` +
        `fill_null` is the fix.
      - ``pl.Enum`` without nulls — empirically works on the CB 1.2.10
        fastpath (reproduced 2026-04-21 — fit + eval_set succeed). Still
        reported in the dump for visibility, but not automatically
        flagged as the culprit.
      - ``pl.List[...]`` — nested types not supported in fastpath.

    Keeps output compact: logs up to ``max_cols_logged`` cat_features
    verbatim; the rest are summarised by dtype count. Safe to call in
    error paths — swallows exceptions and returns a note instead.
    """
    try:
        import polars as _pl
        cat_set = set(cat_features or [])
        text_set = set(text_features or [])
        lines: List[str] = []
        enum_cat_cols: List[str] = []  # the smoking-gun list

        cat_cols = [c for c in df.columns if c in cat_set]
        # Prioritise cat_features for full logging (they're the usual
        # culprit); summarise non-cat columns.
        shown = 0
        for col in cat_cols[:max_cols_logged]:
            dt = df.schema.get(col)
            role = "cat"
            variant = str(dt)
            if isinstance(dt, _pl.Enum):
                variant = f"Enum(n_values={len(dt.categories)})"
                enum_cat_cols.append(col)
            elif dt == _pl.Categorical or (
                hasattr(_pl, "Categorical") and isinstance(dt, type(_pl.Categorical))
            ):
                try:
                    ordering = getattr(dt, "ordering", "?")
                    variant = f"Categorical(ordering={ordering!r})"
                except Exception:
                    variant = "Categorical"
            try:
                nu = df[col].n_unique()
                nn = int(df[col].null_count())
                lines.append(f"    {col} [{role}]: {variant}, n_unique={nu}, nulls={nn}")
            except Exception:
                lines.append(f"    {col} [{role}]: {variant}")
            shown += 1

        if len(cat_cols) > max_cols_logged:
            lines.append(f"    ... +{len(cat_cols) - max_cols_logged} more cat_features")

        # Roll-up of everything else by dtype.
        other_dtype_counts: Dict[str, int] = {}
        for col in df.columns:
            if col in cat_set or col in text_set:
                continue
            dt_str = str(df.schema.get(col))
            other_dtype_counts[dt_str] = other_dtype_counts.get(dt_str, 0) + 1
        if other_dtype_counts:
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(other_dtype_counts.items()))
            lines.append(f"    (non-cat, non-text cols by dtype) {summary}")

        if text_set:
            text_cols_in_df = [c for c in df.columns if c in text_set]
            text_dt_counts: Dict[str, int] = {}
            for col in text_cols_in_df:
                dt_str = str(df.schema.get(col))
                text_dt_counts[dt_str] = text_dt_counts.get(dt_str, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(text_dt_counts.items()))
            lines.append(f"    (text_features by dtype) {summary}")

        nullable_cat_cols = [
            c for c in cat_cols
            if c in df.columns and int(df[c].null_count()) > 0
        ]
        header = f"  Polars schema diagnostic for {df.shape[0]:_}x{df.shape[1]}:"
        if nullable_cat_cols:
            header += (
                f"\n  [!] cat_features with null values: {nullable_cat_cols}. "
                "CatBoost 1.2.x Polars fastpath has no dispatch overload for "
                "Categorical with a validity bitmap; this is the most common "
                "cause of 'No matching signature found'. Fix: "
                "fill_null('__MISSING__') before fit."
            )
        elif enum_cat_cols:
            header += (
                f"\n  (info) cat_features contain pl.Enum columns: {enum_cat_cols}. "
                "Empirically compatible with CB 1.2.10 fastpath when nulls are "
                "filled; reported for visibility only."
            )
        return header + "\n" + "\n".join(lines)
    except Exception as _diag_err:
        return f"  (schema diagnostic failed: {_diag_err!r})"


def _polars_nullable_categorical_cols(df: Any, cat_features: Optional[List[str]] = None) -> "List[str]":
    """Return cat_feature column names with ``null_count > 0`` — the
    set of columns that trigger CatBoost 1.2.x's Polars fastpath
    dispatch miss.

    Root cause (verified 2026-04-19 via direct repro in
    ``bench_polars_cb_nullfrac.py``): CatBoost 1.2.10's
    ``_set_features_order_data_polars_categorical_column`` (Cython
    fused cpdef) has no dispatch signature for a Polars Categorical
    column carrying a validity bitmap. A single null anywhere in any
    cat_feature raises ``TypeError: No matching signature found``.
    Fix: ``pl.col(c).fill_null("__MISSING__")`` before fit — Polars
    auto-extends the category dict, the column loses its validity
    bitmap, CB's fastpath matches the non-nullable signature.

    Null-fraction sweep:
        0.0   → OK        0.5   → FAIL
        0.1   → FAIL      0.99  → FAIL
                          1.0   → FAIL

    Performance: uses ``df.select(cat_cols).null_count()`` which runs a
    SINGLE polars query — one scan over the selected columns, polars
    computes per-column null counts in parallel. The previous
    per-column implementation (``df[c].null_count()`` in a Python loop)
    cost N separate queries and showed up in prod profiling on
    810k-row frames.

    Args:
        df: Polars DataFrame.
        cat_features: Column names to consider. If None, inspects all
            ``pl.Categorical`` / ``pl.Enum`` columns in the schema.

    Returns:
        List of nullable Categorical column names (order-preserving
        against ``cat_features`` when provided). Empty list on any
        exception or non-Polars input — callers can use the list's
        truthiness directly, so the function doubles as a boolean
        detector without requiring a separate wrapper.
    """
    try:
        import polars as _pl
        if not isinstance(df, _pl.DataFrame):
            return []

        schema = df.schema
        # 2026-04-23: extended to include pl.Utf8 / pl.String. Raw Utf8
        # cat_features with nulls trigger the same CB 'Invalid type for
        # cat_feature ... NaN' error on the Polars fastpath — the
        # fill_null('__MISSING__') pre-fit pass must cover them too.
        # Fuzz c0061/c0084/c0096 (cb + polars_utf8 + nulls) all crashed
        # because Utf8 cols weren't in this candidate list.
        def _is_cat_like(dt):
            return (
                dt == _pl.Categorical
                or dt == _pl.Utf8
                or dt == _pl.String
                or (hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum))
            )
        if cat_features:
            candidate = [
                c for c in cat_features
                if c in schema and _is_cat_like(schema[c])
            ]
        else:
            candidate = [
                name for name, dtype in schema.items()
                if dtype == _pl.Categorical
                or (hasattr(_pl, "Enum") and isinstance(dtype, _pl.Enum))
            ]
        if not candidate:
            return []

        # SINGLE-PASS: df.select([...]).null_count() returns a 1-row DF
        # with per-column counts. All cat_features' null counts computed
        # in one scan. Previous per-column loop was N separate queries.
        counts_row = df.select(candidate).null_count().row(0)
        return [c for c, n in zip(candidate, counts_row) if n > 0]
    except Exception:
        return []


def _polars_df_has_null_in_categorical(df: Any, cat_features: Optional[List[str]] = None) -> bool:
    """Boolean wrapper around ``_polars_nullable_categorical_cols`` —
    kept for callers that only need the yes/no answer."""
    return bool(_polars_nullable_categorical_cols(df, cat_features=cat_features))


def _polars_fill_null_in_categorical(
    df: Any,
    nullable_cat_cols: "List[str]",
    sentinel: str = "__MISSING__",
) -> Any:
    """Apply ``pl.col(c).fill_null(sentinel)`` across the listed
    Categorical columns on a Polars DataFrame.

    Separated out so the same expression set can be reused across
    train / val / test (same sentinel → same category code across
    splits) without rebuilding the expr list per split.

    Returns df unchanged if ``nullable_cat_cols`` is empty or df is
    not a Polars DataFrame — caller can unconditionally wrap
    train/val/test without pre-checking.

    2026-04-23 (fuzz c0088 / c0121): ``fill_null(sentinel)`` on a
    ``pl.Enum`` whose category list does NOT already include the
    sentinel is a SILENT NO-OP in polars 1.40 — no error, no warning,
    nulls survive. The caller then hands the still-nullable Enum to
    CB's pandas fallback, which converts null→NaN and crashes with
    ``Invalid type for cat_feature ... =NaN``. Guard: for Enum
    columns we rebuild the Enum with the sentinel appended BEFORE
    filling. For ``pl.Categorical`` the Arrow-level dict auto-extends
    on fill_null, and for ``pl.Utf8/String`` no category list
    constraint applies, so those paths stay as-is.
    """
    try:
        import polars as _pl
        if not nullable_cat_cols or not isinstance(df, _pl.DataFrame):
            return df
        fill_exprs = []
        for c in nullable_cat_cols:
            dt = df.schema.get(c)
            # Enum: rebuild the category list to include the sentinel,
            # cast, THEN fill. Without the cast step the fill is a no-op.
            if dt is not None and hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum):
                orig_cats = list(dt.categories)
                if sentinel not in orig_cats:
                    new_enum = _pl.Enum(orig_cats + [sentinel])
                    fill_exprs.append(
                        _pl.col(c).cast(new_enum).fill_null(sentinel).alias(c)
                    )
                    continue
                # Enum already allowed the sentinel — plain fill_null works.
            fill_exprs.append(_pl.col(c).fill_null(sentinel))
        return df.with_columns(fill_exprs)
    except Exception:
        return df


def _recover_cb_feature_names(model: Any) -> Tuple[List[str], List[str]]:
    """Extract (cat_features, text_features) as column-name lists from a
    fitted CatBoost model.

    At predict time we don't have the original Python-side cat_features /
    text_features lists — the caller is evaluation code with no knowledge
    of how the model was trained. CatBoost exposes its internal
    per-feature metadata via:
      - ``_get_cat_feature_indices()``  — integer indices into feature_names_
      - ``_get_text_feature_indices()`` — ditto
      - ``feature_names_``              — list of column names

    Returns ``([], [])`` on any failure (e.g. non-fitted model, non-CB
    estimator, older CB builds without those private hooks) — callers
    wrap the fallback so missing names just means a less-specific prep
    path, not a crash.
    """
    try:
        feat_names = list(getattr(model, "feature_names_", []) or [])
        cat_idx = getattr(model, "_get_cat_feature_indices", lambda: [])() or []
        text_idx = getattr(model, "_get_text_feature_indices", lambda: [])() or []
        if not feat_names:
            return [], []
        cat_feat = [feat_names[i] for i in cat_idx if 0 <= i < len(feat_names)]
        text_feat = [feat_names[i] for i in text_idx if 0 <= i < len(feat_names)]
        return cat_feat, text_feat
    except Exception:
        return [], []


def _predict_with_fallback(
    model: Any,
    X: Any,
    method: str = "predict_proba",
    verbose: bool = False,
) -> np.ndarray:
    """Call ``model.{method}(X)`` with automatic Polars → pandas fallback
    on CatBoost's Polars-fastpath dispatcher misses.

    Symmetric to ``_train_model_with_fallback``. CatBoost 1.2.x's Polars
    fastpath in ``_set_features_order_data_polars_categorical_column`` is
    a Cython fused cpdef; certain column-dtype combinations raise the
    opaque ``TypeError: No matching signature found``. When training
    already hit this and fell back to pandas for fit, predict time
    gets the same treatment: CB's stored cat-feature indices dispatch
    back through the Polars fastpath when we hand it a pl.DataFrame
    for prediction, and it fails the same way.

    Prior behavior: the caller's ``except (AttributeError, TypeError,
    NotImplementedError)`` block caught the predict_proba failure and
    retried with ``model.predict(X)`` on the SAME pl.DataFrame — which
    hit the same dispatcher and raised again. Not a fallback; a retry
    into the same hole. (2026-04-19 prod log.)

    This helper closes the loop: on a Polars fastpath TypeError with
    a CatBoost model and a pl.DataFrame input, convert the DF to
    pandas (using cat/text feature names recovered from the fitted
    model), run ``prepare_df_for_catboost``, and retry. Non-CatBoost
    TypeErrors and non-Polars inputs propagate unchanged.
    """
    fn = getattr(model, method)
    n_rows = len(X) if hasattr(X, "__len__") else None

    # Self-heal for LGB / sklearn / linear: their sklearn wrappers receive
    # the X arg through _LGBMValidateData → check_array which converts
    # pd.Categorical to numpy object arrays of strings — crashes on the first
    # non-numeric cell. Convert Polars → pandas up front so the wrapper takes
    # its native pandas fastpath. (Mirrors the model.fit self-heal in
    # _train_model_with_fallback.)
    _model_type = type(model).__name__
    if isinstance(X, pl.DataFrame) and "LGBM" in _model_type:
        from .utils import get_pandas_view_of_polars_df
        logger.warning(
            "  [predict] %s.%s received pl.DataFrame; converting to pandas "
            "so LGB's sklearn wrapper takes the pandas-native fastpath.",
            _model_type, method,
        )
        X = get_pandas_view_of_polars_df(X)

    # Fix 9.4.3 correction (2026-04-22): reuse the cached CatBoost val
    # Pool for predict-path calls too. The fit path already stored a
    # Pool in ``_CB_VAL_POOL_CACHE`` keyed on
    # ``(id(val_df), cols, shape, cat_features, text_features,
    #   embedding_features)``. Prior to this fix the metrics path
    # called ``model.predict_proba(val_df)`` with a raw DataFrame,
    # which dispatches into CB's sklearn wrapper → rebuilds a fresh
    # Pool from the frame → on 7.3M rows, 53–66 s wasted per metrics
    # invocation (observed on prod 2026-04-22 log: 53 s at fit, then
    # another 66.7 s at VAL metrics computation for the identical
    # frame). CB's sklearn wrapper short-circuits rebuild when it
    # sees ``isinstance(X, Pool)``, so passing the cached Pool here
    # skips the second build entirely.
    #
    # Lookup uses the full id(X)+cols+shape signature — same frame
    # => same Pool. A stale cache entry that matches the signature
    # but had its label overwritten by fit is still fine for
    # predict_proba / predict (they don't read the label).
    try:
        _model_type = type(model).__name__
        if _model_type in CATBOOST_MODEL_TYPES and hasattr(X, "columns") and hasattr(X, "shape"):
            _cols = tuple(X.columns) if not isinstance(X.columns, tuple) else X.columns
            try:
                _shape = X.shape
                _shape_sig = (int(_shape[0]), int(_shape[1]))
            except Exception:
                _shape_sig = None
            _id = id(X)
            for key, pool in _CB_VAL_POOL_CACHE.items():
                if key[0] == _id and key[1] == _cols and key[2] == _shape_sig:
                    logger.info(
                        "[cb-val-pool-reuse] %s hit on cached val Pool — "
                        "skipping redundant Pool rebuild (saves the 53-66s "
                        "observed on 7M-row prod)",
                        method,
                    )
                    with phase(method, model=_model_type, n_rows=n_rows):
                        return fn(pool)
    except Exception as _exc:
        # Any lookup failure is benign — fall through to normal path.
        logger.debug(
            f"[cb-val-pool-reuse] {method} cache probe failed "
            f"({type(_exc).__name__}: {_exc}); falling through."
        )

    # Short-circuit: if this model already made a Polars-fastpath predict
    # call fail and fell back to pandas (via the except block below, or the
    # parallel path in _train_model_with_fallback), remember it and pre-
    # convert on every subsequent call. Otherwise VAL + TEST + ensemble
    # predict paths each re-hit the same Cython dispatch miss, each emitting
    # a "No matching signature found" WARN + 1-2s wasted conversion. The flag
    # is set on the model instance (not module-level) so different models in
    # the same suite each get their own cache state.
    _sticky_pandas = (
        isinstance(X, pl.DataFrame)
        and type(model).__name__ in CATBOOST_MODEL_TYPES
        and getattr(model, "_mlframe_polars_fastpath_broken", False)
    )
    if _sticky_pandas:
        from mlframe.training.utils import get_pandas_view_of_polars_df
        from mlframe.preprocessing import prepare_df_for_catboost as _prep_cb
        cat_feat, text_feat = _recover_cb_feature_names(model)
        X_pd = get_pandas_view_of_polars_df(X)
        if text_feat:
            for col in text_feat:
                if col in X_pd.columns and isinstance(X_pd[col].dtype, pd.CategoricalDtype):
                    X_pd[col] = X_pd[col].astype("object").fillna("")
        X_pd = _prep_cb(X_pd, cat_features=list(cat_feat), text_features=list(text_feat))
        with phase(method, model=type(model).__name__, n_rows=n_rows):
            return fn(X_pd)

    try:
        with phase(method, model=type(model).__name__, n_rows=n_rows):
            return fn(X)
    except TypeError as e:
        model_type_name = type(model).__name__
        err_str = str(e)
        # Only catch the specific Polars fastpath dispatch miss on a CB
        # model with a pl.DataFrame input. Everything else bubbles up
        # — otherwise we'd mask real type bugs.
        if (
            model_type_name not in CATBOOST_MODEL_TYPES
            or not isinstance(X, pl.DataFrame)
            or "No matching signature found" not in err_str
        ):
            raise

        logger.warning(
            "CatBoost %s Polars fastpath rejected the data (%s); "
            "converting to pandas and retrying.",
            method, err_str.splitlines()[-1][:240],
        )

        # Mark this model instance as "Polars-broken" so the next predict/
        # predict_proba/predict_log_proba call skips the retry dance.
        try:
            model._mlframe_polars_fastpath_broken = True
        except Exception:
            # Non-settable classes (frozen dataclasses, slots) — we simply
            # keep paying the TypeError-retry cost. Not fatal.
            pass

        # Recover cat/text feature names from the fitted model so
        # prepare_df_for_catboost can keep dtypes consistent with fit.
        cat_feat, text_feat = _recover_cb_feature_names(model)
        if verbose or not (cat_feat or text_feat):
            logger.info(
                "  [predict fallback] recovered from model: cat=%d, text=%d",
                len(cat_feat), len(text_feat),
            )

        from mlframe.training.utils import get_pandas_view_of_polars_df
        from mlframe.preprocessing import prepare_df_for_catboost as _prep_cb

        t0 = timer()
        shape_str = f"{X.shape[0]:_}×{X.shape[1]}" if hasattr(X, "shape") else "?"
        X_pd = get_pandas_view_of_polars_df(X)
        logger.info(
            "  [predict fallback] polars→pandas(%s) %s in %.1fs",
            method, shape_str, timer() - t0,
        )

        # Decategorize text columns BEFORE prepare_df_for_catboost
        # (same ordering as _train_model_with_fallback — prep_cb would
        # otherwise hit the CategoricalDtype text cols and either rebuild
        # them via the slow astype path or reject them outright).
        if text_feat:
            for col in text_feat:
                if col in X_pd.columns and isinstance(X_pd[col].dtype, pd.CategoricalDtype):
                    X_pd[col] = X_pd[col].astype("object").fillna("")

        t0 = timer()
        X_pd = _prep_cb(X_pd, cat_features=list(cat_feat), text_features=list(text_feat))
        logger.info(
            "  [predict fallback] prepare_df_for_catboost(%s) in %.1fs",
            method, timer() - t0,
        )

        return fn(X_pd)


# Fix 9.4.3 + Fix Orch-1: process-wide CatBoost Pool cache. Keys: tuple
# of (id(df), cols, shape, sorted cat/text/embedding features). Values:
# the Pool object whose label/weight we mutate in place between fits.
# The train-side cache survives weight schemas and same-type targets;
# the val-side cache adds ~2× on top (val eval_set is rebuilt on every
# fit by _setup_eval_set). Entries stay valid as long as the Python df
# reference is alive; the next train_mlframe_models_suite call produces
# fresh dfs with new id()s and the cache naturally evolves (also
# explicitly cleared at suite entry in core.py). Deliberate plain dict
# (not WeakValueDictionary) so transient GC between weight iterations
# doesn't flush the Pool we're about to reuse.
_CB_POOL_CACHE: "Dict[tuple, Any]" = {}
_CB_VAL_POOL_CACHE: "Dict[tuple, Any]" = {}
_CB_POOL_CACHE_MAX_ENTRIES = 16  # hard cap per cache; ring-buffer eviction oldest-first


def _cb_reuse_capable() -> bool:
    """True iff installed CatBoost Pool exposes both set_label and
    set_weight (the two mutators we rely on for in-place label/weight
    swap between fits)."""
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return False
    return callable(getattr(_Pool, "set_label", None)) and callable(
        getattr(_Pool, "set_weight", None)
    )


def _maybe_get_or_build_cb_pool(
    model_type_name: str,
    model: Any,
    train_df: Any,
    train_target: Any,
    fit_params: Dict[str, Any],
) -> Optional[Any]:
    """Return a cached/freshly-built ``catboost.Pool`` when the CB reuse
    fast-path applies; return None otherwise (caller falls back to
    ``model.fit(train_df, y, **fit_params)``).

    Fast-path activation requires ALL of:
      * ``model_type_name in CATBOOST_MODEL_TYPES``
      * installed CatBoost has Pool.set_label/set_weight
      * train_df is a recognised input type (polars/pandas/numpy)

    Cache-hit: swap label + weight in place, return the cached Pool.
    Cache-miss: build a new Pool, store, return it.
    """
    if model_type_name not in CATBOOST_MODEL_TYPES:
        return None

    # Filter cat/text/embedding features to only those actually present
    # in train_df. Motivation: MRMR and similar selectors can drop columns
    # AFTER fit_params was built, leaving stale feature lists that CB's
    # Pool rejects with ``ValueError: 'feat' is not in list`` from the
    # sklearn-wrapper's ``_get_cat_feature_indices`` (observed 2026-04-21
    # on ``test_mrmr_with_text_column`` / ``_embedding_column``). Applied
    # to CB only — XGB/LGB have their own handling for missing cols.
    try:
        _df_cols = set(train_df.columns) if hasattr(train_df, "columns") else None
    except Exception:
        _df_cols = None
    def _filter_to_df(feats):
        raw = fit_params.get(feats) or []
        if _df_cols is None:
            return tuple(sorted(raw))
        return tuple(sorted(c for c in raw if c in _df_cols))
    cat_features = _filter_to_df("cat_features")
    text_features = _filter_to_df("text_features")
    embedding_features = _filter_to_df("embedding_features")
    # Update fit_params in place so the fallback sklearn path (when reuse
    # is disabled or Pool construction fails) also sees the filtered
    # lists. Callers may rely on the same fit_params dict downstream; we
    # only narrow, never widen.
    if _df_cols is not None:
        if "cat_features" in fit_params and fit_params["cat_features"]:
            fit_params["cat_features"] = list(cat_features)
        if "text_features" in fit_params and fit_params["text_features"]:
            fit_params["text_features"] = list(text_features)
        if "embedding_features" in fit_params and fit_params["embedding_features"]:
            fit_params["embedding_features"] = list(embedding_features)

    if not _cb_reuse_capable():
        return None
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return None

    sample_weight = fit_params.get("sample_weight")

    # Cache key: id(df) alone is unsafe because Python reuses ids after
    # GC. Two tests in the same process that each build a fresh frame
    # may land on the same ``id(train_df)`` value — hitting a cache entry
    # built for a DIFFERENT frame with DIFFERENT cat_features/columns.
    # Include a content signature (columns tuple) so collisions with
    # distinct data produce a miss instead of a corrupted reuse.
    try:
        _cols = tuple(train_df.columns) if hasattr(train_df, "columns") else None
    except Exception:
        _cols = None
    try:
        _shape = getattr(train_df, "shape", None)
        _shape_sig = (int(_shape[0]), int(_shape[1])) if _shape and len(_shape) >= 2 else None
    except Exception:
        _shape_sig = None
    key = (id(train_df), _cols, _shape_sig, cat_features, text_features, embedding_features)

    cached = _CB_POOL_CACHE.get(key)
    if cached is not None:
        # Installed CatBoost 1.2.10 rejects ``Pool.set_label`` on a
        # classification Pool (target type ``Integer``) — the C++
        # ``SetNumericTarget`` path only accepts numeric / unset targets.
        # That means we can only reuse across WEIGHT swaps, not label
        # swaps, for classification pools. Strategy: skip ``set_label``
        # unless the caller actually supplied a different target (by id
        # against the last target we stored). Always mutate weight —
        # ``set_weight`` has no target-type restriction.
        last_target_id = getattr(cached, "_mlframe_last_target_id", None)
        try:
            if last_target_id is None or id(train_target) != last_target_id:
                # Label swap. Cast to float32 — the Pool was built with a
                # float32 label (see build path below), and CB's C++
                # ``SetNumericTarget`` rejects anything but Float/None. If
                # rejection happens anyway, fall through to rebuild.
                try:
                    _label_for_swap = np.asarray(train_target)
                    if _label_for_swap.dtype != np.float32:
                        _label_for_swap = _label_for_swap.astype(np.float32)
                except Exception:
                    _label_for_swap = train_target
                cached.set_label(_label_for_swap)
                cached._mlframe_last_target_id = id(train_target)
            if sample_weight is not None:
                cached.set_weight(sample_weight)
            logger.info(
                f"[cb-pool-reuse] hit key=(id={key[0]},cat={len(cat_features)},"
                f"text={len(text_features)},emb={len(embedding_features)}) "
                f"swapped weight{' + label' if last_target_id != id(train_target) else ''} without rebuild"
            )
            return cached
        except Exception as exc:
            # Drop the stale entry and fall through to rebuild. Typical
            # trigger: classification Pool + set_label on Integer target
            # raises "SetNumericTarget requires numeric or unset target
            # type". Rebuild is safe.
            logger.info(
                f"[cb-pool-reuse] swap path not usable ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:120]}); rebuilding Pool."
            )
            _CB_POOL_CACHE.pop(key, None)

    # Simple FIFO eviction — unlikely to hit during normal runs (<= N
    # models × N tiers entries), but keeps the cache from growing
    # unboundedly across long-running sessions.
    while len(_CB_POOL_CACHE) >= _CB_POOL_CACHE_MAX_ENTRIES:
        _CB_POOL_CACHE.pop(next(iter(_CB_POOL_CACHE)))

    # Cast label to float32 at build time. CatBoost stores the label's
    # raw type on the Pool (Integer vs Float) and later ``Pool.set_label``
    # validates ``ERawTargetType == Float or None`` inside C++
    # ``SetNumericTarget`` — if we built with Integer labels, subsequent
    # label swaps across classification targets would raise
    # ``SetNumericTarget requires numeric or unset target type, got
    # Integer``. Building with float32 pins the Pool's target type to
    # Float upfront; the user's upstream PR's classification tests all
    # pre-cast to float32 for exactly this reason. get_label() still
    # round-trips integer dtype via the Python-level ``target_type``
    # shadow on the Pool.
    try:
        _label_for_pool = np.asarray(train_target)
        if _label_for_pool.dtype.kind in ("i", "u", "b"):
            _label_for_pool = _label_for_pool.astype(np.float32)
        elif _label_for_pool.dtype != np.float32 and _label_for_pool.dtype.kind == "f":
            _label_for_pool = _label_for_pool.astype(np.float32)
    except Exception:
        _label_for_pool = train_target

    try:
        pool = _Pool(
            data=train_df,
            label=_label_for_pool,
            weight=sample_weight,
            cat_features=list(cat_features) or None,
            text_features=list(text_features) or None,
            embedding_features=list(embedding_features) or None,
        )
    except Exception as exc:
        # If Pool rejects the input (e.g. unsupported dtype combo),
        # fall back to the sklearn-wrapper path by returning None. The
        # operator sees the build-logger line above; we don't cache a
        # failed attempt.
        logger.warning(
            f"[cb-pool-reuse] Pool construction failed ({type(exc).__name__}: {exc}); "
            f"falling back to rebuild-every-fit sklearn path."
        )
        return None

    pool._mlframe_last_target_id = id(train_target)
    _CB_POOL_CACHE[key] = pool
    logger.info(
        f"[cb-pool-reuse] miss; stored fresh Pool (cache size={len(_CB_POOL_CACHE)})"
    )
    return pool


def _maybe_rewrite_eval_set_as_cb_pool(fit_params: Dict[str, Any]) -> None:
    """Fix Orch-1 (2026-04-21): for CatBoost, rewrite
    ``fit_params['eval_set']`` from ``[(val_df, val_target)]`` /
    ``(val_df, val_target)`` to a cached ``catboost.Pool`` in place.

    Cache key mirrors the train-side cache (id(val_df) + cols + shape +
    cat/text/embedding features). On a cache hit, swap label + weight on
    the cached Pool; on miss, build a fresh Pool with float32-cast label
    (for CB set_label compatibility across classification targets).

    Called from ``_train_model_with_fallback`` AFTER the train-side
    reuse decision and AFTER ``_setup_eval_set`` has populated
    fit_params['eval_set']. Idempotent per fit: if eval_set already
    contains a Pool, leave it alone.
    """
    if not _cb_reuse_capable():
        return
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return

    es = fit_params.get("eval_set")
    if es is None:
        return

    # Normalise to a list of (df, target) tuples.
    if isinstance(es, tuple) and len(es) == 2:
        es = [es]
    elif isinstance(es, list):
        pass
    else:
        return

    rewritten: list = []
    changed = False
    cat_features = tuple(sorted(fit_params.get("cat_features") or []))
    text_features = tuple(sorted(fit_params.get("text_features") or []))
    embedding_features = tuple(sorted(fit_params.get("embedding_features") or []))

    for entry in es:
        if isinstance(entry, _Pool):
            rewritten.append(entry)
            continue
        if not (isinstance(entry, tuple) and len(entry) == 2):
            rewritten.append(entry)
            continue
        val_df, val_target = entry
        if val_df is None or val_target is None:
            rewritten.append(entry)
            continue

        try:
            _cols = tuple(val_df.columns) if hasattr(val_df, "columns") else None
        except Exception:
            _cols = None
        try:
            _shape = getattr(val_df, "shape", None)
            _shape_sig = (int(_shape[0]), int(_shape[1])) if _shape and len(_shape) >= 2 else None
        except Exception:
            _shape_sig = None
        key = (id(val_df), _cols, _shape_sig, cat_features, text_features, embedding_features)

        cached = _CB_VAL_POOL_CACHE.get(key)
        if cached is not None:
            last_target_id = getattr(cached, "_mlframe_last_target_id", None)
            try:
                if last_target_id != id(val_target):
                    try:
                        _lab = np.asarray(val_target)
                        if _lab.dtype != np.float32:
                            _lab = _lab.astype(np.float32)
                    except Exception:
                        _lab = val_target
                    cached.set_label(_lab)
                    cached._mlframe_last_target_id = id(val_target)
                logger.info(
                    f"[cb-val-pool-reuse] hit key=(id={key[0]},cat={len(cat_features)},"
                    f"text={len(text_features)},emb={len(embedding_features)}) "
                    f"swapped{' label' if last_target_id != id(val_target) else ''} without rebuild"
                )
                rewritten.append(cached)
                changed = True
                continue
            except Exception as exc:
                logger.info(
                    f"[cb-val-pool-reuse] swap failed ({type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:120]}); rebuilding val Pool."
                )
                _CB_VAL_POOL_CACHE.pop(key, None)

        # Miss: build fresh val Pool with float32-cast label.
        while len(_CB_VAL_POOL_CACHE) >= _CB_POOL_CACHE_MAX_ENTRIES:
            _CB_VAL_POOL_CACHE.pop(next(iter(_CB_VAL_POOL_CACHE)))

        try:
            _lab_build = np.asarray(val_target)
            if _lab_build.dtype.kind in ("i", "u", "b"):
                _lab_build = _lab_build.astype(np.float32)
            elif _lab_build.dtype.kind == "f" and _lab_build.dtype != np.float32:
                _lab_build = _lab_build.astype(np.float32)
        except Exception:
            _lab_build = val_target

        try:
            val_pool = _Pool(
                data=val_df,
                label=_lab_build,
                cat_features=list(cat_features) or None,
                text_features=list(text_features) or None,
                embedding_features=list(embedding_features) or None,
            )
        except Exception as exc:
            logger.info(
                f"[cb-val-pool-reuse] Pool build failed ({type(exc).__name__}: {exc}); "
                f"leaving eval_set entry as (df, target) tuple for sklearn-wrapper rebuild."
            )
            rewritten.append(entry)
            continue

        val_pool._mlframe_last_target_id = id(val_target)
        _CB_VAL_POOL_CACHE[key] = val_pool
        rewritten.append(val_pool)
        changed = True
        logger.info(
            f"[cb-val-pool-reuse] miss; stored fresh val Pool (cache size={len(_CB_VAL_POOL_CACHE)})"
        )

    if changed:
        # Preserve original shape — single-tuple or list.
        if len(rewritten) == 1 and not isinstance(es, list):
            fit_params["eval_set"] = rewritten[0]
        else:
            fit_params["eval_set"] = rewritten


def _train_model_with_fallback(
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: Union[pd.DataFrame, np.ndarray],
    train_target: Union[pd.Series, np.ndarray],
    fit_params: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Any, Optional[int]]:
    """Train model with automatic GPU->CPU fallback on OOM errors.

    Parameters
    ----------
    model : Any
        Model to train (may be a Pipeline).
    model_obj : Any
        The actual estimator object (extracted from Pipeline if needed).
    model_type_name : str
        Name of the model type (e.g., 'CatBoostClassifier').
    train_df : pd.DataFrame or np.ndarray
        Training features.
    train_target : pd.Series or np.ndarray
        Training target values.
    fit_params : dict
        Additional parameters for model.fit().
    verbose : bool, default=False
        Whether to log verbose output.

    Returns
    -------
    tuple
        (trained_model, best_iteration) where best_iteration may be None.
    """
    t0_fit = timer()
    # Fix 9.4.3 (CB only, 2026-04-21): reuse a single ``catboost.Pool``
    # across weight schemas and same-target_type targets by mutating the
    # Pool's label/weight in place instead of letting the sklearn wrapper
    # rebuild from X on every fit. Gated on:
    #   * model is CatBoost-family;
    #   * installed CatBoost exposes ``Pool.set_label`` and
    #     ``Pool.set_weight`` (callable);
    #   * ``CatBoostClassifier.fit(X=Pool)`` is the idiomatic native path
    #     (short-circuits rebuild in ``_build_train_pool``).
    # XGB/LGB are not covered this round — their sklearn wrappers don't
    # accept pre-built DMatrix/Dataset yet (upstream FRs drafted in
    # ``reproducers/upstream_feature_requests/``). Only the per-build
    # logging from Fix 9.4.1 makes their rebuild cost visible.
    _cb_pool = _maybe_get_or_build_cb_pool(
        model_type_name=model_type_name,
        model=model,
        train_df=train_df,
        train_target=train_target,
        fit_params=fit_params,
    )
    # Fix Orch-1 (2026-04-21): also reuse the val Pool across fits.
    # Rewrites fit_params['eval_set'] from (val_df, val_target) to a
    # cached Pool so CB's sklearn wrapper short-circuits the val-side
    # rebuild too. Only fires when _cb_pool is active (train-side reuse
    # succeeded) — otherwise the mixed-container path (train=df,
    # eval_set=pool) confuses CB's fit signature.
    if _cb_pool is not None and model_type_name in CATBOOST_MODEL_TYPES:
        _maybe_rewrite_eval_set_as_cb_pool(fit_params)
    # Diagnostic: log the type+module of train_df right before model.fit so
    # silent type drift is visible in the log (Polars vs pandas vs numpy).
    # Critical: type(pl.DataFrame).__name__ == "DataFrame" — same as pandas —
    # so we log the module too, otherwise "DataFrame" can hide a Polars frame
    # that should have been converted upstream.
    _is_polars = isinstance(train_df, pl.DataFrame)
    _is_pandas = isinstance(train_df, pd.DataFrame)
    try:
        if _is_polars:
            _kind = "pl.DataFrame"
        elif _is_pandas:
            _kind = "pd.DataFrame"
        elif isinstance(train_df, np.ndarray):
            _kind = f"np.ndarray(dtype={train_df.dtype})"
        else:
            _kind = type(train_df).__name__
        if hasattr(train_df, "dtypes") and hasattr(train_df, "columns"):
            _dtype_summary = ", ".join(f"{c}={train_df[c].dtype}" for c in list(train_df.columns)[:5])
            if len(train_df.columns) > 5:
                _dtype_summary += f", ... ({len(train_df.columns)} cols total)"
        else:
            _dtype_summary = ""
        logger.info(f"  [pre-fit] train_df type={_kind}, {_dtype_summary}")
    except Exception:
        pass

    # Polars-frame contract: only CatBoost, XGBoost, and HistGradientBoosting
    # accept a Polars frame natively at fit time — their strategies carry
    # ``supports_polars=True``. Everyone else (LGB, sklearn, linear, ridge,
    # ...) MUST arrive with pandas; if a pl.DataFrame gets here for them, the
    # upstream lazy-conversion → pipeline_cache → process_model chain has a
    # leak. Previously the trainer silently ran a second polars→pandas
    # conversion as a "self-heal" — which hid the 2026-04-23 regression where
    # ``pipeline_cache`` crossed streams between XGB (polars-native,
    # ``cache_key="tree" + tier(False,False)``) and LGB (same key) — LGB kept
    # pulling XGB's polars frame out of cache and paying a duplicate 224 s
    # conversion. The pipeline_cache fix (container-kind in key, core.py) is
    # the real fix; this raise is the guard that ensures future leaks are
    # caught at the trainer boundary instead of being papered over.
    _POLARS_NATIVE_FIT_MODEL_PREFIXES = (
        "CatBoost",      # CatBoostClassifier / CatBoostRegressor / CatBoost
        "XGB",           # XGBClassifier / XGBRegressor / XGBRanker
        "HistGradient",  # HistGradientBoostingClassifier / Regressor
    )
    if _is_polars and not any(
        model_type_name.startswith(p) for p in _POLARS_NATIVE_FIT_MODEL_PREFIXES
    ):
        raise RuntimeError(
            f"{model_type_name} received pl.DataFrame at fit time "
            f"(shape={train_df.shape}, id={id(train_df)}). Only Polars-native "
            f"strategies (CatBoost, XGBoost, HistGradientBoosting) may receive "
            f"polars — everyone else needs pandas via the core.py lazy-"
            f"conversion path. Most likely cause: ``pipeline_cache`` returned "
            f"a polars frame cached by a polars-native strategy under a "
            f"``cache_key`` that collides with this strategy's key (see the "
            f"2026-04-23 kind-suffix fix in core.py). Diagnose via "
            f"pipeline_cache keys + id() — do NOT add another silent "
            f"self-heal."
        )

    try:
        with phase("model.fit", model=model_type_name,
                   n_rows=(train_df.shape[0] if hasattr(train_df, "shape") else None),
                   n_cols=(train_df.shape[1] if hasattr(train_df, "shape") else None)):
            if _cb_pool is not None:
                # Reuse path: X=Pool, y omitted (label already on the Pool).
                # fit_params still carries sample_weight, which CB's wrapper
                # ignores when X is a Pool (the Pool already has weight).
                # Filter it explicitly so downstream assertion paths don't
                # flag a "sample_weight with Pool" mismatch.
                _reuse_fit_params = {
                    k: v for k, v in fit_params.items()
                    if k not in ("sample_weight", "cat_features", "text_features", "embedding_features")
                }
                model.fit(_cb_pool, **_reuse_fit_params)
            else:
                model.fit(train_df, train_target, **fit_params)
    except Exception as e:
        try_again = False
        error_str = str(e)

        if "out of memory" in error_str:
            try_again = _handle_oom_error(model_obj, model_type_name)

        elif "User defined callbacks are not supported for GPU" in error_str:
            if "callbacks" in fit_params:
                logger.warning(e)
                try_again = True
                del fit_params["callbacks"]

        elif "CUDA Tree Learner" in error_str:
            logger.warning("CUDA is not enabled in this LightGBM build. Falling back to CPU.")
            model.set_params(device_type="cpu")
            try_again = True
        elif "pandas dtypes must be int, float or bool" in error_str:
            logger.warning(f"Model {model} skipped due to error 'pandas dtypes must be int, float or bool, got {train_df.dtypes}'")
            return None, None

        elif (
            model_type_name in CATBOOST_MODEL_TYPES
            and "Dictionary size is 0" in error_str
        ):
            # CatBoost's text feature estimator failed to build a TF-IDF
            # vocabulary — the column's non-null samples, after the
            # occurrence_lower_bound filter, leave an empty dictionary.
            # Root cause (seen 2026-04-19 in prod): columns auto-promoted
            # to text_features that have >99.9% null rows (e.g.
            # _raw_countries, job_post_source with 6-20 non-null
            # strings out of 810_000). Proactive guard in
            # _auto_detect_feature_types now blocks these at promotion
            # time, but this is the defensive fallback: on the exact
            # CB error, drop text_features from fit_params and retry
            # without text processing. The columns stay in the frame
            # (CB will treat them as plain categorical-by-name or
            # ignore).
            text_feat = fit_params.get("text_features") or []
            if text_feat:
                logger.warning(
                    "CatBoost raised 'Dictionary size is 0' on text_features %s — "
                    "the column(s) have too few non-null samples for CB's TF-IDF "
                    "estimator to build a vocabulary. Dropping text_features from "
                    "fit_params and retrying. Fix upstream: block promotion of "
                    "sparse columns in _auto_detect_feature_types (see the "
                    "min_non_null_for_text_promotion guard), or increase "
                    "non-null coverage of these columns in your feature "
                    "extraction.",
                    text_feat,
                )
                fit_params = {k: v for k, v in fit_params.items() if k != "text_features"}
                try_again = True
            else:
                # Raise — same error without text_features in params is
                # an unexpected variant, not our problem.
                pass

        elif (
            model_type_name in CATBOOST_MODEL_TYPES
            and isinstance(train_df, pl.DataFrame)
            and (
                "No matching signature found" in error_str
                # Catch *both* "Categorical for a numerical feature column" and
                # "Categorical for a text feature column" (the latter surfaces
                # when a column auto-promoted from cat_features -> text_features
                # is still pl.Categorical in the df). Upstream fix casts those
                # columns to pl.String before CB.fit; this is a safety net for
                # any future variant of the same error family.
                or "Unsupported data type Categorical" in error_str
            )
        ):
            # CatBoost's native-Polars fastpath (_set_features_order_data_polars_*)
            # can reject certain categorical column layouts with opaque messages —
            # either "No matching signature found" (fused cpdef dispatch miss) or
            # the categorical/numeric type mismatch above. Fall back to the pandas
            # path: zero-copy Arrow view + `prepare_df_for_catboost` preserves
            # dtypes (post-2026-04-18) and CatBoost's pandas path accepts a wider
            # range of category backings.
            # Full last-line for the one-line message, plus a structured
            # schema dump so the NEXT occurrence is diagnosable from the
            # first log line (prev. we only had the truncated error str,
            # and for opaque dispatch misses that's useless).
            last_line = error_str.splitlines()[-1] if error_str else "<empty>"
            logger.warning(
                "CatBoost Polars fastpath rejected the data (%s); "
                "converting to pandas and retrying.",
                last_line[:240],
            )
            # Mark the model "Polars-broken" so subsequent predict_proba /
            # predict_log_proba calls via _predict_with_fallback go straight
            # to the pandas path — avoids the same Cython dispatch miss on
            # every VAL/TEST/ensemble scoring (one WARN + one ~2s retry per
            # call saved). See the symmetric short-circuit in
            # _predict_with_fallback.
            try:
                model._mlframe_polars_fastpath_broken = True
            except Exception:
                pass
            schema_dump = _polars_schema_diagnostic(
                train_df,
                cat_features=fit_params.get("cat_features"),
                text_features=fit_params.get("text_features"),
            )
            logger.warning("CB Polars fastpath failure — schema context:\n%s", schema_dump)
            from mlframe.training.utils import get_pandas_view_of_polars_df
            from mlframe.preprocessing import prepare_df_for_catboost as _prep_cb

            cat_feat = list(fit_params.get("cat_features") or [])
            text_feat = list(fit_params.get("text_features") or [])

            def _decategorize_text_cols(df):
                """CatBoost's pandas path rejects columns that are pd.Categorical
                but not in cat_features with "column 'X' has dtype 'category' but
                is not in cat_features list". Columns auto-promoted from
                cat_features → text_features keep a pd.Categorical dtype after
                the Polars→pandas zero-copy conversion. Cast those to plain
                object to keep CB happy (and preserve the string content).
                """
                if not text_feat:
                    return df
                for col in text_feat:
                    if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype):
                        df[col] = df[col].astype("object").fillna("")
                return df

            # Per-step timing for the fallback: a production run showed this
            # entire path consumed >1 hour on a 1M × 98 frame with 4
            # high-cardinality text columns — without timing it was impossible
            # to tell which step (Polars→pandas vs. prep_cb vs. decategorize)
            # was responsible. The timer log writes to the trainer logger so
            # the lines interleave with surrounding INFO output.
            t0_fb = timer()
            shape_str = f"{train_df.shape[0]:_}×{train_df.shape[1]}" if hasattr(train_df, "shape") else "?"

            t0 = timer()
            train_df = get_pandas_view_of_polars_df(train_df)
            logger.info(f"  [fallback] polars→pandas(train) {shape_str} in {timer() - t0:.1f}s")

            # IMPORTANT: decategorize text columns BEFORE prepare_df_for_catboost.
            # Otherwise prep_cb hits the pd.Categorical text columns (auto-promoted
            # from cat to text earlier) and runs
            #   df[col].astype(str).fillna("").astype("category")
            # which on a high-cardinality column like skills_text (81k unique
            # values over 810k rows) takes many minutes per column — the
            # production-reproduced hang that motivated this reorder.
            t0 = timer()
            train_df = _decategorize_text_cols(train_df)
            logger.info(f"  [fallback] decategorize text cols(train) in {timer() - t0:.1f}s")

            t0 = timer()
            train_df = _prep_cb(train_df, cat_features=cat_feat, text_features=text_feat)
            logger.info(f"  [fallback] prepare_df_for_catboost(train) in {timer() - t0:.1f}s")

            # eval_set carries the val split for CB — rewrite it too.
            eval_set = fit_params.get("eval_set")
            if eval_set is not None:
                t0_es = timer()
                pairs = eval_set if isinstance(eval_set, list) else [eval_set]
                new_pairs = []
                for pair in pairs:
                    X_val, y_val = pair
                    if isinstance(X_val, pl.DataFrame):
                        X_val = get_pandas_view_of_polars_df(X_val)
                        # Decategorize BEFORE prep_cb (see train_df comment above).
                        X_val = _decategorize_text_cols(X_val)
                        X_val = _prep_cb(X_val, cat_features=cat_feat, text_features=text_feat)
                    else:
                        X_val = _decategorize_text_cols(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                    new_pairs.append((X_val, y_val))
                fit_params["eval_set"] = new_pairs if isinstance(eval_set, list) else new_pairs[0]
                logger.info(f"  [fallback] eval_set rewrite in {timer() - t0_es:.1f}s")

            logger.info(f"  [fallback] total pandas prep for CB in {timer() - t0_fb:.1f}s")
            try_again = True

        elif "unexpected keyword argument" in error_str and any(param in error_str for param in ("X_val", "y_val", "eval_set")):
            # Older sklearn versions don't support validation set in HistGradientBoosting
            val_params = ["X_val", "y_val", "eval_set"]
            removed = [p for p in val_params if p in fit_params]
            if removed:
                logger.warning(
                    f"This sklearn version doesn't support validation set parameters ({', '.join(removed)}) "
                    f"for {model_type_name}. Training without early stopping validation."
                )
                for param in val_params:
                    fit_params.pop(param, None)
                try_again = True

        if try_again:
            _maybe_clean_ram()
            with phase("model.fit", model=model_type_name,
                       n_rows=(train_df.shape[0] if hasattr(train_df, "shape") else None),
                       n_cols=(train_df.shape[1] if hasattr(train_df, "shape") else None),
                       retry=True):
                model.fit(train_df, train_target, **fit_params)
        else:
            raise e

    _maybe_clean_ram()
    fit_elapsed = timer() - t0_fit
    if verbose:
        shape_str = f"{train_df.shape[0]:_}×{train_df.shape[1]}" if hasattr(train_df, "shape") else ""
        logger.info(f"  model.fit({model_type_name}) done — {shape_str}, {fit_elapsed:.1f}s")

    # Apply post-hoc isotonic calibration to binary classifiers that were
    # tagged with ``_mlframe_posthoc_calibrate=True``. Fix 2026-04-15 for the
    # long-standing no-op behavior of ``prefer_calibrated_classifiers=True``
    # on tree models.
    try:
        model = _maybe_apply_posthoc_calibration(model, fit_params, model_type_name, verbose=verbose)
    except Exception as _calib_err:
        logger.warning(f"Post-hoc calibration hook raised: {_calib_err}")

    best_iter = None
    if model is not None:
        try:
            best_iter = get_model_best_iter(model_obj)
            if best_iter and verbose:
                logger.info(f"es_best_iter: {best_iter:_}")
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Could not get best iteration: {e}")

    return model, best_iter


def _filter_categorical_features(fit_params, train_df, val_df=None, test_df=None):
    """Filter cat_features to only include actual categorical columns.

    Uses the UNION of categorical columns across train / val / test
    rather than train alone. Rationale: ``eval_set`` contains val
    (already registered into fit_params at this point). If val has a
    categorical dtype in a column that train doesn't — e.g. upstream
    pipeline cast train but val slipped through a different code path
    — then CB raises
    ``column 'X' has dtype 'category' but is not in cat_features``
    because we pruned X out of cat_features.

    Union-based detection keeps every column that IS categorical in
    ANY split, matching how CB / XGB expect cat_features to be
    declared (a superset list is fine; a subset causes the error).
    """
    if "cat_features" not in fit_params:
        return

    cat_columns: set = set()
    if isinstance(train_df, pd.DataFrame):
        from .strategies import PANDAS_CATEGORICAL_DTYPES
        _pd_cats = list(PANDAS_CATEGORICAL_DTYPES)
        for split in (train_df, val_df, test_df):
            if isinstance(split, pd.DataFrame):
                cat_columns.update(split.select_dtypes(_pd_cats).columns)
    elif isinstance(train_df, pl.DataFrame):
        from .strategies import get_polars_cat_columns
        for split in (train_df, val_df, test_df):
            if isinstance(split, pl.DataFrame):
                cat_columns.update(get_polars_cat_columns(split))
    else:
        return

    fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in cat_columns]


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics and Reporting (imported from evaluation module)
# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Import report functions - these are already in evaluation.py
from .evaluation import (
    report_model_perf,
    report_regression_model_perf,
    report_probabilistic_model_perf,
    get_model_feature_importances,
    plot_model_feature_importances,
)


def _compute_split_metrics(
    split_name: str,
    df,
    target,
    idx,
    model,
    model_type_name: str,
    model_name: str,
    metrics_dict: dict,
    group_ids=None,
    target_label_encoder=None,
    preds=None,
    probs=None,
    figsize=(15, 5),
    nbins: int = 10,
    print_report: bool = True,
    plot_file: str = "",
    show_perf_chart: bool = True,
    show_fi: bool = False,
    fi_kwargs: dict = None,
    subgroups: dict = None,
    custom_ice_metric=None,
    custom_rice_metric=None,
    details: str = "",
    has_other_splits: bool = False,
    n_features: int = None,
):
    """Unified metrics computation for train/val/test splits."""
    # Only skip if we can't compute metrics (no probs AND no df to make predictions)
    if df is None and probs is None:
        return preds, probs, []

    # Derive columns from df if available (for feature importance)
    columns = list(df.columns) if df is not None and hasattr(df, "columns") else []
    df_prepared = _prepare_df_for_model(df, model_type_name) if df is not None else None

    effective_show_fi = show_fi and not has_other_splits
    split_plot_file = f"{plot_file}_{split_name}" if plot_file else ""

    preds, probs = report_model_perf(
        targets=target,
        columns=columns,
        df=df_prepared,
        model_name=model_name,
        model=model,
        target_label_encoder=target_label_encoder,
        preds=preds,
        probs=probs,
        figsize=figsize,
        report_title=" ".join([split_name.upper(), details]).strip(),
        nbins=nbins,
        print_report=print_report,
        plot_file=split_plot_file,
        show_perf_chart=show_perf_chart,
        show_fi=effective_show_fi,
        fi_kwargs=fi_kwargs if fi_kwargs else {},
        subgroups=subgroups,
        subset_index=idx,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        metrics=metrics_dict,
        group_ids=group_ids[idx] if group_ids is not None and idx is not None else None,
        n_features=n_features,
    )
    return preds, probs, columns


def run_confidence_analysis(
    test_df: pd.DataFrame,
    test_target: np.ndarray,
    test_probs: np.ndarray,
    cat_features: List[str] = None,
    confidence_model_kwargs: dict = None,
    fit_params: dict = None,
    use_shap: bool = True,
    max_features: int = 20,
    cmap: str = "coolwarm",
    alpha: float = 0.5,
    title: str = "Confidence Analysis",
    ylabel: str = "Prediction Confidence",
    figsize: Tuple[float, float] = (10, 6),
    verbose: bool = False,
) -> Any:
    """Analyze which features most affect prediction confidence."""
    if test_df is None:
        return None

    if verbose:
        logger.info("Running confidence analysis...")

    if confidence_model_kwargs is None:
        confidence_model_kwargs = {}

    confidence_task_type = "GPU" if CUDA_IS_AVAILABLE else "CPU"
    confidence_model = CatBoostRegressor(verbose=0, eval_fraction=0.1, task_type=confidence_task_type, **confidence_model_kwargs)

    fit_params_copy = {}
    if fit_params:
        fit_params_copy = copy.copy(fit_params)
        if "eval_set" in fit_params_copy:
            del fit_params_copy["eval_set"]

    if cat_features is not None:
        fit_params_copy["cat_features"] = cat_features
    elif "cat_features" not in fit_params_copy:
        fit_params_copy["cat_features"] = get_categorical_columns(test_df, include_string=False)

    fit_params_copy["plot"] = False

    confidence_targets = test_probs[np.arange(test_probs.shape[0]), test_target]

    _maybe_clean_ram()
    try:
        confidence_model.fit(test_df, confidence_targets, **fit_params_copy)
    except Exception as e:
        # CatBoost reports "Environment for task type [GPU] not found" when the
        # host has a CUDA device (so CUDA_IS_AVAILABLE=True) but no CatBoost
        # GPU runtime. Fall back to CPU — the confidence model is small and
        # CPU-adequate; this keeps training from aborting on mixed environments.
        if confidence_task_type == "GPU" and "Environment for task type [GPU] not found" in str(e):
            logger.warning("CatBoost GPU environment unavailable for confidence model; falling back to CPU.")
            confidence_model = CatBoostRegressor(verbose=0, eval_fraction=0.1, task_type="CPU", **confidence_model_kwargs)
            confidence_model.fit(test_df, confidence_targets, **fit_params_copy)
        else:
            raise
    _maybe_clean_ram()

    if use_shap:
        try:
            import shap
            import shap.utils.transformers

            shap.utils.transformers.is_transformers_lm = lambda model: False
        except (ImportError, AttributeError):
            pass
        explainer = shap.TreeExplainer(confidence_model)
        shap_values = explainer(test_df)
        shap.plots.beeswarm(
            shap_values,
            max_display=max_features,
            color=plt.get_cmap(cmap),
            alpha=alpha,
            color_bar_label=ylabel,
            show=False,
        )
        plt.xlabel(title)
        plt.show()
    else:
        plot_model_feature_importances(
            model=confidence_model,
            columns=list(test_df.columns),
            model_name=title,
            num_factors=max_features,
            figsize=(figsize[0] * 0.7, figsize[1] / 2),
        )

    return confidence_model


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Config Building Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


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
    # Control params
    verbose=False,
    use_cache=False,
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
    # Display params
    figsize=(15, 5),
    print_report=True,
    show_perf_chart=True,
    show_fi=True,
    fi_kwargs=None,
    plot_file="",
    data_dir="",
    models_subdir=MODELS_SUBDIR,
    display_sample_size=0,
    show_feature_names=False,
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

    display_config = DisplayConfig(
        figsize=figsize,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
        fi_kwargs=fi_kwargs or {},
        plot_file=plot_file or "",
        data_dir=data_dir or "",
        models_subdir=models_subdir or "models",
        display_sample_size=display_sample_size,
        show_feature_names=show_feature_names,
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

    return data_config, control_config, metrics_config, display_config, naming_config, confidence_config, predictions_container


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main Training Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def train_and_evaluate_model(
    model: object,
    data: DataConfig,
    control: TrainingControlConfig,
    metrics: MetricsConfig,
    display: DisplayConfig,
    naming: NamingConfig,
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
    display : DisplayConfig
        Display configuration (figsize, plot settings, paths).
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
    # Unpack display config
    # ---------------------------------------------------------------------------
    figsize = display.figsize
    print_report = display.print_report
    show_perf_chart = display.show_perf_chart
    show_fi = display.show_fi
    fi_kwargs = dict(display.fi_kwargs) if display.fi_kwargs else {}
    plot_file = display.plot_file
    data_dir = display.data_dir
    models_subdir = display.models_subdir
    display_sample_size = display.display_sample_size
    show_feature_names = display.show_feature_names

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
        logger.info(f"Loading model from file {model_file_name}")
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
                logger.info(f"training dataset shape: {train_df.shape}")

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

            _validate_target_values(train_target, "train")
            if val_target is not None:
                _validate_target_values(val_target, "val")

            if not just_evaluate:
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
                    df=split_df, target=split_target, idx=split_idx,
                    metrics_dict=metrics[split_name],
                    preds=split_preds, probs=split_probs, details=split_details,
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
                df=df, test_df=test_df, test_idx=test_idx, test_target=test_target,
                target=target, real_drop_columns=real_drop_columns,
                model=model, pre_pipeline=pre_pipeline,
                skip_pre_pipeline_transform=skip_pre_pipeline_transform,
                skip_preprocessing=skip_preprocessing,
                selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
            )
            if test_df is not None:
                _orig_test_df = test_df

        # Parallelize val and test metric computation — numba kernels release GIL,
        # Agg matplotlib is thread-safe. Pure-Python parts still block, but the
        # heavy cumtime (binning, AUC, calibration plot save) runs concurrently.
        def _run_val():
            if _val_cfg is None:
                return None
            _, sdf, starg, sidx, spreds, sprobs, sdet, _sc = _val_cfg
            return _compute_split_metrics(
                split_name="val", df=sdf, target=starg, idx=sidx,
                metrics_dict=metrics["val"],
                preds=spreds, probs=sprobs, details=sdet,
                has_other_splits=has_test,
                **common_metrics_params,
            )

        def _run_test_metrics():
            if not _run_test:
                return None
            return _compute_split_metrics(
                split_name="test", df=test_df, target=test_target, idx=test_idx,
                metrics_dict=metrics["test"],
                preds=test_preds, probs=test_probs, details=test_details,
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
        logger.info(f"  Metrics computation done — {timer() - t0_metrics:.1f}s")

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
    """Configure XGBoost model parameters."""
    xgb_configs = cpu_configs if prefer_cpu_for_xgboost else configs

    if use_regression:
        model_cls = flaml_zeroshot.XGBRegressor if use_flaml_zeroshot else XGBRegressor
        model = metamodel_func(model_cls(**xgb_configs.XGB_GENERAL_PARAMS))
    else:
        model_cls = flaml_zeroshot.XGBClassifier if use_flaml_zeroshot else XGBClassifier
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
    """Configure LightGBM model parameters."""
    lgb_configs = cpu_configs if prefer_cpu_for_lightgbm else configs

    if use_regression:
        model_cls = flaml_zeroshot.LGBMRegressor if use_flaml_zeroshot else LGBMRegressor
        model = metamodel_func(model_cls(**lgb_configs.LGB_GENERAL_PARAMS))
        fit_params = {}
    else:
        model_cls = flaml_zeroshot.LGBMClassifier if use_flaml_zeroshot else LGBMClassifier
        model = model_cls(**lgb_configs.LGB_GENERAL_PARAMS)
        fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    return dict(model=model, fit_params=fit_params)


def _configure_mlp_params(
    configs,
    config_params: dict,
    use_regression: bool,
    metamodel_func: callable,
) -> dict:
    """Configure MLP (PyTorch Lightning) model parameters."""
    mlp_kwargs = config_params.get("mlp_kwargs", {})

    mlp_network_params = dict(
        nlayers=20,
        first_layer_num_neurons=100,
        min_layer_neurons=1,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=1.5,
        activation_function=torch.nn.LeakyReLU,
        weights_init_fcn=partial(nn.init.kaiming_normal_, nonlinearity="leaky_relu", a=0.01),
        dropout_prob=0.15,
        inputs_dropout_prob=0.002,
        use_batchnorm=True,
    )
    if mlp_kwargs:
        mlp_network_params.update(mlp_kwargs.get("network_params", {}))

    mlp_general_params = configs.MLP_GENERAL_PARAMS.copy()
    if use_regression:
        mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
        mlp_general_params["model_params"]["loss_fn"] = F.mse_loss
        mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
        mlp_general_params["datamodule_params"]["labels_dtype"] = torch.float32
        mlp_model = PytorchLightningRegressor(network_params=mlp_network_params, **mlp_general_params)
    else:
        mlp_model = PytorchLightningClassifier(network_params=mlp_network_params, **mlp_general_params)

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
    if hasattr(features_train, 'shape'):
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
        if hasattr(features_train, 'shape'):
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
            raise ValueError(f"Unknown recurrent model type: {model_name}. "
                           f"Supported: {list(rnn_type_map.keys())}")

        rnn_type = rnn_type_map[model_name_lower]

        # Create model-specific config
        config = RecurrentConfig(
            input_mode=input_mode,
            rnn_type=rnn_type,
            seq_input_dim=seq_input_dim,
            features_dim=features_dim,
            hidden_size=recurrent_config.hidden_size,
            num_layers=recurrent_config.num_layers,
            dropout=recurrent_config.dropout,
            bidirectional=recurrent_config.bidirectional,
            num_heads=recurrent_config.num_heads,
            use_attention=recurrent_config.use_attention,
            mlp_hidden_dims=recurrent_config.mlp_hidden_dims,
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

    if not use_regression:
        if "catboost_custom_classif_metrics" not in config_params:
            nlabels = len(np.unique(target))
            if nlabels > 2:
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

    cpu_configs = get_training_configs(has_gpu=False, subgroups=indexed_subgroups, **config_params)
    gpu_configs = get_training_configs(has_gpu=None, subgroups=indexed_subgroups, **config_params)

    # Prefer caller-supplied size (typically computed on the Polars frame
    # BEFORE pandas conversion via .estimated_size() — O(cols), microseconds).
    # Fall back to get_df_memory_consumption with deep=False — O(cols) for
    # pandas too. Explicit deep=False avoids the O(rows) deep scan that used
    # to block this site for 3 minutes on frames with millions of unique
    # object-column strings. pyutilz default stays deep=True (back-compat);
    # mlframe opts out at this specific heuristic-only call site.
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

    # Skip expensive GPU probe (nvidia-smi subprocess ~0.5s) when GPU configs
    # are disabled or CatBoost is explicitly on CPU — the result is unused.
    cb_task_type = config_params.get("cb_kwargs", {}).get("task_type")
    cb_devices = config_params.get("cb_kwargs", {}).get("devices")
    if not prefer_gpu_configs or cb_task_type == "CPU":
        all_gpus = {}
        data_fits_gpu_ram = False
        data_fits_cb_gpu_ram = False
    else:
        all_gpus = get_gpuinfo_gpu_info()
        single_gpu_limits = compute_total_gpus_ram(all_gpus)
        data_fits_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < single_gpu_limits.get("gpu_max_ram_total", 0)
        if cb_devices:
            multi_gpu_limits = compute_total_gpus_ram(parse_catboost_devices(cb_devices, all_gpus=all_gpus))
            data_fits_cb_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < multi_gpu_limits.get("gpus_ram_total", 0)
        else:
            data_fits_cb_gpu_ram = data_fits_gpu_ram

    logger.info(f"data_fits_gpu_ram={data_fits_gpu_ram}, data_fits_cb_gpu_ram={data_fits_cb_gpu_ram}, cb_devices={cb_devices}")

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

    hgb_params = None
    if _should_create_model("hgb"):
        hgb_params = dict(
            model=metamodel_func(
                (
                    HistGradientBoostingRegressor(**configs.HGB_GENERAL_PARAMS)
                    if use_regression
                    else HistGradientBoostingClassifier(**configs.HGB_GENERAL_PARAMS)
                ),
            )
        )

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

    mlp_params = None
    if _should_create_model("mlp"):
        mlp_params = _configure_mlp_params(
            configs=configs,
            config_params=config_params,
            use_regression=use_regression,
            metamodel_func=metamodel_func,
        )

    ngb_params = None
    if _should_create_model("ngb"):
        ngb_params = dict(
            model=(
                metamodel_func(
                    (NGBRegressor(**configs.NGB_GENERAL_PARAMS) if use_regression else NGBClassifier(**configs.NGB_GENERAL_PARAMS)),
                )
            ),
            fit_params=(
                {} if config_params.get("early_stopping_rounds") is None
                else dict(early_stopping_rounds=config_params.get("early_stopping_rounds"))
            ),
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
        linear_model_params[model_type] = dict(model=metamodel_func(create_linear_model(model_type, config, use_regression=use_regression)))

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
