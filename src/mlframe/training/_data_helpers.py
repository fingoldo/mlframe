"""Data preparation helpers extracted from ``trainer.py``.

Train/val/test DataFrame manipulation, target extraction, column
validation, infinity checking, and model info/path setup.
"""

from __future__ import annotations

import inspect
import logging
import os
from os import sep as os_sep
from os.path import join
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mlframe.core.helpers import ensure_no_infinity
from pyutilz.pythonlib import get_human_readable_set_size, prefix_dict_elems
from pyutilz.system import ensure_dir_exists
from pyutilz.strings import slugify
from mlframe.config import TABNET_MODEL_TYPES, XGBOOST_MODEL_TYPES

from .callbacks import LightGBMCallback, CatBoostCallback, XGBoostCallback

try:
    from xgboost.callback import TrainingCallback as XGBTrainingCallback
except ImportError:
    XGBTrainingCallback = None  # type: ignore[assignment] -- only used when xgboost is the chosen backend

from .pipeline import _extract_feature_selector, _prepare_test_split
from .utils import filter_existing

logger = logging.getLogger(__name__)

def _validate_trusted_path(path: str, trusted_root: str | None) -> None:
    """Raise ValueError if ``path`` is not inside ``trusted_root`` (absolute commonpath check).

    Matches the convention used in ``mlframe.inference.predict.read_trained_models``. Callers that
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
    except ValueError as exc:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}") from exc
    if common != abs_root:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")


from .models import create_linear_model, LINEAR_MODEL_TYPES

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_function_param_names(func: Callable) -> list[str]:
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
    target: pd.Series | pl.Series | np.ndarray | None,
    idx: np.ndarray | None,
) -> pd.Series | pl.Series | np.ndarray | None:
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
    # Normalise boolean masks to integer positions at the top so both backends see the
    # same idx shape. Pre-fix: pandas branch accepted bool via numpy-style ``.values[mask]``
    # (works because numpy supports bool indexing), but polars ``target.gather(mask)``
    # rejects boolean and raises ``InvalidOperationError``. Same caller code -> opposite
    # behaviour per backend. Convert with np.flatnonzero so both branches take the integer
    # path uniformly.
    _idx_arr = np.asarray(idx)
    if _idx_arr.dtype == np.bool_:
        idx = np.flatnonzero(_idx_arr)
    if isinstance(target, pd.Series):
        # ``.values[idx]`` (np take) is 9x faster than ``.iloc[idx]`` for
        # numeric targets (bench: 0.036s vs 0.324s per 100k-row subset on
        # 1M-row Series x 100 iterations). Wrap the result back in a
        # pd.Series so the contract documented in the
        # return-annotation stays intact - the wrap is a thin view, costs
        # ~10 us vs the 290 ms saved on subset itself.
        return pd.Series(
            target.values[idx],
            index=target.index[idx] if target.index is not None else None,
            name=target.name,
        )
    elif isinstance(target, pl.Series):
        return target.gather(idx)
    # numpy: ``target[idx]`` is already fast — 0.033s vs np.take 0.049s
    return target[idx]


def _subset_dataframe(
    df: pd.DataFrame | pl.DataFrame | None,
    idx: np.ndarray | None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame | pl.DataFrame | None:
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
        # Validate boolean masks eagerly: ``df.iloc[bool_idx]`` with a
        # mismatched-length bool array raises a confusing IndexError deep in
        # pandas, often masked when ``df`` is empty (no error) or when the
        # mismatch is by one row. Surface the precondition violation here.
        _idx_arr = np.asarray(idx) if idx is not None and not isinstance(idx, np.ndarray) else idx
        if isinstance(_idx_arr, np.ndarray) and _idx_arr.dtype == bool:
            if _idx_arr.shape[0] != len(df):
                raise ValueError(
                    f"select_subset: boolean idx length {_idx_arr.shape[0]} "
                    f"does not match dataframe length {len(df)}"
                )
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
    # Coerce target to a plain numpy ndarray for the np.isfinite checks below.
    # Pre-fix shape (pre-2026-05-20 silent-coercion audit): bare
    # ``target.values if pd.Series else target`` returned a pandas
    # ExtensionArray for nullable Int64 / Float64 columns, and the
    # subsequent ``np.isfinite`` raised TypeError that was caught at the
    # outer except and silently set nan_count = inf_count = 0 -- defeating
    # the whole purpose of this NaN-detection function. Same hazard on the
    # polars Int64-with-null branch. Coerce via to_numpy(na_value=nan) so
    # both ExtensionArray-nullable-Int AND polars-Int-with-null appear as
    # NaN to np.isfinite. Float dtypes are unaffected (NaN was already
    # representable natively).
    if isinstance(target, pd.Series):
        try:
            # na_value=nan only valid for float output; integer dtype + has_na -> fall back.
            arr = target.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError):
            arr = target.to_numpy(copy=False)
    else:
        # polars Series or numpy array
        try:
            import polars as _pl
            if isinstance(target, _pl.Series):
                # polars Int with nulls -> cast to Float64 then to_numpy so nulls -> nan.
                if target.null_count() > 0 and not target.dtype.is_float():
                    arr = target.cast(_pl.Float64).to_numpy()
                else:
                    arr = target.to_numpy()
            else:
                arr = np.asarray(target)
        except ImportError:
            arr = np.asarray(target)
    try:
        # Single-pass finiteness check; only re-scan the (typically empty)
        # non-finite subset to split nan_count vs inf_count. Two separate
        # isnan/isinf .sum() calls walked the full array twice on the
        # common all-finite case.
        finite_mask = np.isfinite(arr)
        if finite_mask.all():
            nan_count = inf_count = 0
        else:
            non_finite = arr[~finite_mask]
            nan_count = int(np.isnan(non_finite).sum())
            inf_count = int(np.isinf(non_finite).sum())
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

    iter636 (perf): ``np.asarray(target.tolist())`` instead of the
    previous ``np.stack([np.asarray(c) for c in target])`` listcomp.
    The listcomp crosses the Python<->numpy boundary N times (one
    ``np.asarray`` per row); ``.tolist()`` is one C-level conversion
    to a pure Python nested list and ``np.asarray`` then uses numpy's
    native fast path for nested-list construction. Bench at N=50k,
    K=2: 90.5ms -> 14.5ms (6.2x). Cell ragged-length / dtype mismatch
    is still trapped via the try/except guard so semantics match.
    """
    if target is None or not isinstance(target, np.ndarray):
        return target
    if target.dtype != object or target.ndim != 1 or target.shape[0] == 0:
        return target
    _first = target[0]
    if not (hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))):
        return target
    try:
        return np.asarray(target.tolist())
    except Exception:
        # Fallback for ragged rows / mixed dtypes np.asarray rejects.
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

    if best_iter is not None:
        logger.info("es_best_iter: %s", f"{best_iter:_}")
        model_name = model_name + f" @iter={best_iter:_}"

    return model_name


def _setup_eval_set(
    model_type_name: str,
    fit_params: dict[str, Any],
    val_df: pd.DataFrame | np.ndarray,
    val_target: pd.Series | np.ndarray,
    callback_params: dict[str, Any] | None = None,
    model_obj: Any | None = None,
    model_category: str | None = None,
    extra_eval_sets: list[Any] | None = None,
    sample_weight_val: np.ndarray | None = None,
    base_margin_val: np.ndarray | None = None,
    group_ids_val: np.ndarray | None = None,
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
    extra_eval_sets : list of SliceEvalSet, optional
        Additional per-shard eval-sets to register *after* the full-val eval-set. When provided
        the booster sees a list ``[full_val, shard_0, ..., shard_K-1]`` and the slice-stable
        callback can aggregate per-shard metrics positionally. ``None`` keeps the legacy
        single-val path bit-identical. See ``mlframe.training._slice_helpers.build_slice_eval_sets``.
    sample_weight_val, base_margin_val, group_ids_val : np.ndarray, optional
        Per-row attributes for the *full* val. When ``extra_eval_sets`` is given, the function
        also propagates ``sample_weight_eval_set`` / ``base_margin_eval_set`` / ``eval_qid``
        (XGB) / ``eval_group`` (LGB) lists in parallel so each registered eval-set has its own
        aligned arrays. Without ``extra_eval_sets`` these are ignored (the caller is responsible
        for setting them up in fit_params directly, as before).
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

    # Defensive non-empty val assertion. A 0-row val silently disables early stopping
    # (the booster fits to max_iter on a degenerate / empty eval_set). Outlier detection
    # now raises on val-collapse, but a 0-row val can still reach here from a different
    # source (aging-limit, sequential split, drift filter). Surface it with an actionable
    # error from ANY source instead of training without a working ES signal.
    if val_df is not None:
        _n_val = val_df.shape[0] if hasattr(val_df, "shape") else len(val_df)
        if _n_val == 0:
            raise ValueError(
                f"_setup_eval_set received a 0-row validation set for model_category "
                f"'{model_category}'. A 0-row eval_set silently disables early stopping. "
                f"The val collapsed upstream (outlier detection, trainset_aging_limit, "
                f"sequential/drift split). Investigate the split / filter pipeline OR "
                f"loosen the knob that emptied val."
            )

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

    # Build the eval-set list. Without extra_eval_sets this is the legacy single-tuple/list
    # exactly as before. With shards we always emit a list-of-tuples so the booster registers
    # one eval per element; LGB tuple-only path is upgraded to list (LGB accepts both).
    use_shards = extra_eval_sets is not None and len(extra_eval_sets) > 0
    if use_shards:
        eval_list: list[tuple[Any, Any]] = [(val_df, val_target)]
        for shard in extra_eval_sets:
            eval_list.append((shard.X, shard.y))
        if value_format == "list_of_tuples_values":
            eval_list = [(X.values if hasattr(X, "values") else X,
                          y.values if hasattr(y, "values") else y) for X, y in eval_list]
        if value_format in ("tuple", "list_of_tuples", "list_of_tuples_values"):
            fit_params[param_name] = eval_list
        elif value_format == "separate":
            # HGB / NGB only support a single (X_val, y_val) pair. Slice-stable ES is not supported
            # for these models via the online multi-eval-set path; the caller should route through
            # ``on_unsupported`` policy (default ``posthoc``). Here we just register the full val.
            fit_params["X_val"] = val_df
            fit_params["y_val"] = val_target
        elif value_format == "separate_Y":
            fit_params["X_val"] = val_df
            fit_params["Y_val"] = val_target
        # Parallel-aligned arrays for boosters that read them via separate kwargs.
        if model_category in ("xgb", "lgb", "cb"):
            if sample_weight_val is not None:
                sw_list = [sample_weight_val]
                for shard in extra_eval_sets:
                    sw_list.append(shard.sample_weight if shard.sample_weight is not None else None)
                fit_params["sample_weight_eval_set"] = sw_list
            if base_margin_val is not None and model_category == "xgb":
                bm_list = [base_margin_val]
                for shard in extra_eval_sets:
                    bm_list.append(shard.base_margin if shard.base_margin is not None else None)
                fit_params["base_margin_eval_set"] = bm_list
            if group_ids_val is not None:
                grp_list = [group_ids_val]
                for shard in extra_eval_sets:
                    grp_list.append(shard.group_ids if shard.group_ids is not None else None)
                if model_category == "xgb":
                    fit_params["eval_qid"] = grp_list
                elif model_category == "lgb":
                    # LGB takes per-set group SIZES (not ids). Convert each ids vector to run-lengths.
                    fit_params["eval_group"] = [_groupids_to_sizes(g) for g in grp_list]
    else:
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


def _groupids_to_sizes(group_ids: Any) -> np.ndarray | None:
    """Convert a per-row qid vector into LGB's per-query group-sizes vector.

    Rows must be already sorted by qid -- the standard ranker contract. Returns ``None``
    if input is ``None`` (so the caller can leave gaps in the eval_group list).
    """
    if group_ids is None:
        return None
    arr = np.asarray(group_ids)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    # Run-length encode consecutive equal qids.
    boundaries = np.concatenate([[0], np.flatnonzero(np.diff(arr)) + 1, [arr.size]])
    return np.diff(boundaries).astype(np.int64)


# Model categories that already wire val/eval_set through ``_setup_eval_set`` and have
# their own native ES path (booster callbacks or sklearn-style validation_fraction).
# Models OUTSIDE this set get nothing val-driven by default; the auto-wrap helper below
# folds them into a ``PartialFitESWrapper`` so val is no longer wasted.
_NATIVE_ES_CATEGORIES: frozenset[str] = frozenset(
    {"lgb", "hgb", "ngb", "cb", "xgb", "tabnet", "mlp"}
)

# Per-model budget parameter for the dichotomic-search ES strategy when the model lacks
# ``partial_fit``. ``None`` means no usable budget knob (e.g. plain LinearRegression which
# is closed-form -- no ES is possible at all, the wrapper degrades to a single fit-and-score).
_BUDGET_PARAM_BY_CATEGORY: dict[str, str | None] = {
    "ridge": "max_iter",
    "lasso": "max_iter",
    "elasticnet": "max_iter",
    "huber": "max_iter",
    "ransac": "max_trials",
    "linear": None,         # LinearRegression/LogisticRegression closed-form -- no budget
}


def _detect_budget_param(model_category: str, model_obj: Any) -> str | None:
    """Return the integer budget kwarg name on ``model_obj`` for dichotomic ES, or None.

    Priority: explicit per-category mapping first (so we don't accidentally pick the wrong
    knob on an estimator with multiple iterative params), then a runtime ``get_params``
    probe for common names.
    """
    explicit = _BUDGET_PARAM_BY_CATEGORY.get(model_category)
    if explicit is not None:
        return explicit
    if model_obj is None:
        return None
    try:
        params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}
    except Exception:
        return None
    for cand in ("max_iter", "n_estimators", "max_trials"):
        if cand in params and isinstance(params.get(cand), int):
            return cand
    return None


def maybe_wrap_for_partial_fit_es(
    model_obj: Any,
    *,
    model_category: str,
    X_val: Any,
    y_val: Any,
    is_classification: bool,
    behavior_kwargs: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> tuple[Any, bool]:
    """Wrap a non-native-ES sklearn model in ``PartialFitESWrapper`` when feasible.

    Returns (possibly_wrapped_model, was_wrapped). The wrapper drives val-based ES via
    either ``partial_fit`` (preferred when available) or a dichotomic budget search on
    ``max_iter`` / ``n_estimators`` / ``max_trials``. Models in ``_NATIVE_ES_CATEGORIES``
    are passed through untouched (they already use val via ``_setup_eval_set``). Closed-
    form models with neither capability (plain ``LinearRegression``) are passed through
    too -- no ES is possible.

    Parameters
    ----------
    behavior_kwargs
        Optional dict carrying ``TrainingBehaviorConfig`` ES knobs (patience, min_delta,
        max_iter, budget bounds) forwarded to the wrapper.
    random_state
        Outer suite seed forwarded to the wrapper's internal train/val ES split so ES is
        reproducible per-seed and independent across seeds. ``None`` lets the split vary.
    """
    if model_obj is None or X_val is None or y_val is None:
        return model_obj, False
    if model_category in _NATIVE_ES_CATEGORIES:
        return model_obj, False
    # Already wrapped by a previous call (e.g. nested suite invocation) -- no-op.
    if type(model_obj).__name__ == "PartialFitESWrapper":
        return model_obj, False

    has_partial_fit = hasattr(model_obj, "partial_fit")
    budget_param = None if has_partial_fit else _detect_budget_param(model_category, model_obj)
    if not has_partial_fit and budget_param is None:
        # Closed-form / no usable knob -- nothing to early-stop.
        return model_obj, False

    from ._partial_fit_es_wrapper import PartialFitESWrapper

    kw = dict(behavior_kwargs or {})
    wrapper = PartialFitESWrapper(
        model_obj,
        metric=kw.pop("metric", None),
        patience=int(kw.pop("patience", 10)),
        min_delta=float(kw.pop("min_delta", 0.0)),
        max_iter=int(kw.pop("max_iter", 200)),
        is_classification=is_classification,
        random_state=random_state,
        budget_param=budget_param,
        budget_min=int(kw.pop("budget_min", 1)),
        budget_max=int(kw.pop("budget_max", 1000)),
        external_X_val=X_val,
        external_y_val=y_val,
        verbose=int(kw.pop("verbose", 0)),
    )
    return wrapper, True


def _detect_max_iter(model_category: str, model_obj: Any) -> int | None:
    """Best-effort extraction of the iteration budget from a sklearn-API booster.

    Used by the "best_iter hit max_iter" diagnostic (``UniversalCallback.max_iter``).
    Returns None when the budget is not discoverable.
    """
    if model_obj is None:
        return None
    try:
        params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}
    except Exception:
        params = {}
    if model_category == "cb":
        return params.get("iterations") or params.get("n_estimators")
    if model_category in {"lgb", "xgb"}:
        return params.get("n_estimators")
    return None


def _build_cb_iteration_metrics_callback(fit_params, model_obj, stride):
    """Build the CatBoost per-iteration metric-capture callback from the val eval_set + model target type.

    Requires the build's ``callbacks=`` support and an eval_set in fit_params (the val Pool source). The callback's
    ``iteration_metrics_`` dict is bound by reference onto ``model_obj.iteration_metrics_`` at wiring time so the
    trajectory is readable on the fitted estimator without a post-fit stamp step (CatBoost callbacks have no
    after-training hook). Returns None when capture is not wireable (no eval_set / unsupported build / import fail).
    """
    from .callbacks.monotonic_decline import catboost_callbacks_supported

    if not catboost_callbacks_supported():
        return None
    eval_set = fit_params.get("eval_set")
    if not eval_set:
        return None
    pair = eval_set[0] if isinstance(eval_set, (list, tuple)) and eval_set and isinstance(eval_set[0], (list, tuple)) else eval_set
    try:
        X_val, y_val = pair[0], pair[1]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        from sklearn.base import is_classifier as _sk_is_classifier
        from catboost import Pool

        from .callbacks.iteration_metrics import CBIterationMetricsCallback

        import numpy as _np

        if model_obj is not None and not _sk_is_classifier(model_obj):
            target_type, n_classes = "regression", None
        else:
            n_classes = int(_np.unique(_np.asarray(y_val)).shape[0])
            target_type = "binary_classification" if n_classes <= 2 else "multiclass_classification"
        val_pool = Pool(X_val, y_val)
        cb = CBIterationMetricsCallback(val_pool, y_val, target_type, stride=stride, n_classes=n_classes)
        if model_obj is not None:
            model_obj.iteration_metrics_ = cb.iteration_metrics_  # bound by reference; filled during fit
        return cb
    except Exception as exc:
        logger.debug("CatBoost iteration-metrics capture not wired: %s", exc)
        return None


def _setup_early_stopping_callback(model_category, fit_params, callback_params, model_obj=None):
    """Set up early stopping callback for the given model category."""
    no_callback_list_models = {"xgb", "hgb", "ngb"}

    if model_category not in no_callback_list_models:
        if "callbacks" not in fit_params:
            fit_params["callbacks"] = []

    # Auto-inject the booster's iteration budget so the "best_iter hit max_iter" diagnostic
    # can fire. Caller-provided ``max_iter`` wins if set.
    if isinstance(callback_params, dict) and callback_params.get("max_iter") is None:
        budget = _detect_max_iter(model_category, model_obj)
        if budget:
            callback_params = {**callback_params, "max_iter": int(budget)}

    # Pull the monotonic-decline patience out of callback_params (default-on at 5, mirroring the lgb / xgb
    # shims) before splatting the rest into the UniversalCallback subclass, which does not accept this kwarg.
    # ``None`` disables the fixed-N monotonic stop, leaving the booster's native detector.
    _mono_patience = 7
    if isinstance(callback_params, dict) and "monotonic_decline_patience" in callback_params:
        callback_params = dict(callback_params)
        _mono_patience = callback_params.pop("monotonic_decline_patience")

    # Pull the per-iteration metric-capture knobs out before splatting callback_params into the UniversalCallback
    # subclass (which does not accept them). Wired below for CatBoost (the lgb / xgb shims read them as fit kwargs).
    _cap_iter = False
    _iter_stride = 1
    if isinstance(callback_params, dict) and "capture_iteration_metrics" in callback_params:
        callback_params = dict(callback_params)
        _cap_iter = bool(callback_params.pop("capture_iteration_metrics"))
        _iter_stride = int(callback_params.pop("iteration_metrics_stride", 1))

    if model_category == "lgb":
        es_callback = LightGBMCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
    elif model_category == "cb":
        es_callback = CatBoostCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
        # Monotonic strict-decline stop for CatBoost (default-on at 3) -- same shared rule as lgb / xgb / mlp.
        # Gated on a runtime probe of the installed build's ``callbacks=`` support; older builds fall back to
        # the native od_wait detector gracefully.
        if _mono_patience is not None:
            from .callbacks.monotonic_decline import CBMonotonicDeclineStop, catboost_callbacks_supported
            if catboost_callbacks_supported():
                fit_params["callbacks"].append(CBMonotonicDeclineStop(patience=_mono_patience))
        if _cap_iter:
            _cb = _build_cb_iteration_metrics_callback(fit_params, model_obj, _iter_stride)
            if _cb is not None:
                fit_params["callbacks"].append(_cb)
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


