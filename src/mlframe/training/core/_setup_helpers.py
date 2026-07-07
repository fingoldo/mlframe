"""Suite setup and configuration helpers."""

from __future__ import annotations

import logging
import os
from os.path import join
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from ..configs import (
        PreprocessingConfig,
        TrainingBehaviorConfig,
    )

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

# Mirrors the BorutaShap pattern below -- MRMR transitively pulls in
# the entire mlframe.feature_selection package (numba kernels + filter wrappers
# + sklearn estimators), which adds ~10-25s to first-call import time even when
# the suite doesn't opt into MRMR. Module-level import is deferred to the single
# call site (``_build_pre_pipelines`` in ``_setup_helpers_pre_pipelines``);
# other helpers in this module that only need the class for typing use
# TYPE_CHECKING-guarded references.
if TYPE_CHECKING:
    from mlframe.feature_selection.filters import MRMR  # noqa: F401

from ..configs import TargetTypes
from ..utils import get_pandas_view_of_polars_df
from pyutilz.strings import slugify
from pyutilz.system import ensure_dir_exists

logger = logging.getLogger(__name__)


# Polars-ds Pipeline JSON-roundtrip cache (in-memory + cross-process file cache).
# Re-exported from the sibling so historical
# ``from mlframe.training.core._setup_helpers import _pipeline_disk_cache_path``
# imports keep resolving.
from ._setup_helpers_pipeline_cache import (  # noqa: E402, F401
    _PIPELINE_JSON_ROUNDTRIP_CACHE,
    _PIPELINE_JSON_DISK_CACHE_PATH,
    _PIPELINE_JSON_DISK_CACHE_LOADED,
    _PIPELINE_JSON_DISK_CACHE_MAX_ENTRIES,
    _pipeline_disk_cache_path,
    _pipeline_disk_cache_version_tag,
    _load_pipeline_disk_cache_into_memory,
    _persist_pipeline_disk_cache,
    _PolarsDsPipelineJsonProxy,
    _polars_ds_pipeline_from_json,
)


from mlframe.metrics.core import create_fairness_subgroups

DEFAULT_PROBABILITY_THRESHOLD = 0.5

# Minority-class fraction below which a binary target is treated as imbalanced (and AUTO threshold-tuning fires).
# 0.35 is well clear of the 0.5 balanced point yet catches the moderate-skew regime where a fixed 0.5 already costs F1/balanced-acc.
DECISION_THRESHOLD_IMBALANCE_FRACTION = 0.35

ConfigT = TypeVar("ConfigT")


def is_target_imbalanced(y_true: np.ndarray, *, min_minority_fraction: float = DECISION_THRESHOLD_IMBALANCE_FRACTION) -> bool:
    """Cheap binary-imbalance test on a val/OOF target: True when the minority class is rarer than ``min_minority_fraction``.

    Used by the AUTO decision-threshold gate -- on imbalanced targets a fixed 0.5 produces poor hard labels, so the suite tunes
    the threshold; on roughly balanced targets it leaves 0.5 (val-tuning there only adds variance). Degenerate input
    (empty / single-class) is reported NOT imbalanced so AUTO falls back to the leak-safe 0.5.
    """
    y = np.asarray(y_true).ravel()
    if y.shape[0] == 0:
        return False
    _, counts = np.unique(y, return_counts=True)
    if counts.shape[0] < 2:
        return False
    minority_fraction = counts.min() / counts.sum()
    return bool(minority_fraction < min_minority_fraction)


def should_tune_decision_threshold(mode: bool | str, y_true: np.ndarray) -> bool:
    """Resolve the tri-state ``tune_decision_threshold`` config against a val/OOF target.

    ``True`` -> always tune; ``False`` -> never (force 0.5); ``"auto"`` (default) -> tune only when the target is imbalanced
    (:func:`is_target_imbalanced`). Any other / unknown value is treated as AUTO. Tuning still happens only on val/OOF at the call site.
    """
    if mode is True:
        return True
    if mode is False:
        return False
    return is_target_imbalanced(y_true)


def tune_decision_threshold(
    y_true: np.ndarray,
    pos_proba: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    default: float = DEFAULT_PROBABILITY_THRESHOLD,
    n_candidates: int = 200,
) -> float:
    """Tune a binary decision threshold on a NON-TEST split (val or OOF) by maximising a label metric.

    LEAKAGE-CRITICAL: callers MUST pass val/OOF labels + probabilities only -- never the honest
    test holdout. This function has no knowledge of which split it received; the leak-safety
    contract lives at the call site (the suite stamps the result into metadata from val/OOF).

    For imbalanced / cost-asymmetric targets a fixed 0.5 produces poor hard labels even when the
    probabilities are well-calibrated; sweeping the threshold on val/OOF recovers F1 / balanced
    accuracy. Returns ``default`` (0.5) when the input is degenerate (single class present, empty,
    or non-finite probabilities) so the leak-safe fallback always holds.

    Parameters
    ----------
    y_true : array of {0, 1}
        Ground-truth labels from a val/OOF split.
    pos_proba : array, same length
        Predicted P(y=1) on the same split.
    metric : {"f1", "balanced_accuracy"}
        Objective to maximise over the candidate grid.
    default : float
        Returned unchanged when tuning is not applicable (leak-safe 0.5).
    n_candidates : int
        Number of evenly spaced thresholds in the open interval (0, 1) to evaluate.
    """
    y = np.asarray(y_true).ravel()
    p = np.asarray(pos_proba, dtype=np.float64).ravel()
    if y.shape[0] == 0 or y.shape[0] != p.shape[0] or not np.all(np.isfinite(p)):
        return float(default)
    classes = np.unique(y)
    if classes.shape[0] < 2:
        # Single-class val/OOF: nothing to tune, 0.5 is as good as any.
        return float(default)

    if metric == "f1":
        from sklearn.metrics import f1_score
        scorer = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    elif metric == "balanced_accuracy":
        from mlframe.metrics.core import balanced_accuracy_binary
        scorer = balanced_accuracy_binary
    else:
        raise ValueError(f"tune_decision_threshold: unsupported metric {metric!r}; use 'f1' or 'balanced_accuracy'.")

    candidates = np.linspace(0.0, 1.0, n_candidates + 2)[1:-1]
    best_thr = float(default)
    best_score = scorer(y, (p >= default).astype(np.int8))
    for thr in candidates:
        s = scorer(y, (p >= thr).astype(np.int8))
        if s > best_score:
            best_score = s
            best_thr = float(thr)
    return best_thr


def get_decision_threshold(metadata: dict | None, target_key: str | None = None, default: float = DEFAULT_PROBABILITY_THRESHOLD) -> float:
    """Read a per-target tuned decision threshold stamped into metadata, falling back to 0.5.

    The suite stamps thresholds under ``metadata["decision_thresholds"][target_key]`` (tuned on
    val/OOF by :func:`tune_decision_threshold`). Predict paths call this so they reuse the tuned
    threshold instead of the hardcoded 0.5; an absent / malformed entry yields the leak-safe default.
    """
    if not isinstance(metadata, dict):
        return float(default)
    table = metadata.get("decision_thresholds")
    if not isinstance(table, dict):
        return float(default)
    if target_key is None:
        return float(default)
    val = table.get(target_key)
    try:
        thr = float(val)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(thr) or not (0.0 < thr < 1.0):
        return float(default)
    return thr


def _ensure_config(
    config: ConfigT | dict[str, Any] | None,
    config_class: type,
    kwargs: dict[str, Any],
) -> ConfigT:
    """Convert dict/None to Pydantic config object.

    Dict path is STRICT: a user-supplied config dict that carries a key which
    is neither a declared field nor a whitelisted ``_known_extras`` pass-through
    raises ``ValueError`` so typos (``iteratoins=100``) fail loud instead of
    being silently absorbed by ``extra="allow"``. The None path keeps filtering
    the ambient kwargs to declared fields (those kwargs are the suite's own
    superset, not a user-typed dict, so unknown ones are expected and dropped).
    """
    if isinstance(config, dict):
        obj = config_class(**config)
        extras = getattr(obj, "model_extra", None) or {}
        if extras:
            known = getattr(config_class, "_known_extras", frozenset()) or frozenset()
            unknown = sorted(k for k in extras if k not in known)
            if unknown:
                raise ValueError(
                    f"{config_class.__name__} received unknown config key(s) {unknown}. "
                    f"These are not declared fields and not whitelisted pass-through extras "
                    f"({sorted(known) or '(none)'}). Likely a typo -- fix the key or add it "
                    f"to the model. Declared fields: {sorted(config_class.model_fields)}."
                )
        return obj
    elif config is None:
        return config_class(**{k: v for k, v in kwargs.items() if k in config_class.model_fields})
    return config


# Global outlier-detection helper carved to ``_setup_helpers_outliers``;
# re-exported here so ``from mlframe.training.core._setup_helpers import
# _apply_outlier_detection_global`` keeps working.
from ._setup_helpers_outliers import _apply_outlier_detection_global  # noqa: E402, F401


def _setup_model_directories(
    target_name: str,
    model_name: str,
    target_type: str,
    cur_target_name: str,
    data_dir: str | None,
    models_dir: str | None,
    save_charts: bool = True,
) -> tuple[str | None, str | None]:
    """Set up directories for model artifacts and charts."""
    parts = slugify(target_name), slugify(model_name), slugify(target_type.lower()), slugify(cur_target_name)

    # Falsy check (not `is not None`): empty string data_dir="" means "no persistence", same as None.
    # Truthy "" would leak ./charts / ./models into CWD and re-load stale pickles on later runs.
    if data_dir and save_charts:
        plot_file = join(data_dir, "charts", *parts) + os.path.sep
        ensure_dir_exists(plot_file)
    else:
        plot_file = None

    if data_dir and models_dir:
        model_file = join(data_dir, models_dir, *parts) + os.path.sep
        ensure_dir_exists(model_file)
    else:
        model_file = None

    return plot_file, model_file


def _build_common_params_for_target(
    common_params_dict: dict[str, Any],
    trainset_features_stats: dict | None,
    plot_file: str | None,
    train_od_idx: np.ndarray | None,
    val_od_idx: np.ndarray | None,
    current_train_target: Any | None,
    current_val_target: Any | None,
    outlier_detector: Any | None,
    behavior_config: TrainingBehaviorConfig,
    fairness_subgroups: dict | None,
) -> tuple[dict[str, Any], TrainingBehaviorConfig]:
    """Build common_params and behavior_config for select_target call."""
    if fairness_subgroups is not None:
        current_behavior_config = behavior_config.model_copy(
            update={"_precomputed_fairness_subgroups": fairness_subgroups}
        )
    else:
        current_behavior_config = behavior_config

    # Drop train_target/val_target so they don't conflict when OD applies.
    filtered_params = {k: v for k, v in common_params_dict.items() if k not in ("train_target", "val_target")}
    od_common_params = dict(
        trainset_features_stats=trainset_features_stats,
        plot_file=plot_file,
        train_od_idx=train_od_idx,
        val_od_idx=val_od_idx,
        **filtered_params,
    )

    # With OD applied, pass targets directly to avoid re-subsetting.
    if outlier_detector is not None:
        od_common_params["train_target"] = current_train_target
        if current_val_target is not None:
            od_common_params["val_target"] = current_val_target

    return od_common_params, current_behavior_config


# ``_build_pre_pipelines`` lives in ``_setup_helpers_pre_pipelines``; re-exported here.
from ._setup_helpers_pre_pipelines import _build_pre_pipelines  # noqa: E402, F401


def _build_process_model_kwargs(
    model_file: str,
    model_name_with_weight: str,
    model_file_name:str,
    target_type: TargetTypes,
    pre_pipeline: Any,
    pre_pipeline_name: str,
    cur_target_name: str,
    models: dict,
    model_params: dict[str, Any],
    common_params: dict[str, Any],
    ens_models: list | None,
    trainset_features_stats: dict | None,
    verbose: int,
    cached_dfs: tuple | None,
    polars_pipeline_applied: bool = False,
    mlframe_model_name: str | None = None,
    optimize_storage: bool = True,
    metadata_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs dictionary for process_model call."""
    if mlframe_model_name:
        common_params = common_params.copy()
        common_params["model_category"] = mlframe_model_name

    kwargs = {
        "model_file": model_file,
        "model_name": model_name_with_weight,
        "model_file_name": model_file_name,
        "target_type": target_type,
        "pre_pipeline": pre_pipeline,
        "pre_pipeline_name": pre_pipeline_name,
        "cur_target_name": cur_target_name,
        "models": models,
        "model_params": model_params,
        "common_params": common_params,
        "ens_models": ens_models,
        "trainset_features_stats": trainset_features_stats,
        "verbose": verbose,
        "optimize_storage": optimize_storage,
        "metadata_columns": metadata_columns,
    }

    # Skip scaler/imputer/encoder if Polars-ds pipeline already ran globally; selectors still run.
    if polars_pipeline_applied:
        kwargs["skip_preprocessing"] = True

    if cached_dfs is not None:
        kwargs.update(
            {
                "skip_pre_pipeline_transform": True,
                "cached_train_df": cached_dfs[0],
                "cached_val_df": cached_dfs[1],
                "cached_test_df": cached_dfs[2],
            }
        )

    return kwargs


def _convert_dfs_to_pandas(
    train_df: pd.DataFrame | pl.DataFrame,
    val_df: pd.DataFrame | pl.DataFrame | None,
    test_df: pd.DataFrame | pl.DataFrame | None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Convert DataFrames to pandas format (nominally zero-copy for Polars).

    Per-split timers exist because Polars pl.Categorical columns force a pyarrow round-trip that
    rebuilds each dict with int32 indices (uint32 isn't supported by to_pandas); on
    high-cardinality categoricals this step can take 5+ minutes per split.
    """
    for name, df in [("train_df", train_df), ("val_df", val_df), ("test_df", test_df)]:
        if df is not None and not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError(f"{name} must be pandas DataFrame, polars DataFrame, or None, got {type(df).__name__}")

    def _convert_one(df, name):
        if df is None or isinstance(df, pd.DataFrame):
            return df
        t0 = timer()
        out = get_pandas_view_of_polars_df(df)
        if verbose:
            logger.info(
                "  polars->pandas(%s) %dx%d in %.1fs",
                name, df.shape[0], df.shape[1], timer() - t0,
            )
        return out

    t0_total = timer()
    train_df_pd = _convert_one(train_df, "train")
    val_df_pd = _convert_one(val_df, "val")
    test_df_pd = _convert_one(test_df, "test")
    if verbose:
        logger.info("  polars->pandas total: %.1fs", timer() - t0_total)

    return train_df_pd, val_df_pd, test_df_pd


def _get_pipeline_components(
    preprocessing_config: PreprocessingConfig,
    cat_features: list[str],
    random_seed: int | None = None,
) -> tuple[Any | None, SimpleImputer, StandardScaler]:
    """Get pipeline components (category_encoder, imputer, scaler) from typed config or defaults.

    ``random_seed`` is used to seed the default ``CatBoostEncoder`` so two calls
    with the same seed produce deterministic encodings. CatBoostEncoder draws
    permutations internally; without an explicit ``random_state`` the encoder
    re-shuffles on every fit, breaking determinism across reruns (fix audit
    row FE-P2-5).
    """
    category_encoder = preprocessing_config.category_encoder
    imputer = preprocessing_config.imputer
    scaler = preprocessing_config.scaler

    if category_encoder is None and cat_features:
        _seed = int(random_seed) if random_seed is not None else 42
        # LEAKAGE-CRITICAL: the default target encoder MUST be ordered / CV-style (CatBoostEncoder uses ordered TS). A naive global-mean target encoder leaks the row's own label into its encoding, inflating train-fold scores and overfitting. Do not swap this default for a non-ordered mean encoder. See tests/training/test_target_encoder_leakage_sensor.py.
        category_encoder = ce.CatBoostEncoder(random_state=_seed)

    if imputer is None:
        # keep_empty_features=True: an all-NaN numeric column (degenerate input, or a column that became all-NaN after the pre-split inf->NaN normalisation) must SURVIVE imputation as a zero-filled column, not be silently dropped. The default SimpleImputer() drops such columns, which breaks the column-count contract that ``_NumericOnlyTransformer`` relies on when it reassembles the scaled numeric block back into the original frame (it expects ``inner.transform`` to return exactly ``num_cols`` columns) -- the drop produced "Shape of passed values is (N, k-d), indices imply (N, k)" on the MLP pre-pipeline. Mirrors the imputer config already used in ``_predict_guards``.
        imputer = SimpleImputer(keep_empty_features=True)

    if scaler is None:
        scaler = StandardScaler()

    return category_encoder, imputer, scaler


def _compute_fairness_subgroups(
    df: pd.DataFrame | pl.DataFrame,
    behavior_config: TrainingBehaviorConfig,
) -> tuple[dict | None, list[str]]:
    """Compute fairness subgroups from DataFrame if fairness_features are specified."""
    fairness_features = behavior_config.fairness_features or []
    if not fairness_features:
        return None, fairness_features

    # Select only required columns - memory matters on large frames.
    cols_to_select = [f for f in fairness_features if f not in ("**ORDER**", "**RANDOM**") and f in df.columns]

    if cols_to_select:
        if isinstance(df, pl.DataFrame):
            # Arrow-backed split-blocks bridge: ~32x faster than .to_pandas() default on
            # 9M-row frames -- consolidation copy eliminated for numeric / bool columns.
            # Audit D P1-7 (2026-05-18): the polars->pandas conversion is NEEDED here because
            # ``create_fairness_subgroups`` from ``mlframe.metrics.core`` consumes a pandas
            # frame (pandas groupby / nunique). The conversion cannot be pushed further. Keep
            # the split-blocks bridge so the hop stays at zero-copy on numeric columns.
            df_subset = get_pandas_view_of_polars_df(df.select(cols_to_select))
        else:
            df_subset = df[cols_to_select]
    else:
        # Only **ORDER**/**RANDOM** markers - no actual columns needed.
        df_subset = pd.DataFrame(index=range(len(df)))

    subgroups = create_fairness_subgroups(
        df_subset,
        features=fairness_features,
        cont_nbins=behavior_config.cont_nbins,
        min_pop_cat_thresh=behavior_config.fairness_min_pop_cat_thresh,
    )
    return subgroups, fairness_features


def _should_skip_catboost_metamodel(
    model_or_pipeline_name: str,
    target_type: TargetTypes,
    behavior_config: TrainingBehaviorConfig,
) -> bool:
    """Skip CatBoost regression + metamodel_func: sklearn clone fails on CatBoostRegressor
    (RuntimeError: constructor does not set or modifies parameter custom_metric)."""
    if target_type != TargetTypes.REGRESSION:
        return False
    if behavior_config.metamodel_func is None:
        return False
    return model_or_pipeline_name in ("cb", "cb_rfecv")


def log_chart_summary(metadata: dict | None, *, save_charts: bool, data_dir: str | None) -> str:
    """Emit a one-line INFO at suite end with the chart count + destination, independent of verbose.

    A default ``train_mlframe_models_suite(...)`` saves nothing (``data_dir=""``) and renders nothing
    on a non-interactive run, with no trace -- so an operator cannot tell whether charts were skipped by
    design or lost to a bug. This reads the ``metadata["charts"]`` accounting (the saved/failed lists
    INV-48 stamps) and logs either the count + destination, the "0 charts saved; set
    output_config.data_dir" hint, or the failure count. Returns the message so callers can assert on it.

    Call site is the suite-finalize path (``_phase_finalize`` / orchestrator), owned by the integrator.
    """
    charts = (metadata or {}).get("charts") if isinstance(metadata, dict) else None
    n_saved = len(charts.get("saved", [])) if isinstance(charts, dict) else 0
    n_failed = len(charts.get("failed", [])) if isinstance(charts, dict) else 0

    if n_saved == 0 and not (save_charts and data_dir):
        msg = (
            "[reporting] 0 charts saved; set output_config.data_dir (and keep save_charts=True) "
            "to persist diagnostics to disk."
        )
    else:
        dest = f"{data_dir}/charts" if data_dir else "(not saved)"
        msg = f"[reporting] {n_saved} chart(s) saved to {dest}"
        if n_failed:
            msg += f"; {n_failed} render(s) failed (see WARN logs)"
    logger.info(msg)
    return msg


# Metadata builders / finalizers live in ``_setup_helpers_metadata``;
# re-exported here so historical
# ``from mlframe.training.core._setup_helpers import _finalize_and_save_metadata``
# imports keep resolving.
from ._setup_helpers_metadata import (  # noqa: E402, F401
    _create_initial_metadata,
    _initialize_training_defaults,
    _finalize_and_save_metadata,
)
