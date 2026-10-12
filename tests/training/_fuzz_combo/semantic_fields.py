"""Domain-aware value spaces for the config fields :mod:`field_randomizer` cannot vary from the annotation alone.

``randomize_scalar_fields`` resamples bool / bounded-numeric / Literal / Enum fields because their value space is the annotation. The fields
here have a value space only the consuming code knows: a column the built frame really has, a file the split machinery can replay, a callable
with the call site's signature, a nested payload whose keys the consumer reads, or a ``str`` checked against an allowed set by a validator or an
if-chain. Each generator below builds values from that real source (the frame, the production allowed-value constant, the callee signature),
so a sampled value exercises the lever instead of tripping an unrelated "column not found" / "unknown token" error.

Generators are registered per config class with the tuple of fields they set together. A tuple is skipped when a named FuzzCombo axis already
set any of its fields (``model_fields_set``), so a dedicated axis always wins. Assignment goes through pydantic ``validate_assignment``, so a
generator that produced an invalid value fails loudly at config construction rather than deep inside the suite.

Frame-shape decisions (a row-id column, an integer-epoch ``ts``) are made here too, from the combo seed alone, because the frame builder and the
config generators must agree on them without either seeing the other's output.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

from .combo import FuzzCombo
from .semantic_callables import arcsinh_target_metamodel, passthrough_metamodel

__all__ = [
    "ROW_ID_COLUMN",
    "SEMANTIC_FIELDS",
    "DERIVED_FIELDS",
    "SemanticContext",
    "frame_row_id_column",
    "linear_family_for_combo",
    "randomize_all_fields_enabled",
    "randomize_semantic_fields",
    "register_combo_frame",
    "split_recipe_mode",
    "training_config_for_combo",
    "ts_epoch_unit",
]

RANDOMIZE_ALL_FIELDS_ENV = "MLFRAME_FUZZ_RANDOMIZE_ALL_FIELDS"
ROW_ID_COLUMN = "row_id"

#: Fields with no independent value space: production overwrites them from another field at every construction, so they are exercised exactly
#: when their source field varies (``title_metrics_template`` is a named axis).
DERIVED_FIELDS: Dict[str, str] = {
    "ReportingConfig.title_metrics_tokens": "ReportingConfig.title_metrics_template",
}

_LTR = "learning_to_rank"


def randomize_all_fields_enabled() -> bool:
    """True when ``MLFRAME_FUZZ_RANDOMIZE_ALL_FIELDS`` opts the run into whole-config-surface randomization."""
    return os.environ.get(RANDOMIZE_ALL_FIELDS_ENV, "").lower() in ("1", "true", "yes", "on")


def _decision_rng(combo: FuzzCombo, salt: int) -> np.random.Generator:
    """RNG for a frame-shape decision: seeded off the combo seed only (never ``n_rows``), so a forced-row-count replica decides identically."""
    return np.random.default_rng((int(combo.seed), 9000 + salt))


# ---------------------------------------------------------------------------
# Frame-shape decisions shared with frame_builder
# ---------------------------------------------------------------------------


def ts_epoch_unit(combo: FuzzCombo) -> Optional[str]:
    """Epoch unit the ``ts`` column is emitted in as int64, or None for a datetime column.

    Integer epoch timestamps are what ``TrainingBehaviorConfig.target_temporal_audit_unit`` interprets; on a datetime column that knob is a
    documented no-op, so without this the field could never reach its consumer.
    """
    if not randomize_all_fields_enabled() or not combo.with_datetime_col:
        return None
    return [None, None, "s", "ms", "us", "ns"][int(_decision_rng(combo, 1).integers(0, 6))]


def split_recipe_mode(combo: FuzzCombo) -> str:
    """Which exact-split recipe the combo's ``TrainingSplitConfig`` gets: none / record (id_column only) / replay (id file) / window (dates).

    LTR routes to the ranker suite, which reads only the split fractions, so it always gets ``none``. Date windows need datetime timestamps on
    the extractor (``with_datetime_col`` and not epoch-encoded).
    """
    if not randomize_all_fields_enabled() or combo.target_type == _LTR:
        return "none"
    modes = ["none", "record", "replay"]
    if combo.with_datetime_col and ts_epoch_unit(combo) is None:
        modes += ["window", "window"]
    return modes[int(_decision_rng(combo, 2).integers(0, len(modes)))]


def frame_row_id_column(combo: FuzzCombo) -> Optional[str]:
    """Name of the unique row-key column the frame must carry for ``TrainingSplitConfig.id_column``, or None when no recipe uses one."""
    return ROW_ID_COLUMN if split_recipe_mode(combo) in ("record", "replay") else None


def linear_family_for_combo(combo: FuzzCombo) -> str:
    """Linear estimator family the run actually trains in place of ``"linear"``: the suite takes the family from the
    ``linear_model_type_cfg`` combo axis (see ``combo._canonical_linear_model_type``), never from ``LinearModelConfig.model_type``
    itself (excluded at ``_configure_training_learning_rate_float_trees``) -- mirroring that axis here keeps this field consistent
    with what the suite actually builds, regardless of whether generic-randomizer mode is on."""
    return combo._canonical_linear_model_type()


# ---------------------------------------------------------------------------
# Frame context
# ---------------------------------------------------------------------------

_REGISTERED_FRAME: Dict[Tuple[str, int], Any] = {}
_REPLICA_COLUMNS: Dict[Tuple[str, int], List[str]] = {}


def register_combo_frame(combo: FuzzCombo, df: Any) -> None:
    """Hand the already-built frame to the generators so they read real columns without rebuilding it; keeps only the latest combo's frame."""
    _REGISTERED_FRAME.clear()
    _REGISTERED_FRAME[(combo.short_id(), int(combo.n_rows))] = df


@dataclass
class SemanticContext:
    """What a generator may know about the combo: its frame (registered or rebuilt on demand) and caller-supplied extras."""

    combo: FuzzCombo
    linear_config: Any = None
    _frame: Any = None

    def frame(self) -> Any:
        """The combo's frame: the registered one when the suite already built it, else a fresh deterministic build."""
        if self._frame is None:
            self._frame = _REGISTERED_FRAME.get((self.combo.short_id(), int(self.combo.n_rows)))
        if self._frame is None:
            from .frame_builder import build_frame_for_combo

            self._frame = build_frame_for_combo(self.combo)[0]
        return self._frame

    def columns(self) -> List[str]:
        """Column names of the frame (a small replica when no frame is registered: names do not depend on the row count)."""
        key = (self.combo.short_id(), int(self.combo.n_rows))
        df = _REGISTERED_FRAME.get(key, self._frame)
        if df is not None:
            return [str(c) for c in df.columns]
        if key not in _REPLICA_COLUMNS:
            import dataclasses

            from .frame_builder import build_frame_for_combo

            _REPLICA_COLUMNS.clear()
            replica = build_frame_for_combo(dataclasses.replace(self.combo, n_rows=min(int(self.combo.n_rows), 64)))[0]
            _REPLICA_COLUMNS[key] = [str(c) for c in replica.columns]
        return list(_REPLICA_COLUMNS[key])

    def cat_columns(self) -> List[str]:
        """Categorical feature columns present in the frame (low-cardinality repeating keys: usable as group keys)."""
        return [c for c in self.columns() if c.startswith("cat_")]

    def numeric_feature_columns(self) -> List[str]:
        """The always-present informative numeric features."""
        return [c for c in self.columns() if c in ("num_0", "num_1", "num_2", "num_3")]


def _choice(rng: np.random.Generator, options: Sequence[Any]) -> Any:
    """One element of ``options`` (works for unhashable / mixed-type options, unlike ``rng.choice``)."""
    return options[int(rng.integers(0, len(options)))]


def _subset(rng: np.random.Generator, pool: Sequence[Any], min_size: int = 1, max_size: Optional[int] = None) -> List[Any]:
    """Random non-empty subset of ``pool`` in random order."""
    pool = list(pool)
    hi = len(pool) if max_size is None else min(max_size, len(pool))
    k = int(rng.integers(min_size, hi + 1))
    return [pool[i] for i in rng.permutation(len(pool))[:k]]


_Gen = Callable[[np.random.Generator, SemanticContext, BaseModel], Dict[str, Any]]
_REGISTRY: Dict[str, List[Tuple[Tuple[str, ...], _Gen]]] = {}


def _register(cls_name: str, *fields: str) -> Callable[[_Gen], _Gen]:
    """Register ``fn`` as the generator for ``fields`` of ``cls_name`` (set together, skipped when an axis set any of them)."""

    def deco(fn: _Gen) -> _Gen:
        """Record ``fn`` in the registry for ``cls_name``/``fields`` and return it unchanged."""
        _REGISTRY.setdefault(cls_name, []).append((tuple(fields), fn))
        return fn

    return deco


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

_METAMODEL_FUNCS = (passthrough_metamodel, arcsinh_target_metamodel)


@_register("TrainingConfig", "linear_config", "metamodel_func")
def _gen_training_config(rng, ctx, model) -> Dict[str, Any]:
    """Route the combo's linear config through the aggregate ``training_config=`` carrier, plus a metamodel wrapper on it."""
    out: Dict[str, Any] = {}
    if ctx.linear_config is not None:
        out["linear_config"] = ctx.linear_config
    if rng.random() < 0.5:
        out["metamodel_func"] = _choice(rng, _METAMODEL_FUNCS)
    return out


# ---------------------------------------------------------------------------
# TrainingBehaviorConfig
# ---------------------------------------------------------------------------


@_register("TrainingBehaviorConfig", "metamodel_func")
def _gen_metamodel_func(rng, ctx, model) -> Dict[str, Any]:
    """None (unwrapped) / pass-through wrapper / arcsinh target transform."""
    pick = _choice(rng, (None, *_METAMODEL_FUNCS))
    return {} if pick is None else {"metamodel_func": pick}


def _classification_scorings(target_type: str) -> List[Dict[str, Any]]:
    """``make_scorer`` kwargs dicts valid for ``target_type``'s RFECV scoring (multi-class / multi-label need label-based scorers)."""
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score

    if target_type == "binary_classification":
        return [
            dict(score_func=roc_auc_score, response_method="predict_proba", greater_is_better=True),
            dict(score_func=average_precision_score, response_method="predict_proba", greater_is_better=True),
            dict(score_func=balanced_accuracy_score, response_method="predict", greater_is_better=True),
        ]
    if target_type == "multiclass_classification":
        return [
            dict(score_func=balanced_accuracy_score, response_method="predict", greater_is_better=True),
            dict(score_func=accuracy_score, response_method="predict", greater_is_better=True),
        ]
    return [dict(score_func=accuracy_score, response_method="predict", greater_is_better=True)]


@_register("TrainingBehaviorConfig", "default_classification_scoring")
def _gen_classification_scoring(rng, ctx, model) -> Dict[str, Any]:
    """An RFECV classification scorer valid for the combo's target type."""
    if rng.random() < 0.3:
        return {}
    return {"default_classification_scoring": _choice(rng, _classification_scorings(ctx.combo.target_type))}


@_register("TrainingBehaviorConfig", "default_regression_scoring")
def _gen_regression_scoring(rng, ctx, model) -> Dict[str, Any]:
    """An RFECV regression scorer (loss or score, with the matching ``greater_is_better``)."""
    from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

    options = [
        dict(score_func=mean_squared_error, response_method="predict", greater_is_better=False),
        dict(score_func=median_absolute_error, response_method="predict", greater_is_better=False),
        dict(score_func=r2_score, response_method="predict", greater_is_better=True),
    ]
    return {} if rng.random() < 0.3 else {"default_regression_scoring": _choice(rng, options)}


@_register("TrainingBehaviorConfig", "callback_params")
def _gen_callback_params(rng, ctx, model) -> Dict[str, Any]:
    """``UniversalCallback`` constructor kwargs plus the keys ``_setup_early_stopping_callback`` pops before splatting."""
    options: List[Dict[str, Any]] = [
        {"patience": 15, "min_delta": 0.0, "verbose": 0},
        {"time_budget_mins": 5.0, "reporting_interval_mins": 0.5, "ndigits": 4},
        {"patience": 5, "min_delta": 1e-4, "monotonic_decline_patience": None},
        {"monotonic_decline_patience": 3, "capture_iteration_metrics": True, "iteration_metrics_stride": 2},
    ]
    return {} if rng.random() < 0.3 else {"callback_params": _choice(rng, options)}


@_register("TrainingBehaviorConfig", "cb_fit_params")
def _gen_cb_fit_params(rng, ctx, model) -> Dict[str, Any]:
    """``CatBoost.fit`` kwargs that do not collide with the ones the suite already passes (``plot`` / ``cat_features`` / ``eval_set``)."""
    options: List[Dict[str, Any]] = [{"metric_period": 5}, {"metric_period": 1, "save_snapshot": False}, {"metric_period": 25}]
    return {} if rng.random() < 0.3 else {"cb_fit_params": _choice(rng, options)}


@_register("TrainingBehaviorConfig", "isotonic_risk_kwargs")
def _gen_isotonic_risk(rng, ctx, model) -> Dict[str, Any]:
    """An ``IsotonicRiskConfig`` with a sparse subset of its fields written (unwritten ones keep the callee default)."""
    from mlframe.training.calibration_params.configs import IsotonicRiskConfig

    if rng.random() < 0.3:
        return {}
    pool = {
        "segment_ratio_threshold": float(_choice(rng, (0.01, 0.05, 0.2, 1.0))),
        "remediate": bool(rng.random() < 0.5),
        "density_window": float(_choice(rng, (0.02, 0.05, 0.25))),
    }
    picked = _subset(rng, list(pool))
    return {"isotonic_risk_kwargs": IsotonicRiskConfig(**{k: pool[k] for k in picked})}


@_register("TrainingBehaviorConfig", "threshold_optimizer_kwargs")
def _gen_threshold_optimizer(rng, ctx, model) -> Dict[str, Any]:
    """A ``ThresholdOptimizerConfig``; ``groups`` stays unset (it must be row-aligned with the calib slice, which no config can know)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

    from mlframe.training.calibration_params.configs import ThresholdOptimizerConfig

    if rng.random() < 0.3:
        return {}
    pool = {
        "n_thresholds": int(_choice(rng, (25, 100, 200))),
        "threshold_range": _choice(rng, ((0.0, 1.0), (0.05, 0.95))),
        "min_group_size": int(_choice(rng, (5, 20))),
        "cv": _choice(rng, (None, 3)),
        "cv_seed": int(rng.integers(0, 100)),
        "stability_cv_threshold": float(_choice(rng, (0.05, 0.15, 0.5))),
        "metric_fn": _choice(rng, (f1_score, balanced_accuracy_score, matthews_corrcoef)),
    }
    picked = _subset(rng, list(pool))
    return {"threshold_optimizer_kwargs": ThresholdOptimizerConfig(**{k: pool[k] for k in picked})}


@_register("TrainingBehaviorConfig", "precomputed_fairness_subgroups")
def _gen_precomputed_fairness(rng, ctx, model) -> Dict[str, Any]:
    """Subgroups built by the real ``create_fairness_subgroups`` over this combo's frame.

    Only when ``fairness_features`` is empty: otherwise the suite computes its own subgroups and replaces the hand-off.
    """
    import pandas as pd

    from mlframe.metrics._fairness_metrics import create_fairness_subgroups

    if getattr(model, "fairness_features", None) or rng.random() < 0.4:
        return {}
    n = int(ctx.combo.n_rows)
    thresh = max(1, min(10, n // 4))
    if n <= 200_000 and rng.random() < 0.5:
        df = ctx.frame()
        col = ctx.numeric_feature_columns()[0]
        values = df[col].to_numpy() if hasattr(df[col], "to_numpy") else np.asarray(df[col])
        sub = create_fairness_subgroups(pd.DataFrame({col: values}), features=[col], cont_nbins=3, min_pop_cat_thresh=thresh)
    else:
        sub = create_fairness_subgroups(pd.DataFrame(index=range(n)), features=_subset(rng, ["**ORDER**", "**RANDOM**"]), min_pop_cat_thresh=thresh)
    return {"precomputed_fairness_subgroups": sub} if sub else {}


_EPOCH_UNITS = ("s", "ms", "us", "ns")


@_register("TrainingBehaviorConfig", "target_temporal_audit_unit")
def _gen_audit_unit(rng, ctx, model) -> Dict[str, Any]:
    """The epoch unit for an integer ``ts``: usually the true one (or None = auto-detect), occasionally a wrong one (a degenerate audit)."""
    true_unit = ts_epoch_unit(ctx.combo)
    if true_unit is None:
        # Datetime ts: the override is ignored by contract (coerce_timestamps_for_audit returns datetime64 input as is); still a valid value.
        return {} if rng.random() < 0.5 else {"target_temporal_audit_unit": _choice(rng, _EPOCH_UNITS)}
    pick = _choice(rng, (None, true_unit, true_unit, _choice(rng, _EPOCH_UNITS)))
    return {} if pick is None else {"target_temporal_audit_unit": pick}


@_register("TrainingBehaviorConfig", "mlp_drop_per_group_constants_pattern")
def _gen_mlp_drop_pattern(rng, ctx, model) -> Dict[str, Any]:
    """A drop regex that matches a real column of this frame (so the drop actually fires), switching the drop on with it."""
    if rng.random() < 0.4:
        return {}
    cols = ctx.columns()
    patterns = [r"^num_3$"] + ([r"^cat_0$"] if "cat_0" in cols else []) + [r"^group_.*_(mean|std|min|max)$"]
    out: Dict[str, Any] = {"mlp_drop_per_group_constants_pattern": _choice(rng, patterns)}
    if "mlp_drop_per_group_constants" not in model.model_fields_set:
        out["mlp_drop_per_group_constants"] = True
    return out


# ---------------------------------------------------------------------------
# TrainingSplitConfig
# ---------------------------------------------------------------------------


def _write_split_ids_file(combo: FuzzCombo, rng: np.random.Generator, reuse: Sequence[str]) -> str:
    """Write a ``split_ids.parquet`` keyed on ``ROW_ID_COLUMN`` for this combo's row count and return its path.

    Layout mirrors a real earlier run: oldest rows train, then calib / val / test blocks; a few ids absent from the frame exercise the
    "absent ids" warning path.
    """
    import pandas as pd

    n = int(combo.n_rows)
    labels = np.array(["train"] * n, dtype=object)
    n_test, n_val, n_calib = max(1, n // 10), max(1, n // 10), max(1, n // 20)
    labels[n - n_test :] = "test"
    labels[n - n_test - n_val : n - n_test] = "val"
    labels[n - n_test - n_val - n_calib : n - n_test - n_val] = "calib"
    ids = np.arange(n, dtype=np.int64)
    ghost = np.arange(n, n + 3, dtype=np.int64)
    frame = pd.DataFrame({ROW_ID_COLUMN: np.concatenate([ids, ghost]), "split": np.concatenate([labels, np.array(["test"] * 3, dtype=object)])})
    out_dir = os.path.join(tempfile.gettempdir(), "mlframe_fuzz_split_ids")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{combo.short_id()}_n{n}_{'-'.join(reuse)}.parquet")
    frame.to_parquet(path, index=False)
    return path


def _window_bound(rng: np.random.Generator, ts: Any) -> Any:
    """The same instant in one of the representations the window validator coerces (Timestamp / ISO string / datetime)."""
    return _choice(rng, (ts, ts.isoformat(), ts.to_pydatetime()))


@_register("TrainingSplitConfig", "id_column", "split_ids_path", "reuse_splits", "test_start", "test_end", "val_start", "val_end")
def _gen_split_recipe(rng, ctx, model) -> Dict[str, Any]:
    """One coherent exact-split recipe (see :func:`split_recipe_mode`); fields are emitted in an order every intermediate state validates."""
    import pandas as pd

    from .frame_builder import zero_weight_batch_injected

    mode = split_recipe_mode(ctx.combo)
    if mode == "none":
        return {}
    if mode == "record":
        return {"id_column": ROW_ID_COLUMN}
    if mode == "replay":
        reuse = sorted(_subset(rng, ["test", "val", "calib", "train"], max_size=3))
        return {"id_column": ROW_ID_COLUMN, "reuse_splits": tuple(reuse), "split_ids_path": _write_split_ids_file(ctx.combo, rng, reuse)}
    # Date windows over the hourly ts the frame builder emits; rows splatted to 1900 by the zero-weight-batch axis sit before every window.
    n = int(ctx.combo.n_rows)
    n_valid = n - (max(1, int(n * 0.2)) if zero_weight_batch_injected(ctx.combo) and n >= 5 else 0)
    t0 = pd.Timestamp("2026-01-01")
    hour = lambda i: t0 + pd.Timedelta(hours=int(i))
    val_lo, test_lo = int(n_valid * 0.7), int(n_valid * 0.85)
    # val_end before val_start: an open-ended [val_start, None) window would transiently overlap the test window under validate_assignment.
    out: Dict[str, Any] = {}
    shape = _choice(rng, ("test", "val", "both"))
    if shape in ("val", "both"):
        out["val_end"] = _window_bound(rng, hour(test_lo))
        if rng.random() < 0.7:
            out["val_start"] = _window_bound(rng, hour(val_lo))
    if shape in ("test", "both"):
        out["test_start"] = _window_bound(rng, hour(test_lo))
        if rng.random() < 0.5:
            out["test_end"] = _window_bound(rng, hour(n_valid + 1))
    return out


@_register("PreprocessingConfig", "n_rows", "tail")
def _gen_row_subset(rng, ctx, model) -> Dict[str, Any]:
    """A head (``n_rows``) or tail row cap that keeps most of the frame.

    Owned here because the shape-based randomizer's fallback band for an unbounded ``Optional[int]`` defaulting to None is ``[1, 1]``: it
    capped every randomized frame to ONE row, which every splitter then (correctly) rejects.
    """
    n = int(ctx.combo.n_rows)
    pick = _choice(rng, (None, None, "n_rows", "tail"))
    return {} if pick is None else {pick: max(2, int(n * float(_choice(rng, (0.6, 0.8, 0.95)))))}


# ---------------------------------------------------------------------------
# FeatureTypesConfig / FeatureSelectionConfig
# ---------------------------------------------------------------------------


@_register("FeatureTypesConfig", "datetime_methods")
def _gen_datetime_methods(rng, ctx, model) -> Dict[str, Any]:
    """A non-empty subset of the logical date accessors ``create_date_features`` resolves on both pandas and polars."""
    from mlframe.feature_engineering.basic import _DATE_METHOD_ALIASES

    return {} if rng.random() < 0.3 else {"datetime_methods": set(_subset(rng, sorted(_DATE_METHOD_ALIASES), max_size=6))}


@_register("FeatureSelectionConfig", "ace")
def _gen_ace(rng, ctx, model) -> Dict[str, Any]:
    """Enable the ACE selector with defaults or a small-replicate ``ACEConfig`` payload (its default 20 replicates is a 20x refit)."""
    r = rng.random()
    if r < 0.5:
        return {}
    if r < 0.65:
        return {"ace": True}
    return {
        "ace": {
            "n_replicates": int(_choice(rng, (3, 5))),
            "importance": _choice(rng, ("native", "permutation")),
            "n_perm_repeats": int(_choice(rng, (2, 3))),
            "fdr_control": bool(rng.random() < 0.5),
            "mask_redundant": bool(rng.random() < 0.5),
            "random_state": int(rng.integers(0, 1000)),
        }
    }


# ---------------------------------------------------------------------------
# LinearModelConfig / ModelHyperparamsConfig
# ---------------------------------------------------------------------------


@_register("LinearModelConfig", "learning_rate")
def _gen_linear(rng, ctx, model) -> Dict[str, Any]:
    """``learning_rate`` is an SGD schedule. ``model_type`` is set alongside it to mirror the family the run actually trains (see
    :func:`linear_family_for_combo`), but is NOT independently registered for coverage: it has no value space of its own -- it is
    fully determined by the combo's ``linear_model_type_cfg`` axis, the same "derived, not sampled" pattern as ``DERIVED_FIELDS``."""
    from sklearn.linear_model import SGDRegressor

    schedules = sorted(o for c in SGDRegressor._parameter_constraints["learning_rate"] for o in getattr(c, "options", ()))
    out: Dict[str, Any] = {"model_type": linear_family_for_combo(ctx.combo), "learning_rate": _choice(rng, schedules)}
    # sklearn: the passive-aggressive schedules only work with the epsilon_insensitive loss on SGDRegressor (the classifier builder fixes its own loss).
    if str(out["learning_rate"]).startswith("pa") and "regression" in ctx.combo.target_type and "loss" not in model.model_fields_set:
        out["loss"] = "epsilon_insensitive"
    return out


#: CatBoost classification eval metrics accepted for each target type (multilabel's eval metric is overridden to HammingLoss by the suite).
_CB_CLASSIF_EVAL_METRICS = {
    "binary_classification": ("AUC", "Logloss", "PRAUC", "Accuracy", "BrierScore", "F1", "MCC", "Precision", "Recall"),
    "multiclass_classification": ("AUC", "MultiClass", "Accuracy", "TotalF1", "MCC", "HammingLoss"),
}
#: Regression eval metrics with a mapping on all three boosters (CB native name, LGB ``metric``, XGB ``eval_metric``). MSLE is not here: it is
#: undefined for the negative targets the fuzz frames produce, so CatBoost's NaN on it is correct behaviour, not a bug the fuzz should report.
_REGR_EVAL_METRICS = ("RMSE", "MAE", "MAPE", "Huber")
#: Valid custom-metric lists per family; consumed nowhere today (see the report on this field), but a value must still be a real CB metric.
_CB_CUSTOM_CLASSIF = (["AUC", "BrierScore", "PRAUC"], ["Logloss", "Accuracy"], ["AUC"])
_CB_CUSTOM_REGR = (["RMSE", "MAPE"], ["MAE", "R2"], ["RMSE"])


@_register("ModelHyperparamsConfig", "def_classif_metric")
def _gen_def_classif_metric(rng, ctx, model) -> Dict[str, Any]:
    """A CatBoost classification eval metric valid for this target type."""
    options = _CB_CLASSIF_EVAL_METRICS.get(ctx.combo.target_type)
    return {} if not options or rng.random() < 0.3 else {"def_classif_metric": _choice(rng, options)}


@_register("ModelHyperparamsConfig", "def_regr_metric")
def _gen_def_regr_metric(rng, ctx, model) -> Dict[str, Any]:
    """A regression eval metric every booster maps."""
    return {} if rng.random() < 0.3 else {"def_regr_metric": _choice(rng, _REGR_EVAL_METRICS)}


@_register("ModelHyperparamsConfig", "catboost_custom_classif_metrics", "catboost_custom_regr_metrics")
def _gen_cb_custom_metrics(rng, ctx, model) -> Dict[str, Any]:
    """Real CatBoost metric-name lists for both families."""
    if rng.random() < 0.3:
        return {}
    return {"catboost_custom_classif_metrics": list(_choice(rng, _CB_CUSTOM_CLASSIF)), "catboost_custom_regr_metrics": list(_choice(rng, _CB_CUSTOM_REGR))}


# Generator groups living in sibling modules register themselves on import.
from . import semantic_fields_discovery, semantic_fields_reporting  # noqa: F401

#: ``"ClassName.field"`` for every field a generator here sets.
SEMANTIC_FIELDS: frozenset = frozenset(f"{cls}.{f}" for cls, entries in _REGISTRY.items() for fields, _ in entries for f in fields)


def semantic_owned_fields(model: BaseModel) -> frozenset:
    """Field names of ``model``'s class a semantic generator owns: the shape-based randomizer must skip them even when the generator drew
    "keep the default", or it would resample them with its value-blind fallback band."""
    return frozenset(f for fields, _ in _REGISTRY.get(type(model).__name__, ()) for f in fields)


def randomize_semantic_fields(model: BaseModel, combo: FuzzCombo, rng: np.random.Generator, *, linear_config: Any = None) -> BaseModel:
    """Copy of ``model`` with every registered field group an axis left unset filled by its generator (validated on assignment)."""
    entries = _REGISTRY.get(type(model).__name__)
    if not entries:
        return model
    ctx = SemanticContext(combo=combo, linear_config=linear_config)
    explicit = set(model.model_fields_set)
    out = model.model_copy()
    for fields, gen in entries:
        if explicit.intersection(fields):
            continue
        for name, value in gen(rng, ctx, out).items():
            try:
                setattr(out, name, value)
            except Exception as exc:
                raise ValueError(f"semantic fuzz generator {gen.__name__} produced an invalid {type(model).__name__}.{name}={value!r}: {exc}") from exc
    return out


def training_config_for_combo(combo: FuzzCombo, target_name: str, model_name: str, linear_config: Any) -> Any:
    """A ``TrainingConfig`` carrier for the suite's ``training_config=`` kwarg, or None (env off / not drawn)."""
    if not randomize_all_fields_enabled():
        return None
    rng = np.random.default_rng((int(combo.seed), 7))
    if rng.random() < 0.5:
        return None
    from mlframe.training.configs import TrainingConfig

    return randomize_semantic_fields(TrainingConfig(target_name=target_name, model_name=model_name), combo, rng, linear_config=linear_config)
