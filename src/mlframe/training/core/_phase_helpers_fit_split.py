"""``_phase_fit_pipeline`` + ``_phase_train_val_test_split`` -- the heavy training phases.

Wave 105 (2026-05-21): split out from ``training/core/_phase_helpers.py`` to
keep that file below the 1k-line monolith threshold. Behaviour preserved
bit-for-bit; both functions are re-exported from ``_phase_helpers`` so
existing imports continue to work.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
)

import numpy as np
import pandas as pd

from ..phases import phase

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

# 2026-05-21: wave-105 split-out forgot to mirror the parent's imports +
# NamedTuple defs, so every call into ``_phase_fit_pipeline`` /
# ``_phase_train_val_test_split`` raised NameError. Mirroring the parent
# module's imports here so this file is genuinely self-contained.
if TYPE_CHECKING:
    pass

from ._misc_helpers import (
    _auto_detect_feature_types,
    _df_shape_str,
    _drop_cols_df,
    _elapsed_str,
    _validate_feature_type_exclusivity,
)
from ..preprocessing import (
    create_split_dataframes,
    save_split_artifacts,
)
from ..utils import (
    get_process_rss_mb,
    log_phase,
    log_ram_usage,
    maybe_clean_ram_and_gpu,
)
from ..splitting import make_train_test_split
from ._setup_helpers import _compute_fairness_subgroups


class TrainValTestSplitResult(NamedTuple):
    """Return shape for ``_phase_train_val_test_split`` (mirror of the
    parent module's definition; required here because the split moved
    the function body but left its consumers in both modules).
    """

    train_idx: Any
    val_idx: Any
    test_idx: Any
    train_details: Any
    val_details: Any
    test_details: Any
    train_df: Any
    val_df: Any
    test_df: Any
    fairness_subgroups: Any
    fairness_features: Any
    train_sequences: Any
    val_sequences: Any
    test_sequences: Any
    baseline_rss_mb: Any
    calib_idx: Any = None
    calib_details: Any = None
    calib_df: Any = None


class FitPipelineResult(NamedTuple):
    """Return shape for ``_phase_fit_pipeline`` (see comment on
    ``TrainValTestSplitResult``)."""

    train_df: Any
    val_df: Any
    test_df: Any
    pipeline: Any
    extensions_pipeline: Any
    cat_features: Any
    cat_features_polars: Any
    was_polars_input: Any
    all_models_polars_native: Any
    polars_pipeline_applied: Any
    train_df_polars_pre: Any
    val_df_polars_pre: Any
    test_df_polars_pre: Any
    pipeline_config: Any
    preprocessing_extensions: Any
    train_df_pandas_pre_meta: Any


logger = logging.getLogger(__name__)


def _resolve_timeseries_timestamps(timestamps, split_config, df, *, verbose: bool = False):
    """Return ``timestamps`` to drive a forward-walk main split when a time-series cv_strategy is requested.

    When ``timestamps`` is already supplied upstream, returns it unchanged. Otherwise, when
    ``split_config.cv_strategy`` is ``"timeseries"``/``"purged"`` and ``split_config.time_column`` is set,
    reads that column from ``df`` and returns it as the timestamps array (``make_train_test_split`` then
    splits chronologically). Default ``cv_strategy="random"`` -> returns ``timestamps`` (None) unchanged.
    ("purged" routes the same forward-walk as "timeseries" for the main split, plus an embargo gap:
    ``_apply_purge_embargo`` trims the train->holdout boundary and ``_apply_val_test_embargo`` trims the
    val->test boundary, so both boundaries carry the ``cv_purge`` gap.)
    """
    if timestamps is not None:
        return timestamps
    if getattr(split_config, "cv_strategy", "random") not in ("timeseries", "purged"):
        # Lookahead-leakage foot-gun: a time_column is configured but cv_strategy is left at
        # the "random" default, so the MAIN split shuffles rows across time and the model can
        # train on future rows to predict past ones (val/test metrics inflate; production drops).
        # We do NOT silently override the user's split; we WARN loudly so the choice is deliberate.
        if getattr(split_config, "time_column", None):
            logger.warning(
                "Time-data foot-gun: time_column=%r is configured but cv_strategy=%r -> the "
                "train/val/test split is a RANDOM shuffle, ignoring time order (lookahead-leakage "
                "risk: rows from the future can land in train while their past lands in val/test). "
                "Set cv_strategy='timeseries' (forward-walk) or 'purged' (forward-walk + embargo) "
                "to honor the time column.",
                getattr(split_config, "time_column", None),
                getattr(split_config, "cv_strategy", "random"),
            )
        return timestamps
    tcol = getattr(split_config, "time_column", None)
    if not tcol or df is None:
        return timestamps
    try:
        col = df[tcol]
        ts = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
        if verbose:
            logger.info("E2: routing main split as forward-walk on time_column=%r (cv_strategy=%s).", tcol, split_config.cv_strategy)
        return ts
    except Exception as _ts_err:
        logger.warning("E2: could not read time_column=%r for time-series split (%s); falling back to random split.", tcol, _ts_err)
        return timestamps


def _apply_purge_embargo(train_idx, timestamps, purge: int):
    """Drop the most-recent ``purge`` train rows (largest timestamps) to embargo the train/holdout boundary.

    Pure index trim: returns a subset of ``train_idx`` (train only shrinks), used for cv_strategy="purged" so a
    windowed/recurrent label adjacent to the future val/test block cannot leak. No-op when purge<=0, timestamps
    is None, or the trim would empty train. Does not touch val/test.
    """
    if purge <= 0 or timestamps is None or train_idx is None or len(train_idx) <= purge:
        return train_idx
    ti = np.asarray(train_idx)
    ts = np.asarray(timestamps)[ti]
    keep_order = np.argsort(ts, kind="stable")[: ti.size - int(purge)]  # drop the newest `purge` train rows
    return ti[np.sort(keep_order)]


def _apply_val_test_embargo(val_idx, timestamps, purge: int):
    """Drop the most-recent ``purge`` VAL rows (largest timestamps) to embargo the val/test boundary.

    In a forward-walk split the block order is [train][val][test]; ``_apply_purge_embargo`` only gaps
    train->val/test, leaving NO gap between val and test. A windowed/overlapping label on the newest val
    rows shares its label window with the oldest test rows, so val<->test leaks the same way train<->holdout
    would. Trimming the newest ``purge`` val rows inserts the same embargo gap at that boundary: after the
    trim ``max(val_ts) < min(test_ts)`` with at least the label-window separation the purge encodes.

    Pure index trim (val only shrinks); no-op when purge<=0, timestamps is None, or the trim would empty val.
    Does not touch train/test. Symmetric with ``_apply_purge_embargo``.
    """
    if purge <= 0 or timestamps is None or val_idx is None or len(val_idx) <= purge:
        return val_idx
    vi = np.asarray(val_idx)
    ts = np.asarray(timestamps)[vi]
    keep_order = np.argsort(ts, kind="stable")[: vi.size - int(purge)]  # drop the newest `purge` val rows
    return vi[np.sort(keep_order)]


def _phase_train_val_test_split(
    *,
    df: pl.DataFrame | pd.DataFrame | None,
    target_by_type: dict,
    timestamps: np.ndarray | None,
    group_ids: np.ndarray | pd.Series | None,
    group_ids_raw: np.ndarray | pd.Series | None,
    artifacts: Any,
    sequences: list[np.ndarray] | None,
    split_config: Any,
    behavior_config: Any,
    metadata: dict,
    data_dir: str,
    models_dir: str,
    target_name: str,
    model_name: str,
    df_size_mb: float,
    verbose: bool,
) -> "TrainValTestSplitResult":
    """Train/val/test splitting with auto-stratification + group-aware splitting.

    Mutates ``metadata`` in-place with split sizes + per-split details.
    """
    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info("Making train_val_test split...")

    # E2 routing: a time-series cv_strategy + time_column drives the MAIN split as a chronological forward-walk.
    timestamps = _resolve_timeseries_timestamps(timestamps, split_config, df, verbose=verbose)

    # Auto-stratify by target when no timestamps are present (without stratification,
    # rare-imbalance shuffles produce all-class-0 val slices). Three regimes:
    #   (a) single classification target  -> stratify on its ndarray directly
    #   (b) multiple classification targets (e.g. several binary heads) -> stratify on
    #       a composite key built from the row-tuple, encoded as an int class id.
    #       Gated on combined-cardinality <= MAX_COMPOSITE_CARDINALITY so the
    #       sklearn StratifiedShuffleSplit doesn't reject for sparse classes.
    #   (c) multilabel target (N, K)      -> if iterative-stratification is installed,
    #       pass its ndarray through; otherwise fall back to first-label stratification
    #       as a best-effort over the all-classes-fully-balanced corner case.
    # composite_cardinality_cap configurable via TrainingSplitConfig (default 200). sklearn StratifiedShuffleSplit allocates O(n_classes) buckets and requires >=2 samples per class; >200 classes typically means most have <2 rows and the splitter rejects.
    _MAX_COMPOSITE_CARDINALITY = int(getattr(split_config, "composite_cardinality_cap", 200) or 200)
    _bucket_stratify_enabled = bool(getattr(split_config, "bucket_stratify", True))
    _stratify_y = None
    if timestamps is None and isinstance(target_by_type, dict):
        _classification_targets = []
        _multilabel_target = None
        for _tt, _named in target_by_type.items():
            _tt_name = getattr(_tt, "name", str(_tt)).upper()
            if "MULTILABEL" in _tt_name:
                # Multilabel arrives as (N, K) ndarray under one key; capture and stop.
                if isinstance(_named, dict):
                    _ml_vals = next(iter(_named.values()), None)
                else:
                    _ml_vals = _named
                _multilabel_target = _ml_vals
                continue
            if "CLASS" in _tt_name and isinstance(_named, dict):
                for _tn, _tv in _named.items():
                    if _tv is not None:
                        _classification_targets.append(_tv)
        if _multilabel_target is not None:
            try:
                _ml_arr = np.asarray(_multilabel_target)
                if _ml_arr.ndim == 2 and _ml_arr.shape[1] >= 1:
                    # Prefer the proper iterative-stratification path when available.
                    try:
                        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

                        _stratify_y = _ml_arr
                    except ImportError:
                        # Best-effort fallback: stratify on the first label column. Better
                        # than nothing when one of the K labels is the rare class.
                        _first = _ml_arr[:, 0]
                        _u, _c = np.unique(_first, return_counts=True)
                        if len(_u) >= 2 and _c.min() >= 2:
                            _stratify_y = _first
                        else:
                            logger.warning(
                                "Auto-stratify: multilabel first-label fallback has %d "
                                "unique values with min-count=%d (need >=2 + min-count>=2); "
                                "stratification disabled. val/test slices may be class-degenerate.",
                                len(_u),
                                int(_c.min()) if len(_c) else 0,
                            )
            except Exception as _strat_err:
                logger.warning(
                    "Auto-stratify: multilabel build failed (%s: %s); shuffled-only splits.",
                    type(_strat_err).__name__,
                    _strat_err,
                )
                _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
                    else:
                        # Surface so the rare-imbalance scenario (single-class slice OR
                        # rare-class with one sample) isn't misdiagnosed as random-seed
                        # flakiness when val ends up all-class-0. Pre-fix this branch
                        # silently flipped to shuffled-only.
                        logger.warning(
                            "Auto-stratify: single classification target has %d unique "
                            "classes with min-count=%d (need >=2 + min-count>=2); "
                            "stratification disabled. val/test slices may be class-degenerate.",
                            len(_u),
                            int(_c.min()) if len(_c) else 0,
                        )
            except Exception as _strat_err:
                logger.warning(
                    "Auto-stratify: single-target build failed (%s: %s); shuffled-only splits.",
                    type(_strat_err).__name__,
                    _strat_err,
                )
                _stratify_y = None
        elif len(_classification_targets) > 1:
            try:
                _arrs = [np.asarray(_t) for _t in _classification_targets]
                _n = len(_arrs[0])
                if all(_a.ndim == 1 and len(_a) == _n for _a in _arrs):
                    # Composite key: each row maps to an integer class id from
                    # (val_t0, val_t1, ..., val_tK) tuple. np.unique on stacked (N, K)
                    # returns_inverse for the encoding in one pass.
                    _stack = np.stack(_arrs, axis=1)
                    _, _composite_ids = np.unique(_stack, axis=0, return_inverse=True)
                    _u, _c = np.unique(_composite_ids, return_counts=True)
                    if 2 <= len(_u) <= _MAX_COMPOSITE_CARDINALITY and _c.min() >= 2:
                        _stratify_y = _composite_ids
                    elif len(_u) < 2 or _c.min() < 2:
                        logger.warning(
                            "Auto-stratify: composite key has %d distinct row-tuples with "
                            "min-count=%d (need >=2 + min-count>=2); stratification disabled. "
                            "val/test slices may be class-degenerate.",
                            len(_u),
                            int(_c.min()) if len(_c) else 0,
                        )
                    elif len(_u) > _MAX_COMPOSITE_CARDINALITY:
                        # Surface the silent fallback to shuffled-only splits so operators
                        # know auto-stratification was abandoned on multi-head targets that
                        # exceed the composite-cardinality cap; otherwise they re-discover
                        # the all-class-0-val-slice bug under heavy class imbalance.
                        logger.warning(
                            "Auto-stratify: composite key has %d distinct row-tuples > "
                            "_MAX_COMPOSITE_CARDINALITY=%d; falling back to UNstratified "
                            "shuffle splits. Rare-class imbalance may produce all-one-class "
                            "val/test slices. Reduce the number of classification heads or "
                            "pre-compute a stratify_y manually to restore stratification.",
                            len(_u),
                            _MAX_COMPOSITE_CARDINALITY,
                        )
            except Exception:
                _stratify_y = None
        # Regression bucket-stratify: when no classification stratify_y was set above AND bucket_stratify is enabled (default True), bin regression targets into deciles (quartiles for n<5000) and stratify on bucket ids. Prevents heavy-tail / multimodal regression from concentrating tail rows in val or test (the same all-one-class hazard classification stratification already prevents). Skipped when a classification path already populated _stratify_y, when timestamps drive a temporal split, or when only a single distinct target value is present.
        if _stratify_y is None and _bucket_stratify_enabled and timestamps is None:
            _regression_targets = []
            for _tt, _named in target_by_type.items():
                _tt_name = getattr(_tt, "name", str(_tt)).upper()
                if "REGRESSION" in _tt_name or "QUANTILE" in _tt_name:
                    if isinstance(_named, dict):
                        for _tn, _tv in _named.items():
                            if _tv is not None:
                                _regression_targets.append(_tv)
            if _regression_targets:
                try:
                    _y_reg = np.asarray(_regression_targets[0]).astype(np.float64)
                    if _y_reg.ndim == 1 and len(_y_reg) > 0:
                        _finite = np.isfinite(_y_reg)
                        if _finite.sum() >= 10:
                            _n_bins = 10 if _finite.sum() >= 5000 else 4
                            _quantiles = np.linspace(0.0, 1.0, _n_bins + 1)[1:-1]
                            _edges = np.unique(np.quantile(_y_reg[_finite], _quantiles))
                            _buckets = np.digitize(_y_reg, _edges)
                            _u, _c = np.unique(_buckets, return_counts=True)
                            if len(_u) >= 2 and _c.min() >= 2:
                                _stratify_y = _buckets
                                logger.info(
                                    "Bucket-stratify: regression target binned into %d quantile buckets (min/median/max bucket count=%d/%d/%d). Prevents heavy-tail rows from concentrating in val or test.",
                                    len(_u),
                                    int(_c.min()),
                                    int(np.median(_c)),
                                    int(_c.max()),
                                )
                            else:
                                logger.info(
                                    "Bucket-stratify: regression bucket distribution too sparse for stratification (n_buckets=%d, min_count=%d); fall back to shuffled split.",
                                    len(_u),
                                    int(_c.min()) if len(_c) else 0,
                                )
                except Exception as _bucket_err:
                    logger.warning(
                        "Bucket-stratify: regression binning failed (%s: %s); shuffled-only splits.",
                        type(_bucket_err).__name__,
                        _bucket_err,
                    )
    # Group-aware splitting opt-in: when the extractor produced ``group_ids`` and
    # ``split_config.use_groups`` is set, route through GroupShuffleSplit.
    _groups = group_ids if (split_config.use_groups and group_ids is not None and len(group_ids) > 0) else None
    # Group + bucket combination: ``make_train_test_split`` now routes
    # the 1-D case through sklearn's ``StratifiedGroupKFold`` (sklearn
    # >=1.0) which preserves the class / regression-bucket distribution
    # while keeping whole groups together -- both invariants honoured by
    # the same splitter. The 2-D multilabel case still requires
    # ``MultilabelStratifiedGroupKFold`` from iterative-stratification;
    # the splitter detects + falls back to GroupShuffleSplit when that
    # package is absent.
    if _groups is not None and _stratify_y is not None:
        _strat_arr = np.asarray(_stratify_y)
        if _strat_arr.ndim == 2:
            try:
                from iterstrat.ml_stratifiers import (
                    MultilabelStratifiedGroupKFold,
                )

                _has_multilabel_iterstrat = True
            except ImportError:
                _has_multilabel_iterstrat = False
            if not _has_multilabel_iterstrat:
                logger.warning(
                    "Bucket-stratify: 2-D (multilabel) stratify_y + groups; "
                    "iterative-stratification not installed. Splitter will "
                    "stratify on a derived 1-D composite label-combination id "
                    "(joint balance preserved) when feasible, else fall back to "
                    "GroupShuffleSplit with NO multilabel proportion guarantee. "
                    "pip install iterative-stratification for exact "
                    "MultilabelStratifiedGroupKFold.",
                )
        else:
            logger.info(
                "Bucket-stratify: groups + 1-D stratify_y; using "
                "StratifiedGroupKFold (both invariants honoured: whole "
                "groups stay in one split AND class/bucket proportions "
                "preserved across train/val/test).",
            )
    with phase("split_data"):
        # Dynamically derive the kwargs the splitter accepts by inspecting
        # its signature, then drop any TrainingSplitConfig field not in
        # that set. Static exclude lists drift: observed in prod two
        # consecutive TypeErrors (composite_cardinality_cap
        # then bucket_stratify) because new caller-side fields shipped
        # without exclude updates. Caller-side fields documented:
        #   use_groups -- derives _groups upstream
        #   calib_size -- carves a disjoint calibration slice from train (return_calib=True below)
        #   composite_cardinality_cap -- bucket-stratify gate (line ~139)
        #   bucket_stratify -- selects the bucket-stratify branch
        # Signature-derived filtering catches future additions automatically.
        import inspect as _inspect

        _splitter_kwargs = set(_inspect.signature(make_train_test_split).parameters)
        _explicit_kwargs = {"df", "timestamps", "stratify_y", "groups", "return_calib"}
        _cfg_dict = {k: v for k, v in split_config.model_dump().items() if k in _splitter_kwargs and k not in _explicit_kwargs}
        train_idx, val_idx, test_idx, train_details, val_details, test_details, calib_idx, calib_details = make_train_test_split(
            df=df,
            timestamps=timestamps,
            stratify_y=_stratify_y,
            groups=_groups,
            return_calib=True,
            **_cfg_dict,
        )
        # E2 embargo: for cv_strategy="purged", trim the newest train rows adjacent to the future holdout.
        if getattr(split_config, "cv_strategy", "random") == "purged" and getattr(split_config, "cv_purge", 0):
            _n_before = len(train_idx) if train_idx is not None else 0
            train_idx = _apply_purge_embargo(train_idx, timestamps, int(split_config.cv_purge))
            if verbose and train_idx is not None and len(train_idx) < _n_before:
                logger.info("E2 embargo: dropped %d most-recent train rows (cv_purge=%d).", _n_before - len(train_idx), split_config.cv_purge)
            # val<->test embargo: forward-walk order is [train][val][test]; the train trim above only gaps
            # train->holdout. Trim the newest val rows too so a windowed label on the val tail cannot leak
            # into the test head (guaranteeing max(val_ts) < min(test_ts) by the embargo width).
            _nv_before = len(val_idx) if val_idx is not None else 0
            val_idx = _apply_val_test_embargo(val_idx, timestamps, int(split_config.cv_purge))
            if verbose and val_idx is not None and len(val_idx) < _nv_before:
                logger.info("E2 embargo: dropped %d most-recent val rows to gap val<->test (cv_purge=%d).", _nv_before - len(val_idx), split_config.cv_purge)
    if verbose:
        log_ram_usage()

    if data_dir:
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=group_ids_raw,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name=target_name,
            model_name=model_name,
        )

    metadata.update(
        {
            "train_details": train_details,
            "val_details": val_details,
            "test_details": test_details,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
    )
    if calib_idx is not None and len(calib_idx) > 0:
        metadata["calib_details"] = calib_details
        metadata["calib_size_rows"] = len(calib_idx)

    # Compute fairness subgroups from full df BEFORE splitting.
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info("Computed %d fairness subgroups", len(fairness_subgroups))

    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    # Carve the disjoint calib slice (calib_size>0) as its own raw frame, same schema as train_df, while ``df`` is still
    # alive (the caller nulls ``ctx.df`` right after this phase). The slice is a small fraction of train; subsetting is a
    # format-native ``df[calib_idx]`` / ``.iloc`` view, never a full-frame clone. The trainer transforms it through the
    # same fitted pre_pipeline as test and runs predict_proba so finalize can auto-calibrate.
    calib_df = None
    if calib_idx is not None and len(calib_idx) > 0 and df is not None:
        if isinstance(df, pl.DataFrame):
            calib_df = df[calib_idx]
        else:
            calib_df = df.iloc[calib_idx]
    if verbose:
        logger.info("  Split shapes -- train: %s, val: %s, test: %s", _df_shape_str(train_df), _df_shape_str(val_df), _df_shape_str(test_df))
        logger.info("  PHASE 2 total: %s", _elapsed_str(t0_phase2))

    # Split sequences by train/val/test indices (for recurrent models).
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))

    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    # Refresh baseline so the next maybe_clean_ram_and_gpu in the caller sees the post-del state.
    baseline_rss_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")
    if verbose:
        log_ram_usage()

    return TrainValTestSplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        fairness_subgroups=fairness_subgroups,
        fairness_features=fairness_features,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        baseline_rss_mb=baseline_rss_mb,
        calib_idx=calib_idx,
        calib_details=calib_details,
        calib_df=calib_df,
    )


def _phase_auto_detect_feature_types(
    *,
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    train_df_polars_pre: pl.DataFrame | None,
    val_df_polars_pre: pl.DataFrame | None,
    test_df_polars_pre: pl.DataFrame | None,
    cat_features: list[str],
    cat_features_polars: list[str],
    was_polars_input: bool,
    all_models_polars_native: bool,
    pipeline_config: Any,
    feature_types_config: Any,
    metadata: dict,
    verbose: bool,
    train_df_pandas_pre_meta: dict | None = None,
) -> tuple:
    """Auto-detect text + embedding features, optionally drop high-card columns, validate exclusivity, one-time Polars string->Categorical cast.

    Mutates ``metadata`` in-place with ``columns`` and ``cat_features``.
    """
    # Use pre-pipeline view so auto-detection sees original dtypes BEFORE the ordinal encoder converts strings to int codes.
    # Polars: ``train_df_polars_pre`` (frame alias, always populated; polars frames are conceptually immutable through the
    # public API so the alias is safe). Pandas: ``train_df_pandas_pre_meta`` dict, mutation-immune by construction
    # (column names / dtype-strings / cardinality / non-null counts baked at snapshot time). Fallback to post-pipeline
    # ``train_df`` only when both pre-views are absent (legacy callers / feature_types_first=False).
    if was_polars_input:
        detect_df = train_df_polars_pre
    else:
        detect_df = train_df
    # Auto-flip path: when ``skip_categorical_encoding`` was flipped to True
    # because CB+ordinal+declared-cats was the requested config, the pipeline
    # returns an empty ``cat_features`` (no encoder was fitted). The downstream
    # CB Pool builder still needs to know which columns carry pandas
    # ``category`` dtype - otherwise CatBoost detects them and raises
    # "has dtype 'category' but is not in cat_features list". Recover the
    # list from the pre-pipeline dtype snapshot (mutation-immune) when
    # available, falling back to a live ``select_dtypes`` probe on detect_df.
    _post_flip_pandas_cats: list[str] = []
    if not was_polars_input:
        _pre_meta_dtypes = (train_df_pandas_pre_meta or {}).get("dtypes") if isinstance(train_df_pandas_pre_meta, dict) else None
        if _pre_meta_dtypes:
            _post_flip_pandas_cats = [c for c, dt in _pre_meta_dtypes.items() if "category" in str(dt).lower()]
        elif detect_df is not None and hasattr(detect_df, "select_dtypes"):
            try:
                _post_flip_pandas_cats = detect_df.select_dtypes(include=["category"]).columns.tolist()
            except Exception:
                _post_flip_pandas_cats = []
    # Order-preserving dedup: ``metadata["cat_features"]`` feeds CatBoost/Pool and is
    # serialised into the recipe -- ``list(set(...))`` would reorder per PYTHONHASHSEED,
    # making a fixed-random_state run non-reproducible. Keep first-seen input order.
    _seen_cat: set[str] = set()
    raw_cat_features = [c for c in ((cat_features or []) + (cat_features_polars or []) + _post_flip_pandas_cats) if not (c in _seen_cat or _seen_cat.add(c))]  # type: ignore[func-returns-value]  # intentional order-preserving-dedup idiom: set.add()'s None return is used as the falsy side of `or`
    # Honor only strictly-user-declared pl.Categorical columns as already-assigned.
    if was_polars_input and detect_df is not None:
        user_polars_cats = [c for c, dt in zip(detect_df.columns, detect_df.dtypes) if dt == pl.Categorical]
    else:
        user_polars_cats = []
    text_features, embedding_features, auto_high_card_drop = _auto_detect_feature_types(
        detect_df,
        feature_types_config,
        user_polars_cats,
        verbose=verbose,
        pandas_meta=train_df_pandas_pre_meta if not was_polars_input else None,
    )

    # Capture pre-drop column data so dummy_baselines per_group_mean can use these as group
    # keys downstream (tree models drop them to avoid XGB QuantileDMatrix OOM).
    # Audit D P1-6 (2026-05-18): pre-fix loop ran ``_frame[c].to_numpy()`` per column per
    # split -- N independent Arrow batches per split. Now we do ONE ``_frame.select(cols)``
    # per split, materialise that 2D matrix through ``get_pandas_view_of_polars_df`` (split-
    # blocks Arrow bridge, ~32x faster than naive to_pandas on multi-col selects), then
    # peel the per-column numpy arrays from the resulting DataFrame view. Pandas-branch is
    # unchanged because pandas ``_frame[col]`` is already a Series view (no extra copy).
    dropped_high_card_data = {}
    if auto_high_card_drop:
        # Per-split single-pass materialisation for polars frames; pandas branch is per-col
        # (cheap Series view).
        _per_split_views: dict[str, Any] = {}
        for _label, _frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            if _frame is None:
                continue
            _cols = _frame.columns if hasattr(_frame, "columns") else []
            _present = [c for c in auto_high_card_drop if c in _cols]
            if not _present:
                continue
            if isinstance(_frame, pl.DataFrame):
                try:
                    # Single select over ALL needed columns; Arrow split-blocks bridge.
                    from mlframe.training.utils import get_pandas_view_of_polars_df as _get_pd_view

                    _per_split_views[_label] = _get_pd_view(_frame.select(_present))
                except Exception:
                    # Fallback to bare to_pandas on the multi-col select; still 1 batch vs N.
                    try:
                        _per_split_views[_label] = _frame.select(_present).to_pandas()
                    except Exception:
                        _per_split_views[_label] = None
            else:
                _per_split_views[_label] = _frame
        for _col in auto_high_card_drop:
            _col_frames = {}
            for _label in ("train", "val", "test"):
                _view = _per_split_views.get(_label)
                if _view is None:
                    continue
                if _col not in getattr(_view, "columns", []):
                    continue
                try:
                    _col_frames[_label] = np.asarray(_view[_col])
                except Exception as e:
                    logger.debug("swallowed exception in _phase_helpers_fit_split.py: %s", e)
                    continue
            if _col_frames:
                dropped_high_card_data[_col] = _col_frames
        train_df = _drop_cols_df(train_df, auto_high_card_drop)
        val_df = _drop_cols_df(val_df, auto_high_card_drop)
        test_df = _drop_cols_df(test_df, auto_high_card_drop)
        if was_polars_input:
            if train_df_polars_pre is not None:
                train_df_polars_pre = _drop_cols_df(train_df_polars_pre, auto_high_card_drop)
            if val_df_polars_pre is not None:
                val_df_polars_pre = _drop_cols_df(val_df_polars_pre, auto_high_card_drop)
            if test_df_polars_pre is not None:
                test_df_polars_pre = _drop_cols_df(test_df_polars_pre, auto_high_card_drop)
        raw_cat_features = [c for c in raw_cat_features if c not in auto_high_card_drop]
        metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    text_emb_set = set(text_features) | set(embedding_features)
    effective_cat_features = [c for c in raw_cat_features if c not in text_emb_set]
    _validate_feature_type_exclusivity(text_features, embedding_features, effective_cat_features)
    cat_features = effective_cat_features
    metadata["cat_features"] = cat_features

    # One-time Polars string->Enum cast so XGB's arrow bridge doesn't choke on large_string.
    # Use pl.Enum (per-Series, no global cache impact) keyed off the train-only unique set;
    # val/test cast non-strict so OOV becomes null (matches the alignment semantics elsewhere
    # in the suite). pl.Categorical would widen the process-wide string cache (memory rule:
    # reference_polars_global_string_cache). Fixes audit B-P0-3 / Low-B11.
    if was_polars_input and all_models_polars_native and pipeline_config.skip_categorical_encoding and train_df is not None:
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        _keep_as_string = text_emb_set
        _str_cols = [c for c, dt in zip(train_df.columns, train_df.dtypes) if dt in _string_types and c not in _keep_as_string]
        if _str_cols:
            # Wave 72 (2026-05-21): build per-column Enum domain from train+val
            # uniques (NOT train-only). val is the early-stopping detector --
            # if a val-only categorical value gets cast to null silently, ES is
            # biased away from val-rare-cat-sensitive splits. test stays
            # unseen (built from train+val ONLY). Symmetric with dict-alignment
            # in _phase_polars_fixes.py.
            _enum_domains: dict[str, list[str]] = {}
            # Track per-column val-only category counts so operators see the implicit Enum-domain widening (memory: feedback_observability_loud).
            # Without this log the train+val union is silent. Behaviour stays unchanged; only observability improves.
            _val_only_diag: dict[str, tuple[int, list]] = {}
            for _c in _str_cols:
                try:
                    _u_train = train_df.select(pl.col(_c).drop_nulls().unique())[_c].to_list()
                    _u_val: list = []
                    if val_df is not None and _c in set(val_df.columns):
                        try:
                            _u_val = val_df.select(pl.col(_c).drop_nulls().unique())[_c].to_list()
                        except Exception:
                            _u_val = []
                    _enum_domains[_c] = sorted(set(_u_train) | set(_u_val), key=str)
                    _train_set = set(_u_train)
                    _val_only = [v for v in _u_val if v not in _train_set]
                    if _val_only:
                        _val_only_diag[_c] = (len(_val_only), _val_only[:5])
                except Exception as e:
                    logger.debug("swallowed exception in _phase_helpers_fit_split.py: %s", e)
                    pass
            if _val_only_diag:
                # INFO-level. Per Wave 72 contract this widening is intentional (val=ES detector must not silently null-cast); the log only surfaces what was previously invisible.
                _summary = ", ".join(f"{c}:{n}" for c, (n, _) in _val_only_diag.items())
                _samples = ", ".join(f"{c}={vs}" for c, (_, vs) in list(_val_only_diag.items())[:3])
                logger.info(
                    "[enum-domain] Enum domain widened to include val-only categories on %d col(s): %s. Sample val-only values: %s",
                    len(_val_only_diag),
                    _summary,
                    _samples,
                )

            def _enum_cast(df, strict: bool, split_name: str | None = None):
                if df is None:
                    return df
                _existing = set(df.columns)
                _exprs = []
                _affected_cols = []
                for _c in _str_cols:
                    if _c not in _existing or _c not in _enum_domains:
                        continue
                    _exprs.append(pl.col(_c).cast(pl.Enum(_enum_domains[_c]), strict=strict))
                    _affected_cols.append(_c)
                if not _exprs:
                    return df
                # Wave 72 (2026-05-21): quantify silent OOV-nulling so operators
                # can see how many rows got cast-failed (was invisible before).
                _null_pre = {c: int(df[c].null_count()) for c in _affected_cols}
                out = df.with_columns(_exprs)
                if split_name is not None and not strict:
                    _null_deltas = {c: int(out[c].null_count()) - _null_pre[c] for c in _affected_cols}
                    _nonzero = {c: d for c, d in _null_deltas.items() if d > 0}
                    if _nonzero:
                        logger.info(
                            "[enum-cast] %s split: %d col(s) had OOV nulls cast-failed (cols=%s)",
                            split_name,
                            len(_nonzero),
                            _nonzero,
                        )
                return out

            train_df = _enum_cast(train_df, strict=True)
            # val cast is now strict-but-domain-includes-val by construction;
            # any cast failure here would be a logic bug, so strict=True.
            val_df = _enum_cast(val_df, strict=True)
            test_df = _enum_cast(test_df, strict=False, split_name="test")
            train_df_polars_pre = _enum_cast(train_df_polars_pre, strict=True)
            val_df_polars_pre = _enum_cast(val_df_polars_pre, strict=True)
            test_df_polars_pre = _enum_cast(test_df_polars_pre, strict=False, split_name="test (polars_pre)")
            if verbose:
                logger.info("  Cast Polars string columns -> Enum once (shared across model loop)")

    if verbose and (text_features or embedding_features):
        logger.info("  Feature types -- text: %s, embedding: %s, cat: %s", text_features, embedding_features, cat_features or "(none)")

    return (
        train_df,
        val_df,
        test_df,
        train_df_polars_pre,
        val_df_polars_pre,
        test_df_polars_pre,
        text_features,
        embedding_features,
        cat_features,
        text_emb_set,
        dropped_high_card_data,
    )


# Sibling-module re-export. The 517-LOC ``_phase_fit_pipeline`` body
# lives in ``_phase_helpers_fit_pipeline.py`` so this file stays below
# the 1k-LOC monolith threshold. ``_phase_train_val_test_split`` and
# ``_phase_auto_detect_feature_types`` remain inline above.
from ._phase_helpers_fit_pipeline import _phase_fit_pipeline
