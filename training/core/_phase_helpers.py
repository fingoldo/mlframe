"""Phase helper functions extracted from ``core/utils.py``.

Plot style overrides, composite discovery metadata, outlier detection,
Polars conversion, cardinality logging, feature type detection,
pipeline fitting, train/val/test splitting, data loading.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from sklearn.pipeline import Pipeline

from ..configs import TargetTypes

logger = logging.getLogger(__name__)

def _apply_plot_style_overrides(
    *,
    matplotlib_style=None,
    matplotlib_rcparams=None,
    plotly_template=None,
    verbose: bool = False,
) -> None:
    """Apply ``ReportingConfig.matplotlib_style`` + ``matplotlib_rcparams`` +
    ``plotly_template`` to the process-wide plot-backend state.

    Two independent backends:
    - matplotlib: ``plt.style.use(...)`` + ``plt.rcParams.update(...)``.
    - plotly:     ``plotly.io.templates.default = ...``.

    Each backend's override is applied independently -- the user can set
    just matplotlib, just plotly, or both (recommended for a unified look:
    eg ``matplotlib_style="ggplot"`` + ``plotly_template="ggplot2"``).

    All fields are ``None``-defaultable -- when None, the user's
    pre-suite settings are preserved untouched. When set, they're
    applied PROCESS-WIDE (mirrors the ``plot_inline_display`` plumbing
    semantics: "set once, see everywhere"; not reverted on suite exit so
    a long-running notebook session sees the override across cells).

    Failures are logged at WARNING level and don't abort the suite --
    matplotlib raises ``OSError`` on unknown style names, plotly raises
    ``ValueError`` on unknown template names; both surface as one-line
    warnings so the operator notices the typo without losing the whole
    training run.
    """
    if (matplotlib_style is None and not matplotlib_rcparams
            and plotly_template is None):
        return

    # matplotlib branch
    if matplotlib_style is not None or matplotlib_rcparams:
        try:
            import matplotlib.pyplot as plt
        except Exception as _imp_err:
            logger.warning(
                "[plot_style] matplotlib import failed (%s); matplotlib "
                "style override skipped.", _imp_err,
            )
            plt = None
        if plt is not None:
            if matplotlib_style is not None:
                try:
                    plt.style.use(matplotlib_style)
                    if verbose:
                        logger.info(
                            "[plot_style] matplotlib style: %r", matplotlib_style,
                        )
                except Exception as _style_err:
                    logger.warning(
                        "[plot_style] plt.style.use(%r) failed: %s. "
                        "Continuing with the current matplotlib style.",
                        matplotlib_style, _style_err,
                    )
            if matplotlib_rcparams:
                try:
                    plt.rcParams.update(dict(matplotlib_rcparams))
                    if verbose:
                        logger.info(
                            "[plot_style] applied %d matplotlib rcParams "
                            "override(s): %s",
                            len(matplotlib_rcparams),
                            sorted(matplotlib_rcparams.keys()),
                        )
                except Exception as _rc_err:
                    logger.warning(
                        "[plot_style] plt.rcParams.update(%r) failed: %s. "
                        "Some matplotlib keys may not have been applied.",
                        matplotlib_rcparams, _rc_err,
                    )

    # plotly branch
    if plotly_template is not None:
        try:
            import plotly.io as pio
        except Exception as _imp_err:
            logger.warning(
                "[plot_style] plotly import failed (%s); plotly template "
                "override skipped.", _imp_err,
            )
            return
        try:
            pio.templates.default = plotly_template
            if verbose:
                logger.info(
                    "[plot_style] plotly template: %r", plotly_template,
                )
        except Exception as _tpl_err:
            logger.warning(
                "[plot_style] plotly templates.default = %r failed: %s. "
                "Continuing with the current plotly template.",
                plotly_template, _tpl_err,
            )


def _defensive_copy_and_expand_multilabel_regression(
    *,
    target_by_type,
    composite_target_discovery_config,
    metadata,
):
    """Defensive copy of ``target_by_type`` + multilabel-target expansion.

    Two responsibilities:

    1. **Defensive shallow copy** of the outer dict + per-type inner dicts so
       composite-discovery's downstream additions to
       ``target_by_type[REGRESSION]`` don't mutate the FTE-cached value (FTE
       can be shared across suite invocations).

    2. **R3.18 multilabel expansion** (per-target strategy). 2-D regression
       targets of shape ``(n, K)`` get expanded into K independent 1-D
       targets named ``{target}_out{j}``. Caller can opt out via
       ``multilabel_strategy="skip"`` to preserve the legacy "skip with
       metadata note" behaviour. Records the expansion in
       ``metadata["multilabel_target_expansion"]``.

    Returns
    -------
    new_target_by_type: dict
        Defensive copy with multilabel sub-targets substituted in.
    """
    new_target_by_type = {
        tt: dict(named) if isinstance(named, dict) else named
        for tt, named in target_by_type.items()
    }
    ml_strategy = str(getattr(
        composite_target_discovery_config,
        "multilabel_strategy", "per_target",
    ))
    if ml_strategy == "per_target":
        expanded = dict(new_target_by_type[TargetTypes.REGRESSION])
        ml_expanded_map: Dict[str, List[str]] = {}
        for _tn, _tv in list(new_target_by_type[TargetTypes.REGRESSION].items()):
            _arr = np.asarray(_tv)
            if _arr.ndim == 2 and _arr.shape[1] >= 1:
                sub_names = []
                for _j in range(_arr.shape[1]):
                    _sub_name = f"{_tn}_out{_j}"
                    expanded[_sub_name] = _arr[:, _j]
                    sub_names.append(_sub_name)
                expanded.pop(_tn, None)
                ml_expanded_map[_tn] = sub_names
                logger.info(
                    "[CompositeTargetDiscovery] R3.18: multilabel "
                    "target '%s' (shape=%s) expanded into %d 1-D "
                    "sub-targets: %s",
                    _tn, _arr.shape, _arr.shape[1], sub_names,
                )
        new_target_by_type[TargetTypes.REGRESSION] = expanded
        if ml_expanded_map:
            metadata.setdefault("multilabel_target_expansion", {})[
                str(TargetTypes.REGRESSION)
            ] = ml_expanded_map
    return new_target_by_type


def _init_composite_discovery_metadata(
    *,
    composite_target_discovery_config,
    target_by_type,
    mlframe_models,
    metadata,
):
    """Phase 4.6 (Composite-Target Discovery prologue).

    Initialises the three composite-discovery metadata buckets
    (``composite_target_specs``, ``composite_target_failures``,
    ``composite_target_filter_drops``), snapshots the env signature when
    discovery is enabled (so v2-loaded suites can detect package drift),
    detects which model families are on GPU (for the deferred R10c bug-#6
    warning that fires only when K > 0), and records skip-reasons for every
    non-regression target type so callers can tell "considered & rejected"
    apart from "never looked".

    Mutates ``metadata`` in-place.

    Returns
    -------
    (gpu_families, kept_spec_total):
        ``gpu_families`` -- list of model families detected on GPU (empty
        when discovery disabled); ``kept_spec_total`` -- always 0 here,
        returned so the caller can keep the running counter as the
        per-target discovery loop proceeds.
    """
    metadata["composite_target_specs"] = {}
    metadata["composite_target_failures"] = {}
    metadata["composite_target_filter_drops"] = {}

    gpu_families: List[str] = []
    if composite_target_discovery_config.enabled:
        from ..composite import env_signature as _env_sig, detect_gpu_in_use as _detect_gpu
        metadata["composite_target_env_signature"] = _env_sig()
        gpu_families = _detect_gpu(mlframe_models or [])

    # Skip-reasons for non-regression target types.
    for _tt_skip, _named_skip in target_by_type.items():
        if not isinstance(_named_skip, dict):
            continue
        if _tt_skip == TargetTypes.REGRESSION:
            continue
        reason = None
        if _tt_skip == TargetTypes.LEARNING_TO_RANK:
            reason = "ltr_unsupported_pairwise_breaks_with_residual"
        elif _tt_skip == TargetTypes.MULTICLASS_CLASSIFICATION:
            reason = "multiclass_unsupported_no_residual_semantics"
        elif _tt_skip == TargetTypes.MULTILABEL_CLASSIFICATION:
            reason = "multilabel_classification_unsupported"
        elif _tt_skip == TargetTypes.QUANTILE_REGRESSION:
            reason = "quantile_regression_unsupported_per_quantile_inverse_undefined"
        elif _tt_skip == TargetTypes.BINARY_CLASSIFICATION:
            reason = "binary_classification_unsupported_init_score_logit_offset"
        if reason is not None:
            for _tn_skip in _named_skip:
                metadata["composite_target_failures"].setdefault(
                    str(_tt_skip), {})[_tn_skip] = [{
                        "name": _tn_skip, "kept": False, "rejected": True,
                        "reason": reason,
                    }]
    return gpu_families, 0


def _phase_global_outlier_detection(
    *,
    train_df_pd,
    val_df_pd,
    train_df_polars,
    val_df_polars,
    train_idx,
    val_idx,
    target_by_type,
    outlier_detector,
    od_val_set,
    baseline_rss_mb,
    df_size_mb,
    metadata,
    verbose,
):
    """Phase 4.5 (Global Outlier Detection, once before model loops).

    Steps:
    1. Flatten ``target_by_type`` to ``{target_type/target_name: values}`` for
       the OD class-balance pre-check (so a detector that would eliminate the
       entire minority class for any classification target gets rejected
       upfront).
    2. Run ``_apply_outlier_detection_global`` (handles pandas OD via sklearn,
       returns boolean masks for train/val).
    3. Record metadata: pre/post-OD sizes, n_outliers_dropped per split.
    4. Apply the same boolean mask to the Polars fastpath frames so the
       Polars-native training loop and the OD-filtered targets stay aligned.

    Returns
    -------
    9-tuple:
        filtered_train_df, filtered_val_df,
        filtered_train_idx, filtered_val_idx,
        train_od_idx, val_od_idx,
        outlier_detection_result (dict),
        train_df_polars (filtered), val_df_polars (filtered)
    """
    _targets_flat_for_classbalance = {}
    for _tt, _named in target_by_type.items():
        if isinstance(_named, dict):
            for _tn, _tv in _named.items():
                _targets_flat_for_classbalance[f"{_tt}/{_tn}"] = _tv
    (filtered_train_df, filtered_val_df, filtered_train_idx, filtered_val_idx,
     train_od_idx, val_od_idx) = _apply_outlier_detection_global(
        train_df=train_df_pd,
        val_df=val_df_pd,
        train_idx=train_idx,
        val_idx=val_idx,
        outlier_detector=outlier_detector,
        od_val_set=od_val_set,
        verbose=verbose,
        baseline_rss_mb=baseline_rss_mb,
        df_size_mb=df_size_mb,
        targets_for_classbalance=_targets_flat_for_classbalance or None,
    )

    outlier_detection_result = {
        "train_od_idx": train_od_idx,
        "val_od_idx": val_od_idx,
    }

    if outlier_detector is not None:
        n_train_pre_od = len(train_df_pd) if train_df_pd is not None else None
        n_val_pre_od = len(val_df_pd) if val_df_pd is not None else None
        n_train_post_od = int(train_od_idx.sum()) if train_od_idx is not None else n_train_pre_od
        n_val_post_od = int(val_od_idx.sum()) if val_od_idx is not None else n_val_pre_od
        n_train_dropped = (n_train_pre_od - n_train_post_od) if n_train_pre_od is not None else 0
        n_val_dropped = (n_val_pre_od - n_val_post_od) if (n_val_pre_od is not None and val_od_idx is not None) else 0
        metadata["outlier_detection"] = {
            "applied": True,
            "n_outliers_dropped_train": int(n_train_dropped),
            "n_outliers_dropped_val": int(n_val_dropped),
            "train_size_after_od": int(n_train_post_od) if n_train_post_od is not None else None,
            "val_size_after_od": int(n_val_post_od) if n_val_post_od is not None else None,
        }
    else:
        metadata["outlier_detection"] = {"applied": False}

    # Keep Polars fastpath DFs in sync with OD-filtered targets.
    if train_od_idx is not None and train_df_polars is not None:
        train_df_polars = train_df_polars.filter(pl.Series(train_od_idx))
    if val_od_idx is not None and val_df_polars is not None:
        val_df_polars = val_df_polars.filter(pl.Series(val_od_idx))

    return (
        filtered_train_df, filtered_val_df,
        filtered_train_idx, filtered_val_idx,
        train_od_idx, val_od_idx,
        outlier_detection_result,
        train_df_polars, val_df_polars,
    )


def _phase_pandas_conversion_and_cat_prep(
    *,
    train_df,
    val_df,
    test_df,
    train_df_polars_pre,
    val_df_polars_pre,
    test_df_polars_pre,
    cat_features,
    was_polars_input,
    all_models_polars_native,
    needs_polars_pre_clone,
    mlframe_models,
    recurrent_models,
    rfecv_models,
    baseline_rss_mb,
    df_size_mb,
    verbose,
):
    """Phase 4 pre-loop (pandas conversion + CatBoost cat prep + Polars release).

    Two main responsibilities:

    1. **Pandas conversion gating**. Skip the polars->pandas conversion entirely
       when all models are Polars-native (CB/XGB on supported builds) OR when
       only non-native sklearn models block the fastpath (those do their own
       lazy conversion later). Forces conversion when recurrent_models or
       rfecv_models are requested (those paths predate Polars support).

    2. **CatBoost cat-feature prep + size capture**. Calls
       ``prepare_df_for_catboost`` on the pandas views when actually needed
       (i.e. when conversion wasn't skipped). Captures Polars-side estimated
       sizes BEFORE conversion to avoid the pathological pandas
       ``memory_usage(deep=True)`` scan downstream.

    3. **Post-pandas Polars release**. When a clone was made, frees the
       post-pipeline Polars frames after conversion -- the Arrow-backed
       pandas views hold their own buffer references, so the Polars objects
       are no longer needed (~100 GB peak saved on the user's 4M-row TVT run).

    Returns
    -------
    13-tuple:
        train_df_pd, val_df_pd, test_df_pd,
        train_df_polars, val_df_polars, test_df_polars,
        train_df, val_df, test_df (possibly cleared to None on Polars release),
        train_df_size_bytes_cached, val_df_size_bytes_cached,
        can_skip_pandas_conv, baseline_rss_mb_refreshed
    """
    if verbose:
        logger.info("Zero-copy conversion to pandas...")

    # Pre-pipeline Polars references for the Polars fastpath.
    train_df_polars = train_df_polars_pre
    val_df_polars = val_df_polars_pre
    test_df_polars = test_df_polars_pre

    # Re-resolve strategies locally -- cheap O(M) lookup.
    strategies_for_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []

    _has_rfecv = bool(rfecv_models)
    _has_non_native_mlframe_strategy = was_polars_input and not all_models_polars_native
    can_skip_pandas_conv = (
        was_polars_input
        and not recurrent_models and not _has_rfecv
        and (all_models_polars_native or _has_non_native_mlframe_strategy)
    )

    # Pre-conversion Polars size capture (Fix 3B).
    train_df_size_bytes_cached: Optional[float] = None
    val_df_size_bytes_cached: Optional[float] = None
    if was_polars_input:
        try:
            if isinstance(train_df, pl.DataFrame):
                train_df_size_bytes_cached = float(train_df.estimated_size())
            if val_df is not None and isinstance(val_df, pl.DataFrame):
                val_df_size_bytes_cached = float(val_df.estimated_size())
        except Exception:
            train_df_size_bytes_cached = None
            val_df_size_bytes_cached = None

    if can_skip_pandas_conv:
        train_df_pd, val_df_pd, test_df_pd = train_df, val_df, test_df
        if verbose:
            if all_models_polars_native:
                logger.info("  Skipped pandas conversion -- all models are Polars-native")
            else:
                non_native = [
                    m for m, s in zip(mlframe_models or [], strategies_for_check)
                    if not s.supports_polars
                ]
                logger.info(
                    "  Deferred pandas conversion -- Polars-native models run on the fastpath; "
                    "non-native %s will convert lazily at their strategy branch.",
                    non_native,
                )
    else:
        if verbose:
            reasons = []
            if not was_polars_input:
                reasons.append("input is not a Polars DataFrame")
            if not all_models_polars_native:
                non_native = [
                    m for m, s in zip(mlframe_models or [], strategies_for_check)
                    if not s.supports_polars
                ]
                reasons.append(
                    f"non-Polars-native models requested: {non_native}"
                    if non_native
                    else "all_models_polars_native=False (no strategies)"
                )
            if recurrent_models:
                reasons.append(f"recurrent_models={recurrent_models}")
            if _has_rfecv:
                reasons.append(f"rfecv_models={rfecv_models}")
            logger.info(
                "  polars->pandas conversion needed because: %s",
                "; ".join(reasons) or "unknown",
            )
        train_df_pd, val_df_pd, test_df_pd = _convert_dfs_to_pandas(train_df, val_df, test_df, verbose=verbose)

    # CatBoost cat-feature prep.
    if cat_features and not can_skip_pandas_conv:
        if verbose:
            logger.info("Preparing %d categorical features for CatBoost: %s", len(cat_features), cat_features)
        for df_pd in [train_df_pd, val_df_pd, test_df_pd]:
            if df_pd is not None:
                prepare_df_for_catboost(df_pd, cat_features)
    elif cat_features and can_skip_pandas_conv and verbose:
        logger.info(
            "Skipping pandas-side CatBoost prep for %d categorical "
            "features -- Polars fastpath receives the DFs natively.",
            len(cat_features),
        )

    # B2 -- post-pipeline Polars release.
    if was_polars_input and needs_polars_pre_clone:
        train_df = val_df = test_df = None
        baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-pipeline Polars release")
        if verbose:
            logger.info("  Released post-pipeline Polars DFs (pandas views retained)")

    if verbose:
        log_ram_usage()

    return (
        train_df_pd, val_df_pd, test_df_pd,
        train_df_polars, val_df_polars, test_df_polars,
        train_df, val_df, test_df,
        train_df_size_bytes_cached, val_df_size_bytes_cached,
        can_skip_pandas_conv, baseline_rss_mb,
    )


def _log_cardinality_and_drift_snapshot(
    *,
    train_df,
    val_df,
    test_df,
    cat_features,
    text_features,
    embedding_features,
) -> None:
    """Pre-train cardinality + val/test drift logging (pure side-effect).

    Cardinality: without this, a native XGB/CB crash on high-cardinality
    categoricals leaves us guessing at the input.

    Drift: for time-ordered splits (the common case here), val and test can
    contain category values that never appeared in train -- XGB 3.x on
    Windows crashes silently during val IterativeDMatrix construction when
    this happens (observed 2026-04-20 on prod_jobsdetails). Helper emits a
    WARNING for any column with non-trivial train-vs-val drift along with a
    concrete healing suggestion keyed on the train-side cardinality so the
    operator sees the crash suspect BEFORE the kernel dies.

    Skip if cardinality > 100k (text-sized columns): the anti-join is
    expensive and unseen-category semantics don't cleanly apply to free-text
    columns (CB handles them via TF-IDF, XGB drops them).

    Pure-logging helper -- no return value, no mutation of any inputs.
    """
    all_cat_cols = list(cat_features or []) + list(text_features or []) + list(embedding_features or [])
    if not (all_cat_cols and train_df is not None):
        return
    try:
        _DRIFT_SKIP_CARD = 100_000
        is_polars = isinstance(train_df, pl.DataFrame)
        pairs = []
        for c in all_cat_cols:
            if c not in train_df.columns:
                continue
            if is_polars:
                n_unique = train_df[c].n_unique()
            else:
                n_unique = int(train_df[c].nunique(dropna=False))
            pairs.append((c, n_unique))
        pairs.sort(key=lambda x: -x[1])
        summary = ", ".join(f"{c}:{n:_}" for c, n in pairs)
        logger.info("  Categorical cardinalities (train, n_unique, desc): %s", summary)

        # Drift log: val/test categories not seen in train.
        if is_polars and val_df is not None and test_df is not None and val_df.height > 0:
            drift_rows = []
            for c, card_train in pairs:
                if card_train > _DRIFT_SKIP_CARD:
                    continue
                if c not in val_df.columns or c not in test_df.columns:
                    continue
                tr_uniq = train_df.select(pl.col(c).drop_nulls().unique().alias(c))
                v_uniq  = val_df.select(pl.col(c).drop_nulls().unique().alias(c))
                te_uniq = test_df.select(pl.col(c).drop_nulls().unique().alias(c))
                val_only  = v_uniq.join(tr_uniq, on=c, how="anti").height
                test_only = te_uniq.join(tr_uniq, on=c, how="anti").height
                drift_rows.append((c, card_train, val_only, test_only))

            if drift_rows:
                drift_rows.sort(key=lambda x: -x[2])
                drift_summary = ", ".join(
                    f"{c}:val_only={v},test_only={t}"
                    for c, _, v, t in drift_rows if v > 0 or t > 0
                ) or "(none)"
                logger.info("  Category drift (val/test values missing from train): %s", drift_summary)

                # WARN + healing-suggestion for non-trivial train-vs-val drift.
                # Test-side drift reported above for visibility but NOT used
                # in healing decisions (would leak test info into training).
                for c, card_tr, v_only, t_only in drift_rows:
                    if v_only == 0 and t_only == 0:
                        continue
                    v_frac = v_only / max(card_tr, 1)
                    if v_only >= 5 or v_frac >= 0.05:
                        if card_tr >= 1000:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) hash-bucket via FeatureHasher / target-encoding "
                                f"(card {card_tr:_} >= 1 000 -> model will memorize train-only "
                                f"values and generalize poorly on val/test);\n"
                                f"          b) drop '{c}' from cat_features and keep only the "
                                f"top-K most frequent (K=100-300) as one-hot, route the rest "
                                f"into an '__OTHER__' bucket;\n"
                                f"          c) drop '{c}' entirely if it's an identifier or "
                                f"free-text field -- promote to text_features via use_text_features=True "
                                f"so CatBoost handles it natively and other backends ignore it."
                            )
                        elif card_tr >= 100:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) target-encoding (CatBoostEncoder) to collapse "
                                f"{card_tr:_} levels into a continuous feature;\n"
                                f"          b) keep top-K by train frequency, bucket the rest "
                                f"into '__OTHER__' before fit (K~=30-80)."
                            )
                        else:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) add an explicit '__UNSEEN__' bucket in the "
                                f"Enum domain so val values absent from train resolve to a "
                                f"known category instead of raising;\n"
                                f"          b) widen the training window (temporal split) so "
                                f"val_only categories are observed at fit time."
                            )
                        logger.warning(
                            f"  Category drift suspect: {c} -- val has {v_only} categories "
                            f"({v_frac:.1%} of train card {card_tr:_}) that train never saw. "
                            f"XGB/CB may crash when constructing val DMatrix with ref=train.\n"
                            f"{_healing}"
                        )
    except Exception as _e:
        logger.warning(f"  Failed to compute categorical cardinality/drift: {_e}")


def _phase_auto_detect_feature_types(
    *,
    train_df,
    val_df,
    test_df,
    train_df_polars_pre,
    val_df_polars_pre,
    test_df_polars_pre,
    cat_features,
    cat_features_polars,
    was_polars_input,
    all_models_polars_native,
    pipeline_config,
    feature_types_config,
    metadata,
    verbose,
):
    """Phase 3.5 (Auto-detect text + embedding features) of
    ``train_mlframe_models_suite``.

    Steps:
    1. Run ``_auto_detect_feature_types`` on the pre-pipeline frame (original
       dtypes) using the pre-pipeline cat_features list + user-declared
       Polars categoricals.
    2. When ``use_text_features=False``, drop the high-card columns from the
       train/val/test splits AND the pre-pipeline clones; capture pre-drop
       column data for dummy-baselines ``per_group_mean`` downstream.
    3. Compute ``effective_cat_features`` (raw cat - text/embedding).
    4. Validate feature-type exclusivity.
    5. One-time Polars string -> Categorical cast across all frames so XGB's
       arrow bridge doesn't choke on ``large_string`` later.

    Mutates ``metadata`` in-place with ``columns`` (post-drop) and
    ``cat_features`` (post-effective).

    Returns
    -------
    11-tuple:
        train_df, val_df, test_df,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        text_features, embedding_features, cat_features,
        text_emb_set, dropped_high_card_data
    """
    # Use pre-pipeline DF for auto-detection (original dtypes preserved).
    detect_df = train_df_polars_pre if was_polars_input else train_df
    raw_cat_features = list(set((cat_features or []) + (cat_features_polars or [])))
    # Honor ONLY strictly-user-declared pl.Categorical columns as already-assigned.
    if was_polars_input:
        user_polars_cats = [
            c for c, dt in zip(detect_df.columns, detect_df.dtypes)
            if dt == pl.Categorical
        ]
    else:
        user_polars_cats = []
    text_features, embedding_features, auto_high_card_drop = _auto_detect_feature_types(
        detect_df, feature_types_config, user_polars_cats, verbose=verbose,
    )

    # Capture pre-drop column DATA so dummy_baselines per_group_mean can use
    # these as group keys downstream. Tree models drop them to avoid XGB
    # QuantileDMatrix OOM, but a simple groupby gives an excellent baseline.
    dropped_high_card_data = {}
    if auto_high_card_drop:
        for _col in auto_high_card_drop:
            _col_frames = {}
            for _label, _frame in (("train", train_df), ("val", val_df), ("test", test_df)):
                if _frame is None:
                    continue
                _cols = _frame.columns if hasattr(_frame, "columns") else []
                if _col not in _cols:
                    continue
                try:
                    if isinstance(_frame, pl.DataFrame):
                        _col_frames[_label] = _frame[_col].to_numpy()
                    else:
                        _col_frames[_label] = np.asarray(_frame[_col])
                except Exception:
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

    # One-time Polars string->Categorical cast shared across all models.
    if was_polars_input and all_models_polars_native and pipeline_config.skip_categorical_encoding:
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        _keep_as_string = text_emb_set
        def _precast_strings(df):
            if df is None:
                return df
            str_cols = [c for c, dt in zip(df.columns, df.dtypes)
                        if dt in _string_types and c not in _keep_as_string]
            return df.with_columns([pl.col(c).cast(pl.Categorical) for c in str_cols]) if str_cols else df
        _pre_train = _precast_strings(train_df)
        if _pre_train is not train_df:
            train_df = _pre_train
            val_df = _precast_strings(val_df)
            test_df = _precast_strings(test_df)
            train_df_polars_pre = _precast_strings(train_df_polars_pre)
            val_df_polars_pre = _precast_strings(val_df_polars_pre)
            test_df_polars_pre = _precast_strings(test_df_polars_pre)
            if verbose:
                logger.info("  Cast Polars string columns -> Categorical once (shared across model loop)")

    if verbose and (text_features or embedding_features):
        logger.info("  Feature types -- text: %s, embedding: %s, cat: %s", text_features, embedding_features, cat_features or '(none)')

    return (
        train_df, val_df, test_df,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        text_features, embedding_features, cat_features,
        text_emb_set, dropped_high_card_data,
    )


def _phase_fit_pipeline(
    *,
    train_df,
    val_df,
    test_df,
    mlframe_models,
    pipeline_config,
    preprocessing_config,
    feature_types_config,
    preprocessing_extensions,
    metadata,
    verbose,
):
    """Phase 3 (Pipeline Fitting & Transformation) of ``train_mlframe_models_suite``.

    Decomposes datetime columns BEFORE the pre-pipeline clone, saves
    Polars originals for the fastpath when needed, runs
    ``fit_and_transform_pipeline`` (categorical encoding + imputation +
    scaling + ensure_float32), then applies any
    ``PreprocessingExtensionsConfig`` (custom scaler / poly / dim-reducer).

    Mutates ``metadata`` in-place with ``pipeline``, ``extensions_pipeline``,
    ``cat_features``, ``columns``.

    Returns
    -------
    15-tuple:
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config (possibly updated), preprocessing_extensions (possibly normalised)
    """
    t0_phase3 = timer()
    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    # Track if input is Polars before pipeline transformation
    was_polars_input = isinstance(train_df, pl.DataFrame)

    # Resolve strategies once for subsequent polars-native gating (avoids redundant lookups).
    _strategies_for_polars_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []
    all_models_polars_native = bool(_strategies_for_polars_check) and all(
        s.supports_polars for s in _strategies_for_polars_check
    )

    # Auto-skip categorical encoding when all models handle categoricals natively.
    if was_polars_input and not pipeline_config.skip_categorical_encoding:
        if all_models_polars_native:
            pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
            if verbose:
                logger.info("  All models %s support Polars natively -- skipping categorical encoding in pipeline", mlframe_models)

    # 2026-04-24 (fuzz extension): datetime columns must be decomposed
    # BEFORE the pre-pipeline clone, otherwise ``train_df_polars_pre`` and
    # friends retain the raw datetime and reach downstream (linear
    # pre_pipeline, MRMR, sklearn encoders, CB Pool) where numpy /
    # sklearn / CB all raise on DateTime64DType.
    def _detect_datetime_cols(df_):
        if df_ is None:
            return []
        if isinstance(df_, pl.DataFrame):
            return [name for name, dt in df_.schema.items()
                    if isinstance(dt, (pl.Datetime, pl.Date))]
        if hasattr(df_, "dtypes"):
            return [c for c in df_.columns
                    if pd.api.types.is_datetime64_any_dtype(df_[c])]
        return []

    _dt_cols = _detect_datetime_cols(train_df)
    if _dt_cols:
        from mlframe.feature_engineering.basic import create_date_features
        _dt_methods = {
            "day": np.int8,
            "weekday": np.int8,
            "month": np.int8,
            "hour": np.int8,
        }
        if verbose:
            logger.info(
                "Decomposing %d datetime column(s) into numeric features "
                "(day/weekday/month/hour) before pre-pipeline clone: %s",
                len(_dt_cols), _dt_cols,
            )
        train_df = create_date_features(
            train_df, cols=_dt_cols, delete_original_cols=True,
            methods=_dt_methods,
        )
        if val_df is not None:
            v_cols = [c for c in _dt_cols if c in val_df.columns]
            if v_cols:
                val_df = create_date_features(
                    val_df, cols=v_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )
        if test_df is not None:
            t_cols = [c for c in _dt_cols if c in test_df.columns]
            if t_cols:
                test_df = create_date_features(
                    test_df, cols=t_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )

    # Save pre-pipeline Polars originals for the Polars fastpath.
    needs_polars_pre_clone = (
        was_polars_input
        and not pipeline_config.skip_categorical_encoding
        and pipeline_config.categorical_encoding is not None
    )
    if was_polars_input:
        if needs_polars_pre_clone:
            train_df_polars_pre = train_df.clone()
            val_df_polars_pre = val_df.clone() if isinstance(val_df, pl.DataFrame) else None
            test_df_polars_pre = test_df.clone() if isinstance(test_df, pl.DataFrame) else None
            if verbose:
                logger.info(f"  Cloned pre-pipeline Polars originals (pipeline will modify categoricals)")
        else:
            train_df_polars_pre = train_df
            val_df_polars_pre = val_df if isinstance(val_df, pl.DataFrame) else None
            test_df_polars_pre = test_df if isinstance(test_df, pl.DataFrame) else None
            if verbose:
                logger.info(f"  Skipped pre-pipeline clone (skip_categorical_encoding=True)")
        cat_features_polars = get_polars_cat_columns(train_df)
    else:
        train_df_polars_pre = None
        val_df_polars_pre = None
        test_df_polars_pre = None
        cat_features_polars = []

    t0_fit_pipeline = timer()
    train_df, val_df, test_df, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=pipeline_config,
        ensure_float32=preprocessing_config.ensure_float32_dtypes,
        verbose=verbose,
        text_features=feature_types_config.text_features,
        embedding_features=feature_types_config.embedding_features,
    )
    if verbose:
        logger.info("  fit_and_transform_pipeline done in %s", _elapsed_str(t0_fit_pipeline))

    polars_pipeline_applied = was_polars_input and pipeline_config.prefer_polarsds and pipeline is not None

    # Apply shared sklearn-based extensions
    if preprocessing_extensions is not None and isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    t0_ext = timer()
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df, val_df, test_df, preprocessing_extensions, verbose=verbose,
    )
    if verbose and preprocessing_extensions is not None:
        logger.info("  apply_preprocessing_extensions done in %s", _elapsed_str(t0_ext))
    if extensions_pipeline is not None:
        cat_features = []

    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    if verbose:
        logger.info("  Pipeline done -- train: %s, cat_features: %s", _df_shape_str(train_df), cat_features or '(none)')
        if was_polars_input and cat_features_polars and list(cat_features_polars) != list(cat_features or []):
            logger.info("  Pre-pipeline Polars cat_features: %s", cat_features_polars)
        logger.info("  PHASE 3 total: %s", _elapsed_str(t0_phase3))

    return (
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config, preprocessing_extensions,
    )


def _phase_train_val_test_split(
    *,
    df,
    target_by_type,
    timestamps,
    group_ids,
    group_ids_raw,
    artifacts,
    sequences,
    split_config,
    behavior_config,
    metadata,
    data_dir,
    models_dir,
    target_name,
    model_name,
    df_size_mb,
    verbose,
):
    """Phase 2 (Train/Val/Test Splitting) of ``train_mlframe_models_suite``.

    Resolves auto-stratification + group-aware splitting flags from the
    config + extractor side-channels, calls ``make_train_test_split``, saves
    split artifacts to disk when ``data_dir`` is set, computes fairness
    subgroups on the pre-split frame, materialises ``train/val/test`` splits
    of the dataframe + sequences, frees the original df, refreshes the RSS
    baseline.

    Mutates ``metadata`` in-place with split sizes + train/val/test details.

    Returns
    -------
    15-tuple:
        train_idx, val_idx, test_idx,
        train_details, val_details, test_details,
        train_df, val_df, test_df,
        fairness_subgroups, fairness_features,
        train_sequences, val_sequences, test_sequences,
        baseline_rss_mb_refreshed
    """
    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    # Auto-stratify by target for classification when no timestamps are
    # available. Without this, the unstratified shuffle path can hand
    # an unlucky val/test slice with zero minority-class rows for
    # rare imbalance ratios (fuzz default-seed c0134, seed=99 c0040 --
    # rare_1pct + binary class produces 50 positives out of 5000;
    # random 400-row val_shuf can land all-class-0). Stratification
    # preserves class proportions across train/val/test. Skipped when
    # timestamps are present (the splitter prefers temporal ordering
    # there) or for multitarget setups where picking ONE target as
    # the stratify key is arbitrary.
    _stratify_y = None
    if timestamps is None and isinstance(target_by_type, dict):
        _classification_targets = []
        _has_multilabel = False
        for _tt, _named in target_by_type.items():
            _tt_name = getattr(_tt, "name", str(_tt)).upper()
            if "MULTILABEL" in _tt_name:
                _has_multilabel = True
                break
            if "CLASS" in _tt_name and isinstance(_named, dict):
                for _tn, _tv in _named.items():
                    if _tv is not None:
                        _classification_targets.append(_tv)
        # Multilabel stratification needs the optional ``iterative-
        # stratification`` package. Skip it to avoid forcing the dep on
        # users who don't have it (the ``MultilabelStratifiedShuffleSplit``
        # branch raises ``ModuleNotFoundError`` deep in the splitter).
        # Single-label classification (binary / multiclass) uses sklearn's
        # built-in ``StratifiedShuffleSplit`` which is always available.
        if _has_multilabel:
            _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                # Guard: only stratify when stratification is meaningful --
                # all classes have at least 2 rows, otherwise sklearn's
                # StratifiedShuffleSplit raises "least populated class has
                # only 1 member". Also limit to 1-D targets -- 2-D would
                # route to the multilabel splitter (already excluded above
                # but defense in depth).
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
            except Exception:
                _stratify_y = None
    # Group-aware splitting opt-in. When the extractor produced ``group_ids``
    # (e.g. ``SimpleFeaturesAndTargetsExtractor(group_field="well_id")``) and
    # ``split_config.use_groups`` is True (default), route through
    # ``GroupShuffleSplit`` so that no well straddles train/val/test.
    _groups = group_ids if (split_config.use_groups and group_ids is not None and len(group_ids) > 0) else None
    with phase("split_data"):
        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df=df,
            timestamps=timestamps,
            stratify_y=_stratify_y,
            groups=_groups,
            **split_config.model_dump(exclude={"use_groups"}),
        )
    if verbose:
        log_ram_usage()

    # Save artifacts
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

    # Pre-compute fairness subgroups from full df BEFORE splitting
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info("Computed %d fairness subgroups", len(fairness_subgroups))

    # Create split dataframes
    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    if verbose:
        logger.info("  Split shapes -- train: %s, val: %s, test: %s", _df_shape_str(train_df), _df_shape_str(val_df), _df_shape_str(test_df))
        logger.info("  PHASE 2 total: %s", _elapsed_str(t0_phase2))

    # Split sequences by train/val/test indices (for recurrent models)
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))

    # Delete original df to free RAM (caller already did ``del df`` after
    # we return; we still nudge the GC + arena-release here because the
    # baseline-RSS refresh is meaningful only AFTER the parent frees df).
    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    # Caller's `del df` happens after the return; we refresh baseline using
    # the current RSS (which will reflect the post-del state on the next
    # ``maybe_clean_ram_and_gpu`` call inside the caller).
    baseline_rss_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")
    if verbose:
        log_ram_usage()

    return (
        train_idx, val_idx, test_idx,
        train_details, val_details, test_details,
        train_df, val_df, test_df,
        fairness_subgroups, fairness_features,
        train_sequences, val_sequences, test_sequences,
        baseline_rss_mb,
    )


def _phase_load_and_preprocess(
    *,
    df,
    preprocessing_config,
    features_and_targets_extractor,
    recurrent_models,
    sequences,
    verbose,
):
    """Phase 1 (Data Loading & Preprocessing) of ``train_mlframe_models_suite``.

    Loads the input dataframe (file path or in-memory), runs the
    features-and-targets extractor to surface ``target_by_type`` +
    side-channels (timestamps, group ids, sample weights, artifacts), extracts
    sequences for any recurrent models, then drops the FTE-flagged columns
    and runs the final preprocessing pass.

    Captures the RAM baseline + DF-size estimate AFTER the FTE so the
    downstream ``maybe_clean_ram_and_gpu`` calls have a meaningful
    pre-transient-allocation reference point.

    Returns
    -------
    11-tuple:
        df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts,
        additional_columns_to_drop, sample_weights, baseline_rss_mb,
        df_size_mb, sequences
    """
    if verbose:
        log_phase("PHASE 1: Data Loading & Preprocessing")

    # Load and prepare dataframe
    t0_phase1 = timer()
    with phase("load_and_prepare_dataframe"):
        df = load_and_prepare_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  load_and_prepare_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_phase1))

    # Apply features_and_targets_extractor to extract targets
    if verbose:
        logger.info("Create additional features & extracting targets...")

    t0_fte = timer()
    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop, sample_weights = features_and_targets_extractor.transform(
        df
    )
    if verbose:
        logger.info("  features_and_targets_extractor done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_fte))

    # Capture baseline RSS + DF size NOW -- before any downstream steps that may allocate
    # transient state (get_sequences, drop_columns, preprocess). Used by
    # maybe_clean_ram_and_gpu() at later sites to skip ~0.6s gc calls when memory
    # pressure is low. On 100GB production DFs the growth/free-RAM thresholds trip and
    # clean_ram fires; on small test DFs all sites are skipped.
    baseline_rss_mb = get_process_rss_mb()
    df_size_mb = estimate_df_size_mb(df)

    # Extract sequences for recurrent models (if not provided directly)
    if recurrent_models and sequences is None:
        extracted_sequences = features_and_targets_extractor.get_sequences(df)
        if extracted_sequences is not None:
            sequences = extracted_sequences
            if verbose:
                logger.info("Extracted %d sequences from DataFrame", len(sequences))
        elif verbose:
            logger.warning("recurrent_models specified but no sequences provided or extracted")

    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-FTE")
    if verbose:
        log_ram_usage()

    # Drop columns AFTER features_and_targets_extractor (columns might be needed by features_and_targets_extractor or created by it)
    df = drop_columns_from_dataframe(
        df,
        additional_columns_to_drop=additional_columns_to_drop,
        config_drop_columns=preprocessing_config.drop_columns,
        verbose=verbose,
    )

    # Preprocess dataframe (handle nulls, infinities, constants, dtypes)
    t0_preproc = timer()
    df = preprocess_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  preprocess_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_preproc))
        logger.info("  PHASE 1 total: %s", _elapsed_str(t0_phase1))

    return (
        df, target_by_type, group_ids_raw, group_ids, timestamps,
        artifacts, additional_columns_to_drop, sample_weights,
        baseline_rss_mb, df_size_mb, sequences,
    )


def _build_suite_common_params_dict(
    *,
    reporting_config,
    preprocessing_config,
    confidence_analysis_config,
) -> Dict[str, Any]:
    """Assemble the ``common_params_dict`` carried down through the suite.

    Three sources feed in:
    - ``reporting_config`` -- dumped via ``.model_dump(exclude=...)`` with
      ``title_metrics_tokens`` excluded (derived field auto-recomputed by
      ``_build_configs_from_params``) and ``plot_inline_display`` excluded
      (consumed at suite level only).
    - ``preprocessing_config`` -- conditionally adds ``scaler`` / ``imputer`` /
      ``category_encoder`` when each is non-None.
    - ``confidence_analysis_config`` -- 8 scalar fields under the
      ``confidence_analysis_*`` / ``include_confidence_analysis`` prefix.

    Pure read-only function: no side effects, no mutation of the input
    configs. Tests can unit-check the output dict without spinning up the
    full suite.
    """
    common: Dict[str, Any] = {}
    common.update(
        reporting_config.model_dump(exclude={
            "title_metrics_tokens",
            "plot_inline_display",
            # 2026-05-13: matplotlib / plotly style overrides are
            # consumed at suite-level only (via
            # ``_apply_plot_style_overrides``); the deep consumer
            # ``_build_configs_from_params`` does NOT accept these
            # kwargs. Excluding here so the dump stays compatible with
            # the train_eval signature.
            "matplotlib_style",
            "matplotlib_rcparams",
            "plotly_template",
        })
    )
    if preprocessing_config.scaler is not None:
        common["scaler"] = preprocessing_config.scaler
    if preprocessing_config.imputer is not None:
        common["imputer"] = preprocessing_config.imputer
    if preprocessing_config.category_encoder is not None:
        common["category_encoder"] = preprocessing_config.category_encoder
    common["include_confidence_analysis"] = confidence_analysis_config.include
    common["confidence_analysis_use_shap"] = confidence_analysis_config.use_shap
    common["confidence_analysis_max_features"] = confidence_analysis_config.max_features
    common["confidence_analysis_cmap"] = confidence_analysis_config.cmap
    common["confidence_analysis_alpha"] = confidence_analysis_config.alpha
    common["confidence_analysis_ylabel"] = confidence_analysis_config.ylabel
    common["confidence_analysis_title"] = confidence_analysis_config.title
    common["confidence_model_kwargs"] = dict(confidence_analysis_config.model_kwargs)
    return common


def _maybe_dispatch_to_ltr_ranker_suite(
    *,
    target_type,
    df,
    target_name,
    model_name,
    features_and_targets_extractor,
    mlframe_models,
    use_mlframe_ensembles,
    ranking_config,
    split_config,
    hyperparams_config,
    reporting_config,
    output_config,
    verbose,
):
    """If ``target_type == LEARNING_TO_RANK``, route to the focused ranker suite.

    The standard classification/regression machinery in ``train_mlframe_models_suite``
    doesn't know how to consume per-row scores or per-query metrics, so we hand
    LTR runs off to ``train_mlframe_ranker_suite`` (CB/XGB/LGB native rankers
    + RRF/Borda ensembling). Helper inspects the bag of config objects, mirrors
    the relevant fields onto the ranker-suite signature, and returns its result.

    Returns
    -------
    ``None`` when the call site is NOT LTR (caller continues with the standard
    pipeline); the ranker-suite return tuple otherwise (caller forwards verbatim).
    """
    if target_type is None or target_type != TargetTypes.LEARNING_TO_RANK:
        return None
    from mlframe.training.ranker_suite import train_mlframe_ranker_suite

    # Resolve a save_dir from output_config if available, else None.
    _save_dir = None
    if output_config is not None:
        _data_dir = (
            output_config.get("data_dir") if isinstance(output_config, dict)
            else getattr(output_config, "data_dir", None)
        )
        _models_dir = (
            output_config.get("models_dir") if isinstance(output_config, dict)
            else getattr(output_config, "models_dir", None)
        ) or "models"
        if _data_dir:
            _save_dir = os.path.join(_data_dir, _models_dir, model_name)

    # Pull split sizes from split_config if provided.
    _test_size, _val_size = 0.15, 0.15
    if split_config is not None:
        _test_size = (
            split_config.get("test_size", 0.15) if isinstance(split_config, dict)
            else getattr(split_config, "test_size", 0.15)
        )
        _val_size = (
            split_config.get("val_size", 0.15) if isinstance(split_config, dict)
            else getattr(split_config, "val_size", 0.15)
        )

    # Hyperparams from hyperparams_config if provided.
    _iter, _lr, _es = 200, 0.1, 30
    _mlp_kwargs = None
    if hyperparams_config is not None:
        _iter = (
            hyperparams_config.get("iterations", 200) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "iterations", 200)
        )
        _lr = (
            hyperparams_config.get("learning_rate", 0.1) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "learning_rate", 0.1)
        )
        _es = (
            hyperparams_config.get("early_stopping_rounds", 30) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "early_stopping_rounds", 30)
        )
        # mlp_kwargs forwarded to MLPRanker.__init__ when LTR + 'mlp'
        # is in mlframe_models. Mirrors the non-LTR mlp_kwargs path
        # via _configure_mlp_params; users who want to flip
        # enable_checkpointing, change hidden_layers, etc. for the
        # ranker MLP should put those keys in
        # ``hyperparams_config["mlp_kwargs"]``.
        _mlp_kwargs = (
            hyperparams_config.get("mlp_kwargs", None) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "mlp_kwargs", None)
        )

    # Reporting wiring (auto-emit LTR panel grid per (model, split)).
    _plot_outputs = None
    _ltr_panels = None
    if reporting_config is not None:
        _plot_outputs = (
            reporting_config.get("plot_outputs") if isinstance(reporting_config, dict)
            else getattr(reporting_config, "plot_outputs", None)
        )
        _ltr_panels = (
            reporting_config.get("ltr_panels") if isinstance(reporting_config, dict)
            else getattr(reporting_config, "ltr_panels", None)
        )
    _plot_file = None
    if output_config is not None:
        _plot_file = (
            output_config.get("plot_file") if isinstance(output_config, dict)
            else getattr(output_config, "plot_file", None)
        )

    return train_mlframe_ranker_suite(
        df=df,
        target_name=target_name,
        model_name=model_name,
        features_and_targets_extractor=features_and_targets_extractor,
        mlframe_models=mlframe_models,
        use_mlframe_ensembles=use_mlframe_ensembles,
        ranking_config=ranking_config,
        test_size=_test_size,
        val_size=_val_size,
        iterations=_iter,
        learning_rate=_lr,
        early_stopping_rounds=_es,
        save_dir=_save_dir,
        verbose=verbose,
        plot_file=_plot_file,
        plot_outputs=_plot_outputs,
        ltr_panels=_ltr_panels,
        mlp_kwargs=_mlp_kwargs,
    )


def _ensure_logging_visible(level: int = logging.INFO) -> None:
