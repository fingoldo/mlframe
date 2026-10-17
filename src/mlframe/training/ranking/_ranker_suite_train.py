"""``train_mlframe_ranker_suite`` carved out of
``mlframe.training.ranker_suite``.

Re-imported at the parent's module bottom so historical
``from mlframe.training.ranker_suite import train_mlframe_ranker_suite``
resolves transparently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl


from ._ranker_suite_train_helpers import (  # noqa: F401  -- carved helpers
    logger,
    _train_mlframe_rank_pandas_dataframes_arrow_backed,
    _train_mlframe_rank_without_relevance_column_leaked,
    _train_mlframe_rank_disk_path_feeds_same,
    _train_mlframe_rank_drops_replacements_single_sweep,
    _train_mlframe_rank_try,
    _train_mlframe_rank_selected_features,
    _train_mlframe_rank_lgb_will_try_hash,
    _train_mlframe_rank_embedding_features_kwarg_which,
    _train_mlframe_rank_pandas,
    _train_mlframe_rank_protocol_beyond_just_group,
    _train_mlframe_rank_lockstep_rank_fusion_math,
    _train_mlframe_rank_sees_silent_fallthrough_rather,
    _train_mlframe_rank_save_dir,
)


def _first_nonempty_target_bucket(target_by_type: dict) -> dict:
    """First non-empty target bucket among LTR, regression, binary classification; ``{}`` when none."""
    from mlframe.training.configs import TargetTypes

    for _tt in (TargetTypes.LEARNING_TO_RANK, TargetTypes.REGRESSION, TargetTypes.BINARY_CLASSIFICATION):
        _cand: dict[Any, Any] | None = target_by_type.get(_tt)
        if _cand is not None and len(_cand) > 0:
            return _cand
    return {}


@dataclass
class _LtrSplit:
    """Group-aware LTR split: row indices plus the per-split feature frames, targets and query ids (frames are rebound as preprocessing proceeds)."""

    df_features: Any
    train_idx: Any
    val_idx: Any
    test_idx: Any
    X_tr: Any
    X_va: Any
    X_te: Any
    y_tr: np.ndarray
    y_va: np.ndarray
    y_te: np.ndarray
    g_tr: np.ndarray
    g_va: np.ndarray
    g_te: np.ndarray


def _ltr_extract_inputs(df: Any, target_name: str, features_and_targets_extractor: Any, verbose: int) -> tuple[Any, Any, np.ndarray, Any]:
    """Run the FTE and return ``(df_features, group_ids, y, resolved_target_key)``; raises ValueError on a missing group field / group ids / targets."""
    df_features: Any = None
    if not getattr(features_and_targets_extractor, "group_field", None):
        raise ValueError(
            "train_mlframe_ranker_suite: LEARNING_TO_RANK target type "
            "requires features_and_targets_extractor.group_field to be set "
            "(it identifies queries -- groups). Set group_field='your_query_id_col' "
            "on the extractor, or pick a non-LTR target type."
        )

    # Materialise file-path inputs (parquet) the same way the standard
    # suite does. ``train_mlframe_models_suite`` accepts ``str`` paths
    # for parquet; the LTR fork mirrors that contract so the fuzz suite
    # (which serialises some combos through parquet round-trip to test
    # the on-disk path) feeds the same input here.
    df = _train_mlframe_rank_disk_path_feeds_same(df)

    transformed = features_and_targets_extractor.transform(df)
    # FTE.transform returns a tuple whose precise shape varies by
    # extractor variant; pull the relevant pieces by attribute / index.
    if isinstance(transformed, tuple):
        # Standard contract: (df, target_by_type, group_ids_raw, group_ids,
        # timestamps, ...)
        df_features = transformed[0]
        target_by_type = transformed[1]
        group_ids = transformed[3] if len(transformed) > 3 else None
        # drop the unused timestamps capture. The
        # timeseries-aware ranker path doesn't consume this slot today; if/when
        # it lands, this assignment can be re-added. Removing the F841-bait
        # keeps the static-analysis surface clean.
    else:
        raise ValueError(f"FTE.transform returned {type(transformed)!r}; expected tuple. " "This suite was written against the standard FTE contract.")

    if group_ids is None:
        raise ValueError("FTE produced no group_ids despite group_field being set. " "Check that the configured group_field column exists in df.")

    # Pull the target -- LTR extractors put it under TargetTypes.LEARNING_TO_RANK
    # OR under REGRESSION (graded relevance often slots into the regression
    # bucket by default). Try both.
    target_dict = _first_nonempty_target_bucket(target_by_type)
    if not target_dict:
        raise ValueError(
            "FTE produced no targets. Ensure build_targets registers at "
            "least one entry under TargetTypes.LEARNING_TO_RANK / REGRESSION / "
            "BINARY_CLASSIFICATION."
        )

    # Resolve the target value: prefer exact-name match against ``target_name``
    # (matches the standard suite's contract), fall back to the FTE's
    # configured target_column attribute (the fuzz suite passes
    # combo.short_id() as the suite-level target_name but the FTE keys its
    # output by the dataframe column name like "relevance"), then fall
    # back to the only entry when there's exactly one.
    fte_target_col = getattr(features_and_targets_extractor, "target_column", None)
    if target_name in target_dict:
        _resolved_key = target_name
    elif fte_target_col and fte_target_col in target_dict:
        _resolved_key = fte_target_col
    elif len(target_dict) == 1:
        _resolved_key = next(iter(target_dict))
    else:
        raise ValueError(
            f"target {target_name!r} not found in extractor output. "
            f"Available: {list(target_dict)}. Pass target_name matching "
            "the FTE's target_column, or have the FTE register exactly "
            "one target."
        )
    y = np.asarray(target_dict[_resolved_key])

    if verbose:
        logger.info(
            "LTR suite: %d rows, %d unique queries, target=%r, y range [%s..%s]",
            len(df_features), len(np.unique(group_ids)), target_name,
            y.min(), y.max(),
        )
    return df_features, group_ids, y, _resolved_key


def _ltr_split_and_prepare(
    df_features: Any,
    group_ids: Any,
    y: np.ndarray,
    *,
    features_and_targets_extractor: Any,
    target_name: str,
    resolved_key: Any,
    test_size: float,
    val_size: float,
    random_seed: int,
    verbose: int,
) -> _LtrSplit:
    """Group-aware train/val/test split, Arrow-backed pandas conversion and removal of the group/relevance columns from X."""
    from mlframe.training.splitting import make_train_test_split

    _resolved_key = resolved_key
    # -------------------------------------------------------------
    # 2. Group-aware split
    # -------------------------------------------------------------
    train_idx, val_idx, test_idx, *_ = make_train_test_split(
        df_features,
        test_size=test_size,
        val_size=val_size,
        groups=np.asarray(group_ids),
        random_seed=random_seed,
    )

    # Convert polars to pandas in a single pass: clean inf in polars
    # first (lazy with_columns, zero-copy), then convert via the
    # Arrow-backed bridge. This replaces a 3-copy path (to_pandas,
    # df.copy, df.replace) that cost ~18s on a 1M-row LTR frame.
    # Downstream LGBRanker / XGBRanker / CatBoostRanker all accept
    # pandas DataFrames with Arrow-backed numeric columns.
    df_features = _train_mlframe_rank_pandas_dataframes_arrow_backed(df_features)

    # Drop group column + target column from feature matrix. The group
    # identifier is consumed via group_ids / qid kwarg of the rankers; if
    # it leaks into X, XGB raises (collides with qid kwarg) and CB/LGB
    # would happily memorise it as a "feature" (target leak). The target
    # column is also dropped -- it lives in y_*, not X_*.
    group_col_name = getattr(features_and_targets_extractor, "group_field", None)
    # Also resolve the FTE's target_column for drops (the suite-level
    # ``target_name`` may be a different identifier from the dataframe column).
    fte_target_col_for_drop = getattr(features_and_targets_extractor, "target_column", None)
    # ``_resolved_key`` is the ACTUAL relevance column name the y was read from -- the authoritative drop. FTEs that
    # declare the target via typed lists (e.g. ``learning_to_rank_targets=[...]``) expose no ``target_column``, so
    # without this the relevance column leaked into X and the rankers memorised it (perfect target leak).
    cols_to_drop_for_X = [c for c in (group_col_name, target_name, fte_target_col_for_drop, _resolved_key) if c is not None]
    X_te, X_tr, X_va = _train_mlframe_rank_without_relevance_column_leaked(df_features, cols_to_drop_for_X, train_idx, val_idx, test_idx)

    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    g_tr = np.asarray(group_ids)[train_idx]
    g_va = np.asarray(group_ids)[val_idx]
    g_te = np.asarray(group_ids)[test_idx]

    if verbose:
        logger.info(
            "LTR split: train=%d rows / %d queries; val=%d / %d; test=%d / %d",
            len(train_idx), len(np.unique(g_tr)),
            len(val_idx), len(np.unique(g_va)),
            len(test_idx), len(np.unique(g_te)),
        )
    return _LtrSplit(
        df_features=df_features, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te, g_tr=g_tr, g_va=g_va, g_te=g_te,
    )


def _ltr_select_features(split: _LtrSplit, *, feature_selection_config: Any, rfecv_models: list[str] | None, random_seed: int, verbose: int) -> Optional[list]:
    """Optional LTR feature selection on the train split; rebinds the X frames of the split to the selected subset and returns the selected columns."""
    X_te, X_tr, X_va = split.X_te, split.X_tr, split.X_va
    y_tr, g_tr = split.y_tr, split.g_tr
    # -------------------------------------------------------------
    # 2b. Feature selection -- driven by the COMMON FeatureSelectionConfig (use_mrmr_fs / rfecv_models /
    # use_boruta_shap), so LTR uses the same FS settings as every other target type. The graded relevance is the
    # selection target; ranking._ranker_fs builds the standard selectors (target-type-aware) and fits them on the
    # TRAIN split, then every ranker trains on the selected subset. Core selector procedures are NOT modified.
    # -------------------------------------------------------------
    from mlframe.training.configs import TargetTypes

    selected_features: Optional[list] = None
    if feature_selection_config is not None:
        from ._ranker_fs import select_ltr_features

        selected_features = select_ltr_features(
            X_tr, y_tr, g_tr,  # g_tr = per-query groups -> group-aware (per-query) relevance MI, the correct LtR signal
            feature_selection_config=feature_selection_config,
            rfecv_models=rfecv_models,
            target_type=TargetTypes.LEARNING_TO_RANK,
            fs_random_seed=random_seed, verbose=verbose,
        )
        X_te, X_tr, X_va = _train_mlframe_rank_selected_features(selected_features, X_tr, X_va, X_te, verbose)
    split.X_te, split.X_tr, split.X_va = X_te, X_tr, X_va
    return selected_features


def _ltr_encode_features(split: _LtrSplit) -> list[str]:
    """Detect categorical columns, drop embedding columns and label-encode string cats on the X frames of the split; returns ``cat_features``."""
    X_te, X_tr, X_va = split.X_te, split.X_tr, split.X_va
    # Detect categorical columns for cat_features kwarg (CB/LGB use, XGB
    # uses enable_categorical). Heuristic: pandas object/category cols, but
    # excluding embedding-like columns (object cells that are array-like /
    # nested lists) - those must NOT go into cat_features or downstream CB /
    # LGB will try to hash an unhashable cell.
    cat_features: list[str] = []
    _embedding_cols: list[str] = []
    _train_mlframe_rank_lgb_will_try_hash(X_tr, cat_features, _embedding_cols)
    # Drop embedding columns from the X frames before model fit - native
    # rankers (CatBoostRanker / XGBRanker / LGBMRanker) all treat them as
    # numeric and raise on the array cell. Native embedding support requires
    # the ``embedding_features`` kwarg which the LTR dispatch does not plumb.
    X_te, X_tr, X_va = _train_mlframe_rank_embedding_features_kwarg_which(_embedding_cols, X_tr, X_va, X_te)

    # LightGBM (`LGBMRanker` + `Booster.predict`) rejects pandas object-
    # dtype columns at both fit and predict time with
    # ``ValueError: pandas dtypes must be int, float or bool. Fields with
    # bad pandas dtypes: cat_0: object, ...``. The main classifier/
    # regressor path's CatBoostEncoder pre-pipeline does the object ->
    # CategoricalDtype upgrade upstream, but the LTR dispatch
    # (train_mlframe_ranker_suite) consumes the post-FTE frame directly.
    # Label-encode object-dtype cat columns to int32 codes ONCE here
    # using a shared train+val+test vocabulary so all downstream
    # consumers (_fit_lgb_ranker, predict_ranker_scores, dummy_baselines)
    # see int codes uniformly. LGB treats each unique int code as a
    # category. NaN/null cells map to -1 which LGB treats as missing.
    #
    # Use ``is_string_dtype`` rather than ``dtype == object`` because
    # pandas 2.1+ with ``future.infer_string=True`` (default in pd 3.x)
    # and pyarrow-backed strings report ``pd.StringDtype()`` instead of
    # numpy ``object``. ``is_string_dtype`` is True for BOTH, so the
    # gate fires on legacy + modern + pyarrow string columns alike.
    # Pre-fix the modern envs silently skipped encoding -> LGB saw
    # string columns -> the original "pandas dtypes must be int, float
    # or bool" crash returned (test_object_cats_encoded... regressed on
    # pandas 2.3+).
    X_te, X_tr, X_va = _train_mlframe_rank_pandas(X_tr, cat_features, X_va, X_te)
    split.X_te, split.X_tr, split.X_va = X_te, X_tr, X_va
    return cat_features


def _ltr_run_dummy_baselines(
    split: _LtrSplit, *, features_and_targets_extractor: Any, target_name: str, dummy_baselines_config: Any, plot_file: str | None
) -> None:
    """Compute and log the dummy / trivial-baseline floor for LTR; a baseline failure never blocks training."""
    X_te, X_tr, X_va = split.X_te, split.X_tr, split.X_va
    y_tr, y_va, y_te = split.y_tr, split.y_va, split.y_te
    g_tr, g_va, g_te = split.g_tr, split.g_va, split.g_te
    df_features = split.df_features
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
    # Dummy / trivial-baseline floor for LTR (random_within_query /
    # identity_input_order / mean_relevance). One verdict line at INFO,
    # full table at DEBUG. Wrapped in try/except so a baseline failure
    # never blocks training.
    try:
        from mlframe.training.configs import DummyBaselinesConfig
        from mlframe.training.baselines import compute_dummy_baselines
        from mlframe.training.phases import phase as _phase_ctx
        _db_cfg = DummyBaselinesConfig() if dummy_baselines_config is None else dummy_baselines_config
        if _db_cfg.enabled and "learning_to_rank" in _db_cfg.apply_to_target_types:
            # Optional: pull per-row doc_ids for the popularity baseline
            # when the FTE has a ``doc_field`` set (extends the LTR
            # protocol beyond just ``group_field`` = qid).
            _doc_field = getattr(features_and_targets_extractor, "doc_field", None)
            _doc_tr = _doc_va = _doc_te = None
            _doc_te, _doc_tr, _doc_va = _train_mlframe_rank_protocol_beyond_just_group(_doc_field, df_features, train_idx, val_idx, test_idx, _doc_te, _doc_tr, _doc_va)
            with _phase_ctx("dummy_baselines:learning_to_rank", target=target_name):
                _db_report = compute_dummy_baselines(
                    target_type="learning_to_rank",
                    target_name=target_name,
                    train_X=X_tr, val_X=X_va, test_X=X_te,
                    train_y=y_tr, val_y=y_va, test_y=y_te,
                    group_ids_train=g_tr, group_ids_val=g_va, group_ids_test=g_te,
                    doc_ids_train=_doc_tr, doc_ids_val=_doc_va, doc_ids_test=_doc_te,
                    config=_db_cfg,
                    plot_file_prefix=(plot_file or ""),
                )
            logger.info(_db_report.format_text())
            logger.debug(
                "[dummy-baselines] target='%s' full table:\n%s",
                target_name, _db_report.table.to_string(),
            )
    except Exception as _db_err:
        logger.warning(
            "[DUMMY_BASELINES] FAILED target='%s' (learning_to_rank): %s. " "Training continues without baseline floor.",
            target_name,
            _db_err,
        )


def _ltr_train_flavors(
    split: _LtrSplit,
    selected: list[str],
    *,
    ranking_config: Any,
    cat_features: list[str],
    iterations: int,
    learning_rate: float,
    early_stopping_rounds: int,
    eval_at: tuple[int, ...],
    mlp_kwargs: dict[str, Any] | None,
    model_name: str,
    plot_file: str | None,
    plot_outputs: str | None,
    ltr_panels: str | None,
    verbose: int,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], list[str]]:
    """Fit each selected native ranker, score val/test, optionally emit panels; returns ``(models_dict, val_scores, test_scores, flavor_order)``."""
    from .ranker_suite import _strategy_for_model
    from .ranking import fit_ranker, predict_ranker_scores
    from mlframe.metrics.ranking import compute_ranking_summary

    X_te, X_tr, X_va = split.X_te, split.X_tr, split.X_va
    y_tr, y_va, y_te = split.y_tr, split.y_va, split.y_te
    g_tr, g_va, g_te = split.g_tr, split.g_va, split.g_te

    models_dict: dict[str, Any] = {}
    val_scores_per_model: list[np.ndarray] = []
    test_scores_per_model: list[np.ndarray] = []
    flavor_order: list[str] = []

    for flavor in selected:
        strategy = _strategy_for_model(flavor)
        iter_kw = "iterations" if flavor == "cb" else "n_estimators"
        # The suite-level ``learning_rate`` default (0.1) is calibrated
        # for tree boosters (CB / XGB / LGB) where 0.1 is a sensible
        # shrinkage. For MLP it is DISASTROUS: AdamW + LayerNorm + ReLU
        # at lr=0.1 blows up the weights within the first few steps,
        # the post-Linear pre-activations saturate negative, every
        # ReLU dies, and the final Linear collapses to outputting its
        # bias for every input (observed:
        # std(val_scores)=4.77e-07, identical -4.430379 per row). The
        # zero-variance gate in the rank-fusion ensemble then drops
        # MLP entirely. Skip the suite-level lr forward for MLP and
        # let MLPRanker's own default (1e-3) apply; callers wanting
        # MLP-specific lr override via ``mlp_kwargs={"learning_rate": ...}``.
        model_kwargs: dict[str, Any]
        if flavor == "mlp":
            model_kwargs = {iter_kw: iterations}
        else:
            model_kwargs = {iter_kw: iterations, "learning_rate": learning_rate}
        # MLPRanker-specific knobs (e.g. enable_checkpointing,
        # hidden_layers, dropout) are forwarded only for the mlp
        # flavor; for cb / xgb / lgb they would be unknown init
        # kwargs and raise a TypeError.
        if flavor == "mlp" and mlp_kwargs:
            model_kwargs.update(mlp_kwargs)
        if verbose >= 2:
            logger.info("Training %s ranker...", flavor)
        fitted = fit_ranker(
            strategy, X_tr, y_tr, g_tr,
            X_val=X_va, y_val=y_va, group_ids_val=g_va,
            ranking_config=ranking_config,
            cat_features=list(cat_features) if cat_features else None,
            model_kwargs=model_kwargs,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose >= 2,
        )

        val_scores = predict_ranker_scores(fitted, X_va)
        test_scores = predict_ranker_scores(fitted, X_te)

        val_metrics = compute_ranking_summary(y_va, val_scores, g_va, eval_at=eval_at)
        test_metrics = compute_ranking_summary(y_te, test_scores, g_te, eval_at=eval_at)

        fitted["val_scores"] = val_scores
        fitted["test_scores"] = test_scores
        fitted["val_metrics"] = val_metrics
        fitted["test_metrics"] = test_metrics
        models_dict[flavor] = fitted
        flavor_order.append(flavor)
        val_scores_per_model.append(val_scores)
        test_scores_per_model.append(test_scores)

        if verbose:
            logger.info(
                "  %s: val NDCG@10=%.4f / test NDCG@10=%.4f",
                flavor, val_metrics["ndcg@10"], test_metrics["ndcg@10"],
            )

        # Auto-emit LTR panel grid per (flavor, split). No-op when caller
        # didn't supply plot_file / plot_outputs / ltr_panels (legacy
        # callers see no behavioural change).
        if plot_file and plot_outputs and ltr_panels:
            from mlframe.reporting import render_multi_target_panels
            render_multi_target_panels(
                targets=y_va, preds=val_scores, group_ids=g_va,
                plot_outputs=plot_outputs, ltr_panels=ltr_panels,
                base_path=f"{plot_file}_{model_name}_{flavor}_val",
                suptitle=f"VAL {model_name} {flavor}",
            )
            render_multi_target_panels(
                targets=y_te, preds=test_scores, group_ids=g_te,
                plot_outputs=plot_outputs, ltr_panels=ltr_panels,
                base_path=f"{plot_file}_{model_name}_{flavor}_test",
                suptitle=f"TEST {model_name} {flavor}",
            )
    return models_dict, val_scores_per_model, test_scores_per_model, flavor_order


def _ltr_build_ensemble(
    split: _LtrSplit,
    models_dict: dict[str, Any],
    val_scores_per_model: list[np.ndarray],
    test_scores_per_model: list[np.ndarray],
    flavor_order: list[str],
    *,
    use_mlframe_ensembles: bool,
    ensemble_method: str | None,
    ranking_config: Any,
    eval_at: tuple[int, ...],
    model_name: str,
    plot_file: str | None,
    plot_outputs: str | None,
    ltr_panels: str | None,
    verbose: int,
) -> None:
    """Gate the members, rank-fuse them and add the ``"ensemble"`` entry to ``models_dict`` (in place) when at least two members survive."""
    from .ranking import ensemble_ranker_scores
    from mlframe.metrics.ranking import compute_ranking_summary

    y_va, y_te = split.y_va, split.y_te
    g_va, g_te = split.g_va, split.g_te

    # Arch-5: apply quality / diversity / zero-crossing / NaN gates BEFORE the rank-fusion step.
    # Each gate drops members from val_scores_per_model + test_scores_per_model + flavor_order in
    # lockstep so the rank-fusion math sees a coherent post-gate set. Drops are logged at WARN.
    _gated_flavor_order, _gated_test_scores, _gated_val_scores = _train_mlframe_rank_lockstep_rank_fusion_math(use_mlframe_ensembles, val_scores_per_model, flavor_order, test_scores_per_model, models_dict)

    if use_mlframe_ensembles and len(_gated_flavor_order) >= 2:
        # Resolution order for the ensembling method:
        # 1. Explicit function-arg ``ensemble_method`` (back-compat, highest priority).
        # 2. ``ranking_config.ensemble_method`` when the legacy field was customised
        #    (i.e. set to something other than the default "rrf") -- this preserves
        #    callers who already pinned the loose field to "score_mean" or "borda".
        # 3. ``ranking_config.ltr_ensemble_method`` (typed Literal["rrf","borda"];
        #    new preferred source for fresh configurations).
        #
        # When BOTH the legacy and typed fields are set to non-default conflicting
        # values, the legacy wins per priority above, but we WARN so the operator
        # sees the silent fallthrough rather than discovering it via metrics.
        _legacy = getattr(ranking_config, "ensemble_method", "rrf")
        _typed = getattr(ranking_config, "ltr_ensemble_method", "rrf")
        method = _train_mlframe_rank_sees_silent_fallthrough_rather(ensemble_method, _legacy, _typed)
        ens_val = ensemble_ranker_scores(
            _gated_val_scores, g_va,
            method=method, rrf_k=ranking_config.rrf_k,
            assume_comparable_scales=ranking_config.assume_comparable_scales,
        )
        ens_test = ensemble_ranker_scores(
            _gated_test_scores, g_te,
            method=method, rrf_k=ranking_config.rrf_k,
            assume_comparable_scales=ranking_config.assume_comparable_scales,
        )
        ens_val_metrics = compute_ranking_summary(y_va, ens_val, g_va, eval_at=eval_at)
        ens_test_metrics = compute_ranking_summary(y_te, ens_test, g_te, eval_at=eval_at)
        models_dict["ensemble"] = {
            "method": method,
            "members": _gated_flavor_order,
            "members_pre_gate": flavor_order,
            "val_scores": ens_val,
            "test_scores": ens_test,
            "val_metrics": ens_val_metrics,
            # WARN: ``test_metrics`` is for REPORTING ONLY; do NOT use it to pick the LTR
            # ensemble method. The ``method`` here is resolved from config BEFORE the test
            # set is seen (val/test split honoured); iterating on ``method`` until
            # ``test_metrics`` improves would convert test into a model-selection surface.
            # Selection-grade metrics belong to ``val_metrics``.
            "test_metrics_for_reporting_only": True,
            "test_metrics": ens_test_metrics,
        }
        if verbose:
            logger.warning(
                "  ensemble (%s, N=%d): val NDCG@10=%.4f / test NDCG@10=%.4f  "
                "[test_metrics for reporting only -- do NOT use to pick ensemble method]",
                method, len(flavor_order),
                ens_val_metrics["ndcg@10"], ens_test_metrics["ndcg@10"],
            )

        if plot_file and plot_outputs and ltr_panels:
            from mlframe.reporting import render_multi_target_panels
            render_multi_target_panels(
                targets=y_va, preds=ens_val, group_ids=g_va,
                plot_outputs=plot_outputs, ltr_panels=ltr_panels,
                base_path=f"{plot_file}_{model_name}_ensemble_val",
                suptitle=f"VAL {model_name} ensemble[{method}]",
            )
            render_multi_target_panels(
                targets=y_te, preds=ens_test, group_ids=g_te,
                plot_outputs=plot_outputs, ltr_panels=ltr_panels,
                base_path=f"{plot_file}_{model_name}_ensemble_test",
                suptitle=f"TEST {model_name} ensemble[{method}]",
            )


def _ltr_build_metadata(
    split: _LtrSplit,
    models_dict: dict[str, Any],
    flavor_order: list[str],
    *,
    target_name: str,
    model_name: str,
    ranking_config: Any,
    cat_features: list[str],
    selected_features: Optional[list],
) -> dict[str, Any]:
    """Suite-level metadata dict (schema-compatible with the other target types)."""
    X_tr = split.X_tr
    g_tr, g_va, g_te = split.g_tr, split.g_va, split.g_te

    # Build a per-model schema dict so downstream fuzz / regression
    # assertions on metadata schema don't have to special-case LTR.
    _model_schemas = {
        flavor: {
            "flavor": models_dict[flavor]["flavor"],
            "objective_kwargs": models_dict[flavor]["objective_kwargs"],
            "n_features": int(X_tr.shape[1]) if hasattr(X_tr, "shape") else None,
        }
        for flavor in flavor_order
    }
    _columns = list(X_tr.columns) if hasattr(X_tr, "columns") else list(range(X_tr.shape[1])) if hasattr(X_tr, "shape") else []
    metadata: dict[str, Any] = {
        "target_type": "learning_to_rank",
        "target_name": target_name,
        "model_name": model_name,
        "ranking_config_dump": ranking_config.model_dump() if hasattr(ranking_config, "model_dump") else dict(vars(ranking_config)),
        "n_train_queries": len(np.unique(g_tr)),
        "n_val_queries": len(np.unique(g_va)),
        "n_test_queries": len(np.unique(g_te)),
        "per_model_test_metrics": {f: models_dict[f]["test_metrics"] for f in flavor_order},
        # Schema-compatible keys so the fuzz suite's metadata-shape
        # assertions (`columns`, `cat_features`, `outlier_detection`,
        # `model_schemas`) don't need an LTR-specific branch.
        "columns": _columns,
        "cat_features": cat_features,
        "outlier_detection": None,
        "model_schemas": _model_schemas,
        # LTR-local feature selection result (None when feature_selection was off): the raw columns the rankers
        # were trained on. Mirrors the main suite's selected_features_ so downstream reports can see the FS subset.
        "selected_features": selected_features,
    }
    if "ensemble" in models_dict:
        metadata["ensemble_test_metrics"] = models_dict["ensemble"]["test_metrics"]
    return metadata


def train_mlframe_ranker_suite(
    df: pd.DataFrame | pl.DataFrame,
    target_name: str,
    model_name: str,
    features_and_targets_extractor: Any,
    *,
    mlframe_models: list[str] | None = None,
    use_mlframe_ensembles: bool = True,
    ranking_config: Any = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    iterations: int = 200,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 30,
    eval_at: tuple[int, ...] = (1, 5, 10),
    ensemble_method: str | None = None,
    save_dir: str | None = None,
    random_seed: int = 42,
    verbose: int = 1,
    plot_file: str | None = None,
    plot_outputs: str | None = None,
    ltr_panels: str | None = None,
    mlp_kwargs: dict[str, Any] | None = None,
    dummy_baselines_config: Any = None,
    feature_selection_config: Any = None,
    rfecv_models: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train a suite of native rankers + (optionally) ensemble them.

    Parameters
    ----------
    df : DataFrame
        Source frame. Passed through the FTE to extract features /
        target / group_ids / timestamps.
    target_name : str
        Column to predict. Must be present in the frame produced by the
        FTE (typically the FTE itself adds it via ``build_targets``).
    model_name : str
        Identifier for the trained models (used in artefact paths).
    features_and_targets_extractor : FeaturesAndTargetsExtractor
        MUST have ``group_field`` set -- groups are how queries are
        identified for ranking. Raises if absent.
    mlframe_models : list[str], optional
        Subset of ``{"cb", "xgb", "lgb"}``. Defaults to all three.
    use_mlframe_ensembles : bool
        Whether to RRF / Borda the per-model scores.
    ranking_config : LearningToRankConfig, optional
        Per-library objective / loss / ensemble knobs. Defaults applied
        when None.
    test_size, val_size : float
        Group-aware split fractions.
    iterations, learning_rate, early_stopping_rounds : int / float / int
        Per-model hyperparameters (passed through to all three rankers).
    eval_at : tuple
        NDCG@k / MAP@k cutoffs to report.
    ensemble_method : str, optional
        Override the config-resolved method (highest priority). When None,
        the suite picks ``ranking_config.ensemble_method`` only if that
        legacy field was customised away from "rrf"; otherwise it reads
        the typed ``ranking_config.ltr_ensemble_method`` (Literal of
        "rrf" / "borda"). Valid values across the union: ``rrf`` (default)
        / ``borda`` / ``score_mean`` (last one requires
        ``assume_comparable_scales=True``).
    save_dir : str, optional
        If provided, save each fitted ranker as
        ``<save_dir>/<model_name>_<flavor>.joblib`` plus a metadata json.
    random_seed : int
        For reproducibility of the group split.
    verbose : int
        0 = silent, 1 = INFO, 2 = library-level verbose.

    Returns
    -------
    (models_dict, metadata_dict)
        ``models_dict`` keyed by flavor (``"cb"``, ``"xgb"``, ``"lgb"``,
        ``"ensemble"``); each value is the ``fit_ranker`` return dict
        augmented with ``"val_scores"``, ``"test_scores"``,
        ``"val_metrics"``, ``"test_metrics"``.
        ``metadata_dict`` carries the suite-level summary
        (``target_type``, ``ranking_config_dump``, per-model NDCG@k).
    """
    from mlframe.training.configs import LearningToRankConfig
    from .ranker_suite import _filter_models_for_ranking

    if ranking_config is None:
        ranking_config = LearningToRankConfig()

    df_features, group_ids, y, resolved_key = _ltr_extract_inputs(df, target_name, features_and_targets_extractor, verbose)
    split = _ltr_split_and_prepare(
        df_features, group_ids, y,
        features_and_targets_extractor=features_and_targets_extractor, target_name=target_name, resolved_key=resolved_key,
        test_size=test_size, val_size=val_size, random_seed=random_seed, verbose=verbose,
    )
    selected_features = _ltr_select_features(
        split, feature_selection_config=feature_selection_config, rfecv_models=rfecv_models, random_seed=random_seed, verbose=verbose,
    )
    selected = _filter_models_for_ranking(mlframe_models)
    cat_features = _ltr_encode_features(split)
    _ltr_run_dummy_baselines(
        split, features_and_targets_extractor=features_and_targets_extractor, target_name=target_name,
        dummy_baselines_config=dummy_baselines_config, plot_file=plot_file,
    )
    models_dict, val_scores_per_model, test_scores_per_model, flavor_order = _ltr_train_flavors(
        split, selected, ranking_config=ranking_config, cat_features=cat_features, iterations=iterations, learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds, eval_at=eval_at, mlp_kwargs=mlp_kwargs, model_name=model_name,
        plot_file=plot_file, plot_outputs=plot_outputs, ltr_panels=ltr_panels, verbose=verbose,
    )
    _ltr_build_ensemble(
        split, models_dict, val_scores_per_model, test_scores_per_model, flavor_order,
        use_mlframe_ensembles=use_mlframe_ensembles, ensemble_method=ensemble_method, ranking_config=ranking_config, eval_at=eval_at,
        model_name=model_name, plot_file=plot_file, plot_outputs=plot_outputs, ltr_panels=ltr_panels, verbose=verbose,
    )
    metadata = _ltr_build_metadata(
        split, models_dict, flavor_order, target_name=target_name, model_name=model_name, ranking_config=ranking_config,
        cat_features=cat_features, selected_features=selected_features,
    )
    _train_mlframe_rank_save_dir(save_dir, model_name, flavor_order, models_dict, verbose, metadata)

    return models_dict, metadata
