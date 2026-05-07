"""Top-level Learning-to-Rank training suite.

Mirrors the shape of ``train_mlframe_models_suite`` but with LTR-specific
dispatch:

- Accepts the same ``df`` + ``FeaturesAndTargetsExtractor`` inputs.
- ``mlframe_models`` is filtered to ``{cb, xgb, lgb}`` (HGB / Linear have
  no native ranker; warns and drops).
- ``group_ids`` from the FTE's ``group_field`` is REQUIRED. Raises a
  helpful error otherwise.
- Splits via ``make_train_test_split(groups=...)`` so query integrity is
  preserved end-to-end.
- For each model: ``fit_ranker`` + ``predict_ranker_scores`` + ranking
  metrics (NDCG@k, MAP@k, MRR).
- When ``use_mlframe_ensembles=True``, runs RRF / Borda ensembling over
  the model scores.
- Saves models via joblib for sklearn-wrapper consistency. Metadata
  (target_type, ranking_config, per-model NDCG@k) persisted alongside.

This is invoked by ``train_mlframe_models_suite`` when the target type
resolves to ``LEARNING_TO_RANK``; users can also call it directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _filter_models_for_ranking(mlframe_models: Optional[List[str]]) -> List[str]:
    """Drop models without native rankers (HGB, Linear, etc.) with WARN.

    Supported: CatBoost / XGBoost / LightGBM ship native rankers; MLP gets
    a custom RankNet/ListNet pairwise/listwise loss
    (mlframe.training.neural.ranker.MLPRanker).
    """
    if not mlframe_models:
        return ["cb", "xgb", "lgb"]
    requested = [m.lower() for m in mlframe_models]
    supported = {"cb", "xgb", "lgb", "mlp"}
    kept = [m for m in requested if m in supported]
    dropped = [m for m in requested if m not in supported]
    if dropped:
        logger.warning(
            "LTR target_type: dropping %d model(s) without native ranker: %s. "
            "Supported rankers: cb / xgb / lgb (native) + mlp (RankNet/ListNet); "
            "HGB / Linear / sklearn would need a regression-then-rerank "
            "workaround (not implemented). Surviving models: %s",
            len(dropped), dropped, kept,
        )
    if not kept:
        raise NotImplementedError(
            f"LTR target_type: every requested model {requested} lacks a "
            "native ranker. Pick at least one of cb / xgb / lgb / mlp."
        )
    return kept


def _strategy_for_model(model_name: str):
    """Return the strategy instance for a given model short-tag."""
    from mlframe.training.strategies import (
        CatBoostStrategy, XGBoostStrategy, TreeModelStrategy, NeuralNetStrategy,
    )
    name = model_name.lower()
    if name == "cb":
        return CatBoostStrategy()
    if name == "xgb":
        return XGBoostStrategy()
    if name == "lgb":
        return TreeModelStrategy()
    if name == "mlp":
        return NeuralNetStrategy()
    raise ValueError(f"unknown ranker model {model_name!r}")


def train_mlframe_ranker_suite(
    df: Union[pd.DataFrame, "pl.DataFrame"],
    target_name: str,
    model_name: str,
    features_and_targets_extractor,
    *,
    mlframe_models: Optional[List[str]] = None,
    use_mlframe_ensembles: bool = True,
    ranking_config=None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    iterations: int = 200,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 30,
    eval_at: Tuple[int, ...] = (1, 5, 10),
    ensemble_method: Optional[str] = None,
    save_dir: Optional[str] = None,
    random_seed: int = 42,
    verbose: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        Override ranking_config.ensemble_method. ``rrf`` (default) /
        ``borda`` / ``score_mean`` (latter requires
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
    from mlframe.training.configs import TargetTypes, LearningToRankConfig
    from mlframe.training.splitting import make_train_test_split
    from mlframe.training.ranking import (
        fit_ranker, predict_ranker_scores, ensemble_ranker_scores,
    )
    from mlframe.ranking_metrics import compute_ranking_summary

    if ranking_config is None:
        ranking_config = LearningToRankConfig()

    # -------------------------------------------------------------
    # 1. Extract features + targets + group_ids via the FTE
    # -------------------------------------------------------------
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
    if isinstance(df, str):
        if df.lower().endswith(".parquet"):
            try:
                import polars as _pl
                df = _pl.read_parquet(df)
            except Exception:
                df = pd.read_parquet(df)
        else:
            raise ValueError(
                f"train_mlframe_ranker_suite: file-path input must be .parquet, "
                f"got {df!r}"
            )

    transformed = features_and_targets_extractor.transform(df)
    # FTE.transform returns a tuple whose precise shape varies by
    # extractor variant; pull the relevant pieces by attribute / index.
    if isinstance(transformed, tuple):
        # Standard contract: (df, target_by_type, group_ids_raw, group_ids,
        # timestamps, ...)
        df_features = transformed[0]
        target_by_type = transformed[1]
        group_ids = transformed[3] if len(transformed) > 3 else None
        timestamps = transformed[4] if len(transformed) > 4 else None
    else:
        raise ValueError(
            f"FTE.transform returned {type(transformed)!r}; expected tuple. "
            "This suite was written against the standard FTE contract."
        )

    if group_ids is None:
        raise ValueError(
            "FTE produced no group_ids despite group_field being set. "
            "Check that the configured group_field column exists in df."
        )

    # Pull the target -- LTR extractors put it under TargetTypes.LEARNING_TO_RANK
    # OR under REGRESSION (graded relevance often slots into the regression
    # bucket by default). Try both.
    target_dict = (
        target_by_type.get(TargetTypes.LEARNING_TO_RANK)
        or target_by_type.get(TargetTypes.REGRESSION)
        or target_by_type.get(TargetTypes.BINARY_CLASSIFICATION)
        or {}
    )
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

    # 2026-05-04: convert polars frames to pandas before ranker fit. The
    # standard suite goes through a Polars-fastpath alignment step that
    # converts pl.Utf8 columns to pl.Enum etc.; the LTR fork doesn't
    # currently mirror that machinery, so the simplest robust path is
    # to materialise pandas + let the rankers handle their own cat
    # dispatch. Future optimisation: thread the alignment through.
    try:
        import polars as _pl
        if isinstance(df_features, _pl.DataFrame):
            df_features = df_features.to_pandas()
    except ImportError:
        pass

    # Drop group column + target column from feature matrix. The group
    # identifier is consumed via group_ids / qid kwarg of the rankers; if
    # it leaks into X, XGB raises (collides with qid kwarg) and CB/LGB
    # would happily memorise it as a "feature" (target leak). The target
    # column is also dropped -- it lives in y_*, not X_*.
    group_col_name = getattr(features_and_targets_extractor, "group_field", None)
    # Also resolve the FTE's target_column for drops (the suite-level
    # ``target_name`` may be a different identifier from the dataframe column).
    fte_target_col_for_drop = getattr(features_and_targets_extractor, "target_column", None)
    cols_to_drop_for_X = [
        c for c in (group_col_name, target_name, fte_target_col_for_drop)
        if c is not None
    ]
    if isinstance(df_features, pd.DataFrame):
        existing = [c for c in cols_to_drop_for_X if c in df_features.columns]
        df_X = df_features.drop(columns=existing) if existing else df_features
        # XGB / CB / LGB rankers need numeric or pd.Categorical; drop
        # datetime columns + auto-cast object columns to pd.Categorical.
        # (Standard suite does this via the polars-alignment step; LTR
        # fork takes the simple route here.)
        for col in list(df_X.columns):
            dt = df_X[col].dtype
            if str(dt).startswith("datetime"):
                df_X = df_X.drop(columns=[col])
            elif dt == object:
                # Fill nulls BEFORE astype("category") so the missing
                # sentinel becomes a category level (CatBoost rejects NaN
                # in cat_features; the standard suite uses the same
                # ``__MISSING__`` sentinel pattern).
                df_X[col] = df_X[col].fillna("__MISSING__").astype("category")
            elif isinstance(dt, pd.CategoricalDtype):
                # Existing category column with NaN: add the sentinel.
                if df_X[col].isna().any():
                    if "__MISSING__" not in df_X[col].cat.categories:
                        df_X[col] = df_X[col].cat.add_categories("__MISSING__")
                    df_X[col] = df_X[col].fillna("__MISSING__")
        X_tr = df_X.iloc[train_idx].reset_index(drop=True)
        X_va = df_X.iloc[val_idx].reset_index(drop=True)
        X_te = df_X.iloc[test_idx].reset_index(drop=True)
    else:
        try:
            import polars as pl
            if isinstance(df_features, pl.DataFrame):
                existing = [c for c in cols_to_drop_for_X if c in df_features.columns]
                df_X = df_features.drop(existing) if existing else df_features
                X_tr = df_X[train_idx.tolist()]
                X_va = df_X[val_idx.tolist()]
                X_te = df_X[test_idx.tolist()]
            else:
                X_tr = df_features[train_idx]
                X_va = df_features[val_idx]
                X_te = df_features[test_idx]
        except ImportError:
            X_tr = df_features[train_idx]
            X_va = df_features[val_idx]
            X_te = df_features[test_idx]

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

    # -------------------------------------------------------------
    # 3. Filter models + train each
    # -------------------------------------------------------------
    selected = _filter_models_for_ranking(mlframe_models)

    models_dict: Dict[str, Any] = {}
    val_scores_per_model: List[np.ndarray] = []
    test_scores_per_model: List[np.ndarray] = []
    flavor_order: List[str] = []

    # Detect categorical columns for cat_features kwarg (CB/LGB use, XGB
    # uses enable_categorical). Heuristic: pandas object/category cols.
    cat_features: List[str] = []
    if isinstance(X_tr, pd.DataFrame):
        for col in X_tr.columns:
            if isinstance(X_tr[col].dtype, pd.CategoricalDtype) or X_tr[col].dtype == object:
                cat_features.append(col)

    for flavor in selected:
        strategy = _strategy_for_model(flavor)
        iter_kw = "iterations" if flavor == "cb" else "n_estimators"
        model_kwargs = {iter_kw: iterations, "learning_rate": learning_rate}
        if verbose >= 2:
            logger.info("Training %s ranker...", flavor)
        fitted = fit_ranker(
            strategy, X_tr, y_tr, g_tr,
            X_val=X_va, y_val=y_va, group_ids_val=g_va,
            ranking_config=ranking_config,
            cat_features=cat_features if cat_features else None,
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

    # -------------------------------------------------------------
    # 4. Ensemble
    # -------------------------------------------------------------
    if use_mlframe_ensembles and len(val_scores_per_model) >= 2:
        method = ensemble_method or ranking_config.ensemble_method
        ens_val = ensemble_ranker_scores(
            val_scores_per_model, g_va,
            method=method, rrf_k=ranking_config.rrf_k,
            assume_comparable_scales=ranking_config.assume_comparable_scales,
        )
        ens_test = ensemble_ranker_scores(
            test_scores_per_model, g_te,
            method=method, rrf_k=ranking_config.rrf_k,
            assume_comparable_scales=ranking_config.assume_comparable_scales,
        )
        ens_val_metrics = compute_ranking_summary(y_va, ens_val, g_va, eval_at=eval_at)
        ens_test_metrics = compute_ranking_summary(y_te, ens_test, g_te, eval_at=eval_at)
        models_dict["ensemble"] = {
            "method": method,
            "members": flavor_order,
            "val_scores": ens_val,
            "test_scores": ens_test,
            "val_metrics": ens_val_metrics,
            "test_metrics": ens_test_metrics,
        }
        if verbose:
            logger.info(
                "  ensemble (%s, N=%d): val NDCG@10=%.4f / test NDCG@10=%.4f",
                method, len(flavor_order),
                ens_val_metrics["ndcg@10"], ens_test_metrics["ndcg@10"],
            )

    # -------------------------------------------------------------
    # 5. Save artefacts
    # -------------------------------------------------------------
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
    _columns = (
        list(X_tr.columns) if hasattr(X_tr, "columns") else
        list(range(X_tr.shape[1])) if hasattr(X_tr, "shape") else []
    )
    metadata: Dict[str, Any] = {
        "target_type": "learning_to_rank",
        "target_name": target_name,
        "model_name": model_name,
        "ranking_config_dump": ranking_config.model_dump() if hasattr(ranking_config, "model_dump") else dict(vars(ranking_config)),
        "n_train_queries": int(len(np.unique(g_tr))),
        "n_val_queries": int(len(np.unique(g_va))),
        "n_test_queries": int(len(np.unique(g_te))),
        "per_model_test_metrics": {f: models_dict[f]["test_metrics"] for f in flavor_order},
        # Schema-compatible keys so the fuzz suite's metadata-shape
        # assertions (`columns`, `cat_features`, `outlier_detection`,
        # `model_schemas`) don't need an LTR-specific branch.
        "columns": _columns,
        "cat_features": cat_features,
        "outlier_detection": None,
        "model_schemas": _model_schemas,
    }
    if "ensemble" in models_dict:
        metadata["ensemble_test_metrics"] = models_dict["ensemble"]["test_metrics"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import joblib
        for flavor in flavor_order:
            artefact_path = os.path.join(save_dir, f"{model_name}_{flavor}.joblib")
            joblib.dump(models_dict[flavor]["model"], artefact_path)
            if verbose:
                logger.info("  saved %s -> %s", flavor, artefact_path)
        # Metadata json
        import json
        meta_path = os.path.join(save_dir, f"{model_name}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            # numpy types aren't json-serialisable; coerce.
            def _coerce(o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return o
            json.dump(metadata, f, indent=2, default=_coerce)
        if verbose:
            logger.info("  saved metadata -> %s", meta_path)

    return models_dict, metadata


__all__ = ["train_mlframe_ranker_suite"]
