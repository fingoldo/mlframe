"""``train_mlframe_ranker_suite`` carved out of
``mlframe.training.ranker_suite``.

Re-imported at the parent's module bottom so historical
``from mlframe.training.ranker_suite import train_mlframe_ranker_suite``
resolves transparently.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def train_mlframe_ranker_suite(
    df: pd.DataFrame | pl.DataFrame,
    target_name: str,
    model_name: str,
    features_and_targets_extractor,
    *,
    mlframe_models: list[str] | None = None,
    use_mlframe_ensembles: bool = True,
    ranking_config=None,
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
    dummy_baselines_config=None,
    feature_selection_config=None,
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
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .ranker_suite import _filter_models_for_ranking, _strategy_for_model
    from mlframe.training.configs import TargetTypes, LearningToRankConfig
    from mlframe.training.splitting import make_train_test_split
    from .ranking import (
        fit_ranker, predict_ranker_scores, ensemble_ranker_scores,
    )
    from mlframe.metrics.ranking import compute_ranking_summary

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
                logger.debug("polars read_parquet failed for %r; falling back to pandas", df, exc_info=True)
                df = pd.read_parquet(df)
        else:
            raise ValueError(f"train_mlframe_ranker_suite: file-path input must be .parquet, " f"got {df!r}")

    transformed = features_and_targets_extractor.transform(df)
    # FTE.transform returns a tuple whose precise shape varies by
    # extractor variant; pull the relevant pieces by attribute / index.
    if isinstance(transformed, tuple):
        # Standard contract: (df, target_by_type, group_ids_raw, group_ids,
        # timestamps, ...)
        df_features = transformed[0]
        target_by_type = transformed[1]
        group_ids = transformed[3] if len(transformed) > 3 else None
        # Wave 63 (2026-05-20): drop the unused timestamps capture. The
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

    # Convert polars to pandas in a single pass: clean inf in polars
    # first (lazy with_columns, zero-copy), then convert via the
    # Arrow-backed bridge. This replaces a 3-copy path (to_pandas,
    # df.copy, df.replace) that cost ~18s on a 1M-row LTR frame.
    # Downstream LGBRanker / XGBRanker / CatBoostRanker all accept
    # pandas DataFrames with Arrow-backed numeric columns.
    import polars as _pl
    if isinstance(df_features, _pl.DataFrame):
        # Clean inf in polars (lazy with_columns, one pass)
        _inf_exprs = []
        for _cn, _dt in zip(df_features.columns, df_features.dtypes):
            if _dt.is_numeric():
                _inf_exprs.append(_pl.when(_pl.col(_cn).is_infinite()).then(None).otherwise(_pl.col(_cn)).alias(_cn))
        if _inf_exprs:
            df_features = df_features.with_columns(_inf_exprs)
        # Convert to pandas via the Arrow-backed bridge (utils.get_pandas_view_of_polars_df).
        # split_blocks=True keeps numeric / bool columns as zero-copy Arrow buffer views
        # instead of consolidating into fresh numpy blocks -- ~32x faster on multi-million-row
        # frames (bench: 30s -> 0.95s on 7.3M x 118 with 18 dict cols). String/Categorical
        # columns still materialise to pandas-native dtypes that CatBoost can ingest.
        from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
        df_features = _get_pandas_view(df_features)
    elif isinstance(df_features, pd.DataFrame):
        # Pandas path: inf cleaning (pandas .replace on full frame, one copy).
        # Rare in fuzz (the suite sends polars by default); kept for completeness.
        _num_cols = df_features.select_dtypes(include=[np.number]).columns
        if len(_num_cols) > 0:
            # Avoid SettingWithCopy: boolean-mask replace, no .copy() needed
            # because the mask targets the columns explicitly.
            _inf_mask = df_features[_num_cols].isin([np.inf, -np.inf])
            if _inf_mask.any().any():
                df_features[_num_cols] = df_features[_num_cols].mask(
                    _inf_mask, np.nan,
                )

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
    if isinstance(df_features, pd.DataFrame):
        existing = [c for c in cols_to_drop_for_X if c in df_features.columns]
        df_X = df_features.drop(columns=existing) if existing else df_features
        # XGB / CB / LGB rankers need numeric or pd.Categorical; drop
        # datetime columns + auto-cast object columns to pd.Categorical.
        # (Standard suite does this via the polars-alignment step; LTR
        # fork takes the simple route here.)
        # Audit D P1-4 (2026-05-18): pre-fix did ``df_X = df_X.drop`` and ``df_X[col] = ...`` per
        # column, which triggers a full DataFrame copy on every assignment under pandas'
        # BlockManager -- O(n_cols^2) memory churn on 100+ column frames. Now we accumulate
        # drops + replacements in a single sweep, then commit them in two batch operations.
        _drop_cols: list[str] = []
        _replacements: dict[str, pd.Series] = {}
        for col in df_X.columns:
            dt = df_X[col].dtype
            if str(dt).startswith("datetime"):
                _drop_cols.append(col)
            elif dt is object:
                # object-dtype columns can contain either scalar strings (cat features ->
                # astype('category') OK) OR nested arrays/lists (embedding features from
                # pl.List(pl.Float32) round-trip -> astype('category') raises
                # ``TypeError: unhashable type: 'numpy.ndarray'`` because pandas factorize
                # can't hash arrays). Drop nested-element columns silently: the rankers can't
                # consume them as numeric anyway, and CB/XGB/LGB sklearn wrappers reject them
                # at fit time.
                _sample = next((v for v in df_X[col] if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
                if _sample is not None and isinstance(_sample, (list, tuple, np.ndarray)):
                    # Surface the silent drop: pre-fix the comment said "drop silently" but
                    # an operator who shipped an embedding column unflagged loses a feature
                    # without any log line. Log once per column at INFO so the column count
                    # mismatch downstream traces back.
                    logger.info(
                        "ranker_suite: dropping object-dtype column %r containing nested "
                        "elements (sample type=%s) -- rankers can't consume embeddings; "
                        "use the FTE's embedding_columns slot to route them properly.",
                        col, type(_sample).__name__,
                    )
                    _drop_cols.append(col)
                    continue
                # Fill nulls BEFORE astype("category") so the missing sentinel becomes a
                # category level (CatBoost rejects NaN in cat_features; the standard suite
                # uses the same ``__MISSING__`` sentinel pattern).
                _replacements[col] = df_X[col].fillna("__MISSING__").astype("category")
            elif isinstance(dt, pd.CategoricalDtype):
                # Existing category column with NaN: add the sentinel.
                if df_X[col].isna().any():
                    _series = df_X[col]
                    if "__MISSING__" not in _series.cat.categories:
                        _series = _series.cat.add_categories("__MISSING__")
                    _replacements[col] = _series.fillna("__MISSING__")
        if _drop_cols:
            df_X = df_X.drop(columns=_drop_cols)
        if _replacements:
            # Single-pass concat preserves column order, avoids O(n_cols^2) rebuilds.
            _orig_order = list(df_X.columns)
            _unchanged = [c for c in _orig_order if c not in _replacements]
            df_X = pd.concat(
                [df_X[_unchanged]] + [_replacements[c].rename(c) for c in _orig_order if c in _replacements],
                axis=1,
            )[_orig_order]
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
    # 2b. Feature selection -- driven by the COMMON FeatureSelectionConfig (use_mrmr_fs / rfecv_models /
    # use_boruta_shap), so LTR uses the same FS settings as every other target type. The graded relevance is the
    # selection target; ranking._ranker_fs builds the standard selectors (target-type-aware) and fits them on the
    # TRAIN split, then every ranker trains on the selected subset. Core selector procedures are NOT modified.
    # -------------------------------------------------------------
    from mlframe.training.configs import TargetTypes as _TT

    selected_features: Optional[list] = None
    if feature_selection_config is not None:
        from ._ranker_fs import select_ltr_features

        selected_features = select_ltr_features(
            X_tr, y_tr, g_tr,  # g_tr = per-query groups -> group-aware (per-query) relevance MI, the correct LtR signal
            feature_selection_config=feature_selection_config,
            rfecv_models=rfecv_models,
            target_type=_TT.LEARNING_TO_RANK,
            fs_random_seed=random_seed, verbose=verbose,
        )
        if selected_features:
            def _subset_cols(Xf):
                return Xf.select(selected_features) if hasattr(Xf, "select") else Xf[selected_features]
            X_tr, X_va, X_te = _subset_cols(X_tr), _subset_cols(X_va), _subset_cols(X_te)
            if verbose:
                logger.info("LTR feature selection: %d features selected for ranker training.", len(selected_features))

    # -------------------------------------------------------------
    # 3. Filter models + train each
    # -------------------------------------------------------------
    selected = _filter_models_for_ranking(mlframe_models)

    models_dict: dict[str, Any] = {}
    val_scores_per_model: list[np.ndarray] = []
    test_scores_per_model: list[np.ndarray] = []
    flavor_order: list[str] = []

    # Detect categorical columns for cat_features kwarg (CB/LGB use, XGB
    # uses enable_categorical). Heuristic: pandas object/category cols, but
    # excluding embedding-like columns (object cells that are array-like /
    # nested lists) - those must NOT go into cat_features or downstream CB /
    # LGB will try to hash an unhashable cell.
    cat_features: list[str] = []
    _embedding_cols: list[str] = []
    if isinstance(X_tr, pd.DataFrame):
        for col in X_tr.columns:
            _dt = X_tr[col].dtype
            if isinstance(_dt, pd.CategoricalDtype):
                cat_features.append(col)
                continue
            # ``_dt == object`` misses pandas-3 / future.infer_string columns
            # whose dtype is ``StringDtype(na_value=nan)`` (dtype.name == "str")
            # or ``StringDtype`` (dtype.name == "string"). Broaden the test so
            # those land in cat_features and the downstream LGB ranker can
            # cast them to pandas Categorical before fit.
            _dn = str(_dt)
            _is_str_like = _dt == object or _dn in ("string", "str") or "string" in _dn.lower()  # noqa: E721 -- pandas dtype `== object` comparison is intended
            if _is_str_like:
                _probe = X_tr[col].dropna()
                if len(_probe) == 0:
                    cat_features.append(col)
                    continue
                _first = _probe.iloc[0]
                # array-likes (np.ndarray, list, tuple of numbers) are NOT
                # categoricals; they are embedding columns mistakenly routed
                # through the pandas object dtype by polars list-of-float
                # materialisation.
                if isinstance(_first, (np.ndarray, list, tuple)):
                    _embedding_cols.append(col)
                    continue
                cat_features.append(col)
    # Drop embedding columns from the X frames before model fit - native
    # rankers (CatBoostRanker / XGBRanker / LGBMRanker) all treat them as
    # numeric and raise on the array cell. Native embedding support requires
    # the ``embedding_features`` kwarg which the LTR dispatch does not plumb.
    if _embedding_cols:
        logger.warning(
            "[ranker_suite] dropping %d embedding-like object-dtype column(s) "
            "(%s) from the X frames before native ranker fit; LTR dispatch "
            "does not currently plumb ``embedding_features``.",
            len(_embedding_cols), _embedding_cols,
        )
        if isinstance(X_tr, pd.DataFrame):
            X_tr = X_tr.drop(columns=_embedding_cols, errors="ignore")
        if isinstance(X_va, pd.DataFrame):
            X_va = X_va.drop(columns=_embedding_cols, errors="ignore")
        if isinstance(X_te, pd.DataFrame):
            X_te = X_te.drop(columns=_embedding_cols, errors="ignore")

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
    if isinstance(X_tr, pd.DataFrame) and cat_features:
        _to_encode = [c for c in cat_features if c in X_tr.columns and pd.api.types.is_string_dtype(X_tr[c])]
        if _to_encode:
            _splits_for_vocab = [X_tr]
            if isinstance(X_va, pd.DataFrame):
                _splits_for_vocab.append(X_va)
            if isinstance(X_te, pd.DataFrame):
                _splits_for_vocab.append(X_te)
            _vocabs: dict[str, dict] = {}
            _skip_cols: set[str] = set()
            for _c in _to_encode:
                _vals: set = set()
                _abort = False
                for _split in _splits_for_vocab:
                    if _c in _split.columns:
                        try:
                            # set.update on a C-level list comprehension is ~1.45x
                            # faster than a Python ``for _v in ...: _vals.add(_v)``
                            # loop at 200k rows x 15 cat cols (bench
                            # ``profiling/bench_ranker_suite_vocab_build.py``,
                            # 720ms -> 490ms). Unhashable cells (numpy arrays /
                            # lists inside an object column mis-labelled as
                            # cat_feature) still raise TypeError, which we catch
                            # the same way as the prior per-cell try.
                            _vals.update(_split[_c].dropna().tolist())
                        except TypeError:
                            _abort = True
                            break
                if _abort:
                    _skip_cols.add(_c)
                    continue
                # Stable code assignment via sorted string repr -- avoids
                # run-to-run drift on dict-ordering of insertion sets. ``key=str``
                # (not the equivalent ``lambda x: str(x)``) sidesteps the per-call
                # Python-frame overhead.
                _vocabs[_c] = {v: i for i, v in enumerate(sorted(_vals, key=str))}
            # Per-col ``_split_local[_c] = ...`` triggers BlockManager rebuilds
            # for each column. Build a {col: new_series} dict in one pass, then
            # assemble the result frame with a single ``pd.concat`` so the
            # BlockManager rebuilds once.
            for _split_name, _split in (("train", X_tr), ("val", X_va), ("test", X_te)):
                if not isinstance(_split, pd.DataFrame):
                    continue
                _orig_cols = list(_split.columns)
                _new_series: dict[str, pd.Series] = {}
                for _c in _to_encode:
                    if _c in _skip_cols:
                        continue
                    if _c in _split.columns:
                        _vmap = _vocabs[_c]
                        # Cast to object before .map(): Categorical[string]
                        # propagates its dtype through .map(), and the resulting
                        # Categorical[int] rejects .fillna(-1) ("Cannot setitem
                        # on a Categorical with a new category") because -1 is
                        # not in the mapped vocabulary. Plain object dtype
                        # demotes to float64 on missing cells and lets fillna(-1)
                        # land cleanly.
                        _new_series[_c] = _split[_c].astype(object).map(_vmap).fillna(-1).astype("int32")
                if _new_series:
                    _kept = [c for c in _orig_cols if c not in _new_series]
                    _split_local = pd.concat(
                        [_split[_kept]] + [_new_series[c].rename(c) for c in _orig_cols if c in _new_series],
                        axis=1,
                    )[_orig_cols]
                else:
                    _split_local = _split
                if _split_name == "train":
                    X_tr = _split_local
                elif _split_name == "val":
                    X_va = _split_local
                elif _split_name == "test":
                    X_te = _split_local

    # Dummy / trivial-baseline floor for LTR (random_within_query /
    # identity_input_order / mean_relevance). One verdict line at INFO,
    # full table at DEBUG. Wrapped in try/except so a baseline failure
    # never blocks training.
    try:
        from mlframe.training.configs import DummyBaselinesConfig
        from mlframe.training.baselines import compute_dummy_baselines
        from mlframe.training.phases import phase as _phase_ctx
        _db_cfg = dummy_baselines_config or DummyBaselinesConfig()
        if _db_cfg.enabled and "learning_to_rank" in _db_cfg.apply_to_target_types:
            # Optional: pull per-row doc_ids for the popularity baseline
            # when the FTE has a ``doc_field`` set (extends the LTR
            # protocol beyond just ``group_field`` = qid).
            _doc_field = getattr(features_and_targets_extractor, "doc_field", None)
            _doc_tr = _doc_va = _doc_te = None
            if _doc_field and isinstance(df_features, pd.DataFrame) and _doc_field in df_features.columns:
                try:
                    _doc_full = np.asarray(df_features[_doc_field])
                    _doc_tr = _doc_full[train_idx]
                    _doc_va = _doc_full[val_idx]
                    _doc_te = _doc_full[test_idx]
                except Exception:
                    logger.debug("failed to slice doc_ids field %r for LTR popularity baseline; disabling doc_ids", _doc_field, exc_info=True)
                    _doc_tr = _doc_va = _doc_te = None
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

    for flavor in selected:
        strategy = _strategy_for_model(flavor)
        iter_kw = "iterations" if flavor == "cb" else "n_estimators"
        # The suite-level ``learning_rate`` default (0.1) is calibrated
        # for tree boosters (CB / XGB / LGB) where 0.1 is a sensible
        # shrinkage. For MLP it is DISASTROUS: AdamW + LayerNorm + ReLU
        # at lr=0.1 blows up the weights within the first few steps,
        # the post-Linear pre-activations saturate negative, every
        # ReLU dies, and the final Linear collapses to outputting its
        # bias for every input (observed 2026-05-21:
        # std(val_scores)=4.77e-07, identical -4.430379 per row). The
        # zero-variance gate in the rank-fusion ensemble then drops
        # MLP entirely. Skip the suite-level lr forward for MLP and
        # let MLPRanker's own default (1e-3) apply; callers wanting
        # MLP-specific lr override via ``mlp_kwargs={"learning_rate": ...}``.
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

    # -------------------------------------------------------------
    # 4. Ensemble
    # -------------------------------------------------------------
    # Arch-5: apply quality / diversity / zero-crossing / NaN gates BEFORE the rank-fusion step.
    # Each gate drops members from val_scores_per_model + test_scores_per_model + flavor_order in
    # lockstep so the rank-fusion math sees a coherent post-gate set. Drops are logged at WARN.
    if use_mlframe_ensembles and len(val_scores_per_model) >= 2:
        _gate_log: list[str] = []
        _keep_idx = list(range(len(flavor_order)))

        # NaN gate: members whose val OR test scores contain any NaN cannot rank-fuse cleanly
        # (RRF / Borda map NaN to last place silently which biases the fused order).
        _nan_drop = [
            i for i in _keep_idx
            if not np.all(np.isfinite(np.asarray(val_scores_per_model[i], dtype=np.float64)))
            or not np.all(np.isfinite(np.asarray(test_scores_per_model[i], dtype=np.float64)))
        ]
        if _nan_drop:
            _gate_log.append(f"NaN: dropped {[flavor_order[i] for i in _nan_drop]}")
            _keep_idx = [i for i in _keep_idx if i not in set(_nan_drop)]

        # Zero-variance gate: scores with std<=eps cannot inform rank order; including them
        # adds a phantom tie-breaker that lowers the fused NDCG.
        _zero_var_eps = 1e-12
        _zv_drop = [i for i in _keep_idx if float(np.std(np.asarray(val_scores_per_model[i], dtype=np.float64))) <= _zero_var_eps]
        if _zv_drop:
            _gate_log.append(f"zero_variance: dropped {[flavor_order[i] for i in _zv_drop]}")
            _keep_idx = [i for i in _keep_idx if i not in set(_zv_drop)]

        # Quality gate: drop members whose val NDCG@10 is below half of the best surviving
        # member. The exact threshold is a heuristic (half-best is the same rule
        # ``ensemble_probabilistic_predictions`` applies for outlier members); the goal is to
        # avoid dragging the fused score below the best single member.
        if len(_keep_idx) >= 2:
            _ndcgs = {i: float(models_dict[flavor_order[i]]["val_metrics"].get("ndcg@10", 0.0)) for i in _keep_idx}
            _ndcg_finite = {i: v for i, v in _ndcgs.items() if np.isfinite(v)}
            if _ndcg_finite:
                _best = max(_ndcg_finite.values())
                _floor = 0.5 * _best
                _q_drop = [i for i, v in _ndcg_finite.items() if v < _floor]
                if _q_drop and len(_keep_idx) - len(_q_drop) >= 2:
                    _gate_log.append(f"quality: dropped {[flavor_order[i] for i in _q_drop]} " f"(val_ndcg@10 below 0.5x best={_best:.4f})")
                    _keep_idx = [i for i in _keep_idx if i not in set(_q_drop)]

        # Diversity gate: when two members have val_score Spearman correlation > 0.99 they
        # contribute almost no independent signal; keep the higher-NDCG of each near-duplicate
        # pair. Spearman (rank correlation) is the rank-fusion-appropriate measure -- Pearson on
        # raw scores would over-flag flavours with different score scales but identical orders.
        if len(_keep_idx) >= 2:
            from scipy.stats import spearmanr  # local import: heavy dep only when ensembling
            _drop_div: set[int] = set()
            _kept_sorted = sorted(
                _keep_idx,
                key=lambda i: float(models_dict[flavor_order[i]]["val_metrics"].get("ndcg@10", 0.0)),
                reverse=True,
            )
            for _ix, _i in enumerate(_kept_sorted):
                if _i in _drop_div:
                    continue
                _vi = np.asarray(val_scores_per_model[_i], dtype=np.float64)
                for _j in _kept_sorted[_ix + 1 :]:
                    if _j in _drop_div:
                        continue
                    _vj = np.asarray(val_scores_per_model[_j], dtype=np.float64)
                    try:
                        _rho, _ = spearmanr(_vi, _vj)
                    except Exception:
                        logger.debug("spearmanr failed for diversity gate between models %d and %d; skipping pair", _i, _j, exc_info=True)
                        continue
                    if _rho is not None and np.isfinite(_rho) and _rho > 0.99:
                        _drop_div.add(_j)
            if _drop_div and len(_keep_idx) - len(_drop_div) >= 2:
                _gate_log.append(f"diversity: dropped {[flavor_order[i] for i in sorted(_drop_div)]} " f"(val Spearman > 0.99 vs higher-NDCG sibling)")
                _keep_idx = [i for i in _keep_idx if i not in _drop_div]

        if _gate_log:
            logger.warning(
                "[ranker_ensemble gates] %s. Surviving members for rank-fusion: %s.",
                "; ".join(_gate_log),
                [flavor_order[i] for i in _keep_idx],
            )
        # Materialise the post-gate set. The legacy `flavor_order` list keeps the full set so
        # per-member reporting under `models_dict[<flavour>]` stays intact; the fusion step
        # iterates over the post-gate slices below.
        _gated_flavor_order = [flavor_order[i] for i in _keep_idx]
        _gated_val_scores = [val_scores_per_model[i] for i in _keep_idx]
        _gated_test_scores = [test_scores_per_model[i] for i in _keep_idx]
        if len(_gated_flavor_order) < 2:
            logger.warning(
                "[ranker_ensemble gates] only %d member(s) survived post-gate " "(need >=2 to build a rank-fusion ensemble); skipping ensemble step.",
                len(_gated_flavor_order),
            )
    else:
        _gated_flavor_order = []
        _gated_val_scores = []
        _gated_test_scores = []

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
        if ensemble_method is not None:
            method = ensemble_method
        elif _legacy != "rrf":
            if _typed != "rrf" and _typed != _legacy:
                logger.warning(
                    "train_mlframe_ranker_suite: ranking_config.ensemble_method=%r and "
                    "ranking_config.ltr_ensemble_method=%r both set to conflicting non-default "
                    "values; using legacy ensemble_method=%r (typed ltr_ensemble_method ignored). "
                    "Pass ensemble_method=... explicitly or align the two fields to silence.",
                    _legacy, _typed, _legacy,
                )
            method = _legacy
        else:
            method = _typed
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
    _columns = list(X_tr.columns) if hasattr(X_tr, "columns") else list(range(X_tr.shape[1])) if hasattr(X_tr, "shape") else []
    metadata: dict[str, Any] = {
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
        # LTR-local feature selection result (None when feature_selection was off): the raw columns the rankers
        # were trained on. Mirrors the main suite's selected_features_ so downstream reports can see the FS subset.
        "selected_features": selected_features,
    }
    if "ensemble" in models_dict:
        metadata["ensemble_test_metrics"] = models_dict["ensemble"]["test_metrics"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import joblib
        # Wave 46 (2026-05-20): raw caller-supplied model_name plumbed into a path
        # basename is a traversal vector (e.g. model_name="../../evil" produces
        # "save_dir/../../evil_cb.joblib" which os.path.join leaves traversable).
        from pyutilz.strings import slugify as _slugify
        _safe_model_name = _slugify(model_name)
        for flavor in flavor_order:
            artefact_path = os.path.join(save_dir, f"{_safe_model_name}_{flavor}.joblib")
            joblib.dump(models_dict[flavor]["model"], artefact_path)
            # Wave 19 P0 #3: write the .meta.json sidecar that records the
            # booster + mlframe library versions at save time. Without this,
            # the agent's analysis: "CB/LGB/XGB minor upgrades silently
            # mis-restore booster internals" -- load-side has no way to
            # detect the skew until predict() crashes deep with a cryptic
            # AttributeError. Reuses the io.py helper for consistent shape.
            try:
                from ..io import _write_save_meta_sidecar as _wsms
                _wsms(artefact_path, durable=False)
            except Exception as _meta_e:
                logger.warning(
                    "ranker_suite: failed to write .meta.json sidecar for "
                    "%s: %s. Booster artefact saved; load-time version "
                    "validation will fall through to back-compat path.",
                    artefact_path, _meta_e,
                )
            if verbose:
                logger.info("  saved %s -> %s", flavor, artefact_path)
        # Metadata json
        import json
        meta_path = os.path.join(save_dir, f"{_safe_model_name}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            # numpy types aren't json-serialisable; coerce.
            # NOTE: no shared numpy->json coercer exists in mlframe.utils or
            # pyutilz today; not worth a new util for this single call site.
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
