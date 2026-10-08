"""Helpers carved out of ``_ranker_suite_train`` to keep that module under its size budget."""
from __future__ import annotations

import logging
import functools
import os
from typing import Any

import numpy as np
import pandas as pd

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _train_mlframe_rank_pandas_dataframes_arrow_backed(df_features):
    """Block of train_mlframe_ranker_suite starting at ``if isinstance(df_features, _pl.DataFrame):``."""
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
        from mlframe.training.utils import get_pandas_view_of_polars_df as _get_pandas_view
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
    return df_features


def _train_mlframe_rank_without_relevance_column_leaked(df_features, cols_to_drop_for_X, train_idx, val_idx, test_idx):
    """Block of train_mlframe_ranker_suite starting at ``if isinstance(df_features, pd.DataFrame):``."""
    X_va: Any = None
    X_tr: Any = None
    X_te: Any = None
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
        _train_mlframe_rank_drops_replacements_single_sweep(df_X, _drop_cols, _replacements)
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
        X_te, X_tr, X_va = _train_mlframe_rank_try(df_features, cols_to_drop_for_X, train_idx, val_idx, test_idx, X_te, X_tr, X_va)
    return X_te, X_tr, X_va


def _train_mlframe_rank_disk_path_feeds_same(df):
    """Block of train_mlframe_ranker_suite starting at ``if isinstance(df, str):``."""
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
    return df


def _train_mlframe_rank_drops_replacements_single_sweep(df_X, _drop_cols, _replacements):
    """Block of train_mlframe_ranker_suite starting at ``for col in df_X.columns:``."""
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


def _train_mlframe_rank_try(df_features, cols_to_drop_for_X, train_idx, val_idx, test_idx, X_te, X_tr, X_va):
    """Block of train_mlframe_ranker_suite starting at ``try:``."""
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
    return X_te, X_tr, X_va


def _train_mlframe_rank_selected_features(selected_features, X_tr, X_va, X_te, verbose):
    """Block of train_mlframe_ranker_suite starting at ``if selected_features:``."""
    if selected_features:
        def _subset_cols(Xf):
            """Restrict a frame to ``selected_features``, format-agnostic (polars ``.select`` when available, else plain column indexing)."""
            return Xf.select(selected_features) if hasattr(Xf, "select") else Xf[selected_features]
        X_tr, X_va, X_te = _subset_cols(X_tr), _subset_cols(X_va), _subset_cols(X_te)
        if verbose:
            logger.info("LTR feature selection: %d features selected for ranker training.", len(selected_features))
    return X_te, X_tr, X_va


def _train_mlframe_rank_lgb_will_try_hash(X_tr, cat_features, _embedding_cols):
    """Block of train_mlframe_ranker_suite starting at ``if isinstance(X_tr, pd.DataFrame):``."""
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


def _train_mlframe_rank_embedding_features_kwarg_which(_embedding_cols, X_tr, X_va, X_te):
    """Block of train_mlframe_ranker_suite starting at ``if _embedding_cols:``."""
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
    return X_te, X_tr, X_va


def _train_mlframe_rank_pandas(X_tr, cat_features, X_va, X_te):
    """Block of train_mlframe_ranker_suite starting at ``if isinstance(X_tr, pd.DataFrame) and cat_features:``."""
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
    return X_te, X_tr, X_va


def _train_mlframe_rank_protocol_beyond_just_group(_doc_field, df_features, train_idx, val_idx, test_idx, _doc_te, _doc_tr, _doc_va):
    """Block of train_mlframe_ranker_suite starting at ``if _doc_field and isinstance(df_features, pd.DataFrame) and _doc_field``."""
    if _doc_field and isinstance(df_features, pd.DataFrame) and _doc_field in df_features.columns:
        try:
            _doc_full = np.asarray(df_features[_doc_field])
            _doc_tr = _doc_full[train_idx]
            _doc_va = _doc_full[val_idx]
            _doc_te = _doc_full[test_idx]
        except Exception:
            logger.debug("failed to slice doc_ids field %r for LTR popularity baseline; disabling doc_ids", _doc_field, exc_info=True)
            _doc_tr = _doc_va = _doc_te = None
    return _doc_te, _doc_tr, _doc_va


def _train_mlframe_rank_lockstep_rank_fusion_math(use_mlframe_ensembles, val_scores_per_model, flavor_order, test_scores_per_model, models_dict):
    """Block of train_mlframe_ranker_suite starting at ``if use_mlframe_ensembles and len(val_scores_per_model) >= 2:``."""
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
    return _gated_flavor_order, _gated_test_scores, _gated_val_scores


def _train_mlframe_rank_sees_silent_fallthrough_rather(ensemble_method, _legacy, _typed):
    """Block of train_mlframe_ranker_suite starting at ``if ensemble_method is not None:``."""
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
    return method


def _train_mlframe_rank_save_dir(save_dir, model_name, flavor_order, models_dict, verbose, metadata):
    """Block of train_mlframe_ranker_suite starting at ``if save_dir:``."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import joblib
        from mlframe.training.io import atomic_write_bytes
        # Wave 46 (2026-05-20): raw caller-supplied model_name plumbed into a path
        # basename is a traversal vector (e.g. model_name="../../evil" produces
        # "save_dir/../../evil_cb.joblib" which os.path.join leaves traversable).
        from pyutilz.strings import slugify as _slugify
        _safe_model_name = _slugify(model_name)
        for flavor in flavor_order:
            artefact_path = os.path.join(save_dir, f"{_safe_model_name}_{flavor}.joblib")
            _model_obj = models_dict[flavor]["model"]
            atomic_write_bytes(artefact_path, functools.partial(joblib.dump, _model_obj))
            # Wave 19 P0 #3: write the .meta.json sidecar that records the
            # booster + mlframe library versions at save time. Without this,
            # the agent's analysis: "CB/LGB/XGB minor upgrades silently
            # mis-restore booster internals" -- load-side has no way to
            # detect the skew until predict() crashes deep with a cryptic
            # AttributeError. Reuses the io.py helper for consistent shape.
            try:
                from mlframe.training.io import _write_save_meta_sidecar as _wsms
                _wsms(artefact_path, durable=False)
            except Exception as _meta_e:
                log_throttle(
                    logger,
                    "ranker_suite_meta_sidecar_write_failed",
                    logging.WARNING,
                    "ranker_suite: failed to write .meta.json sidecar for "
                    "%s: %s. Booster artefact saved; load-time version "
                    "validation will fall through to back-compat path.",
                    artefact_path, _meta_e,
                )
            if verbose:
                logger.info("  saved %s -> %s", flavor, artefact_path)
        # Metadata json
        import orjson

        meta_path = os.path.join(save_dir, f"{_safe_model_name}_metadata.json")
        with open(meta_path, "wb") as f:
            # numpy types aren't json-serialisable; coerce.
            # NOTE: no shared numpy->json coercer exists in mlframe.utils or
            # pyutilz today; not worth a new util for this single call site.
            def _coerce(o):
                """``orjson.dumps`` ``default=`` hook: coerce numpy scalar/array types to plain Python int/float/list so the ranker run metadata serialises."""
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return o
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2, default=_coerce))
        if verbose:
            logger.info("  saved metadata -> %s", meta_path)
