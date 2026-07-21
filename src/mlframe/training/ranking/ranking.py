"""Learning-to-Rank fit/predict + pre-fit data prep, single module per strategy.

Wraps native rankers (CatBoostRanker / XGBRanker / LGBMRanker) with a
uniform mlframe-shaped contract:

    fitted = fit_ranker(strategy, X_train, y_train, group_ids_train,
                        X_val, y_val, group_ids_val,
                        ranking_config=LearningToRankConfig(),
                        cat_features=..., model_kwargs=...)

    scores = predict_ranker_scores(fitted, X_test)

The per-strategy data prep (CB row-sort, XGB qid, LGB per-query group
sizes) is hidden behind these two calls so the suite-level code doesn't
need to remember each library's quirks.

Verified against installed CatBoost 1.2.10, XGBoost 3.x, LightGBM 4.6.0.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Core: per-strategy pre-fit prep
# ----------------------------------------------------------------------------------


def qid_to_group_sizes(group_ids: np.ndarray) -> np.ndarray:
    """Convert per-row qid array to per-query sizes (LGB / native XGB API).

    Assumes ``group_ids`` are CONTIGUOUS-by-query (i.e. all rows of one
    query are adjacent). For arbitrary qid arrays, sort by group first.
    Uses ``np.diff(np.flatnonzero(...))`` for an O(N) pass.

    Example:
        group_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        returns      [3, 2, 4]
    """
    if len(group_ids) == 0:
        return np.array([], dtype=np.intp)
    arr = np.asarray(group_ids)
    boundaries = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    edges = np.concatenate(([0], boundaries, [len(arr)]))
    return np.diff(edges).astype(np.intp)


def _validate_ranking_inputs(X: Any, y: np.ndarray, group_ids: np.ndarray, *, name: str) -> None:
    """Common shape / contract checks for ranking inputs."""
    n = len(y) if hasattr(y, "__len__") else None
    n_x = X.shape[0] if hasattr(X, "shape") else (len(X) if hasattr(X, "__len__") else None)
    n_g = len(group_ids) if hasattr(group_ids, "__len__") else None

    if n is None or n_x is None or n_g is None:
        raise ValueError(
            f"{name}: X, y, group_ids must all have a length. Got " f"types {type(X).__name__}, {type(y).__name__}, " f"{type(group_ids).__name__}."
        )
    if not (n == n_x == n_g):
        raise ValueError(f"{name}: length mismatch -- y={n}, X={n_x}, group_ids={n_g}")
    if n == 0:
        raise ValueError(f"{name}: empty inputs")


def prepare_cb_inputs(X: Any, y: np.ndarray, group_ids: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """CatBoost requires rows of one query to be contiguous.

    Sorts X / y / group_ids by group_id (stable sort, preserves within-group
    ordering). Returns ``(X_sorted, y_sorted, group_ids_sorted, sort_idx)``
    where ``sort_idx`` lets callers map predictions back to the original
    row order via ``preds[np.argsort(sort_idx)]`` (or simply unsort with
    ``np.empty_like(preds); out[sort_idx] = preds``).

    Already-sorted input is detected and returned as-is (no copy) so
    repeated fits on the same prep'd frame don't pay the sort cost.
    """
    _validate_ranking_inputs(X, y, group_ids, name="prepare_cb_inputs")
    arr = np.asarray(group_ids)

    # Detect already-contiguous (cheaper than sort + compare).
    is_sorted_groups = bool(np.all(arr[1:] >= arr[:-1])) if len(arr) > 1 else True
    if is_sorted_groups:
        return X, np.asarray(y), arr, np.arange(len(arr), dtype=np.intp)

    sort_idx = np.argsort(arr, kind="stable")
    if isinstance(X, pd.DataFrame):
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    elif isinstance(X, np.ndarray):
        X_sorted = X[sort_idx]
    else:
        # Polars or other. Best-effort: convert to numpy positions.
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X_sorted = X[sort_idx.tolist()]
            else:
                X_sorted = np.asarray(X)[sort_idx]
        except ImportError:
            X_sorted = np.asarray(X)[sort_idx]
    y_sorted = np.asarray(y)[sort_idx]
    g_sorted = arr[sort_idx]
    return X_sorted, y_sorted, g_sorted, sort_idx


def prepare_xgb_inputs(X: Any, y: np.ndarray, group_ids: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
    """XGBoost ``XGBRanker.fit(X, y, qid=...)`` accepts per-row qid.

    No sort required (XGB handles arbitrary qid order internally).
    Returns ``(X, y, qid)`` -- qid is just ``group_ids`` reshaped.
    """
    _validate_ranking_inputs(X, y, group_ids, name="prepare_xgb_inputs")
    return X, np.asarray(y), np.asarray(group_ids)


def prepare_lgb_inputs(X: Any, y: np.ndarray, group_ids: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """LightGBM ``LGBMRanker.fit(X, y, group=per_query_sizes)``.

    Sorts by group_id first (LGB's group= shape requires contiguous
    queries) then computes per-query sizes via np.diff on boundaries.
    Returns ``(X_sorted, y_sorted, group_sizes, sort_idx)``.

    Already-sorted input is detected; sort skipped.
    """
    _validate_ranking_inputs(X, y, group_ids, name="prepare_lgb_inputs")
    arr = np.asarray(group_ids)
    is_sorted_groups = bool(np.all(arr[1:] >= arr[:-1])) if len(arr) > 1 else True

    if is_sorted_groups:
        sort_idx = np.arange(len(arr), dtype=np.intp)
        X_sorted, y_sorted, g_sorted = X, np.asarray(y), arr
    else:
        sort_idx = np.argsort(arr, kind="stable")
        if isinstance(X, pd.DataFrame):
            X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        elif isinstance(X, np.ndarray):
            X_sorted = X[sort_idx]
        else:
            try:
                import polars as pl
                if isinstance(X, pl.DataFrame):
                    X_sorted = X[sort_idx.tolist()]
                else:
                    X_sorted = np.asarray(X)[sort_idx]
            except ImportError:
                X_sorted = np.asarray(X)[sort_idx]
        y_sorted = np.asarray(y)[sort_idx]
        g_sorted = arr[sort_idx]

    group_sizes = qid_to_group_sizes(g_sorted)
    return X_sorted, y_sorted, group_sizes, sort_idx


# ----------------------------------------------------------------------------------
# Native ranker fit (per strategy)
# ----------------------------------------------------------------------------------


def _strategy_flavor(strategy) -> str:
    """Map strategy class name to one of {'catboost','xgboost','lightgbm','mlp'}."""
    name = type(strategy).__name__
    if name == "CatBoostStrategy":
        return "catboost"
    if name == "XGBoostStrategy":
        return "xgboost"
    if name == "TreeModelStrategy":
        return "lightgbm"
    if name == "NeuralNetStrategy":
        return "mlp"
    raise NotImplementedError(
        f"Strategy {name!r} has no native ranker. LTR target_type requires "
        f"CatBoost / XGBoost / LightGBM / MLP. Drop this strategy from "
        f"mlframe_models or pick a non-LTR target."
    )


def fit_ranker(
    strategy,
    X_train: Any,
    y_train: np.ndarray,
    group_ids_train: np.ndarray,
    X_val: Any | None = None,
    y_val: np.ndarray | None = None,
    group_ids_val: np.ndarray | None = None,
    ranking_config=None,
    cat_features: list[str | int] | None = None,
    model_kwargs: dict | None = None,
    early_stopping_rounds: int | None = 50,
    verbose: int | bool = False,
) -> dict:
    """Fit a native ranker for the given strategy.

    Returns a dict with::

        {
            "model": <fitted ranker>,
            "flavor": "catboost"|"xgboost"|"lightgbm",
            "objective_kwargs": {...},   # what the dispatcher resolved
            "sort_idx_train": ndarray,   # for unscrambling preds (CB/LGB)
        }

    Caller invokes ``predict_ranker_scores`` with the dict + new X.
    """
    if not getattr(strategy, "supports_native_ranking", False):
        raise NotImplementedError(f"{type(strategy).__name__} does not support native ranking.")

    flavor = _strategy_flavor(strategy)
    y_max = float(np.asarray(y_train).max()) if len(y_train) else None
    obj_kwargs = strategy.get_ranker_objective_kwargs(
        ranking_config=ranking_config, y_max=y_max,
    )
    model_kwargs = dict(model_kwargs or {})

    if flavor == "catboost":
        return _fit_cb_ranker(
            X_train, y_train, group_ids_train,
            X_val, y_val, group_ids_val,
            obj_kwargs, model_kwargs, cat_features,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose,
        )
    elif flavor == "xgboost":
        return _fit_xgb_ranker(
            X_train, y_train, group_ids_train,
            X_val, y_val, group_ids_val,
            obj_kwargs, model_kwargs, cat_features,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose,
        )
    elif flavor == "mlp":
        return _fit_mlp_ranker(
            X_train, y_train, group_ids_train,
            X_val, y_val, group_ids_val,
            obj_kwargs, model_kwargs, cat_features,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose,
        )
    elif flavor == "lightgbm":
        return _fit_lgb_ranker(
            X_train, y_train, group_ids_train,
            X_val, y_val, group_ids_val,
            obj_kwargs, model_kwargs, cat_features,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose,
        )
    raise RuntimeError("unreachable: unhandled ranker family after backend dispatch")


def _fit_cb_ranker(
    X_train, y_train, group_ids_train,
    X_val, y_val, group_ids_val,
    obj_kwargs, model_kwargs, cat_features,
    *, early_stopping_rounds, verbose,
) -> dict:
    """Sorts inputs into CatBoost's required group-contiguous order, fills NA categoricals, and fits a ``CatBoostRanker`` via its Pool API."""
    from catboost import CatBoostRanker

    X_tr, y_tr, g_tr, sort_idx_tr = prepare_cb_inputs(X_train, y_train, group_ids_train)

    # CatBoost's Pool raises ``must be real number, not NoneType`` on an OBJECT-dtype
    # cat column carrying Python ``None`` (from null_fraction_cats), even when the
    # column is declared cat_features. Fill None with a string sentinel so CB treats
    # missing categoricals as their own level. Applied to fit + eval + (via the stored
    # column list) predict so the fit / predict category spaces stay aligned.
    _CB_NA = "__MLFRAME_NA__"

    def _fill_obj_cat_nones(_df):
        """Replaces null cells in declared cat columns (object-dtype OR pandas CategoricalDtype) with
        the sentinel so CatBoost's Pool doesn't choke on them. Mirrors the identical
        object-dtype-vs-CategoricalDtype handling in ``_ranker_suite_train.py``'s FTE prep and the main
        CB training path's own ``__MISSING__``-sentinel convention."""
        if not isinstance(_df, pd.DataFrame) or not cat_features:
            return _df
        _upd = {}
        for _c in cat_features:
            if _c not in _df.columns:
                continue
            _col = _df[_c]
            if _col.dtype == object:
                if _col.isna().any():
                    _upd[_c] = _col.where(_col.notna(), _CB_NA)
            elif isinstance(_col.dtype, pd.CategoricalDtype):
                if _col.isna().any():
                    _series = _col
                    if _CB_NA not in _series.cat.categories:
                        _series = _series.cat.add_categories(_CB_NA)
                    _upd[_c] = _series.fillna(_CB_NA)
        return _df.assign(**_upd) if _upd else _df

    X_tr = _fill_obj_cat_nones(X_tr)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None and group_ids_val is not None:
        X_va, y_va, g_va, _ = prepare_cb_inputs(X_val, y_val, group_ids_val)
        X_va = _fill_obj_cat_nones(X_va)
        # CatBoostRanker.fit accepts eval_set as a tuple of (X, y) plus
        # eval_group_id is set on the Pool internally when X is a Pool.
        # Easiest: build a Pool for eval and pass that.
        from catboost import Pool
        eval_pool = Pool(
            data=X_va, label=y_va, group_id=g_va, cat_features=cat_features,
        )
        fit_kwargs["eval_set"] = eval_pool
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    # obj_kwargs encodes the ranking objective / loss function resolved by the strategy;
    # silently letting model_kwargs override them (the prior {**obj_kwargs, **model_kwargs}
    # ordering) would let a caller-passed loss_function break the ranker assumption. Detect
    # conflicts, log them, and let obj_kwargs win.
    _conflicts = {k: model_kwargs[k] for k in model_kwargs if k in obj_kwargs and model_kwargs[k] != obj_kwargs[k]}
    if _conflicts:
        logger.warning(
            "_fit_cb_ranker: model_kwargs override of ranker objective kwargs ignored: %s "
            "(obj_kwargs from strategy: %s). Drop these keys from model_kwargs to silence.",
            _conflicts, {k: obj_kwargs[k] for k in _conflicts},
        )
    init_kwargs = {**model_kwargs, **obj_kwargs}
    init_kwargs.setdefault("verbose", verbose if isinstance(verbose, int) else (100 if verbose else False))

    model = CatBoostRanker(**init_kwargs)
    # CB ranker sklearn API: fit(X, y, group_id=..., cat_features=...).
    model.fit(
        X_tr, y_tr,
        group_id=g_tr,
        cat_features=cat_features if cat_features else None,
        **fit_kwargs,
    )
    return {
        "model": model,
        "flavor": "catboost",
        "objective_kwargs": obj_kwargs,
        "sort_idx_train": sort_idx_tr,
        "cb_cat_features": list(cat_features) if cat_features else [],
        "cb_na_sentinel": _CB_NA,
    }


def _fit_xgb_ranker(
    X_train, y_train, group_ids_train,
    X_val, y_val, group_ids_val,
    obj_kwargs, model_kwargs, cat_features,
    *, early_stopping_rounds, verbose,
) -> dict:
    """Builds per-row qid arrays (no sort required), coerces declared categoricals to a shared category universe, and fits an ``XGBRanker``."""
    from xgboost import XGBRanker

    X_tr, y_tr, qid_tr = prepare_xgb_inputs(X_train, y_train, group_ids_train)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None and group_ids_val is not None:
        X_va, y_va, qid_va = prepare_xgb_inputs(X_val, y_val, group_ids_val)
        # XGBRanker accepts eval_set as list of (X, y); eval_qid is a sibling list.
        fit_kwargs["eval_set"] = [(X_va, y_va)]
        fit_kwargs["eval_qid"] = [qid_va]

    init_kwargs = {
        "enable_categorical": bool(cat_features),
        **obj_kwargs,
        **model_kwargs,
    }
    if early_stopping_rounds is not None:
        init_kwargs.setdefault("early_stopping_rounds", early_stopping_rounds)
    if verbose:
        init_kwargs.setdefault("verbosity", 1 if verbose is True else int(verbose))

    # XGBRanker with enable_categorical needs a category (not raw object) dtype.
    # Cast declared cat columns to pandas Categorical for fit AND store the dtype
    # so predict re-casts an object / per-DF-category inference frame to the same
    # category universe (otherwise XGB predict raises on object columns / mismatched
    # categories). The shared train+val union keeps fit and eval consistent.
    _xgb_cat_dtypes: dict = {}
    if cat_features and isinstance(X_tr, pd.DataFrame):
        from pandas.api.types import CategoricalDtype as _CatDtype
        _val_for_union = X_va if (X_val is not None and isinstance(X_va, pd.DataFrame)) else None
        for _c in cat_features:
            if _c not in X_tr.columns:
                continue
            _levels: list = []
            _seen: set = set()
            for _frame in [X_tr] + ([_val_for_union] if _val_for_union is not None else []):
                if _c not in _frame.columns:
                    continue
                _col = _frame[_c]
                _uniq = _col.cat.categories.tolist() if isinstance(_col.dtype, _CatDtype) else _col.dropna().unique().tolist()
                for _v in _uniq:
                    if _v not in _seen:
                        _seen.add(_v)
                        _levels.append(_v)
            if _levels:
                _xgb_cat_dtypes[_c] = _CatDtype(categories=sorted(_levels, key=str))

    def _apply_xgb_cats(_df):
        """Casts declared cat columns to the fit-time ``CategoricalDtype`` union so train/eval/predict agree on category codes."""
        if not isinstance(_df, pd.DataFrame) or not _xgb_cat_dtypes:
            return _df
        _upd = {_c: _df[_c].astype(_dt) for _c, _dt in _xgb_cat_dtypes.items() if _c in _df.columns}
        return _df.assign(**_upd) if _upd else _df

    X_tr = _apply_xgb_cats(X_tr)
    if fit_kwargs.get("eval_set"):
        fit_kwargs["eval_set"] = [(_apply_xgb_cats(_xv), _yv) for _xv, _yv in fit_kwargs["eval_set"]]

    model = XGBRanker(**init_kwargs)
    model.fit(X_tr, y_tr, qid=qid_tr, **fit_kwargs)
    return {
        "model": model,
        "flavor": "xgboost",
        "objective_kwargs": obj_kwargs,
        "sort_idx_train": np.arange(len(y_tr), dtype=np.intp),
        "xgb_cat_dtypes": _xgb_cat_dtypes,
    }


def _fit_lgb_ranker(
    X_train, y_train, group_ids_train,
    X_val, y_val, group_ids_val,
    obj_kwargs, model_kwargs, cat_features,
    *, early_stopping_rounds, verbose,
) -> dict:
    """Groups rows into contiguous group-size runs, casts declared categoricals (LightGBM rejects raw object/string dtype), and fits an ``LGBMRanker``."""
    from lightgbm import LGBMRanker

    X_tr, y_tr, group_sizes_tr, sort_idx_tr = prepare_lgb_inputs(X_train, y_train, group_ids_train)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None and group_ids_val is not None:
        X_va, y_va, group_sizes_va, _ = prepare_lgb_inputs(X_val, y_val, group_ids_val)
        fit_kwargs["eval_set"] = [(X_va, y_va)]
        fit_kwargs["eval_group"] = [group_sizes_va]

    init_kwargs = {**obj_kwargs, **model_kwargs}
    if verbose:
        init_kwargs.setdefault("verbose", 1 if verbose is True else int(verbose))
    else:
        init_kwargs.setdefault("verbose", -1)

    model = LGBMRanker(**init_kwargs)

    # LGB uses callbacks for early stopping in newer versions.
    callbacks: list = []
    if early_stopping_rounds is not None and X_val is not None:
        try:
            from lightgbm import early_stopping as _es
            callbacks.append(_es(stopping_rounds=early_stopping_rounds, verbose=False))
        except ImportError:
            pass

    if cat_features:
        # LGB accepts categorical_feature as list of column names or indices.
        # LGB rejects object / str / StringDtype dtype columns even when listed
        # as categorical_feature: ``_check_for_bad_pandas_dtypes`` runs BEFORE
        # the categorical metadata is consulted. Cast each declared cat column
        # to pandas Categorical so LGB's numeric-only data path accepts the
        # codes view. Same coercion applied to eval_set if present.
        import pandas as _pd
        def _coerce_cats(_df):
            """Casts declared cat columns with object/string dtype to pandas ``category`` so LightGBM's dtype pre-check accepts them."""
            if not isinstance(_df, _pd.DataFrame):
                return _df
            _to_cast = {}
            for _c in cat_features:
                if _c not in _df.columns:
                    continue
                _dn = str(_df[_c].dtype)
                if _dn in ("object", "string", "str") or "string" in _dn.lower():
                    _to_cast[_c] = _df[_c].astype("category")
            return _df.assign(**_to_cast) if _to_cast else _df

        X_tr = _coerce_cats(X_tr)
        if fit_kwargs.get("eval_set"):
            fit_kwargs["eval_set"] = [(_coerce_cats(_xv), _yv) for _xv, _yv in fit_kwargs["eval_set"]]
        fit_kwargs["categorical_feature"] = cat_features

    model.fit(
        X_tr, y_tr, group=group_sizes_tr,
        callbacks=callbacks if callbacks else None,
        **fit_kwargs,
    )
    # Persist the train-time CategoricalDtype per cat column so predict can re-cast
    # the inference frame to the SAME category universe. LightGBM compares the
    # predict frame's ``pandas_categorical`` against the model's training metadata
    # and raises "train and valid dataset categorical_feature do not match" when a
    # predict-time category column (object / per-DF category) differs.
    _train_cat_dtypes: dict = {}
    if cat_features and isinstance(X_tr, pd.DataFrame):
        for _c in cat_features:
            if _c in X_tr.columns and isinstance(X_tr[_c].dtype, pd.CategoricalDtype):
                _train_cat_dtypes[_c] = X_tr[_c].dtype
    return {
        "model": model,
        "flavor": "lightgbm",
        "objective_kwargs": obj_kwargs,
        "sort_idx_train": sort_idx_tr,
        "lgb_cat_dtypes": _train_cat_dtypes,
    }


def _fit_mlp_ranker(
    X_train, y_train, group_ids_train,
    X_val, y_val, group_ids_val,
    obj_kwargs, model_kwargs, cat_features,
    *, early_stopping_rounds, verbose,
) -> dict:
    """Fit an MLPRanker (PyTorch Lightning + RankNet/ListNet loss).

    cat_features: MLPRanker doesn't accept raw categoricals (it operates
    on numeric tensors). Caller must encode them to numeric upstream.
    """
    from mlframe.training.neural.ranker import MLPRanker

    # Map our cross-library obj_kwargs to MLPRanker init kwargs.
    # NeuralNetStrategy.get_ranker_objective_kwargs returns
    # {"loss_fn": "ranknet"} (or "listnet"); MLPRanker signature uses
    # the same key.
    init_kwargs = dict(model_kwargs or {})
    if "loss_fn" in obj_kwargs:
        init_kwargs.setdefault("loss_fn", obj_kwargs["loss_fn"])
    if early_stopping_rounds is not None:
        init_kwargs.setdefault("early_stopping_patience", early_stopping_rounds)
    init_kwargs.setdefault("verbose", 1 if verbose else 0)

    # If caller passed CB/XGB/LGB-style kwargs (iterations / n_estimators /
    # learning_rate), propagate -> MLPRanker normalises to n_estimators.
    if "iterations" in init_kwargs and "n_estimators" not in init_kwargs:
        init_kwargs["n_estimators"] = init_kwargs.pop("iterations")
    elif "iterations" in init_kwargs:
        init_kwargs.pop("iterations")  # silently drop dup

    if cat_features:
        logger.info(
            "MLPRanker: cat_features=%s passed but ignored -- MLPRanker " "operates on numeric tensors; caller must pre-encode.",
            cat_features,
        )

    model = MLPRanker(**init_kwargs)
    model.fit(
        X_train, y_train, group_ids_train,
        X_val=X_val, y_val=y_val, group_ids_val=group_ids_val,
        cat_features=cat_features,
    )
    return {
        "model": model,
        "flavor": "mlp",
        "objective_kwargs": obj_kwargs,
        "sort_idx_train": np.arange(len(y_train), dtype=np.intp),
    }


# ----------------------------------------------------------------------------------
# Predict (uniform shape regardless of flavor)
# ----------------------------------------------------------------------------------


def predict_ranker_scores(fitted: dict, X: Any, group_ids: np.ndarray | None = None) -> np.ndarray:
    """Return per-row scores in the SAME order as input rows.

    For CB/LGB, the model was trained on sorted-by-group rows; ``predict``
    on a raw X however does not require sorting (the per-row score is
    independent of order at inference time -- the sort was for the FIT-
    time group_id constraint only).

    Returns a 1-D ``(n_rows,)`` numpy array.
    """
    model = fitted["model"]
    flavor = fitted["flavor"]
    if flavor == "catboost":
        _cb_cats = fitted.get("cb_cat_features") or []
        _cb_na = fitted.get("cb_na_sentinel")
        if _cb_cats and _cb_na is not None and isinstance(X, pd.DataFrame):
            _upd = {}
            for _c in _cb_cats:
                if _c in X.columns and X[_c].dtype == object and X[_c].isna().any():
                    _upd[_c] = X[_c].where(X[_c].notna(), _cb_na)
            if _upd:
                X = X.assign(**_upd)
        scores = model.predict(X)
    elif flavor == "xgboost":
        _cat_dtypes = fitted.get("xgb_cat_dtypes") or {}
        if _cat_dtypes and isinstance(X, pd.DataFrame):
            _upd = {_c: X[_c].astype(_dt) for _c, _dt in _cat_dtypes.items() if _c in X.columns}
            if _upd:
                X = X.assign(**_upd)
        scores = model.predict(X)
    elif flavor == "lightgbm":
        # Re-cast cat columns to the EXACT train-time CategoricalDtype so LightGBM's
        # predict-time ``pandas_categorical`` matches the model's training metadata
        # (otherwise an object / per-DF-category predict frame trips "train and valid
        # dataset categorical_feature do not match").
        _cat_dtypes = fitted.get("lgb_cat_dtypes") or {}
        if _cat_dtypes and isinstance(X, pd.DataFrame):
            _upd = {_c: X[_c].astype(_dt) for _c, _dt in _cat_dtypes.items() if _c in X.columns}
            if _upd:
                X = X.assign(**_upd)
        scores = model.predict(X)
    elif flavor == "mlp":
        scores = model.predict(X)
    else:
        raise ValueError(f"unknown ranker flavor {flavor!r}; expected one of catboost/xgboost/lightgbm/mlp")
    return np.asarray(scores).ravel()


# ----------------------------------------------------------------------------------
# Ensembling for ranking scores
# ----------------------------------------------------------------------------------


def _ranks_within_group(scores: np.ndarray, group_starts: np.ndarray, *, descending: bool = True) -> np.ndarray:
    """Per-group AVERAGE rank assignment.

    For each group's slice ``scores[group_starts[i]:group_starts[i+1]]``,
    assign rank 1 to the highest score (when ``descending=True``), 2 to
    the next, etc. Genuinely TIED items get EQUAL ranks -- the average of
    the 1-based positions they span (scipy ``rankdata(method="average")``
    semantics). This is the canonical RRF/Borda tie handling: two items
    with identical scores must contribute identical reciprocal-rank mass
    regardless of their array index. The prior dense-positional scheme
    broke ties by input position, injecting index-dependent fusion noise
    on genuine ties (a correctness fix, not bit-identical on ties).

    Returns a 1-D ``(n_rows,)`` array of float ranks (1-indexed).

    Vectorised via a single lexsort over (-score, group_id_proxy): the
    per-group Python loop with argsort-per-group was O(n_groups) Python
    calls and dominated wall-clock on >50k-query workloads. Tie-averaging
    is applied vectorised over the sorted order (no per-group loop).
    """
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    n_groups = len(group_starts) - 1
    if n_groups == 0:
        return np.empty(0, dtype=np.float64)

    # Build a per-row group_id proxy: repeat group index by group size.
    group_sizes = np.diff(group_starts)
    group_ids = np.repeat(np.arange(n_groups, dtype=np.intp), group_sizes)

    # lexsort sorts by LAST key as primary; sort by (group_id asc, score)
    # so within-group ordering is score-desc (or asc).
    if descending:
        primary = -scores
    else:
        primary = scores
    order = np.lexsort((primary, group_ids))  # group asc, then score

    # Position within group = (cumulative count) - (group start) after order.
    # Each element's group_id appears `group_sizes[g]` times; in `order`,
    # the i-th element of each group lands at position group_starts[g] + i.
    # So dense 1-based rank = (its index in order) - group_starts[group_id] + 1.
    pos = np.empty(n, dtype=np.intp)
    pos[order] = np.arange(n, dtype=np.intp)
    dense_ranks = (pos - group_starts[group_ids] + 1).astype(np.float64)

    # Average-rank tie correction, vectorised over the sorted order. Two
    # sorted-adjacent rows tie iff they share a group AND an equal score.
    # Ranks in sorted order are strictly increasing per group, so for each
    # maximal tied run we replace the dense ranks by their mean. Compute
    # run boundaries in sorted order, then broadcast the per-run mean back.
    ranks_sorted = dense_ranks[order]  # dense ranks laid out in sorted order
    prim_sorted = primary[order]
    gid_sorted = group_ids[order]
    # A new tied run starts at position i (>0) when group or score changes.
    same_as_prev = np.empty(n, dtype=bool)
    same_as_prev[0] = False
    same_as_prev[1:] = (gid_sorted[1:] == gid_sorted[:-1]) & (prim_sorted[1:] == prim_sorted[:-1])
    run_id = np.cumsum(~same_as_prev) - 1  # 0-based run index in sorted order
    n_runs = int(run_id[-1]) + 1
    run_sum = np.bincount(run_id, weights=ranks_sorted, minlength=n_runs)
    run_cnt = np.bincount(run_id, minlength=n_runs)
    run_mean = run_sum / run_cnt
    avg_sorted = run_mean[run_id]

    # Scatter averaged ranks back to original row order.
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_sorted
    return ranks


def _group_starts_from_ids(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort group_ids and return (sort_idx, group_starts).

    ``group_starts`` defines query slices in the SORTED order; callers
    that want results back in original-row order use ``sort_idx`` for the
    inverse permutation.
    """
    arr = np.asarray(group_ids)
    if len(arr) == 0:
        return np.array([], dtype=np.intp), np.array([0], dtype=np.intp)
    sort_idx = np.argsort(arr, kind="stable")
    sorted_groups = arr[sort_idx]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    group_starts = np.concatenate(([0], boundaries, [len(arr)])).astype(np.intp)
    return sort_idx, group_starts


def ensemble_ranker_scores(
    scores_per_model: list[np.ndarray],
    group_ids: np.ndarray,
    method: str = "rrf",
    rrf_k: int = 60,
    assume_comparable_scales: bool = False,
) -> np.ndarray:
    """Combine per-row ranker scores from multiple models.

    Parameters
    ----------
    scores_per_model : list of (n_rows,) score arrays
        One array per model. All same length and aligned to ``group_ids``.
    group_ids : (n_rows,) array
        Per-row query identifier. Ranks are computed within each query.
    method : {"rrf", "borda", "score_mean"}
        - **rrf**: Reciprocal Rank Fusion -- ``Σ 1/(rrf_k + rank_i)`` per
          row, where rank_i is the rank assigned by model i WITHIN the
          row's query. Higher = better. Scale-invariant (TREC default).
        - **borda**: Per-query rank averaging -- ``Σ rank_i``. LOWER is
          better, so we negate at the end so "higher = better" matches
          the other methods.
        - **score_mean**: Arithmetic mean of raw scores. Requires
          ``assume_comparable_scales=True`` (else WARN + fall back to
          rrf, since CB YetiRank emits ~[0,1], XGB rank:ndcg ~[-10,+10],
          LGB lambdarank arbitrary range -- raw mean is meaningless
          across libraries).
    rrf_k : int
        RRF damping constant. 60 is the TREC default.
    assume_comparable_scales : bool
        Acknowledge that score_mean is safe (you've calibrated externally).

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Ensembled per-row scores, aligned to original row order. Higher
        score = more relevant.
    """
    if not scores_per_model:
        raise ValueError("scores_per_model is empty")
    n_rows = len(scores_per_model[0])
    for i, s in enumerate(scores_per_model):
        if len(s) != n_rows:
            raise ValueError(f"score length mismatch: model 0 has {n_rows}, model {i} has {len(s)}")
    if len(group_ids) != n_rows:
        raise ValueError(f"group_ids length {len(group_ids)} != score length {n_rows}")

    method = method.lower()
    if method not in {"rrf", "borda", "score_mean"}:
        raise ValueError(f"unknown ensemble method {method!r}; must be rrf | borda | score_mean")

    if method == "score_mean" and not assume_comparable_scales:
        # C-P1-4: hard-fail instead of silently mutating method='score_mean' -> method='rrf'. The previous
        # silent fallback meant operators saw 'score_mean' in their config and metadata while RRF math
        # actually executed; method choice is a contract, not a suggestion.
        raise ValueError(
            "ensemble_ranker_scores: method='score_mean' requires assume_comparable_scales=True. "
            "Raw scores from CB/XGB/LGB rankers are NOT comparable -- different objectives emit "
            "different ranges. To enable score_mean, calibrate scores externally and pass "
            "assume_comparable_scales=True. To use rank-fusion instead, pass method='rrf' or "
            "method='borda' explicitly."
        )

    sort_idx, group_starts = _group_starts_from_ids(group_ids)
    inv_sort = np.empty_like(sort_idx)
    inv_sort[sort_idx] = np.arange(len(sort_idx))

    if method == "score_mean":
        out = np.mean(np.asarray(scores_per_model), axis=0)
        return np.asarray(out)

    # rrf and borda both work in rank-space (per-query).
    aggregate = np.zeros(n_rows, dtype=np.float64)
    for s in scores_per_model:
        s_sorted = np.asarray(s, dtype=np.float64)[sort_idx]
        ranks_sorted = _ranks_within_group(s_sorted, group_starts, descending=True)
        # Map ranks back to original row order.
        ranks_orig = ranks_sorted[inv_sort]
        if method == "rrf":
            aggregate += 1.0 / (rrf_k + ranks_orig)
        elif method == "borda":
            aggregate += ranks_orig

    if method == "borda":
        # Borda: lower sum-of-ranks is better. Negate so higher = better
        # (matches contract).
        aggregate = -aggregate

    return aggregate


__all__ = [
    "qid_to_group_sizes",
    "prepare_cb_inputs",
    "prepare_xgb_inputs",
    "prepare_lgb_inputs",
    "fit_ranker",
    "predict_ranker_scores",
    "ensemble_ranker_scores",
]
