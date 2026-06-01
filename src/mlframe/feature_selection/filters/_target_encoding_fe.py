"""K-Fold target encoding for categorical features (Layer 33, 2026-05-31).

Modal production tabular ML pipelines target-encode medium / high-cardinality
categorical columns (mean of y per category). The naive single-pass encoding
leaks y into X (the Layer 17 leakage pattern) -- the per-row encoded value
is computed from a histogram that INCLUDES the row's own y. K-fold OOF
target encoding is the standard leakage-safe pattern:

  * For each fold F, compute per-category mean(y) using rows in folds != F.
  * Apply that to the rows in fold F.
  * At transform time (no y), apply the stored full-data per-category mean.

Smoothing (Micci-Barreca 2001) shrinks rare-category estimates toward the
global mean: ``te = (n_c * raw + alpha * global) / (n_c + alpha)`` with
``alpha = smoothing``. Categories never seen during fit fall back to the
global mean at transform time (no NaN propagation).

Sibling to ``_cat_target_encoding_and_weighted._compute_target_encoding``,
which target-encodes MERGED k-way categorical CELLS inside the cat-FE
pair-search kernel; that path is gated on cat-interactions, not on raw
single-column categoricals, and its output is folded into the merged-class
factorize lookup rather than a recipe per source column.

The recipe (kind ``"kfold_target_encoded"``) carries only the per-category
``te_value`` lookup + ``global_mean`` + ``smoothing`` -- no y reference at
replay time. ``MRMR.transform`` recomputes each column deterministically.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "auto_detect_te_cols",
    "kfold_target_encode_fit",
    "apply_target_encoding",
    "kfold_target_encode_with_recipes",
    "engineered_name_te",
]


# ---------------------------------------------------------------------------
# Naming + auto-detection
# ---------------------------------------------------------------------------


def engineered_name_te(col: str) -> str:
    """Stable engineered column name for a target-encoded source column.

    Suffix ``_te`` matches the prior ``_compute_target_encoding`` convention
    used by the cat-FE pair-search emit path; downstream consumers that
    grep ``__te`` in column names already exist (sklearn pipelines /
    plotting helpers)."""
    return f"{col}__te"


def auto_detect_te_cols(
    X: pd.DataFrame,
    *,
    min_card: int = 5,
    max_card: int = 500,
) -> list[str]:
    """Pick columns that are good candidates for k-fold target encoding.

    Heuristics:
      * Object / category / string dtype: ALWAYS candidate when cardinality
        in ``[min_card, max_card]``.
      * Low-cardinality integer columns ARE NOT auto-selected here. The
        existing ``composite_auto_detect.detect_group_column_candidates``
        already heuristically promotes int-low-card to "categorical-ish",
        but it's calibrated for linear_residual_grouped and would
        false-positive for low-cardinality identifiers like ``year`` or
        ``count_top_decile`` where mean-of-y is meaningless. Caller can
        pass an explicit list to bypass auto-detect for those.

    Cardinality bounds (5 .. 500): below 5 the column is better one-hot-
    encoded (no target leakage risk at one-hot); above 500 the per-category
    sample count is too small for stable mean estimates even with
    smoothing.
    """
    if not isinstance(X, pd.DataFrame):
        return []
    candidates: list[str] = []
    for col in X.columns:
        dt = X[col].dtype
        if not (dt == object or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])):
            continue
        # nunique() with dropna=True is fast (Cython on pandas).
        try:
            card = int(X[col].nunique(dropna=True))
        except Exception:
            continue
        if min_card <= card <= max_card:
            candidates.append(col)
    return candidates


# ---------------------------------------------------------------------------
# K-fold OOF encoder (fit-time)
# ---------------------------------------------------------------------------


def _smooth(raw_mean: float, count: float, global_mean: float, smoothing: float) -> float:
    """Micci-Barreca shrinkage toward ``global_mean`` with strength ``smoothing``."""
    if count <= 0.0:
        return global_mean
    return (count * raw_mean + smoothing * global_mean) / (count + smoothing)


def _column_to_str(col: pd.Series) -> np.ndarray:
    """Coerce a categorical / object column to a numpy array of Python
    strings. NaN values map to a sentinel ``"__nan__"`` so they form their
    own implicit category at fit AND transform time (no NaN propagation
    when the test row has NaN in the source column)."""
    arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if v is None:
            out[i] = "__nan__"
        elif isinstance(v, float) and v != v:  # NaN
            out[i] = "__nan__"
        else:
            out[i] = str(v)
    return out


def kfold_target_encode_fit(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: Sequence[str],
    *,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit K-fold out-of-fold target encoding for each column in ``cat_cols``.

    Parameters
    ----------
    X : pd.DataFrame
        Input frame containing every column in ``cat_cols``.
    y : ndarray, shape (n,)
        Target. For binary classification this is treated as {0, 1};
        per-cell mean of y is then per-cell P(y=1). For regression any
        numeric y works (mean is mean).
    cat_cols : sequence of str
        Categorical columns to encode. ``auto_detect_te_cols`` may be used
        to pick these.
    n_folds : int, default 5
        K-fold split. Must be >= 2.
    smoothing : float, default 10.0
        Micci-Barreca shrinkage strength. With ``alpha = smoothing`` a
        category with one row gets weight 1 / (1 + alpha) on its raw mean
        and ``alpha / (1 + alpha)`` on the global mean. At alpha = 10, a
        category needs ~10 rows before its raw estimate dominates the
        prior.
    random_state : int, default 0
        Seeds the fold assignment.

    Returns
    -------
    te_df : pd.DataFrame
        Shape (n, len(cat_cols)). Column names: ``{col}__te`` for each
        ``col`` in ``cat_cols``. Per-row OOF target-encoded value.
    recipes : dict
        ``{col: {"lookup": {category: te_value}, "global_mean": float,
                  "smoothing": float}}``. ``lookup`` is built from the
        FULL training data (no fold split) -- this is the deterministic
        replay table used by ``apply_target_encoding`` at transform time.
        Categories not present in ``lookup`` map to ``global_mean``.

    Raises
    ------
    ValueError
        On invalid n_folds, missing columns, or zero-length y.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    if len(X) == 0:
        raise ValueError("kfold_target_encode_fit: X is empty")
    if len(y) != len(X):
        raise ValueError(
            f"kfold_target_encode_fit: len(y)={len(y)} != len(X)={len(X)}"
        )
    missing = [c for c in cat_cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"kfold_target_encode_fit: columns missing from X: {missing}"
        )

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(X)
    global_mean = float(y_arr.mean())

    # Deterministic fold assignment via numpy generator. Round-robin over
    # SHUFFLED indices so categories that happen to cluster in the input
    # ordering don't all land in the same fold (which would make their OOF
    # estimate identical to the in-fold estimate -- defeats leakage guard).
    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n) % int(n_folds)

    encoded_cols: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}

    for col in cat_cols:
        cats = _column_to_str(X[col])
        # Unique categories with stable integer codes.
        unique_cats, inverse = np.unique(cats, return_inverse=True)
        n_cats = unique_cats.shape[0]

        # OOF encoding: for each fold f, build per-category sum/count from
        # rows in folds != f and apply to rows in fold f. Single pass over
        # the data per fold; total O(n * K) which is fine for K=5.
        oof_values = np.full(n, global_mean, dtype=np.float64)
        for f in range(int(n_folds)):
            train_mask = fold_ids != f
            test_mask = ~train_mask
            if not train_mask.any():
                continue
            cell_sum = np.zeros(n_cats, dtype=np.float64)
            cell_cnt = np.zeros(n_cats, dtype=np.float64)
            train_idx = np.where(train_mask)[0]
            for row in train_idx:
                c = int(inverse[row])
                cell_sum[c] += y_arr[row]
                cell_cnt[c] += 1.0
            cell_means_fold = np.full(n_cats, global_mean, dtype=np.float64)
            for c in range(n_cats):
                if cell_cnt[c] > 0.0:
                    raw = cell_sum[c] / cell_cnt[c]
                    cell_means_fold[c] = _smooth(raw, cell_cnt[c], global_mean, smoothing)
            test_idx = np.where(test_mask)[0]
            for row in test_idx:
                oof_values[row] = cell_means_fold[int(inverse[row])]
        encoded_cols[engineered_name_te(col)] = oof_values

        # Full-data lookup for transform-time replay. This is the table
        # ``apply_target_encoding`` consults; not used for training rows
        # (those got their OOF values above).
        full_sum = np.zeros(n_cats, dtype=np.float64)
        full_cnt = np.zeros(n_cats, dtype=np.float64)
        for row in range(n):
            c = int(inverse[row])
            full_sum[c] += y_arr[row]
            full_cnt[c] += 1.0
        lookup: dict[str, float] = {}
        for c in range(n_cats):
            cat_str = str(unique_cats[c])
            if full_cnt[c] > 0.0:
                raw = full_sum[c] / full_cnt[c]
                lookup[cat_str] = _smooth(raw, full_cnt[c], global_mean, smoothing)
            else:
                lookup[cat_str] = global_mean
        recipes[col] = {
            "lookup": lookup,
            "global_mean": global_mean,
            "smoothing": float(smoothing),
        }

    te_df = pd.DataFrame(encoded_cols, index=X.index)
    return te_df, recipes


# ---------------------------------------------------------------------------
# Transform-time replay
# ---------------------------------------------------------------------------


def apply_target_encoding(
    X_test: pd.DataFrame | np.ndarray,
    col: str,
    recipe: dict,
) -> np.ndarray:
    """Deterministically apply the stored TE lookup to a test column.

    Categories not in ``recipe["lookup"]`` map to ``recipe["global_mean"]``
    (no NaN). NaN values in the source column map to ``"__nan__"`` which
    is itself a category in the lookup (it was treated as such at fit
    time); if NaN was never seen at fit, the lookup miss falls back to
    global_mean exactly like any other unseen category.

    Parameters
    ----------
    X_test : pd.DataFrame or ndarray with column-name access
        Test frame.
    col : str
        Column name to encode.
    recipe : dict
        Per-column recipe from ``kfold_target_encode_fit``. Must contain
        ``lookup`` and ``global_mean``.

    Returns
    -------
    encoded : ndarray, shape (n_test,)
        Float64 encoded column.
    """
    if "lookup" not in recipe or "global_mean" not in recipe:
        raise KeyError(
            f"apply_target_encoding: recipe for col {col!r} is missing "
            f"'lookup' or 'global_mean'. Re-fit to regenerate."
        )
    if isinstance(X_test, pd.DataFrame):
        col_series = X_test[col]
    elif hasattr(X_test, "__getitem__") and not isinstance(X_test, np.ndarray):
        # polars or similar; fall back to repeated single-column extract.
        col_series = pd.Series(X_test[col].to_numpy())
    else:
        raise TypeError(
            f"apply_target_encoding: X_test must be a DataFrame with "
            f"named columns; got {type(X_test).__name__}"
        )
    cats = _column_to_str(col_series)
    lookup: dict = recipe["lookup"]
    global_mean: float = float(recipe["global_mean"])
    out = np.empty(len(cats), dtype=np.float64)
    for i, c in enumerate(cats):
        out[i] = lookup.get(c, global_mean)
    return out


# ---------------------------------------------------------------------------
# End-to-end wrapper for MRMR.fit auto-wiring
# ---------------------------------------------------------------------------


def kfold_target_encode_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
    auto_min_card: int = 5,
    auto_max_card: int = 500,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
):
    """End-to-end: detect / accept cat cols, fit OOF encoding, build
    ``EngineeredRecipe`` objects ready for ``MRMR.transform`` replay.

    Returns
    -------
    X_augmented : pd.DataFrame
        ``X`` with the ``{col}__te`` columns appended. Source columns are
        kept (caller can drop them later if desired -- MRMR's screening
        will treat the encoded col as numeric and the source col as
        categorical, and may keep / drop either).
    encoded_columns : list of str
        Names of the appended TE columns (in append order).
    recipes : list of EngineeredRecipe
        One per appended column, kind ``"kfold_target_encoded"``.
    """
    from .engineered_recipes import build_kfold_target_encoded_recipe

    if cat_cols is None or len(cat_cols) == 0:
        cat_cols = auto_detect_te_cols(X, min_card=auto_min_card, max_card=auto_max_card)
    if not cat_cols:
        return X.copy(), [], []

    te_df, raw_recipes = kfold_target_encode_fit(
        X, y, cat_cols,
        n_folds=n_folds,
        smoothing=smoothing,
        random_state=random_state,
    )

    # Tier-1 local MI floor (Layer 91): drop target-encoded columns whose
    # MI(col; y) falls below the raw-baseline noise floor, keep top-K. Bounds
    # the pool before it reaches MRMR's relevance screen.
    if mi_gate and not te_df.empty:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(te_df, y, raw_X=X, top_k=mi_gate_top_k))
        if not keep:
            return X.copy(), [], []
        kept_src = [c for c in cat_cols if engineered_name_te(c) in keep]
        te_df = te_df[[engineered_name_te(c) for c in kept_src]]
        cat_cols = kept_src

    # Append the encoded columns without disturbing the source columns
    # (MRMR's screening handles them as ordinary numeric features).
    X_aug = pd.concat([X, te_df], axis=1)
    appended = list(te_df.columns)

    recipes = []
    for col in cat_cols:
        rec = build_kfold_target_encoded_recipe(
            name=engineered_name_te(col),
            src_name=col,
            lookup=raw_recipes[col]["lookup"],
            global_mean=raw_recipes[col]["global_mean"],
            smoothing=raw_recipes[col]["smoothing"],
        )
        recipes.append(rec)

    return X_aug, appended, recipes
