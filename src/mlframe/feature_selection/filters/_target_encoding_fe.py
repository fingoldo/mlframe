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
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "auto_detect_te_cols",
    "kfold_target_encode_fit",
    "apply_target_encoding",
    "kfold_target_encode_with_recipes",
    "engineered_name_te",
    "engineered_name_te_stat",
    "TE_SUPPORTED_STATS",
]

# Per-cell target STATISTICS the encoder can emit (beyond the plain mean). std / skew / kurtosis of y within a
# category carry signal the mean cannot when the cell MODULATES a raw feature (heteroscedastic / varying-slope
# regimes): measured +0.04..+0.09 OOS R^2 on varying-slope regression with the encoded stats fed alongside the
# raw feature (bench_multistat_cell_encoding). For a pure mean-shift / homoscedastic / binary target the extra
# moments are redundant (Bernoulli moments are functions of the mean), so ``("mean",)`` stays the default.
TE_SUPPORTED_STATS = ("mean", "std", "skew", "kurt")


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


def engineered_name_te_stat(col: str, stat: str) -> str:
    """Engineered column name for a target-encoded source column + statistic. ``mean`` keeps the historical
    ``{col}__te`` name (back-compat with existing recipes / grep consumers); other stats get ``{col}__te_{stat}``."""
    return engineered_name_te(col) if stat == "mean" else f"{col}__te_{stat}"


def _per_category_stats_smoothed(
    inverse: np.ndarray, y_arr: np.ndarray, n_cats: int, stats: Sequence[str],
    global_stats: dict, smoothing: float,
) -> dict:
    """Vectorised per-category target statistics (mean/std/skew/kurt) from raw moments via ``np.bincount``,
    each shrunk toward its global value (Micci-Barreca) so rare categories stay robust. Returns ``{stat: arr}``
    of length ``n_cats``. O(n) per fold -- replaces the prior per-row Python loop (a real speedup on wide cat sets)."""
    cnt = np.bincount(inverse, minlength=n_cats).astype(np.float64)
    safe = np.maximum(cnt, 1.0)
    s1 = np.bincount(inverse, weights=y_arr, minlength=n_cats)
    mean = s1 / safe
    need_hi = any(s in ("std", "skew", "kurt") for s in stats)
    out: dict = {}
    if need_hi:
        s2 = np.bincount(inverse, weights=y_arr * y_arr, minlength=n_cats)
        m2 = np.maximum(s2 / safe - mean * mean, 0.0)  # variance (clip tiny negatives from fp error)
        std = np.sqrt(m2)
    for stat in stats:
        if stat == "mean":
            raw = mean
        elif stat == "std":
            raw = std
        elif stat == "skew":
            s3 = np.bincount(inverse, weights=y_arr ** 3, minlength=n_cats)
            m3 = s3 / safe - 3.0 * mean * (s2 / safe) + 2.0 * mean ** 3
            raw = np.where(std > 1e-9, m3 / (std ** 3 + 1e-12), 0.0)
        elif stat == "kurt":
            s3 = np.bincount(inverse, weights=y_arr ** 3, minlength=n_cats)
            s4 = np.bincount(inverse, weights=y_arr ** 4, minlength=n_cats)
            m4 = (s4 / safe - 4.0 * mean * (s3 / safe) + 6.0 * mean ** 2 * (s2 / safe) - 3.0 * mean ** 4)
            raw = np.where(m2 > 1e-12, m4 / (m2 * m2 + 1e-12) - 3.0, 0.0)  # excess kurtosis
        else:
            raise ValueError(f"target-encoding stat {stat!r} not in {TE_SUPPORTED_STATS}")
        g = float(global_stats[stat])
        # Shrink toward the global statistic; empty categories (cnt==0) -> global value.
        smoothed = np.where(cnt > 0, (cnt * raw + smoothing * g) / (cnt + smoothing), g)
        out[stat] = smoothed
    return out


def _global_target_stats(y_arr: np.ndarray, stats: Sequence[str]) -> dict:
    """Global (all-rows) value of each requested statistic -- the shrink target / unseen-category fallback."""
    from scipy.stats import kurtosis as _kurt, skew as _skew
    g = {}
    sd = float(np.std(y_arr))
    for stat in stats:
        if stat == "mean":
            g[stat] = float(np.mean(y_arr))
        elif stat == "std":
            g[stat] = sd
        elif stat == "skew":
            g[stat] = float(_skew(y_arr)) if (y_arr.size > 2 and sd > 1e-12) else 0.0
        elif stat == "kurt":
            g[stat] = float(_kurt(y_arr)) if (y_arr.size > 3 and sd > 1e-12) else 0.0
    return {k: (v if np.isfinite(v) else 0.0) for k, v in g.items()}


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
        if not (dt == object or isinstance(dt, pd.CategoricalDtype)
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
    from ._internals import canonical_group_token

    arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
    # Integer / unsigned / bool columns can never hold None or NaN, so the
    # sentinel branch is dead. Convert only the distinct values to canonical
    # tokens and gather (runs per-unique, not per-row): ~5-7x on low-card cat
    # keys. Canonical tokens collapse integral int/float so a fit-int /
    # predict-float dtype drift still hits the per-category encoding instead of
    # the global fallback (the bare str() made '1' and '1.0' distinct keys).
    if arr.dtype.kind in ("i", "u", "b"):
        uniq, inv = np.unique(arr, return_inverse=True)
        toks = np.array([canonical_group_token(u) for u in uniq], dtype=object)
        return toks[inv]
    # object / mixed dtype: canonicalise per-UNIQUE then gather (was a per-ROW
    # Python loop -- 200k calls of ``canonical_group_token`` per 200k-row object
    # column collapse to one call per distinct value). ``pd.factorize`` tolerates
    # the unorderable mixed-type object arrays that ``np.unique`` rejects, and
    # collapses None + NaN into a single sentinel category (use_na_sentinel=False
    # keeps it as a real code, not -1). The per-unique token map is bit-identical
    # to the old per-row map: None / float-NaN uniques -> "__nan__" (the old
    # sentinel), every other unique -> canonical_group_token.
    #
    # GATE: factorize keys on Python equality, so ``True`` collapses with ``1``
    # / ``1.0`` (all == 1) into ONE category -- but the old per-row map emits
    # DISTINCT tokens "True" vs "1" for them. That divergence only arises when
    # the column actually mixes bool with equal-valued numerics; in that case
    # fall back to the exact per-row loop. (Pure-string / pure-numeric / NaN
    # object columns -- the overwhelming common case -- take the fast path.)
    codes, uniq = pd.factorize(arr, use_na_sentinel=False)
    # factorize keys on Python equality, so a bool collapses with an equal-valued
    # numeric / string (``True == 1 == 1.0``) into ONE code -- but the per-row map
    # emits DISTINCT tokens ("True" vs "1"). A lone bool survives as its own unique
    # (caught by the isinstance scan); a COLLIDED bool hides behind a surviving
    # unique that compares == 0 or == 1. So when no unique is bool AND none equals
    # 0/1, no collision is possible and the per-unique fast path is bit-identical;
    # otherwise fall back to the exact per-row loop (rare: bool-in-object column).
    _bool_risk = any(isinstance(v, (bool, np.bool_)) for v in uniq) or any(
        (not (isinstance(v, float) and v != v)) and (v == 0 or v == 1) for v in uniq
    )
    if not _bool_risk:
        toks = np.empty(len(uniq), dtype=object)
        for j, v in enumerate(uniq):
            if v is None or (isinstance(v, float) and v != v):  # None or NaN
                toks[j] = "__nan__"
            else:
                toks[j] = canonical_group_token(v)
        return toks[codes]
    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if v is None:
            out[i] = "__nan__"
        elif isinstance(v, float) and v != v:  # NaN
            out[i] = "__nan__"
        else:
            out[i] = canonical_group_token(v)
    return out


def kfold_target_encode_fit(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: Sequence[str],
    *,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
    stats: Sequence[str] = ("mean",),
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

    stats = tuple(stats) if stats else ("mean",)
    bad = [s for s in stats if s not in TE_SUPPORTED_STATS]
    if bad:
        raise ValueError(f"kfold_target_encode_fit: unsupported stats {bad}; supported {TE_SUPPORTED_STATS}")

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(X)
    global_stats = _global_target_stats(y_arr, stats)

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

        # OOF encoding: for each fold f, compute per-category statistics from rows in folds != f (vectorised via
        # np.bincount moments -- O(n) per fold, no per-row Python loop) and apply to rows in fold f.
        oof = {s: np.full(n, global_stats[s], dtype=np.float64) for s in stats}
        for f in range(int(n_folds)):
            train_mask = fold_ids != f
            if not train_mask.any():
                continue
            per_cat = _per_category_stats_smoothed(
                inverse[train_mask], y_arr[train_mask], n_cats, stats, global_stats, smoothing,
            )
            test_idx = np.where(~train_mask)[0]
            inv_test = inverse[test_idx]
            for s in stats:
                oof[s][test_idx] = per_cat[s][inv_test]

        # Full-data lookups for transform-time replay (one table per statistic).
        full_per_cat = _per_category_stats_smoothed(inverse, y_arr, n_cats, stats, global_stats, smoothing)
        cat_strs = [str(unique_cats[c]) for c in range(n_cats)]
        stat_lookups: dict[str, dict] = {}
        for s in stats:
            stat_lookups[s] = {cat_strs[c]: float(full_per_cat[s][c]) for c in range(n_cats)}
            encoded_cols[engineered_name_te_stat(col, s)] = oof[s]

        recipes[col] = {
            # Back-compat: ``lookup`` / ``global_mean`` are the MEAN statistic (historical single-stat shape).
            "lookup": stat_lookups.get("mean", stat_lookups[stats[0]]),
            "global_mean": float(global_stats.get("mean", global_stats[stats[0]])),
            "smoothing": float(smoothing),
            # Multi-stat payload: per-statistic lookup table + global fallback, in emit order.
            "stats": list(stats),
            "stat_lookups": stat_lookups,
            "global_stats": {s: float(global_stats[s]) for s in stats},
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
    # Vectorized lookup: pd.Series.map resolves the dict once per row in C, with
    # unseen categories -> NaN -> global_mean, replacing the per-row Python
    # dict.get loop. Bit-identical (same key -> same value; the str-keyed lookup
    # and NaN-fill reproduce the dict.get(default) semantics exactly).
    out = (
        pd.Series(cats, copy=False)
        .map(lookup)
        .fillna(global_mean)
        .to_numpy(dtype=np.float64)
    )
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
    reject_sink: Optional[Callable[..., None]] = None,
    stats: Sequence[str] = ("mean",),
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

    stats = tuple(stats) if stats else ("mean",)
    te_df, raw_recipes = kfold_target_encode_fit(
        X, y, cat_cols,
        n_folds=n_folds,
        smoothing=smoothing,
        random_state=random_state,
        stats=stats,
    )

    # Tier-1 local MI floor (Layer 91): drop target-encoded columns whose
    # MI(col; y) falls below the raw-baseline noise floor, keep top-K. Bounds
    # the pool before it reaches MRMR's relevance screen.
    if mi_gate and not te_df.empty:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(te_df, y, raw_X=X, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X.copy(), [], []
        # Gate operates per OUTPUT column (one per (col, stat)); keep the columns it admits.
        te_df = te_df[[c for c in te_df.columns if c in keep]]
        cat_cols = [c for c in cat_cols if any(engineered_name_te_stat(c, s) in keep for s in stats)]

    # Append the encoded columns without disturbing the source columns
    # (MRMR's screening handles them as ordinary numeric features).
    X_aug = pd.concat([X, te_df], axis=1)
    appended = list(te_df.columns)
    _kept = set(appended)

    # One recipe per appended (col, stat) output column. A std / skew / kurt recipe is structurally identical to
    # the mean recipe -- same replay path -- just a different per-category lookup table and global fallback.
    recipes = []
    for col in cat_cols:
        rec_info = raw_recipes[col]
        for s in rec_info.get("stats", ["mean"]):
            out_name = engineered_name_te_stat(col, s)
            if out_name not in _kept:
                continue
            rec = build_kfold_target_encoded_recipe(
                name=out_name,
                src_name=col,
                lookup=rec_info["stat_lookups"][s],
                global_mean=rec_info["global_stats"][s],
                smoothing=rec_info["smoothing"],
            )
            recipes.append(rec)

    return X_aug, appended, recipes
