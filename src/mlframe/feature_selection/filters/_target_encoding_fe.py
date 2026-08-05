"""K-Fold target encoding for categorical features (Layer 33, 2026-05-31).

Modal production tabular ML pipelines target-encode medium / high-cardinality
categorical columns (mean of y per category). The naive single-pass encoding
leaks y into X (the Layer 17 leakage pattern) - the per-row encoded value
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
``te_value`` lookup + ``global_mean`` + ``smoothing`` - no y reference at
replay time. ``MRMR.transform`` recomputes each column deterministically.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit

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
# Robust / order-statistic stats (median, trimmed_mean, q10, q90, iqr, min, max) live in _target_encoding_order_stats:
# they need y sorted within each category (not expressible from raw moments), and carry their own per-stat stability
# floors so rare cells fall back to the global value. Promoting the order-stat winners to the ctor default is a tracked
# follow-up - it needs a multi-seed sweep confirming the biz_value win generalises before the default flips.
from ._target_encoding_order_stats import ORDER_STATS as _TE_ORDER_STATS, global_order_stats, per_category_order_stats

_TE_MOMENT_STATS = ("mean", "std", "skew", "kurt")
TE_SUPPORTED_STATS = _TE_MOMENT_STATS + _TE_ORDER_STATS


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


@njit(cache=True)
def _per_cat_centered_moments_njit(inverse: np.ndarray, y: np.ndarray, n_cats: int) -> tuple:
    """One-pass-per-stage, numerically-STABLE per-category ``(cnt, mean, m2, m3, m4)`` where
    ``m2/m3/m4`` are sums of CENTRED powers ``(y - mean[cat])**k``, not raw power sums.

    bench-attempt-rejected (2026-08-04): the original per-category moment path here (and its sibling in
    ``_binned_numeric_agg_fe.py``'s ``_derive_cell_stats``) derived skew/kurt from RAW power sums
    (``sum(y)``/``sum(y**2)``/``sum(y**3)``/``sum(y**4)``) via the textbook binomial expansion -- measured
    CATASTROPHICALLY WRONG (errors up to 5.8e13) on categories whose ``y`` has a large mean relative to its
    spread (e.g. offset~1e4, scale~1e-1..1e-3), the classic large-nearly-equal-numbers cancellation this
    project has already hit and fixed twice this session (``_global_stats_all`` in `_binned_numeric_agg_fe.py`,
    the fused bootstrap bundle). Target-encoding a real regression target (price, revenue, counts - rarely
    centred at 0) with ``stats=(...,"skew","kurt")`` hits this exact regime. Two-pass centred accumulation
    (mean first, then ``(y-mean)**k`` directly - no algebraic expansion, no cancellation) matches what scipy's
    own skew/kurtosis do internally, at the SAME O(n) cost (mean pass + one fused centred-power pass)."""
    n = inverse.shape[0]
    cnt = np.zeros(n_cats, dtype=np.float64)
    s1 = np.zeros(n_cats, dtype=np.float64)
    for i in range(n):
        c = inverse[i]
        cnt[c] += 1.0
        s1[c] += y[i]
    mean = np.zeros(n_cats, dtype=np.float64)
    for c in range(n_cats):
        if cnt[c] > 0.0:
            mean[c] = s1[c] / cnt[c]
    m2 = np.zeros(n_cats, dtype=np.float64)
    m3 = np.zeros(n_cats, dtype=np.float64)
    m4 = np.zeros(n_cats, dtype=np.float64)
    for i in range(n):
        c = inverse[i]
        d = y[i] - mean[c]
        d2 = d * d
        m2[c] += d2
        m3[c] += d2 * d
        m4[c] += d2 * d2
    return cnt, mean, m2, m3, m4


def _smooth_moments_from_centered(
    cnt: np.ndarray, mean: np.ndarray, m2: np.ndarray, m3: np.ndarray, m4: np.ndarray,
    moment_stats: Sequence[str], global_stats: dict, smoothing: float,
) -> dict:
    """Derive smoothed (Micci-Barreca) per-category moment stats from the centred moments in
    :func:`_per_cat_centered_moments_njit`. Pure function of those moments - callable on either the
    FULL-data moments or a fold's TRAIN-only moments (computed directly on the train subset, since centred
    moments -- unlike raw power sums -- are NOT additive/subtractable across row subsets: the train subset's
    own mean differs from the full-data mean, so ``moments(full) - moments(test) != moments(train)`` for any
    centred quantity. This trades the old raw-sum ``full - test`` O(n/n_folds) shortcut for a direct
    O(n_train) pass per fold - still O(n) total across all folds, just without the ~(n_folds-1)/2x row-visit
    reduction the (buggy) raw-sum subtraction bought; correctness comes first.

    No ``+eps`` denominator padding (unlike this function's raw-moment predecessor): the ``np.where`` guards
    (``std > 1e-9`` / ``var > 1e-12``) already ensure the denominator is bounded away from zero before it's
    used, so an ADDITIVE epsilon pad only corrupts small-but-legitimate variances (e.g. var~1e-6 has
    var**2~1e-12, on the same order as a naively-added 1e-12 pad) instead of protecting against a genuine
    div-by-zero it can no longer reach."""
    out: dict = {}
    if not moment_stats:
        return out
    safe = np.maximum(cnt, 1.0)
    need_hi = any(s in ("std", "skew", "kurt") for s in moment_stats)
    if need_hi:
        var = m2 / safe
        std = np.sqrt(var)
    for stat in moment_stats:
        if stat == "mean":
            rawv = mean
        elif stat == "std":
            rawv = std
        elif stat == "skew":
            m3n = m3 / safe
            rawv = np.where(std > 1e-9, m3n / std**3, 0.0)
        elif stat == "kurt":
            m4n = m4 / safe
            rawv = np.where(var > 1e-12, m4n / (var * var) - 3.0, 0.0)  # excess kurtosis
        else:
            raise ValueError(f"target-encoding stat {stat!r} not in {TE_SUPPORTED_STATS}")
        g = float(global_stats[stat])
        # Shrink toward the global statistic; empty categories (cnt==0) -> global value.
        smoothed = np.where(cnt > 0, (cnt * rawv + smoothing * g) / (cnt + smoothing), g)
        out[stat] = smoothed
    return out


def _global_target_stats(y_arr: np.ndarray, stats: Sequence[str]) -> dict:
    """Global (all-rows) value of each requested statistic - the shrink target / unseen-category / rare-cell fallback."""
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
    g.update(global_order_stats(y_arr, stats))
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
        if not (dt == object or isinstance(dt, pd.CategoricalDtype)  # noqa: E721 - pandas dtype `== object` comparison is intended
                or pd.api.types.is_string_dtype(X[col])):
            continue
        # nunique() with dropna=True is fast (Cython on pandas).
        try:
            card = int(X[col].nunique(dropna=True))
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
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
    # Python loop - 200k calls of ``canonical_group_token`` per 200k-row object
    # column collapse to one call per distinct value). ``pd.factorize`` tolerates
    # the unorderable mixed-type object arrays that ``np.unique`` rejects, and
    # collapses None + NaN into a single sentinel category (use_na_sentinel=False
    # keeps it as a real code, not -1). The per-unique token map is bit-identical
    # to the old per-row map: None / float-NaN uniques -> "__nan__" (the old
    # sentinel), every other unique -> canonical_group_token.
    #
    # GATE: factorize keys on Python equality, so ``True`` collapses with ``1``
    # / ``1.0`` (all == 1) into ONE category - but the old per-row map emits
    # DISTINCT tokens "True" vs "1" for them. That divergence only arises when
    # the column actually mixes bool with equal-valued numerics; in that case
    # fall back to the exact per-row loop. (Pure-string / pure-numeric / NaN
    # object columns - the overwhelming common case - take the fast path.)
    codes, uniq = pd.factorize(arr, use_na_sentinel=False)
    # factorize keys on Python equality, so a bool collapses with an equal-valued
    # numeric / string (``True == 1 == 1.0``) into ONE code - but the per-row map
    # emits DISTINCT tokens ("True" vs "1"). A lone bool survives as its own unique
    # (caught by the isinstance scan); a COLLIDED bool hides behind a surviving
    # unique that compares == 0 or == 1. So when no unique is bool AND none equals
    # 0/1, no collision is possible and the per-unique fast path is bit-identical;
    # otherwise fall back to the exact per-row loop (rare: bool-in-object column).
    # The factorize/per-row token divergence (True/1/1.0 collapse to one code, but the per-row map emits distinct
    # "True" vs "1") requires a bool AND an ==-equal numeric (0/1) to COEXIST - a 0/1 WITHOUT a bool factorizes to
    # its own canonical token, bit-identical to the per-row map. So gate on AND, not OR, and vectorise the 0/1 test
    # (object elementwise; NaN/str -> False), so a high-card numeric-object column that merely CONTAINS a 0/1 value
    # takes the fast path (~8x @100k) instead of the per-row fallback; the bool scan is short-circuited away entirely
    # when no 0/1 is present (the overwhelming common case).
    try:
        _has_01 = bool(np.asarray((uniq == 0) | (uniq == 1)).any())
    except Exception as e:
        logger.debug("vectorized 0/1 check failed, falling back to the per-element loop: %s", e)
        _has_01 = any((not (isinstance(v, float) and v != v)) and (v == 0 or v == 1) for v in uniq)
    # The bool-instance scan must run over the RAW array, not ``uniq``: factorize keeps only ONE representative
    # per equivalence class, and when a collided bool loses that slot to an equal-valued int (e.g. array order
    # puts ``1`` before ``True``), ``uniq`` never contains the bool at all - scanning ``uniq`` here silently
    # missed exactly the collision case this gate exists to catch. Restrict the scan to the 0/1-valued rows of
    # ``arr`` (not the full array) to keep the common high-cardinality case cheap.
    if _has_01:
        try:
            _zero_one_mask = np.asarray((arr == 0) | (arr == 1))
        except Exception as e:
            logger.debug("vectorized 0/1 mask failed, falling back to the per-element loop: %s", e)
            _zero_one_mask = np.array([(not (isinstance(v, float) and v != v)) and (v == 0 or v == 1) for v in arr], dtype=bool)
        _bool_risk = any(isinstance(v, (bool, np.bool_)) for v in arr[_zero_one_mask])
    else:
        _bool_risk = False
    if not _bool_risk:
        toks = np.empty(len(uniq), dtype=object)
        for j, v in enumerate(uniq):
            if v is None or (isinstance(v, float) and v != v):  # None or NaN
                toks[j] = "__nan__"
            else:
                toks[j] = canonical_group_token(v)
        return np.asarray(toks[codes])
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
        FULL training data (no fold split) - this is the deterministic
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
        raise ValueError(f"kfold_target_encode_fit: len(y)={len(y)} != len(X)={len(X)}")
    missing = [c for c in cat_cols if c not in X.columns]
    if missing:
        raise ValueError(f"kfold_target_encode_fit: columns missing from X: {missing}")

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
    # estimate identical to the in-fold estimate - defeats leakage guard).
    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n) % int(n_folds)

    encoded_cols: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}

    # Fold membership depends ONLY on ``f`` (not the column), so precompute the train masks + test indices
    # ONCE instead of recomputing ``fold_ids != f`` (O(n) bool) and ``np.where`` per (col, fold). Bit-identical.
    _fold_ne = [fold_ids != f for f in range(int(n_folds))]
    _fold_test_idx = [np.where(fold_ids == f)[0] for f in range(int(n_folds))]
    moment_stats = [s for s in stats if s in _TE_MOMENT_STATS]
    order_stats_wanted = [s for s in stats if s in _TE_ORDER_STATS]
    from ._fe_deadline import fe_deadline_passed

    for col in cat_cols:
        # Optional-enrichment wall-clock budget: stop the per-column K-fold target-encoding fit once
        # MRMR.fit's deadline passes; return whatever columns/recipes were engineered so far. No-op
        # without a budget (mirrors the orth-univariate/pair-cross/extra-basis generators' internal
        # deadline check).
        if fe_deadline_passed():
            break
        cats = _column_to_str(X[col])
        # Unique categories with stable integer codes.
        unique_cats, inverse = np.unique(cats, return_inverse=True)
        n_cats = unique_cats.shape[0]

        # Full-data counts, needed for order stats + the persisted replay lookup below.
        full_counts = np.bincount(inverse, minlength=n_cats) if (moment_stats or order_stats_wanted) else None

        # OOF encoding: for each fold f, compute per-category statistics from rows in folds != f and apply to
        # rows in fold f.
        oof = {s: np.full(n, global_stats[s], dtype=np.float64) for s in stats}
        for f in range(int(n_folds)):
            test_idx = _fold_test_idx[f]
            # Equivalent to the old ``_fold_ne[f].any()`` full-array-scan gate (O(n), done once per
            # (col, fold)) without touching the array: train is empty iff every row fell into this
            # fold's test set, i.e. test_idx (already computed) covers the whole dataset.
            if test_idx.size == n:
                continue
            train_mask = _fold_ne[f]
            per_cat: dict = {}
            if moment_stats:
                # Direct O(n_train) pass on the TRAIN subset via train_mask (already available) -- NOT the old
                # full-minus-test raw-sum subtraction (see _smooth_moments_from_centered's docstring for why
                # that shortcut can't carry over to centred moments: the train subset's own mean differs from
                # the full-data mean, so centred moments aren't additive/subtractable the way raw power sums
                # were).
                t_cnt, t_mean, t_m2, t_m3, t_m4 = _per_cat_centered_moments_njit(
                    np.ascontiguousarray(inverse[train_mask]), np.ascontiguousarray(y_arr[train_mask], dtype=np.float64), n_cats,
                )
                per_cat.update(_smooth_moments_from_centered(t_cnt, t_mean, t_m2, t_m3, t_m4, moment_stats, global_stats, smoothing))
            if order_stats_wanted:
                # Order stats need y actually sorted within each TRAIN category (not expressible from additive
                # sums), so this branch still rescans the train rows - unchanged from the pre-existing behaviour.
                per_cat.update(per_category_order_stats(inverse[train_mask], y_arr[train_mask], n_cats, order_stats_wanted, global_stats))
            inv_test = inverse[test_idx]
            for s in stats:
                oof[s][test_idx] = per_cat[s][inv_test]

        # Full-data lookups for transform-time replay (one table per statistic).
        full_per_cat: dict = {}
        if moment_stats:
            f_cnt, f_mean, f_m2, f_m3, f_m4 = _per_cat_centered_moments_njit(
                np.ascontiguousarray(inverse), np.ascontiguousarray(y_arr, dtype=np.float64), n_cats,
            )
            full_per_cat.update(_smooth_moments_from_centered(f_cnt, f_mean, f_m2, f_m3, f_m4, moment_stats, global_stats, smoothing))
        if order_stats_wanted:
            full_per_cat.update(per_category_order_stats(inverse, y_arr, n_cats, order_stats_wanted, global_stats, counts=full_counts))
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
        raise KeyError(f"apply_target_encoding: recipe for col {col!r} is missing " f"'lookup' or 'global_mean'. Re-fit to regenerate.")
    if isinstance(X_test, pd.DataFrame):
        col_series = X_test[col]
    elif hasattr(X_test, "__getitem__") and not isinstance(X_test, np.ndarray):
        # polars or similar; fall back to repeated single-column extract.
        col_series = pd.Series(X_test[col].to_numpy())
    else:
        raise TypeError(f"apply_target_encoding: X_test must be a DataFrame with " f"named columns; got {type(X_test).__name__}")
    lookup: dict = recipe["lookup"]
    global_mean: float = float(recipe["global_mean"])
    # Integer / unsigned / bool source columns (the common high-cardinality
    # categorical case: user_id / merchant_id / device fingerprint) never hold
    # None/NaN, so ``_column_to_str`` would materialise a length-n OBJECT array of
    # canonical string tokens purely so the str-keyed ``lookup`` can be ``.map``-ed
    # per row - two passes over n rows (build-tokens + per-row hash). Fuse them:
    # canonicalise + resolve the lookup once PER DISTINCT integer value (a few
    # hundred), then gather via the ``np.unique`` inverse codes. The int ``np.unique``
    # is cheap (object ``np.unique`` is NOT - hence this fast path is gated to the
    # integral kinds only; object/mixed columns keep the ``_column_to_str`` + ``.map``
    # path). Bit-identical to the ``.map`` path BY CONSTRUCTION: the per-unique
    # ``canonical_group_token`` is the exact token ``_column_to_str`` emits for the
    # same value, and ``lookup.get(token, global_mean)`` reproduces map->NaN->fillna.
    _arr = col_series.to_numpy() if hasattr(col_series, "to_numpy") else np.asarray(col_series)
    if _arr.dtype.kind in ("i", "u", "b"):
        from ._internals import canonical_group_token

        uniq, inv = np.unique(_arr, return_inverse=True)
        vals = np.array(
            [lookup.get(canonical_group_token(u), global_mean) for u in uniq],
            dtype=np.float64,
        )
        return vals[inv]
    cats = _column_to_str(col_series)
    # Vectorized lookup: pd.Series.map resolves the dict once per row in C, with
    # unseen categories -> NaN -> global_mean, replacing the per-row Python
    # dict.get loop. Bit-identical (same key -> same value; the str-keyed lookup
    # and NaN-fill reproduce the dict.get(default) semantics exactly).
    out = pd.Series(cats, copy=False).map(lookup).fillna(global_mean).to_numpy(dtype=np.float64)
    return np.asarray(out)


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
        kept (caller can drop them later if desired - MRMR's screening
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
        return X, [], []

    stats = tuple(stats) if stats else ("mean",)
    te_df, raw_recipes = kfold_target_encode_fit(
        X, y, cat_cols,
        n_folds=n_folds,
        smoothing=smoothing,
        random_state=random_state,
        stats=stats,
    )
    # kfold_target_encode_fit's internal per-column loop honours MRMR.fit's optional wall-clock deadline
    # and may return early with fewer columns than requested - narrow cat_cols to what raw_recipes
    # actually has (order-preserving) so the recipe-building loop below never KeyErrors on a column the
    # fit didn't reach.
    cat_cols = [c for c in cat_cols if c in raw_recipes]

    # Tier-1 local MI floor (Layer 91): drop target-encoded columns whose
    # MI(col; y) falls below the raw-baseline noise floor, keep top-K. Bounds
    # the pool before it reaches MRMR's relevance screen.
    if mi_gate and not te_df.empty:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(te_df, y, raw_X=X, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X, [], []
        # Gate operates per OUTPUT column (one per (col, stat)); keep the columns it admits.
        te_df = te_df[[c for c in te_df.columns if c in keep]]
        cat_cols = [c for c in cat_cols if any(engineered_name_te_stat(c, s) in keep for s in stats)]

    # Append the encoded columns without disturbing the source columns
    # (MRMR's screening handles them as ordinary numeric features).
    X_aug = pd.concat([X, te_df], axis=1)
    appended = list(te_df.columns)
    _kept = set(appended)

    # One recipe per appended (col, stat) output column. A std / skew / kurt recipe is structurally identical to
    # the mean recipe - same replay path - just a different per-category lookup table and global fallback.
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
