"""Layer 34 (2026-05-31): COUNT / FREQUENCY encoding + CATEGORICAL x NUMERIC
interaction via OOF target-mean residual.

Sibling to ``_target_encoding_fe`` (Layer 33): same production-grade
discipline (recipe-based replay, no y at transform time, default OFF in
``MRMR``), targeting the next two categorical patterns that show up in
production tabular pipelines:

* **Count encoding** ``count_encode(X, cat_col)``: replace each row's
  category with the integer count of occurrences seen at fit time.
  Captures the "rare vs frequent identifier" signal (a user / sku /
  region seen 5 times is meaningfully different from one seen 5000
  times). No y reference at fit OR replay; cleanly leakage-free.

* **Frequency encoding** ``frequency_encode(X, cat_col)``: same lookup
  scaled by 1 / n_samples. Equivalent up to affine scaling for tree
  models but distinct for linear models that consume it raw.

* **Cat x Num interaction** ``cat_num_interaction(X, y, cat_col, num_col,
  n_folds, smoothing)``: per-row residual
  ``num_col[i] - mean(num_col | cat_col = X[cat_col][i], OOF)`` with
  Micci-Barreca shrinkage. "Deviation from group norm" is a strong
  production signal: a price that's 2 sigma above its category's mean
  predicts very differently from one that's at the mean. K-fold OOF
  protects against leakage exactly the same way the Layer 33 mean-of-y
  encoder does (each row's per-cell mean is computed from rows in folds
  other than its own).

The recipe payloads (``count_encoded``, ``frequency_encoded``,
``cat_num_residual``) are deterministic functions of X alone at replay:

* count_encoded: ``{lookup: dict[str, int], default: int = 0}``
* frequency_encoded: ``{lookup: dict[str, float], default: float = 0.0}``
* cat_num_residual: ``{lookup: dict[str, float], global_mean: float,
                       smoothing: float, num_col: str}``

Unseen categories at transform map to ``default`` (count / freq) or
``global_mean`` for residual subtraction (matches the Layer 33 unseen-
category contract: no NaN propagation, no KeyError).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._target_encoding_fe import _column_to_str, _smooth

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_count",
    "engineered_name_freq",
    "engineered_name_cat_num_residual",
    "count_encode_fit",
    "frequency_encode_fit",
    "cat_num_interaction_fit",
    "apply_count_encoding",
    "apply_frequency_encoding",
    "apply_cat_num_residual",
    "count_encode_with_recipes",
    "frequency_encode_with_recipes",
    "cat_num_interaction_with_recipes",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def engineered_name_count(col: str) -> str:
    """Stable engineered name for the count-encoded column."""
    return f"{col}__count"


def engineered_name_freq(col: str) -> str:
    """Stable engineered name for the frequency-encoded column."""
    return f"{col}__freq"


def engineered_name_cat_num_residual(cat_col: str, num_col: str) -> str:
    """Stable engineered name for the cat x num residual column.

    ``{num}__resid_by__{cat}`` reads as "deviation of num from its
    cat-conditional mean", matching the underlying signal semantics.
    """
    return f"{num_col}__resid_by__{cat_col}"


# ---------------------------------------------------------------------------
# Count / Frequency encoders -- pure X-only kernels (no y reference)
# ---------------------------------------------------------------------------


def count_encode_fit(
    X: pd.DataFrame,
    cat_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit count encoding for each column in ``cat_cols``.

    Returns
    -------
    enc_df : pd.DataFrame
        Shape (n, len(cat_cols)), columns named ``{col}__count``. Per-row
        integer count of occurrences in the fit set.
    recipes : dict
        ``{col: {"lookup": {category: int_count}, "default": 0}}``.
        Replay-only payload (no y reference).
    """
    if len(X) == 0:
        raise ValueError("count_encode_fit: X is empty")
    missing = [c for c in cat_cols if c not in X.columns]
    if missing:
        raise ValueError(f"count_encode_fit: columns missing from X: {missing}")
    encoded_cols: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    for col in cat_cols:
        cats = _column_to_str(X[col])
        unique_cats, inverse, counts = np.unique(
            cats, return_inverse=True, return_counts=True,
        )
        # Per-row count: inverse maps each row to its category code; counts
        # is the per-category count. counts[inverse] is one int per row.
        row_counts = counts[inverse].astype(np.int64, copy=False)
        encoded_cols[engineered_name_count(col)] = row_counts
        lookup = {str(unique_cats[c]): int(counts[c]) for c in range(unique_cats.shape[0])}
        recipes[col] = {"lookup": lookup, "default": 0}
    enc_df = pd.DataFrame(encoded_cols, index=X.index)
    return enc_df, recipes


def frequency_encode_fit(
    X: pd.DataFrame,
    cat_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit frequency encoding (count / n_samples) for each column in cat_cols.

    Returns
    -------
    enc_df : pd.DataFrame
        Shape (n, len(cat_cols)), columns named ``{col}__freq``.
    recipes : dict
        ``{col: {"lookup": {category: float_freq}, "default": 0.0}}``.
    """
    if len(X) == 0:
        raise ValueError("frequency_encode_fit: X is empty")
    missing = [c for c in cat_cols if c not in X.columns]
    if missing:
        raise ValueError(f"frequency_encode_fit: columns missing from X: {missing}")
    n = len(X)
    encoded_cols: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    for col in cat_cols:
        cats = _column_to_str(X[col])
        unique_cats, inverse, counts = np.unique(
            cats, return_inverse=True, return_counts=True,
        )
        freqs = counts.astype(np.float64) / float(n)
        row_freqs = freqs[inverse]
        encoded_cols[engineered_name_freq(col)] = row_freqs
        lookup = {str(unique_cats[c]): float(freqs[c]) for c in range(unique_cats.shape[0])}
        recipes[col] = {"lookup": lookup, "default": 0.0}
    enc_df = pd.DataFrame(encoded_cols, index=X.index)
    return enc_df, recipes


def apply_count_encoding(
    X_test: pd.DataFrame, col: str, recipe: dict,
) -> np.ndarray:
    """Replay count encoding from the stored lookup. Unseen categories map
    to ``recipe['default']`` (default 0). NaN values map to ``"__nan__"``
    which is itself a category in the lookup if seen at fit; otherwise
    falls back to default."""
    if "lookup" not in recipe:
        raise KeyError(f"apply_count_encoding: recipe for col {col!r} is missing 'lookup'.")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_count_encoding: X_test must be a DataFrame; " f"got {type(X_test).__name__}")
    cats = np.asarray(_column_to_str(X_test[col]))
    lookup: dict = recipe["lookup"]
    default: int = int(recipe.get("default", 0))
    if cats.size == 0:
        return np.empty(0, dtype=np.int64)
    # Resolve int(lookup.get(...)) once per DISTINCT category and gather by code.
    # pd.factorize is an O(n) HASHTABLE pass (no sort) -- measured ~6-8x faster
    # than both the per-row dict.get loop and an np.unique(sort)-based map on
    # object/string columns (np.unique sorts Python strings in interpreted code,
    # which is SLOWER than the dict loop). Bit-identical to the per-row loop
    # (same lookup+default per distinct key). _column_to_str maps NaN -> "__nan__"
    # (a real key), so pd.factorize never emits its -1 NaN sentinel here.
    codes, uniques = pd.factorize(cats)
    vals = np.array([int(lookup.get(u, default)) for u in uniques], dtype=np.int64)
    return vals[codes]


def apply_frequency_encoding(
    X_test: pd.DataFrame, col: str, recipe: dict,
) -> np.ndarray:
    """Replay frequency encoding from the stored lookup. Unseen categories
    map to ``recipe['default']`` (default 0.0)."""
    if "lookup" not in recipe:
        raise KeyError(f"apply_frequency_encoding: recipe for col {col!r} is missing 'lookup'.")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_frequency_encoding: X_test must be a DataFrame; " f"got {type(X_test).__name__}")
    cats = np.asarray(_column_to_str(X_test[col]))
    lookup: dict = recipe["lookup"]
    default: float = float(recipe.get("default", 0.0))
    if cats.size == 0:
        return np.empty(0, dtype=np.float64)
    # Vectorized replay: pd.factorize (O(n) hashtable, no sort) + one
    # float(lookup.get(...)) per distinct category, gathered by code. ~6-8x over
    # the per-row loop; bit-identical. See apply_count_encoding for why factorize
    # beats both an np.unique(sort) map on string columns and an njit gather.
    codes, uniques = pd.factorize(cats)
    vals = np.array([float(lookup.get(u, default)) for u in uniques], dtype=np.float64)
    return vals[codes]


# ---------------------------------------------------------------------------
# Categorical x numeric interaction: OOF target-mean residual
# ---------------------------------------------------------------------------
#
# The signal: ``num_col - E[num_col | cat_col]``. A row whose numeric
# value deviates from its category's typical numeric value carries
# information that neither raw column has alone -- e.g. a high price
# in a "luxury" category is normal; the same price in a "budget"
# category is anomalous and often predictive. This mirrors the
# Layer 33 OOF mean-of-y discipline but conditions on cat to compute a
# per-cell mean of num (not of y), then subtracts that mean from the row's
# own num value.
#
# Note re: ``y`` parameter -- y is NOT consumed for the residual
# expression; it appears only as the OOF FOLD KEY (each row is assigned
# a fold from a y-stratified split when available, else random). The
# recipe stores ONLY the full-data per-category mean of num, so
# transform() is a pure function of (X[cat], X[num]) -- no y dependency.


def cat_num_interaction_fit(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_col: str,
    num_col: str,
    *,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
) -> tuple[np.ndarray, dict]:
    """Fit OOF target-mean residual for the (cat_col, num_col) pair.

    Parameters
    ----------
    X : DataFrame
        Must contain both ``cat_col`` and ``num_col``.
    y : ndarray
        Used ONLY to derive a stratified fold assignment when the target is
        binary; otherwise random. Not consumed into the residual expression.
    cat_col, num_col : str
    n_folds : int, default 5
    smoothing : float, default 10.0
        Micci-Barreca shrinkage of the per-category mean of num toward the
        global mean of num. ``alpha = smoothing``: a category needs ~alpha
        rows before its raw per-cell mean dominates the prior.
    random_state : int, default 0

    Returns
    -------
    residual : ndarray, shape (n,)
        Per-row ``num_col[i] - smoothed_mean(num_col | cat_col = X[cat_col][i], OOF)``.
    recipe : dict
        ``{lookup: dict[str, float], global_mean: float, smoothing: float,
           num_col: str}``. Replay subtracts ``lookup.get(cat, global_mean)``
        from the row's num value -- no y at replay.

    Raises
    ------
    ValueError
        On invalid ``n_folds``, empty X, length mismatch, or num_col not
        numeric.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    if len(X) == 0:
        raise ValueError("cat_num_interaction_fit: X is empty")
    if len(y) != len(X):
        raise ValueError(f"cat_num_interaction_fit: len(y)={len(y)} != len(X)={len(X)}")
    for c in (cat_col, num_col):
        if c not in X.columns:
            raise ValueError(f"cat_num_interaction_fit: column {c!r} missing from X")
    num_vals = X[num_col].to_numpy()
    if not np.issubdtype(num_vals.dtype, np.number):
        raise ValueError(f"cat_num_interaction_fit: num_col {num_col!r} is not numeric " f"(dtype={num_vals.dtype!r})")
    num_vals = num_vals.astype(np.float64, copy=False)
    n = len(X)

    # NaN handling for the numeric column: rows with NaN num contribute
    # nothing to the per-cell mean (they're skipped in the sum/cnt
    # accumulators) AND their residual output stays 0.0 (no deviation
    # information available; the per-cell mean would have to be the
    # row's own num to make it 0 anyway). Mirrors the apply path.
    finite_mask = np.isfinite(num_vals)
    global_mean = float(num_vals[finite_mask].mean()) if finite_mask.any() else 0.0

    cats = _column_to_str(X[cat_col])
    unique_cats, inverse = np.unique(cats, return_inverse=True)
    n_cats = unique_cats.shape[0]

    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n) % int(n_folds)

    oof_pred = np.full(n, global_mean, dtype=np.float64)
    inv_arr = inverse.astype(np.int64, copy=False)
    for f in range(int(n_folds)):
        train_mask = (fold_ids != f) & finite_mask
        if not train_mask.any():
            continue
        # Per-category sum / count from rows in folds != f (and finite num).
        cell_sum = np.zeros(n_cats, dtype=np.float64)
        cell_cnt = np.zeros(n_cats, dtype=np.float64)
        # Vectorized via np.add.at: O(n_train) without an explicit Python loop.
        np.add.at(cell_sum, inv_arr[train_mask], num_vals[train_mask])
        np.add.at(cell_cnt, inv_arr[train_mask], 1.0)
        cell_means_fold = np.full(n_cats, global_mean, dtype=np.float64)
        nz = cell_cnt > 0.0
        if nz.any():
            raw = np.where(nz, cell_sum / np.maximum(cell_cnt, 1.0), global_mean)
            # Vectorized Micci-Barreca shrinkage.
            shrunk = (cell_cnt * raw + smoothing * global_mean) / (cell_cnt + smoothing)
            cell_means_fold = np.where(nz, shrunk, global_mean)
        test_mask = fold_ids == f
        oof_pred[test_mask] = cell_means_fold[inv_arr[test_mask]]

    # Residual: num - OOF per-cell mean. Rows with NaN num get residual 0
    # (no information available for them).
    residual = np.where(finite_mask, num_vals - oof_pred, 0.0).astype(np.float64, copy=False)

    # Full-data lookup for transform-time replay (smoothed per-category mean
    # of num computed on the entire training set).
    full_sum = np.zeros(n_cats, dtype=np.float64)
    full_cnt = np.zeros(n_cats, dtype=np.float64)
    np.add.at(full_sum, inv_arr[finite_mask], num_vals[finite_mask])
    np.add.at(full_cnt, inv_arr[finite_mask], 1.0)
    lookup: dict[str, float] = {}
    for c in range(n_cats):
        if full_cnt[c] > 0.0:
            raw_mean = full_sum[c] / full_cnt[c]
            lookup[str(unique_cats[c])] = _smooth(
                raw_mean, full_cnt[c], global_mean, smoothing,
            )
        else:
            lookup[str(unique_cats[c])] = global_mean

    recipe = {
        "lookup": lookup,
        "global_mean": global_mean,
        "smoothing": float(smoothing),
        "num_col": str(num_col),
    }
    return residual, recipe


def apply_cat_num_residual(
    X_test: pd.DataFrame, cat_col: str, num_col: str, recipe: dict,
) -> np.ndarray:
    """Replay cat x num residual: ``X[num] - lookup.get(X[cat], global_mean)``.

    Unseen categories use ``global_mean`` (subtracting the unconditional
    mean of num is the natural fallback). NaN num rows emit 0.0 residual.
    """
    for key in ("lookup", "global_mean"):
        if key not in recipe:
            raise KeyError(f"apply_cat_num_residual: recipe missing {key!r}")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_cat_num_residual: X_test must be a DataFrame; " f"got {type(X_test).__name__}")
    cats = _column_to_str(X_test[cat_col])
    num_vals = X_test[num_col].to_numpy()
    if not np.issubdtype(num_vals.dtype, np.number):
        raise TypeError(f"apply_cat_num_residual: num_col {num_col!r} must be numeric " f"(dtype={num_vals.dtype!r})")
    num_vals = num_vals.astype(np.float64, copy=False)
    lookup: dict = recipe["lookup"]
    global_mean = float(recipe["global_mean"])
    cats = np.asarray(cats)
    if cats.size == 0:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(num_vals)
    # Vectorized replay: pd.factorize (O(n) hashtable, no sort) resolves one float(lookup.get(...)) per DISTINCT
    # category, gathered by code -- bit-identical to the per-row loop (measured 9.36x @10M). Mirrors the count/freq
    # apply paths above; _column_to_str maps NaN -> "__nan__" so factorize never emits its -1 sentinel here.
    codes, uniques = pd.factorize(cats)
    cell = np.array([float(lookup.get(u, global_mean)) for u in uniques], dtype=np.float64)
    return np.where(finite, num_vals - cell[codes], 0.0).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# End-to-end wrappers for MRMR.fit auto-wiring (recipe-emitting variants)
# ---------------------------------------------------------------------------


def _gate_enc(enc_df, y, raw_X, mi_gate, mi_gate_top_k, reject_sink=None):
    """Tier-1 local MI floor over an ``enc_df`` of engineered columns. Returns
    the (possibly pruned) ``enc_df`` keeping only columns that clear the raw-
    baseline noise floor (top-K by MI). No-op when ``mi_gate`` is False."""
    if not mi_gate or enc_df is None or enc_df.empty:
        return enc_df
    from ._unified_fe_gate import local_mi_gate

    keep = local_mi_gate(enc_df, y, raw_X=raw_X, top_k=mi_gate_top_k, reject_sink=reject_sink)
    return enc_df[keep]


def count_encode_with_recipes(
    X: pd.DataFrame,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append ``{col}__count`` columns to X and emit one recipe per col.

    When ``mi_gate=True`` (and ``y`` supplied) the emitted columns are filtered
    by the Tier-1 local MI floor (Layer 91): drop columns whose ``MI(col; y)``
    is below the raw-baseline noise floor, keep top-``mi_gate_top_k``.
    """
    from .engineered_recipes import build_count_encoded_recipe

    if not cat_cols:
        return X.copy(), [], []
    cat_cols = [c for c in cat_cols if c in X.columns]
    if not cat_cols:
        return X.copy(), [], []
    enc_df, raw_recipes = count_encode_fit(X, cat_cols)
    if mi_gate and y is not None:
        enc_df = _gate_enc(enc_df, y, X, mi_gate, mi_gate_top_k, reject_sink=reject_sink)
        if enc_df.empty:
            return X.copy(), [], []
        kept = set(enc_df.columns)
        cat_cols = [c for c in cat_cols if engineered_name_count(c) in kept]
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [
        build_count_encoded_recipe(
            name=engineered_name_count(col),
            src_name=col,
            lookup=raw_recipes[col]["lookup"],
            default=raw_recipes[col]["default"],
        )
        for col in cat_cols
    ]
    return X_aug, appended, recipes


def frequency_encode_with_recipes(
    X: pd.DataFrame,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append ``{col}__freq`` columns to X and emit one recipe per col.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91).
    """
    from .engineered_recipes import build_frequency_encoded_recipe

    if not cat_cols:
        return X.copy(), [], []
    cat_cols = [c for c in cat_cols if c in X.columns]
    if not cat_cols:
        return X.copy(), [], []
    enc_df, raw_recipes = frequency_encode_fit(X, cat_cols)
    if mi_gate and y is not None:
        enc_df = _gate_enc(enc_df, y, X, mi_gate, mi_gate_top_k, reject_sink=reject_sink)
        if enc_df.empty:
            return X.copy(), [], []
        kept = set(enc_df.columns)
        cat_cols = [c for c in cat_cols if engineered_name_freq(c) in kept]
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [
        build_frequency_encoded_recipe(
            name=engineered_name_freq(col),
            src_name=col,
            lookup=raw_recipes[col]["lookup"],
            default=raw_recipes[col]["default"],
        )
        for col in cat_cols
    ]
    return X_aug, appended, recipes


def cat_num_interaction_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append ``{num}__resid_by__{cat}`` columns for the Cartesian product
    of ``cat_cols`` x ``num_cols`` (only valid combinations: cat in X,
    num in X and numeric).

    ``mi_gate=True`` applies the Tier-1 local MI floor (Layer 91) over the
    Cartesian residual pool (which is O(|cat| * |num|) and the most
    explosion-prone of the three L34 emitters)."""
    from .engineered_recipes import build_cat_num_residual_recipe

    if not cat_cols or not num_cols:
        return X.copy(), [], []
    cat_cols = [c for c in cat_cols if c in X.columns]
    # ``X[c]`` is a DataFrame (no ``.dtype``) when ``c`` is a duplicated column name; skip such ambiguous columns
    # (a single numeric dtype can't be determined) instead of raising AttributeError and losing the whole FE pass.
    num_cols = [c for c in num_cols if c in X.columns and getattr(X[c], "ndim", 2) == 1 and np.issubdtype(X[c].dtype, np.number)]
    if not cat_cols or not num_cols:
        return X.copy(), [], []
    appended: list[str] = []
    recipes: list = []
    new_cols: dict[str, np.ndarray] = {}
    for cat in cat_cols:
        for num in num_cols:
            name = engineered_name_cat_num_residual(cat, num)
            residual, raw_rec = cat_num_interaction_fit(
                X, y, cat, num,
                n_folds=n_folds,
                smoothing=smoothing,
                random_state=random_state,
            )
            new_cols[name] = residual
            appended.append(name)
            recipes.append(
                build_cat_num_residual_recipe(
                    name=name,
                    cat_name=cat,
                    num_name=num,
                    lookup=raw_rec["lookup"],
                    global_mean=raw_rec["global_mean"],
                    smoothing=raw_rec["smoothing"],
                )
            )
    new_df = pd.DataFrame(new_cols, index=X.index)
    if mi_gate and not new_df.empty:
        new_df = _gate_enc(new_df, y, X, mi_gate, mi_gate_top_k, reject_sink=reject_sink)
        if new_df.empty:
            return X.copy(), [], []
        kept = set(new_df.columns)
        appended = [n for n in appended if n in kept]
        recipes = [r for r in recipes if r.name in kept]
    X_aug = pd.concat([X, new_df], axis=1)
    return X_aug, appended, recipes
