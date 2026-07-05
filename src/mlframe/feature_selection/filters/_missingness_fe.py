"""Layer 37 (2026-05-31): MISSING-VALUE-AWARE FE -- surface missingness as
predictive signal where it carries information (MNAR pattern).

In production tabular pipelines, the FACT that a value is missing is
often a stronger predictor than the imputed-then-modeled value:

* Credit scoring: ``credit_history is NULL`` -> applicant is thin-file,
  a strong predictor of risk.
* Healthcare: ``lab_panel is NULL`` -> patient never had the panel,
  predicts both prior diagnosis history AND the cost trajectory.
* Churn: ``last_login is NULL`` -> user never engaged after onboarding,
  near-deterministic churn signal.
* Marketing: ``income is NULL`` -> high-net-worth refuser cluster, often
  the most valuable segment.

MRMR's ``nan_strategy='separate_bin'`` (Layer 7 contract) handles MNAR
at the BINNING level inside the MI estimator. Layer 37 complements that
by EMITTING missingness signals as their OWN engineered columns the
downstream model can consume directly:

* ``missing_indicator_encode``: per-source ``is_missing__{col}`` binary
  column. Stateless (just an isna check at fit AND replay).
* ``missingness_count_encode``: per-row count of missing values across
  a fixed subset of columns. Captures the "this record has many missing
  fields" signal (e.g. a fragmentary survey response indicates churn
  intent).
* ``missingness_pattern_encode``: per-row integer label encoding the
  TOP-K most frequent missingness patterns from fit; unseen patterns at
  transform map to the "other" bucket. Lets a downstream linear model
  separate distinct MNAR clusters (e.g. "high-net-worth refuser" vs
  "thin-file young").

All three replays are pure functions of X (no y reference at fit OR
transform): recipes carry only column names and (for the pattern
encoder) the top-K signature -> label map.

extra layout:
* missing_indicator  : {}  (src_names=(col,) is all that's needed)
* missingness_count  : {cols: tuple[str, ...]}
* missingness_pattern: {cols: tuple[str, ...], pattern_to_label: dict,
                        other_label: int, top_k: int}
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit, prange

logger = logging.getLogger(__name__)


@njit(cache=True, parallel=True)
def _bitpack_rows_njit(arr: np.ndarray) -> np.ndarray:
    """Fused per-row bit-pack of an (n, k) bool/uint8 isna block into one int64 per row.

    Bit-identical to ``(arr.astype(int64) * (1 << arange(k))).sum(axis=1)`` but never materialises the two (n, k) int64 temporaries that broadcast-multiply
    + row-reduce allocate -- at 10M rows those temporaries (~80MB each per column) are the memory-bandwidth bottleneck, not the packing arithmetic.
    """
    n, k = arr.shape
    out = np.empty(n, dtype=np.int64)
    for i in prange(n):
        acc = 0
        for j in range(k):
            if arr[i, j]:
                acc |= 1 << j
        out[i] = acc
    return out

__all__ = [
    "engineered_name_missing_indicator",
    "engineered_name_missingness_count",
    "engineered_name_missingness_pattern",
    "auto_detect_missing_cols",
    "missing_indicator_fit",
    "missingness_count_fit",
    "missingness_pattern_fit",
    "apply_missing_indicator",
    "apply_missingness_count",
    "apply_missingness_pattern",
    "missing_indicator_with_recipes",
    "missingness_count_with_recipes",
    "missingness_pattern_with_recipes",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def engineered_name_missing_indicator(col: str) -> str:
    """Stable engineered name for the per-column missingness indicator."""
    return f"is_missing__{col}"


def engineered_name_missingness_count() -> str:
    """Stable engineered name for the per-row missingness count."""
    return "missingness_count"


def engineered_name_missingness_pattern() -> str:
    """Stable engineered name for the per-row missingness pattern label."""
    return "missingness_pattern"


# ---------------------------------------------------------------------------
# Auto-detect: pick columns whose NaN rate clears a minimum threshold.
# ---------------------------------------------------------------------------


def auto_detect_missing_cols(
    X: pd.DataFrame,
    *,
    min_nan_rate: float = 0.01,
    max_nan_rate: float = 0.99,
) -> list[str]:
    """Pick columns with non-trivial missingness.

    Heuristics:
      * NaN rate in ``[min_nan_rate, max_nan_rate]`` -- below the floor
        the indicator is constant-zero (no MI), above the ceiling the
        indicator is constant-one (also no MI).
      * Any dtype works (numeric / object / categorical / datetime); we
        test via ``isna()`` which respects pandas' missing-value contract
        for each dtype.

    Default ``min_nan_rate=0.01`` (1%): below this threshold the column
    is "essentially never missing" and the indicator carries effectively
    no information on a typical n in [1k, 100k]; the bound matches the
    Layer 33 ``min_card`` floor philosophy.
    """
    if not isinstance(X, pd.DataFrame):
        return []
    candidates: list[str] = []
    n = len(X)
    if n == 0:
        return []
    for col in X.columns:
        try:
            nan_rate = float(X[col].isna().sum()) / float(n)
        except Exception:
            continue
        if min_nan_rate <= nan_rate <= max_nan_rate:
            candidates.append(col)
    return candidates


def _count_row_nans(X: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Per-row count of NaNs across ``cols`` as int32.

    Bit-identical to ``X.loc[:, cols].isna().sum(axis=1)`` but avoids the pandas row-wise ``.sum(axis=1)`` reduction, which at 10M rows is ~10x slower than
    accumulating each column's boolean ``isna()`` mask directly into an int32 buffer (the 2-D bool block + row reduction is the bottleneck, not ``isna()``).
    """
    out = np.zeros(len(X), dtype=np.int32)
    for c in cols:
        out += X[c].isna().to_numpy()
    return out


# ---------------------------------------------------------------------------
# Per-column indicator: ``is_missing__{col}`` -- stateless (just isna)
# ---------------------------------------------------------------------------


def missing_indicator_fit(
    X: pd.DataFrame,
    cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit per-column missingness indicators.

    Returns
    -------
    enc_df : pd.DataFrame
        Shape (n, len(cols)), columns named ``is_missing__{col}`` with
        int8 values (0/1).
    recipes : dict
        ``{col: {}}`` -- no fit-time state; replay just re-runs ``isna()``.
    """
    if len(X) == 0:
        raise ValueError("missing_indicator_fit: X is empty")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"missing_indicator_fit: columns missing from X: {missing}")
    encoded_cols: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    for col in cols:
        encoded_cols[engineered_name_missing_indicator(col)] = X[col].isna().to_numpy().astype(np.int8, copy=False)
        recipes[col] = {}
    enc_df = pd.DataFrame(encoded_cols, index=X.index)
    return enc_df, recipes


def apply_missing_indicator(
    X_test: pd.DataFrame, col: str, recipe: dict,
) -> np.ndarray:
    """Replay per-column missingness indicator. No fit state needed; the
    recipe argument is kept for parity with other encoders."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_missing_indicator: X_test must be a DataFrame; " f"got {type(X_test).__name__}")
    if col not in X_test.columns:
        raise KeyError(f"apply_missing_indicator: column {col!r} missing from X_test")
    return X_test[col].isna().to_numpy().astype(np.int8, copy=False)


# ---------------------------------------------------------------------------
# Per-row count of missing fields across a fixed subset of columns.
# ---------------------------------------------------------------------------


def missingness_count_fit(
    X: pd.DataFrame,
    cols: Sequence[str],
) -> tuple[np.ndarray, dict]:
    """Fit per-row missingness count over ``cols``.

    Returns
    -------
    counts : ndarray, shape (n,), int32
    recipe : dict
        ``{cols: tuple[str, ...]}`` -- replay just re-counts ``isna()``
        across the same columns.
    """
    if len(X) == 0:
        raise ValueError("missingness_count_fit: X is empty")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"missingness_count_fit: columns missing from X: {missing}")
    if not cols:
        # Degenerate but allowed; returns all-zero column.
        counts = np.zeros(len(X), dtype=np.int32)
        return counts, {"cols": ()}
    counts = _count_row_nans(X, cols)
    return counts, {"cols": tuple(str(c) for c in cols)}


def apply_missingness_count(
    X_test: pd.DataFrame, recipe: dict,
) -> np.ndarray:
    """Replay per-row missingness count from the recipe's ``cols`` list.
    Missing columns at test time contribute 0 (not 1) -- the recipe
    contract is "count NaNs in the original column set", and a column
    that doesn't exist at test time is a schema bug, not an
    informative missing value."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_missingness_count: X_test must be a DataFrame; " f"got {type(X_test).__name__}")
    cols = tuple(recipe.get("cols", ()))
    if not cols:
        return np.zeros(len(X_test), dtype=np.int32)
    # Intersect with X_test.columns; absent columns count as 0 (safer
    # than KeyError, matches the "graceful schema-drift" contract).
    cols_present = [c for c in cols if c in X_test.columns]
    if not cols_present:
        return np.zeros(len(X_test), dtype=np.int32)
    return _count_row_nans(X_test, cols_present)


# ---------------------------------------------------------------------------
# Per-row top-K pattern label.
# ---------------------------------------------------------------------------


def _row_pattern_signature(isna_block: np.ndarray) -> np.ndarray:
    """Convert an (n, k) bool / int isna block into one int64 hash per row.

    Uses a deterministic packing: for k <= 63 we pack bits into an int64;
    for k > 63 we fall back to a Python-level tuple hash (slower but
    unambiguous). MRMR is typically called with small ``cols`` sets so
    the bitpack path covers the common case.
    """
    arr = np.asarray(isna_block, dtype=np.uint8)
    n, k = arr.shape
    if k == 0:
        return np.zeros(n, dtype=np.int64)
    if k <= 63:
        # Bit-pack: each column j contributes bit j. Fused njit prange avoids the two (n, k) int64 broadcast temporaries the numpy form allocates.
        return _bitpack_rows_njit(np.ascontiguousarray(arr))
    # Fallback: per-row tuple hash. Determinism guaranteed by sorting cols
    # caller-side before passing the block in.
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = hash(tuple(arr[i].tolist()))
    return out


def missingness_pattern_fit(
    X: pd.DataFrame,
    cols: Sequence[str],
    *,
    top_k: int = 5,
) -> tuple[np.ndarray, dict]:
    """Fit per-row missingness-pattern label.

    Each row's missingness pattern across ``cols`` is hashed; the
    ``top_k`` most frequent patterns get integer labels ``0..top_k-1``;
    all rare patterns map to the "other" bucket label ``top_k``.

    Returns
    -------
    labels : ndarray, shape (n,), int32
    recipe : dict
        ``{cols: tuple[str, ...], pattern_to_label: dict[int, int],
           other_label: int, top_k: int}``
    """
    if len(X) == 0:
        raise ValueError("missingness_pattern_fit: X is empty")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"missingness_pattern_fit: columns missing from X: {missing}")
    cols_t = tuple(str(c) for c in cols)
    n = len(X)
    if not cols_t:
        labels = np.zeros(n, dtype=np.int32)
        return labels, {
            "cols": (),
            "pattern_to_label": {},
            "other_label": 0,
            "top_k": int(top_k),
        }
    sub = X.loc[:, list(cols_t)]
    isna_block = sub.isna().to_numpy()
    sigs = _row_pattern_signature(isna_block)
    # Frequency-rank patterns at fit time. ``return_counts`` lets us pick
    # the top-K without a Counter pass.
    uniq, counts = np.unique(sigs, return_counts=True)
    # Sort descending by count, ties broken by signature value for
    # determinism across platforms.
    order = np.lexsort((uniq, -counts))
    top_signatures = uniq[order][: int(top_k)]
    pattern_to_label: dict[int, int] = {int(sig): i for i, sig in enumerate(top_signatures)}
    other_label = int(top_k)
    # Replay-time label assignment vectorisation: build a mapping array
    # for fast lookup at fit-time labelling.
    labels = np.full(n, other_label, dtype=np.int32)
    for sig_val, lbl in pattern_to_label.items():
        labels[sigs == sig_val] = lbl
    return labels, {
        "cols": cols_t,
        "pattern_to_label": pattern_to_label,
        "other_label": other_label,
        "top_k": int(top_k),
    }


def apply_missingness_pattern(
    X_test: pd.DataFrame, recipe: dict,
) -> np.ndarray:
    """Replay per-row pattern label. Unseen patterns -> ``other_label``."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_missingness_pattern: X_test must be a DataFrame; " f"got {type(X_test).__name__}")
    cols = tuple(recipe.get("cols", ()))
    other_label = int(recipe.get("other_label", 0))
    pattern_to_label = dict(recipe.get("pattern_to_label", {}))
    n = len(X_test)
    if not cols:
        return np.full(n, other_label, dtype=np.int32)
    # If some fit-time cols are missing at test time, treat their values
    # as not-missing (False) in the signature -- the same "graceful
    # schema-drift" contract used by missingness_count. The pattern
    # then resolves to "other" if it never appeared at fit.
    blocks = []
    for c in cols:
        if c in X_test.columns:
            blocks.append(X_test[c].isna().to_numpy().astype(np.uint8))
        else:
            blocks.append(np.zeros(n, dtype=np.uint8))
    isna_block = np.column_stack(blocks)
    sigs = _row_pattern_signature(isna_block)
    labels = np.full(n, other_label, dtype=np.int32)
    # pattern_to_label keys may have been coerced to str during pickle
    # round-trip; coerce both sides to int for a robust lookup.
    pattern_to_label_int = {int(k): int(v) for k, v in pattern_to_label.items()}
    if not pattern_to_label_int:
        return labels
    # Vectorised mapping via sorted-key searchsorted; avoids a per-row Python loop that at 10M rows dominates apply wall.
    keys = np.fromiter(pattern_to_label_int.keys(), dtype=np.int64, count=len(pattern_to_label_int))
    vals = np.fromiter(pattern_to_label_int.values(), dtype=np.int32, count=len(pattern_to_label_int))
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    vals_sorted = vals[order]
    pos = np.searchsorted(keys_sorted, sigs)
    pos = np.clip(pos, 0, len(keys_sorted) - 1)
    hit = keys_sorted[pos] == sigs
    labels[hit] = vals_sorted[pos[hit]]
    return labels


# ---------------------------------------------------------------------------
# End-to-end wrappers (recipe-emitting) for MRMR.fit auto-wiring.
# ---------------------------------------------------------------------------


def missing_indicator_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    raw_X: Optional[pd.DataFrame] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append ``is_missing__{col}`` columns to X and emit one recipe per col.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91):
    per-source indicator columns are the explosion-prone L37 emitter (one
    column per source), so the floor drops indicators that carry no signal on y
    (the common MAR case) and keeps top-``mi_gate_top_k`` informative ones.

    ``raw_X`` anchors that noise floor on the RAW input columns. When ``X`` has
    already had earlier-stage engineered columns appended (e.g. the adaptive
    Fourier basis), their plug-in-MI-inflated values would push the floor far
    above a genuine MNAR indicator's MI and silently drop it; pass the pre-FE
    raw frame here so the floor reflects the true raw-MI distribution. Defaults
    to ``X`` when not supplied (back-compat).
    """
    from .engineered_recipes import build_missing_indicator_recipe

    if not cols:
        return X.copy(), [], []
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return X.copy(), [], []
    enc_df, _raw_recipes = missing_indicator_fit(X, cols)
    if mi_gate and y is not None and not enc_df.empty:
        from ._unified_fe_gate import local_mi_gate

        _floor_ref = raw_X if isinstance(raw_X, pd.DataFrame) and raw_X.shape[1] else X
        keep = set(local_mi_gate(enc_df, y, raw_X=_floor_ref, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X.copy(), [], []
        cols = [c for c in cols if engineered_name_missing_indicator(c) in keep]
        enc_df = enc_df[[engineered_name_missing_indicator(c) for c in cols]]
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [
        build_missing_indicator_recipe(
            name=engineered_name_missing_indicator(col),
            src_name=col,
        )
        for col in cols
    ]
    return X_aug, appended, recipes


def missingness_count_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
):
    """Append one ``missingness_count`` column to X and emit a single
    recipe carrying the source-column subset."""
    from .engineered_recipes import build_missingness_count_recipe

    if not cols:
        return X.copy(), [], []
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return X.copy(), [], []
    counts, raw_recipe = missingness_count_fit(X, cols)
    name = engineered_name_missingness_count()
    X_aug = X.copy()
    X_aug[name] = counts
    recipe = build_missingness_count_recipe(
        name=name, cols=tuple(raw_recipe["cols"]),
    )
    return X_aug, [name], [recipe]


# ---------------------------------------------------------------------------
# Recipe-apply adapters: thin wrappers consumed by ``engineered_recipes.
# apply_recipe`` via lazy import (the recipe dispatcher dispatches on
# ``recipe.kind`` and forwards to one of the helpers below).
# ---------------------------------------------------------------------------


def _apply_missing_indicator_recipe(recipe, X) -> np.ndarray:
    if len(recipe.src_names) != 1:
        raise ValueError(f"missing_indicator recipe '{recipe.name}' must have exactly 1 " f"src_name; got {len(recipe.src_names)}")
    src = recipe.src_names[0]
    # The apply helper accepts a DataFrame; coerce other X shapes here so
    # the dispatcher's contract (pandas / polars / structured) matches the
    # other kinds in engineered_recipes.
    if isinstance(X, pd.DataFrame):
        return apply_missing_indicator(X, src, dict(recipe.extra))
    # Polars / structured ndarray fallback: build a one-column pandas view.
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return apply_missing_indicator(
                pd.DataFrame({src: X[src].to_numpy()}), src, dict(recipe.extra),
            )
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return apply_missing_indicator(
            pd.DataFrame({src: X[src]}), src, dict(recipe.extra),
        )
    raise TypeError(f"missing_indicator recipe '{recipe.name}': cannot extract column " f"{src!r} from X of type {type(X).__name__}")


def _coerce_X_to_pandas_with_cols(X, cols, recipe_name) -> pd.DataFrame:
    """Shared X coercion helper for count / pattern replay."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            data = {}
            for c in cols:
                if c in X.columns:
                    data[c] = X[c].to_numpy()
            return pd.DataFrame(data)
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        data = {}
        for c in cols:
            if c in X.dtype.names:
                data[c] = X[c]
        return pd.DataFrame(data)
    raise TypeError(f"recipe '{recipe_name}': cannot extract columns from X of type " f"{type(X).__name__}")


def _apply_missingness_count_recipe(recipe, X) -> np.ndarray:
    cols = tuple(recipe.extra.get("cols", ()))
    X_view = _coerce_X_to_pandas_with_cols(X, cols, recipe.name)
    return apply_missingness_count(X_view, {"cols": cols})


def _apply_missingness_pattern_recipe(recipe, X) -> np.ndarray:
    cols = tuple(recipe.extra.get("cols", ()))
    X_view = _coerce_X_to_pandas_with_cols(X, cols, recipe.name)
    return apply_missingness_pattern(
        X_view,
        {
            "cols": cols,
            "pattern_to_label": dict(recipe.extra.get("pattern_to_label", {})),
            "other_label": int(recipe.extra.get("other_label", 0)),
            "top_k": int(recipe.extra.get("top_k", 0)),
        },
    )


def missingness_pattern_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    top_k: int = 5,
):
    """Append one ``missingness_pattern`` column to X and emit a single
    recipe carrying the top-K pattern signatures."""
    from .engineered_recipes import build_missingness_pattern_recipe

    if not cols:
        return X.copy(), [], []
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return X.copy(), [], []
    labels, raw_recipe = missingness_pattern_fit(X, cols, top_k=top_k)
    name = engineered_name_missingness_pattern()
    X_aug = X.copy()
    X_aug[name] = labels
    recipe = build_missingness_pattern_recipe(
        name=name,
        cols=tuple(raw_recipe["cols"]),
        pattern_to_label=dict(raw_recipe["pattern_to_label"]),
        other_label=int(raw_recipe["other_label"]),
        top_k=int(raw_recipe["top_k"]),
    )
    return X_aug, [name], [recipe]
