"""Row-wise ordinal-pattern (Bandt-Pompe permutation) encoding (mrmr_audit_2026-07-20
fe_expansion.md "Row-wise ordinal-pattern (Bandt-Pompe permutation) encoding").

Bandt & Pompe (2002, "Permutation Entropy: A Natural Complexity Measure for Time Series"): for a
chosen K-tuple of columns ``(x_i1, ..., x_iK)``, encode each row by the FULL RELATIVE ORDER
(permutation) of the K values -- i.e. which of the ``K!`` orderings the row realizes.

Why this catches a shape the catalog misses: the existing row-argmax operator
(``_conditional_gate_fe.apply_row_argmax``) only reports WHICH column is the row maximum -- for
K=3 that collapses all 6 possible total orderings into 3 buckets, discarding the second-vs-third
order entirely. Concrete scenario: ``y = 1{x1 > x2 > x3}`` (exactly one of the 6 orderings is
positive) -- ``argmax(x1,x2,x3)`` only tells you x1 is the max in 2 of those 6 orderings
(x1>x2>x3 AND x1>x3>x2) and cannot distinguish the target-positive one from the target-negative
one; only the full permutation id resolves it. This is operator #3 in the argmax/conditional-gate
family (generalizing #1, argmax, to the full ranking).

This module computes only the permutation-id CATEGORICAL itself (not the downstream target
encoding) -- the caller feeds the resulting low-cardinality categorical through the existing
K-fold target-encoding / count-encoding machinery (``_cat_target_encoding_and_weighted.py``)
exactly like any other synthetic categorical column, per the audit's own sketch.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = [
    "ordinal_pattern_ids",
    "ordinal_pattern_lexicographic_rank",
    "engineered_name_ordinal_pattern",
    "generate_ordinal_pattern_te_features",
    "apply_ordinal_pattern_te",
    "build_ordinal_pattern_te_recipe",
    "hybrid_ordinal_pattern_te_fe",
]


def ordinal_pattern_lexicographic_rank(perm: tuple) -> int:
    """Lexicographic rank (0-indexed) of a permutation of ``range(len(perm))`` among all ``K!``
    permutations of that size -- the base-``K!`` integer id Bandt-Pompe patterns are conventionally
    numbered by. E.g. for K=3, ``(0,1,2)`` -> 0 (the identity, lexicographically first) and
    ``(2,1,0)`` -> 5 (lexicographically last)."""
    k = len(perm)
    available = list(range(k))
    rank = 0
    fact = [1] * (k + 1)
    for i in range(1, k + 1):
        fact[i] = fact[i - 1] * i
    for i, p in enumerate(perm):
        pos = available.index(p)
        rank += pos * fact[k - 1 - i]
        available.pop(pos)
    return rank


def ordinal_pattern_ids(X_cols: np.ndarray, *, tie_policy: str = "nan") -> np.ndarray:
    """Row-wise Bandt-Pompe permutation id for a ``(n, K)`` block of columns.

    For each row, ``np.argsort(X_cols[row], kind="stable")`` gives the permutation of column
    indices that sorts the row's values ascending; this permutation's lexicographic rank in
    ``0 .. K!-1`` is computed via a vectorized Lehmer-code reduction (the same rank definition
    :func:`ordinal_pattern_lexicographic_rank` computes per-permutation, but applied to all rows
    at once via a loop over ``K``, not over ``n``).

    Parameters
    ----------
    X_cols : (n, K) array
        The K candidate columns for this ordinal-pattern tuple (``K`` matching the existing
        triplet/quadruplet arity cap, typically 3-5).
    tie_policy : {"nan", "ignore"}
        ``"nan"``: a row with any exactly-tied values among its K columns gets ``np.nan`` (the
        ordering is not well-defined for that row -- honest missingness, matching the row-argmax
        operator's own NaN-propagation-on-ambiguity convention). ``"ignore"``: ties are broken by
        ``np.argsort``'s own stable (first-occurrence) rule, silently picking one of the tied
        orderings -- only safe when the caller has already verified ties are rare/irrelevant.

    Returns
    -------
    (n,) float64 array of permutation ids in ``0 .. K!-1`` (NaN for tied rows under
    ``tie_policy="nan"``).
    """
    if tie_policy not in ("nan", "ignore"):
        raise ValueError(f"ordinal_pattern_ids: tie_policy must be 'nan' or 'ignore', got {tie_policy!r}")
    X_cols = np.asarray(X_cols, dtype=np.float64)
    if X_cols.ndim != 2:
        raise ValueError(f"ordinal_pattern_ids: X_cols must be 2-D (n, K); got shape {X_cols.shape}")
    n, k = X_cols.shape
    if k < 2:
        raise ValueError(f"ordinal_pattern_ids: K must be >= 2; got K={k}")

    # Vectorized Lehmer-code rank (replaces an earlier per-row Python dict-lookup loop, which
    # cProfiled as the dominant cost at n=100k -- see bench_ordinal_pattern_cprofile.py). Loops only
    # over K (small, 3-5), never over n: for each position i, count how many of the REMAINING
    # positions (i+1..K-1) hold a smaller column index than position i's -- exactly the Lehmer code
    # digit -- then combine via the standard factorial-number-system weights (bit-identical to the
    # per-row dict lookup, since both compute the same lexicographic rank definition).
    order = np.argsort(X_cols, axis=1, kind="stable")
    fact = [1] * (k + 1)
    for i in range(1, k + 1):
        fact[i] = fact[i - 1] * i
    out = np.zeros(n, dtype=np.float64)
    for i in range(k - 1):
        less_count = np.sum(order[:, i + 1 :] < order[:, i : i + 1], axis=1)
        out += less_count * fact[k - 1 - i]

    if tie_policy == "nan":
        sorted_vals = np.take_along_axis(X_cols, order, axis=1)
        has_tie = np.any(np.diff(sorted_vals, axis=1) == 0.0, axis=1)
        out[has_tie] = np.nan

    nan_rows = ~np.isfinite(X_cols).all(axis=1)
    if nan_rows.any():
        out[nan_rows] = np.nan
    return out


def engineered_name_ordinal_pattern(cols: Sequence[str]) -> str:
    """Deterministic engineered-column name for an ordinal-pattern K-tuple."""
    return "opat__" + "_".join(str(c) for c in cols)


def _fit_perm_id_te_lookup(perm_id: np.ndarray, y: np.ndarray, *, smoothing: float) -> tuple[dict, float]:
    """Micci-Barreca-shrunk target-encoding lookup for an integer-coded categorical (the perm_id),
    mirroring ``_target_encoding_fe._smooth`` exactly so this fused recipe's replay-time arithmetic
    matches the codebase's one canonical K-fold-TE shrinkage formula."""
    from ._target_encoding_fe import _smooth

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    global_mean = float(np.nanmean(y_arr))
    finite = np.isfinite(perm_id)
    lookup: dict = {}
    for pid in np.unique(perm_id[finite]):
        mask = finite & (perm_id == pid)
        cnt = float(mask.sum())
        raw_mean = float(y_arr[mask].mean())
        lookup[int(pid)] = _smooth(raw_mean, cnt, global_mean, smoothing)
    return lookup, global_mean


def generate_ordinal_pattern_te_features(
    X: "pd.DataFrame",
    col_tuples: Sequence[Sequence[str]],
    y: np.ndarray,
    *,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
) -> "tuple[pd.DataFrame, dict[str, dict]]":
    """For every K-tuple of columns in ``col_tuples``, compute the row-wise ordinal-pattern id and
    K-fold OOF target-encode it -- a SINGLE fused feature per tuple (the intermediate perm_id
    categorical is never exposed as its own column / recipe, avoiding a 2-deep nested-recipe replay
    the codebase's 1-deep replay convention cannot order at transform() time).

    Returns ``(enc_df, raw_recipes)``; each recipe payload stores the column tuple + the full-data
    TE lookup (``{perm_id: te_value}``) + ``global_mean`` + ``smoothing`` -- replay recomputes
    perm_id fresh from the raw source columns (a deterministic pure function) and looks up its TE
    value, so the payload never captures ``y``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_ordinal_pattern_te_features: X must be a pandas DataFrame; got {type(X).__name__}")
    if len(X) == 0:
        raise ValueError("generate_ordinal_pattern_te_features: X is empty")
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if y_arr.shape[0] != len(X):
        raise ValueError(f"generate_ordinal_pattern_te_features: len(y)={y_arr.shape[0]} != len(X)={len(X)}")

    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    rng = np.random.default_rng(random_state)
    for tup in col_tuples:
        tup = [c for c in tup if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        if len(tup) < 2:
            continue
        X_cols = X[tup].to_numpy(dtype=np.float64)
        perm_id = ordinal_pattern_ids(X_cols)
        finite = np.isfinite(perm_id)
        if finite.sum() < 2 or float(np.nanstd(perm_id[finite])) <= 1e-12:
            continue  # degenerate: all rows tied, or a single realized pattern -- no information

        # OOF fold assignment (same construction as kfold_target_encode_fit): each row's TE value is
        # looked up from a lookup fit on the OTHER folds only, so the emitted TRAIN column is leak-safe.
        n = len(X)
        perm = rng.permutation(n)
        fold_ids = np.empty(n, dtype=np.int64)
        fold_ids[perm] = np.arange(n) % int(n_folds)
        vals = np.full(n, np.nan, dtype=np.float64)
        for f in range(int(n_folds)):
            train_mask = fold_ids != f
            test_mask = fold_ids == f
            if not train_mask.any() or not test_mask.any():
                continue
            fold_lookup, fold_global = _fit_perm_id_te_lookup(perm_id[train_mask], y_arr[train_mask], smoothing=smoothing)
            test_pid = perm_id[test_mask]
            test_vals = np.array([fold_lookup.get(int(p), fold_global) if np.isfinite(p) else np.nan for p in test_pid], dtype=np.float64)
            vals[test_mask] = test_vals

        name = engineered_name_ordinal_pattern(tup)
        encoded[name] = vals
        full_lookup, full_global = _fit_perm_id_te_lookup(perm_id, y_arr, smoothing=smoothing)
        raw_recipes[name] = {"cols": tuple(tup), "lookup": full_lookup, "global_mean": full_global, "smoothing": float(smoothing)}

    return pd.DataFrame(encoded, index=X.index), raw_recipes


def apply_ordinal_pattern_te(X_test: "pd.DataFrame", recipe: dict) -> np.ndarray:
    """Replay one ordinal-pattern-TE column: recompute the perm_id fresh from the raw source
    columns, then look up its frozen TE value (categories unseen at fit time map to
    ``global_mean``). Reads only X (the lookup itself was built from y at fit time, but the
    recipe's own stored table is fit-time-frozen and carries no live y -> transform() is leak-safe
    the same way every K-fold-TE recipe in this codebase is)."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_ordinal_pattern_te: X_test must be a DataFrame; got {type(X_test).__name__}")
    cols = list(recipe["cols"])
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise KeyError(f"apply_ordinal_pattern_te: missing column(s) {missing} from X_test")
    lookup = recipe["lookup"]
    global_mean = float(recipe["global_mean"])
    X_cols = X_test[cols].to_numpy(dtype=np.float64)
    perm_id = ordinal_pattern_ids(X_cols)
    out = np.full(perm_id.shape[0], global_mean, dtype=np.float64)
    finite = np.isfinite(perm_id)
    out[finite] = np.array([lookup.get(int(p), global_mean) for p in perm_id[finite]], dtype=np.float64)
    out[~finite] = np.nan
    return out


def build_ordinal_pattern_te_recipe(*, name: str, cols: Sequence[str], lookup: dict, global_mean: float, smoothing: float) -> "EngineeredRecipe":
    """Frozen recipe for one ordinal-pattern-TE column. Stores the source column tuple + the
    frozen TE lookup; replay reads only X (recomputes perm_id from the raw columns), so
    transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="ordinal_pattern_te",
        src_names=tuple(str(c) for c in cols),
        extra={
            "cols": tuple(str(c) for c in cols),
            "lookup": dict(lookup),
            "global_mean": float(global_mean),
            "smoothing": float(smoothing),
        },
    )


def _apply_ordinal_pattern_te_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    cols = list(recipe.extra["cols"])
    if isinstance(X, pd.DataFrame):
        X_view = X
    else:
        try:
            import polars as _pl

            if isinstance(X, _pl.DataFrame):
                X_view = pd.DataFrame({c: X[c].to_numpy() for c in cols})
            else:
                X_view = pd.DataFrame({c: np.asarray(X[c]) for c in cols})
        except ImportError:
            X_view = pd.DataFrame({c: np.asarray(X[c]) for c in cols})
    return apply_ordinal_pattern_te(X_view, recipe.extra)


def hybrid_ordinal_pattern_te_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    k: int = 3,
    max_cols_for_tuples: int = 5,
    n_folds: int = 5,
    smoothing: float = 10.0,
    top_k: int = 5,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    random_state: int = 0,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """End-to-end ordinal-pattern FE: bound the column pool by top raw-MI (mirrors the existing
    triplet/quadruplet arity-cap cost guard), enumerate all C(max_cols_for_tuples, k) tuples among
    them, K-fold-TE-encode each tuple's ordinal pattern, MI-gate against the raw baseline, keep top
    ``top_k``.

    Returns ``(X_aug, appended, recipes, scores)``. ``y`` is consumed only by the tuple-ranking +
    OOF encoding + MI gate; recipes carry a frozen (fit-time) lookup table, not ``y`` itself.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_ordinal_pattern_te_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [c for c in num_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) < int(k):
        return X.copy(), [], [], pd.DataFrame()

    if y is not None:
        from ._extra_fe_families import _top_mi_num_cols  # lazy: break parent<->sibling cycle

        num_cols = _top_mi_num_cols(X, num_cols, y, max_cols_for_tuples)
    else:
        num_cols = list(num_cols)[: int(max_cols_for_tuples)]
    if len(num_cols) < int(k):
        return X.copy(), [], [], pd.DataFrame()

    col_tuples = list(combinations(num_cols, int(k)))
    enc_df, raw_recipes = generate_ordinal_pattern_te_features(X, col_tuples, y, n_folds=n_folds, smoothing=smoothing, random_state=random_state)
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        _gate_top_k = int(mi_gate_top_k) if mi_gate_top_k else int(top_k)
        winners = local_mi_gate(enc_df, y, raw_X=X, top_k=_gate_top_k, reject_sink=reject_sink)
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_ordinal_pattern_te_recipe(
            name=name,
            cols=raw_recipes[name]["cols"],
            lookup=raw_recipes[name]["lookup"],
            global_mean=raw_recipes[name]["global_mean"],
            smoothing=raw_recipes[name]["smoothing"],
        )
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df
