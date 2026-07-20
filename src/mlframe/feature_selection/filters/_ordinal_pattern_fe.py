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

import numpy as np

__all__ = ["ordinal_pattern_ids", "ordinal_pattern_lexicographic_rank"]


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
