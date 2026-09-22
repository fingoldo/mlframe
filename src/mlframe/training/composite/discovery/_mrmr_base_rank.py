"""Min-redundancy-max-relevance (MRMR) ordering of composite base candidates.

``_auto_base`` ranks base candidates by pure relevance MI(base, y) and takes
the top-K. When several strong candidates are mutually redundant (``y_prev``,
``y_prev_smoothed``, ``y_lag2`` all near-collinear) that fills the shortlist
with near-duplicates -- wasting discovery budget and inflating ensemble
correlation. MRMR greedily prefers candidates that are relevant to y AND
non-redundant with the already-picked bases:

    score(c) = relevance(c) - beta * mean_{p in picked} redundancy(c, p)

This is a pure, kernel-agnostic helper: the caller supplies the relevance
vector and the redundancy source (an NxN matrix or a callable) so the MI
kernels already used by ``_auto_base`` are reused, never duplicated here.

cProfile: greedy selection is O(k * n_candidates) lookups on an already-built
redundancy source (each lookup O(1) for a matrix, or one memoised MI call for
the callable). At composite scales (n_candidates <= a few dozen, k <= ~10) the
whole loop is sub-millisecond and dwarfed by the MI-matrix build in the caller
-- no actionable speedup, so no dispatcher/ladder is warranted.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def mrmr_rank_bases(
    candidates: Sequence[Any],
    relevance: Sequence[float],
    redundancy_fn_or_matrix: Callable[[int, int], float] | np.ndarray | Sequence[Sequence[float]],
    k: int,
    *,
    beta: float = 1.0,
) -> list[Any]:
    """Greedy MRMR ordering of ``candidates`` (most-relevant-and-diverse first).

    ``relevance[i]`` is MI(candidates[i], y). ``redundancy_fn_or_matrix`` gives
    MI(candidates[i], candidates[j]) either as an indexable NxN matrix or as a
    callable ``fn(i, j) -> float`` over candidate indices. The first pick is the
    max-relevance candidate; each subsequent pick maximises
    ``relevance[c] - beta * mean redundancy(c, picked)``. ``beta=0`` reduces to
    pure relevance ordering. Returns at most ``k`` candidates (fewer if the pool
    is smaller); ``k <= 0`` returns ``[]``.

    Both terms are rescaled to the pool's own maximum before they are combined.
    The two are different mutual-information quantities: relevance is feature-to-target,
    redundancy is feature-to-feature, and the second is routinely far larger,
    especially against a low-cardinality target. Subtracting them raw at
    ``beta=1`` let the redundancy term dominate by its scale alone, so the order
    was driven almost entirely by diversity and the genuinely most relevant
    candidates could be pushed down the shortlist - the opposite of trading a
    little relevance for diversity. Rescaled, ``beta`` means what it reads as:
    the weight of diversity against relevance, both in [0, 1].
    """
    n = len(candidates)
    if n == 0 or k <= 0:
        return []
    rel = np.asarray(relevance, dtype=np.float64)
    if rel.shape[0] != n:
        raise ValueError(f"relevance length {rel.shape[0]} != n_candidates {n}")

    matrix = None
    if not callable(redundancy_fn_or_matrix):
        matrix = np.asarray(redundancy_fn_or_matrix, dtype=np.float64)
        if matrix.shape != (n, n):
            raise ValueError(f"redundancy matrix shape {matrix.shape} != ({n}, {n})")

    def _red(i: int, j: int) -> float:
        """Redundancy between candidates ``i`` and ``j``: a matrix lookup when ``redundancy_fn_or_matrix`` is a precomputed matrix, else the pairwise callable evaluated on demand."""
        if matrix is not None:
            return float(matrix[i, j])
        assert callable(redundancy_fn_or_matrix)
        return float(redundancy_fn_or_matrix(i, j))

    target_k = min(k, n)
    remaining = list(range(n))
    # First pick = pure max relevance (no redundancy term yet); ties broken by
    # original candidate order via a stable argmax over the remaining pool.
    first = max(remaining, key=lambda i: (rel[i], -i))
    picked = [first]
    remaining.remove(first)
    # Running redundancy sum per remaining candidate vs the picked set, so each
    # step is O(remaining) instead of re-summing over all picked candidates.
    red_sum = {i: _red(i, first) for i in remaining}
    # Put both terms on one scale. The relevance scale is known up front; the redundancy scale comes from the first pick's row, which the
    # line above has already evaluated, so this costs nothing extra even when redundancy is a callable.
    _rel_scale = float(np.max(np.abs(rel))) if rel.size else 0.0
    _rel_scale = _rel_scale if _rel_scale > 0.0 else 1.0
    _red_scale = max((abs(v) for v in red_sum.values()), default=0.0)
    _red_scale = _red_scale if _red_scale > 0.0 else 1.0
    while len(picked) < target_k and remaining:
        n_picked = len(picked)
        nxt = max(
            remaining,
            key=lambda i: (rel[i] / _rel_scale - beta * (red_sum[i] / n_picked) / _red_scale, -i),
        )
        picked.append(nxt)
        remaining.remove(nxt)
        for i in remaining:
            red_sum[i] += _red(i, nxt)
    return [candidates[i] for i in picked]
