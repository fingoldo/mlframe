"""Micro-bench: O(C*P) ``cols.index`` per-name resolution vs an O(C+P) ``name->index`` map.

In ``_fit_impl_core._fit_impl`` the post-categorize step resolved every target name and every categorical
feature name to its column index with ``[cols.index(col) for col in names]``. ``list.index`` is a linear scan,
so resolving P names against C columns is O(C*P). On a wide frame (thousands of columns, hundreds of
categoricals) that is a measurable, avoidable cost on the fit hot path.

The fix builds ``name_to_idx = {c: i for i, c in enumerate(cols)}`` once (O(C)) and resolves each name in O(1),
making the whole resolution O(C+P). The result is bit-identical: same indices, same order.

This bench measures both styles, warm, best-of-N, at a realistic wide-frame shape and reports the speedup.

Usage:

    python -m mlframe.feature_selection._benchmarks.bench_name_to_idx_lookup
"""

from __future__ import annotations

import time


def _old_lookup(cols, names):
    """Pre-fix: O(C*P) -- a linear ``list.index`` scan per name."""
    return [cols.index(c) for c in names]


def _new_lookup(cols, names):
    """Post-fix: O(C+P) -- one dict build, O(1) lookups."""
    name_to_idx = {c: i for i, c in enumerate(cols)}
    return [name_to_idx[c] for c in names]


def _best_of(fn, cols, names, repeats: int = 7) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(cols, names)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    # Realistic wide engineered frame: thousands of columns, a few hundred categoricals + targets to resolve.
    shapes = [
        (2_000, 200),
        (5_000, 500),
        (10_000, 1_000),
    ]
    print(f"{'C (cols)':>10} {'P (names)':>10} {'old (ms)':>12} {'new (ms)':>12} {'speedup':>10}")
    for n_cols, n_names in shapes:
        cols = [f"col_{i}" for i in range(n_cols)]
        # Resolve names spread across the column space (worst case for list.index is late names).
        step = max(1, n_cols // n_names)
        names = [cols[(i * step) % n_cols] for i in range(n_names)]

        # Warm both (build any caches / JIT-free here, just steady-state Python).
        assert _old_lookup(cols, names) == _new_lookup(cols, names), "lookups must be bit-identical"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input

        old_ms = _best_of(_old_lookup, cols, names) * 1e3
        new_ms = _best_of(_new_lookup, cols, names) * 1e3
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")
        print(f"{n_cols:>10} {n_names:>10} {old_ms:>12.4f} {new_ms:>12.4f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
