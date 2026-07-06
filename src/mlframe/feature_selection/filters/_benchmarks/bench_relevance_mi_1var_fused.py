"""A/B bench + bit-identity check for the fused single-var relevance-MI fast path in mi_direct.

The analytic-null branch of ``permutation.py:mi_direct`` (engaged by default at n >= 25k, raw MI)
computed the relevance I(X_j; Y) via ``merge_vars(x)`` (an O(n) accumulate pass that builds the
length-n ``classes_x`` relabel array + a possible O(n) lookup-remap pass) followed by
``compute_mi_from_classes`` (another O(n) joint-histogram pass) -- and then DISCARDED ``classes_x``,
using only the MI scalar + occupied-x-bin count. ``_relevance_mi_1var_fused`` fuses those into ONE
O(n) pass for the single-variable relevance x (every MRMR/FE caller passes x=(var,) / x=[0]).

Run:
    python -m mlframe.feature_selection.filters._benchmarks.bench_relevance_mi_1var_fused

Reports paired process_time A/B (OLD merge_vars+compute vs NEW fused) at wellbore-like shapes and
asserts bit-identical MI + occupied-bin count.
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.info_theory import merge_vars, compute_mi_from_classes
from mlframe.feature_selection.filters.permutation import _relevance_mi_1var_fused


def _legacy(factors_data, ix, factors_nbins, classes_y, freqs_y, dtype=np.int32):
    ax_classes, ax_freqs, _ = merge_vars(
        factors_data=factors_data, vars_indices=np.array([ix], dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    mi = compute_mi_from_classes(ax_classes, ax_freqs, classes_y, freqs_y, dtype=dtype)
    return mi, int(ax_freqs.shape[0])


def _make(n, nb_x, nb_y, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nb_x, size=n).astype(np.int32)
    # y partly correlated with x so MI is nonzero and realistic.
    y = ((x + rng.integers(0, nb_y, size=n)) % nb_y).astype(np.int32)
    fd = np.column_stack([x, y]).astype(np.int32)
    factors_nbins = np.array([nb_x, nb_y], dtype=np.int64)
    cy, fy, _ = merge_vars(
        factors_data=fd, vars_indices=np.array([1], dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=np.int32,
    )
    return fd, factors_nbins, cy, fy


def main():
    shapes = [(50_000, 16, 10), (200_000, 20, 10), (998_327, 20, 12)]
    print(f"{'n':>9} {'nb_x':>5} {'nb_y':>5} {'old_ms':>9} {'new_ms':>9} {'speedup':>8} {'maxdiff':>10} {'bx_ok':>6}")
    # warm JIT
    fd, fnb, cy, fy = _make(2000, 8, 6, 0)
    _legacy(fd, 0, fnb, cy, fy)
    _relevance_mi_1var_fused(fd, 0, 8, cy, fy)

    for n, nb_x, nb_y in shapes:
        fd, fnb, cy, fy = _make(n, nb_x, nb_y, 42)
        # bit-identity
        mi_old, bx_old = _legacy(fd, 0, fnb, cy, fy)
        mi_new, bx_new = _relevance_mi_1var_fused(fd, 0, nb_x, cy, fy)
        maxdiff = abs(mi_old - mi_new)
        bx_ok = bx_old == bx_new
        assert maxdiff == 0.0, f"MI not bit-identical at n={n}: {mi_old} vs {mi_new} (diff {maxdiff})"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input
        assert bx_ok, f"bx mismatch at n={n}: {bx_old} vs {bx_new}"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input

        N, REP = 11, 20
        old_t, new_t = [], []
        for _ in range(N):
            t0 = time.perf_counter()
            for _ in range(REP):
                _legacy(fd, 0, fnb, cy, fy)
            old_t.append((time.perf_counter() - t0) / REP)
            t0 = time.perf_counter()
            for _ in range(REP):
                _relevance_mi_1var_fused(fd, 0, nb_x, cy, fy)
            new_t.append((time.perf_counter() - t0) / REP)
        om, nm = np.median(old_t) * 1e3, np.median(new_t) * 1e3
        sp = om / nm if nm > 0 else float("nan")
        print(f"{n:>9} {nb_x:>5} {nb_y:>5} {om:>9.3f} {nm:>9.3f} {sp:>7.2f}x {maxdiff:>10.1e} {str(bx_ok):>6}")


if __name__ == "__main__":
    main()
