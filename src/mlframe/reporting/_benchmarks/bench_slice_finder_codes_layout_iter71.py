"""iter71 A/B: slice_finder ``_aggregate_combo`` arity-2 column gather -- C-order vs F-order ``codes`` matrix.

The arity-2 fast path (the dominant regime: thousands of feature pairs) does
``np.ascontiguousarray(codes[:, feat_idx[k]])`` per call. With the current C-contiguous ``(n, p)`` ``codes`` a column
slice is strided, so ``ascontiguousarray`` COPIES ``n`` int64 every call -- 2 length-n copies x N_pairs. Storing
``codes`` column-major (Fortran-order) makes ``codes[:, j]`` already contiguous, so ``ascontiguousarray`` returns the
view with ZERO copy. Bit-identical by construction (identical values, only memory layout differs).

This bench measures the per-combo gather+reduce cost for both layouts at the 200k-scale diag regime (DIAG_ROW_CAP
sub-sample n=100k, ~300 feature pairs), warm (numba JIT pre-compiled), best-of-N.

Run: python -m mlframe.reporting._benchmarks.bench_slice_finder_codes_layout_iter71
"""
import time

import numpy as np

from mlframe.reporting.charts.slice_finder import _aggregate_combo, _bin_matrix


def _make(n: int, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float64), rng.standard_normal(n).astype(np.float64)


def _run(codes, err, pairs):
    out = []
    for combo in pairs:
        sums, counts, _ = _aggregate_combo(codes, err, combo, [4, 4])
        out.append((sums, counts))
    return out


def main():
    n, p = 100_000, 30
    n_pairs = 300
    mat, err = _make(n, p)
    err = np.ascontiguousarray(err)
    codes_c, _edges = _bin_matrix(mat, 4)  # current: C-order
    codes_f = np.asfortranarray(codes_c)  # proposed: F-order
    assert codes_c.flags["C_CONTIGUOUS"] and codes_f.flags["F_CONTIGUOUS"]  # nosec B101 - internal invariant check in src/mlframe/reporting/_benchmarks, not reachable with untrusted input
    rng = np.random.default_rng(1)
    allpairs = [(int(a), int(b)) for a in range(p) for b in range(a + 1, p)]
    pairs = [allpairs[i] for i in rng.choice(len(allpairs), size=n_pairs, replace=False)]

    # Warm numba.
    _run(codes_c, err, pairs[:2]); _run(codes_f, err, pairs[:2])

    # Identity gate.
    rc, rf = _run(codes_c, err, pairs), _run(codes_f, err, pairs)
    for (sc, cc), (sf, cf) in zip(rc, rf):
        assert np.array_equal(sc, sf) and np.array_equal(cc, cf), "layout changed output!"  # nosec B101 - internal invariant check in src/mlframe/reporting/_benchmarks, not reachable with untrusted input
    print("IDENTITY: bit-identical sums+counts across all", len(pairs), "pairs")

    N = 30
    tc = []; tf = []
    for _ in range(N):
        t0 = time.perf_counter(); _run(codes_c, err, pairs); tc.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _run(codes_f, err, pairs); tf.append(time.perf_counter() - t0)
    mc, mf = min(tc), min(tf)
    medc, medf = sorted(tc)[N // 2], sorted(tf)[N // 2]
    faster = sum(1 for a, b in zip(tc, tf) if b < a)
    print(f"n={n} p={p} pairs={len(pairs)}  best-of-{N}")
    print(f"  C-order (current):  min={mc*1e3:.2f}ms median={medc*1e3:.2f}ms")
    print(f"  F-order (proposed): min={mf*1e3:.2f}ms median={medf*1e3:.2f}ms")
    print(f"  speedup min={mc/mf:.3f}x median={medc/medf:.3f}x  F faster in {faster}/{N} paired trials")


if __name__ == "__main__":
    main()
