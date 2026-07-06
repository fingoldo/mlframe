"""Microbench (CPX25-A): mixed-radix cell-id decode in slice_finder.find_weak_slices.

The OLD code (slice_finder.py ~289-295) decodes each valid+degraded cell id back to its
per-feature bins with a Python loop over the combo arity, inside a Python loop over the cell
ids -- for every combo. The NEW path decodes the whole batch of cell ids for a combo at once
with vectorized floor-div / mod against the stride vector (the np.unravel_index arithmetic),
which is trivially bit-identical (pure integer index arithmetic, same formula).

Run:  python src/mlframe/reporting/charts/_benchmarks/bench_slice_decode_broadcast.py
"""
from __future__ import annotations

import time
import numpy as np


def decode_old(cell_ids, strides):
    """Exact copy of the per-cell Python-loop decode (slice_finder.py ~289-295)."""
    m = len(strides)
    out = []
    for cid in cell_ids:
        rem = int(cid)
        bins_of_cell = []
        for k in range(m):
            bins_of_cell.append(rem // int(strides[k]))
            rem = rem % int(strides[k])
        out.append(bins_of_cell)
    return out


def decode_new(cell_ids, strides):
    """Vectorized batch decode: bins[:,k] = (cid // stride_k) % nbins_k, expressed via the
    same mixed-radix recurrence (floor-div by stride, then take remainder for the next digit)."""
    cid = np.asarray(cell_ids, dtype=np.int64)
    st = np.asarray(strides, dtype=np.int64)
    m = st.size
    bins = np.empty((cid.size, m), dtype=np.int64)
    rem = cid.copy()
    for k in range(m):
        bins[:, k] = rem // st[k]
        rem = rem % st[k]
    return bins


def _make_case(rng, arity, nbins, n_cells_decoded):
    strides = np.ones(arity, dtype=np.int64)
    for k in range(arity - 1, 0, -1):
        strides[k - 1] = strides[k] * nbins
    ncells = int(strides[0]) * nbins
    cell_ids = rng.integers(0, ncells, size=n_cells_decoded, dtype=np.int64)
    return cell_ids, strides


def bench():
    rng = np.random.default_rng(0)
    # Realistic: thousands of combos, each emitting a handful..dozens of degraded cells to decode.
    # We aggregate the decode work across all combos in one search.
    configs = [
        ("arity2 nbins10  3000 combos x 8 cells", 2, 10, 3000 * 8),
        ("arity3 nbins8   2000 combos x 12 cells", 3, 8, 2000 * 12),
        ("arity2 nbins20  5000 combos x 20 cells", 2, 20, 5000 * 20),
    ]
    for label, arity, nbins, total in configs:
        cell_ids, strides = _make_case(rng, arity, nbins, total)

        # identity
        old = decode_old(cell_ids, strides)
        new = decode_new(cell_ids, strides)
        ok = np.array_equal(np.asarray(old, dtype=np.int64), new)
        assert ok, f"identity FAIL {label}"  # nosec B101 - internal invariant check in src/mlframe/reporting/charts/_benchmarks, not reachable with untrusted input

        # We bench per-combo batches (decode is called once per combo with that combo's cell_ids).
        # Simulate by splitting total into batches matching the config.
        n_combos = {2: 3000, 3: 2000}[arity] if "3000" in label or "2000" in label else 5000
        if "5000" in label:
            n_combos = 5000
        per = max(1, total // n_combos)
        batches = [cell_ids[i : i + per] for i in range(0, total, per)]

        best_old = best_new = 1e9
        for _ in range(7):
            t = time.perf_counter()
            for b in batches:
                decode_old(b, strides)
            best_old = min(best_old, time.perf_counter() - t)
            t = time.perf_counter()
            for b in batches:
                decode_new(b, strides)
            best_new = min(best_new, time.perf_counter() - t)
        print(f"{label:42s}  identity={'OK' if ok else 'FAIL'}  " f"OLD={best_old*1e3:8.3f}ms  NEW={best_new*1e3:8.3f}ms  speedup={best_old/best_new:5.2f}x")


if __name__ == "__main__":
    bench()
