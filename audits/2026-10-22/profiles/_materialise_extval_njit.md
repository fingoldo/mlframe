# `_materialise_extval_njit` loop order, 2026-10-29

Whole-fit profile: 19 calls, 0.92 s own time. The kernel looped external factor, then op, then row, writing `out[r, col]` into an (n, n_ext * n_ops) array,
so every store jumped n_ext * n_ops doubles.

Change: rows outermost, then external factor, then op, so each row's block of columns is written contiguously; `a`, `sign(a)` and the NaN flag are computed once per (row, factor).
Arithmetic per element is unchanged, so output is bit-identical (`D:/Temp/extval_bench.py`, `np.array_equal(..., equal_nan=True)`, NaN and exact-zero inputs).

Kernel time, n=20000, 9 ops, old vs new: n_ext=4 3.6 ms vs 1.5 ms; n_ext=20 29.2 ms vs 7.1 ms; n_ext=60 125 ms vs 20.9 ms (2.4x to 6x).
Pinned by `tests/feature_selection/fe/test_extval_kernel_row_order_parity.py` (every column against the numpy op) plus the four existing extval test files (88 passed).
