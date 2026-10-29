# `null_max_mi` CPU rank path (permutation null of `local_mi_gate`), 2026-10-29

Whole-fit profile (n=20000): 7 calls, 273 permutations; `_mi_from_codes_rows_njit` 0.77 s, `_rank_codes_rows_njit` 0.34 s.

Tried: one kernel over all (permutation, column) pairs (`prange` over B*k) in place of one `prange`-over-columns call per permutation (`D:/Temp/null_perm_bench.py`, B=40, n=20000).
Results are bit-identical (`np.array_equal` on the null maxima). Times, old vs batched: k=8 13 ms vs 5 ms; k=30 25 ms vs 15 ms; k=120 72 ms vs 69 ms.

Not applied: the gain exists only for narrow matrices, and the real calls spend about 2.8 ms per permutation, i.e. they are already column-parallel and compute-bound;
the saving on the real fit is bounded by a few tens of milliseconds. Rank coding (0.34 s) is already `prange` over columns with a per-column argsort.
