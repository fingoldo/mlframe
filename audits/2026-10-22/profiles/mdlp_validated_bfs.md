# Validated MDLP (`_mdlp_recurse_validated_bfs`): the CPU edge work left in the FE worker pools

Warm all-families 100k x 100 fit (seed 10): 153 `mdlp_bin_edges` calls, 10.4 s of CPU summed over the worker threads, all with distinct (x, y, settings),
so an in-memory memo saves nothing (measured: 0.0 s duplicate time). Worker-thread stack samples put the time in `_decide_pending_level` (the per-level
permutation-null significance test), `_dedupe_xy` and `compute_col_edges`; the FE worker pools are CPU-saturated during the concurrent stages, so this CPU
time competes with the other families (the JMIM device path and the batched W1 won in full fits for the same reason, 9b6cc63b1 and f5cfae680).

Device design, not built yet:
- Per level, every pending node of every column: the observed best split and `n_permutations` permuted best-split gains, the same statistic the honest-null
  root-split kernels (`_honest_null_k4`, `_honest_null_device`) compute for the root; batched over (column, node, permutation).
- Blocker: the permutations come from numba's Mersenne generator through `validated_seed`, and the serial / batch twins must stay identical. A device
  version either reproduces that stream on the device or moves both twins to the counter-based `_numba_utils_shuffle` stream (as the honest null did on
  2026-11-02); the second changes the edges, so it needs a selection-level check against the current path, not equality.

## 2026-11-04: xorshift draws for the permutation null

`_permutation_null_gain_njit` and its batch twin `_mdlp_permutation_batch_njit` re-seeded numba's Mersenne Twister per permutation (`np.random.seed(s)`
initialises a 624-word state) and drew every Fisher-Yates index with `randint`. Both now draw from the `_numba_utils_shuffle` xorshift stream with the same
seeds (`seed + p`, `base_seeds[node] + p`), so the twins stay equal to each other. 12 `mdlp_bin_edges` calls at 100k rows (binary and 3-class targets):
0.296 -> 0.254 s, best of 3. The permutations differ from the Mersenne ones (valid and uniform), so the edges on significant columns move slightly
(29 -> 32 and 20 -> 23 edges on the two informative columns of the bench; the noise columns keep 2).
