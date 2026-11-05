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

## 2026-11-05: correction - the serial twin must draw the same permutations

The xorshift commit (57762e7a5, pushed in 8abc2061b) moved `_permutation_null_gain_njit` and `_mdlp_permutation_batch_njit` to xorshift draws and claimed the twins stay equal. They did not:
`_accept_split_serial` (the serial twin of the batched level test, used by the column-outer honest null of `_numba_utils_cross_module`) still drew from numba's Mersenne
`np.random.seed`. Verdicts of the two stochastic tests then disagreed on borderline splits, and `test_honest_candidate_spec_route::test_candidate_nfailed_identical_on_both_routes`
failed (categorise route 3, batched route 2 exceedances for column 4; bisect from 3e8021b6e^: first bad commit 57762e7a5). Verdict agreement on clear-cut nodes hid it: a twin test
that compares accept flags passes with either stream (`test_mdlp_serial_batch_twins` first version passed with the old serial twin).

Fix: `_numba_utils_shuffle.shuffle_labels_into(src, dst, seed)` is the one permutation draw of the MDLP split tests (= `src[seeded_permutation(n, seed)]`, the honest-null stream); the
per-node null, the batched level test, the serial twin and the benchmark reference all call it, so they cannot drift. The remaining numba-RNG users
(`root_null_gain_threshold`, `_mdlp_oos_honest_njit`) have no twin in another stream and stay.
