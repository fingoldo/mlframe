
## 2026-11-03: early-stopping permutation null

Whole-fit py-spy (100000 x 100, nbins_strategy="mdlp"): the edge-pool workers spent 3.9 thread-s in `_decide_pending_level` (the batched permutation null of
the MDLP nodes) out of 15 busy thread-s. The verdict needs only a count (accept iff at most `n_permutations - 1 - q_idx` nulls reach the gain), so the
permutations now run in chunks of 16 over the undecided nodes and a node stops once decided; each draw keeps its own seed. Same verdicts (0 mismatches on
591 fit nodes and 366 synthetic ones); in the fit 3.88 -> 3.56 s (most cost is accepted nodes, which need nearly every draw).
`_dedupe_xy`: 2.1 thread-s of the edge-pool workers in the whole-fit py-spy, a `lexsort((y, x))` per column even when `x` arrives sorted (`mdlp_bin_edges`
passes its sorted column). A strictly increasing `x` has no repeated (x, y) pair, so one O(n) check returns the input as the lexsort path would: 100000 rows
10.6 -> 0.04 ms; tied or unsorted input keeps the lexsort (same output, test_mdlp_dedupe_fast_path).

## 2026-11-03: shared xorshift shuffle in the MDLP permutation nulls - rejected (selection changed)

Per permutation of a 100000-row node, numba's `np.random.seed` + per-element `np.random.randint` Fisher-Yates cost 2.74 ms against 1.09 ms for the best-split
scan. Switching `_mdlp_permutation_batch_njit`, `_permutation_null_gain_njit` and their serial twin `_accept_split_serial` to the shared xorshift draw
(`_numba_utils_shuffle`, same per-permutation seeds) made one column 0.098 -> 0.073 s and passed the 180 MDLP / discretization tests, but the permutations are
different draws: one test column got 58 edges instead of 60, and of two 20000 x 30 all-family fits one selected the same features and the other a different set
(2 features against 5). Reverted. The MDLP-binned selection is sensitive to which permutations are drawn on this data; a faster shuffle that keeps the
selection needs the same draws (the Mersenne Twister stream with numba's rejection sampling), not a different generator.
