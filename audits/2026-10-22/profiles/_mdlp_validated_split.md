
## 2026-11-03: early-stopping permutation null

Whole-fit py-spy (100000 x 100, nbins_strategy="mdlp"): the edge-pool workers spent 3.9 thread-s in `_decide_pending_level` (the batched permutation null of
the MDLP nodes) out of 15 busy thread-s. The verdict needs only a count (accept iff at most `n_permutations - 1 - q_idx` nulls reach the gain), so the
permutations now run in chunks of 16 over the undecided nodes and a node stops once decided; each draw keeps its own seed. Same verdicts (0 mismatches on
591 fit nodes and 366 synthetic ones); in the fit 3.88 -> 3.56 s (most cost is accepted nodes, which need nearly every draw).
`_dedupe_xy`: 2.1 thread-s of the edge-pool workers in the whole-fit py-spy, a `lexsort((y, x))` per column even when `x` arrives sorted (`mdlp_bin_edges`
passes its sorted column). A strictly increasing `x` has no repeated (x, y) pair, so one O(n) check returns the input as the lexsort path would: 100000 rows
10.6 -> 0.04 ms; tied or unsorted input keeps the lexsort (same output, test_mdlp_dedupe_fast_path).
