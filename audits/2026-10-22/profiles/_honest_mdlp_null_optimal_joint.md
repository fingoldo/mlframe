# Honest null with nbins_strategy=optimal_joint / cv (2026-11-03)

100k x 100 all-families GPU fit, nbins_strategy sweep: mdlp ~96 s, quantile 105.6 s, fd 122.0 s, uniform 136.5 s, optimal_joint > 29 min (stopped),
cv > 11 min (stopped). 30000-row optimal_joint fit (stage_sample3b, SS_NBINS=optimal_joint): 120.6 s, of which `honest_fdr_floor` 53.4 s and
`honest_candidate_nfailed` 3.4 s. The non-mdlp floor re-bins the whole pool per permutation (`_honest_perm_best` -> `PermutedTargetBinner.codes` ->
`prescore_best_m`, then `corrected_mi_best`) in a serial Python loop; plain mdlp takes the `spec` route instead (`_njit_fdr_floor`: prange over the
permutations inside numba).
Done: permutations of the floor and of the per-candidate confirmation on the edge pool (0bd799885, aa861275e; same floor / counts): floor 53.4 -> 35.0 s,
fit 120.6 -> 103.0 s (repeat 34.4 s / 105.1 s).
Why only 1.5x on 8 threads (py-spy record, 70 s, per-thread raw stacks): every pool thread was active ~4 s of the ~35 s floor; the active samples are the
binner's Python per-permutation work (`codes`, `_honest_null_codes`: ~300), `prescore_best_m` (~400) and `corrected_mi_best` (~470). numba's threading
layer here is TBB, so `parallel_kernel_entry` is a no-op: not a lock problem. Tried and reverted: a serial nogil twin of `corrected_mi_best` for the pool
threads (floor 37.0 s, noise).
Next (the real fix): a fused numba null for optimal_joint / cv like the mdlp spec route - per permutation, pick the candidate bin count of every pool
column from its cached plan (`_oj_pick_candidates` on the packed plans), re-code the pool with the winners' edges, and take the corrected-MI maximum,
all inside one prange-over-permutations kernel; the Python binner then runs once per fit instead of once per permutation.
