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

## Fused kernel (`_honest_oj_fused.py`), 2026-11-03
Code rows of every (re-binned column, candidate count) built once through the binner's own `_edges_slot` (the count planted as the plan's winner), then one
prange-over-permutations kernel per block of 64 host-drawn permutations: fold targets, winner per column (the `_oj_pick_candidates` loop calling the same
`_plug_in_mi_njit`), code lookup, the corrected-MI maximum. Same floor as the per-permutation path: 3000 x 10 fit 0.005560508526380476 on all three
paths; 30000 x 100 fit, both floor calls 0.0006034131614497845 (test_honest_oj_fused checks every floor call of a small fit).
Time per floor call, 30000 x 100 (999 permutations, 85 pool columns): pooled per-permutation 18.7 / 22.2 s -> fused 14.6 / 13.8 s. What remains is the
statistic itself (per permutation 85 x 4 x 3 fold MIs over 10k rows and 85 MIs over 30k rows), now inside numba.
Declines (per-permutation path) for a continuous / wide / negative raw target, uncached plans, a column without a scored candidate;
MLFRAME_HONEST_OJ_FUSED=0 turns it off.

## 2026-11-03: fused floor declined at 100k (plan-cache budget)

At 100k rows x 107 re-binned columns the default `OptimalJointPlanCache` (512 MB) stored the column facts first and refused 5 plans
(382 MB in use, the next plans did not fit), so `oj_fused_floor` declined and the per-permutation floor ran. Fix (dd229eeae): `put`
evicts column facts (one sort to rebuild) to make room for a plan; the fused floor reads plans from the cache and places columns with
target-free edges like fixed ones. 30k check unchanged: fused 13.8-14.6 s vs pooled 18.7-22.2 s, identical floor 0.0006034131614497845.
