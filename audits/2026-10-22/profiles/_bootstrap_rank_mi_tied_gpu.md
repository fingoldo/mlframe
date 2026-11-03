# _bootstrap_rank_mi_tied_gpu (device counting sort of the bootstrap rank MI)

100000 x 100 every-family fit, numeric decomposition: 856 columns (rounding at 1 / 0.1 / 0.01 / 0.001, digits 0-2, raw), 10 resamples of 80000 draws.
Host `_bootstrap_rank_mi_f32` 2.9-3.0 s, uniform across the groups (3.3 ms per column; 10 to 6000 distinct levels). The device distinct walk does not apply (ties).

Design: host dense ranks (`dense_ranks_f32`, the host's radix sort), device passes per (column, resample) block: level counts in shared memory (per-warp copies
for at most 64 levels), exclusive scan, draws of a level inside one bin histogrammed in any order, draws of the at most `n_bins - 1` straddling levels given their
ordinal by a per-warp coalesced walk with ballots and a scan over the warps. Bins from the host `binpos`, MI summed in the host order; difference to the host 1.8e-15.

Steps (device part of the 846 tied columns, then whole call):
- one warp walking all draws in order for the ordinals: kernel 2.86 s (slower than the host)
- per-thread contiguous chunks + block scan per straddling level: 2.51 s (uncoalesced `idx` reads)
- per-warp histograms and level counts (atomic contention): 2.64 s, no gain: contention was not the limit
- int16 ranks: 2.25 s
- per-warp coalesced segments with ballot ordinals: 1.17 s; call 2.43 s against host 2.95 s
- host ranks of block i+1 overlapped with the kernel of block i: call 1.75 s against host 2.98 s (1.7x)

Decomposition family warm wall 3.95 -> 2.82 s, same five columns. Dispatch: KTC `fe_bootstrap_rank_tied_device` (fallback device from 30000 rows),
`MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE=0|1` forces. Columns above 6144 levels and distinct columns keep the host path.
