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

## nsys of the kernel inside the 100000 x 100 every-family fit (2026-11-06, window around `binned_numeric_agg_with_recipes`, nsys 2022.4.2, GTX 1050 Ti)

`tied_replicate_mi`: 4 launches in the window, 143 to 159 ms (grid 1280, block 256), 211 ms (1280, 128), 187 ms (1090, 128) and 169 ms (1200, 256): 1.02 s in all, 39 percent of the kernel time of the window (about 2.6 s) and the largest single kernel in it; `null_mi_perms_bits` 0.22 s is the next. The kernel is already launch-width-derived and KTC-dispatched (see above); no hardware-counter profile exists for it because `ncu` refuses Pascal (CUDA.md section 6), so the occupancy and stall causes inside the 150 ms are not measured. A candidate for the newer-GPU machine's `ncu` pass (CUDA.md section 9). No change made.

### Hardware counters (nvprof, 2026-11-06; the session runs elevated, so the counter metrics were collected)

`nvprof --profile-from-start off --kernels tied_replicate_mi --metrics ...` over the same 100000 x 100 fit (`bna_nsys.py`), 7 launches (the warm-up fit included), min / avg / max:

| metric | min | avg | max |
|---|---|---|---|
| achieved_occupancy | 0.238 | 0.384 | 0.613 |
| sm_efficiency | 95.7 % | 98.3 % | 99.3 % |
| warp_execution_efficiency | 71.2 % | 75.3 % | 79.4 % |
| gld_efficiency | 14.6 % | 15.2 % | 16.0 % |
| gst_efficiency | 0.01 % | 7.2 % | 25 % |
| shared_efficiency | 1.0 % | 2.0 % | 2.9 % |
| issue_slot_utilization | 12.9 % | 17.5 % | 21.1 % |
| ipc | 0.55 | 0.75 | 0.91 |
| stall_memory_dependency | 62.9 % | 77.6 % | 87.0 % |
| stall_exec_dependency | 9.6 % | 16.8 % | 28.3 % |
| stall_sync | 0.4 % | 0.8 % | 1.2 % |
| dram_utilization | Low (2) | Low (3) | Mid (5) |
| local_load_transactions | 6.0e3 | 1.06e6 | 2.36e6 |
| local_store_transactions | 4.2e3 | 6.6e5 | 3.06e6 |

Reading: the multiprocessors are active almost all of the time (98 %) but issue an instruction in only 17.5 % of the slots, and four fifths of the stalls are waits for data. The data is the random gather `rj[ib[t]]` of a 2-byte rank at a resample's draw (15 % load efficiency: each 32-byte sector carries one useful value; inherent to a bootstrap draw unless the draws of a resample are processed in rank order, which is what the host's counting sort does with two passes), then `ycode[r]` and `bp[...]` of the same row. Occupancy is 38 % because the dynamic shared table (levels, per-warp histograms and level counts) limits the resident blocks; shared efficiency of 2 % comes from the per-warp atomics on a tiny `(bin, class)` table (bank conflicts on a 20-cell histogram). One million local-memory transactions in the heavier launches are register or array spills, from the per-thread loops over straddling levels. Candidates, to be tried with a same-process A/B: prefetching `ib[t]` / `ycode` in a block of lanes before the dependent loads (more loads in flight), a smaller `PRIV_LEVELS` or per-warp tables to lift occupancy, and removing the spill array. None applied.

### Same-process A/B of three kernel changes (2026-11-06, `tied_ab.py`: 256 columns x 100000 rows with ties, 10 resamples of 80000 draws, `tied_replicate_mi_stream` including the host dense ranks, warm, 3 repeats)

Upper bound: replacing the random reads `rj[ib[t]]` and `ycode[r]` by sequential ones (wrong results, timing only) takes 0.77 s to 0.49 s, so the random gather is about a third of the stream's time at most. Tried, all with results equal to the base kernel (`array_equal`):

| change | base | variant |
|---|---|---|
| count pass unrolled by 4 (four independent gathers in flight) | 0.82 s | 0.81 s |
| software prefetch of the next draw's `ib` / `rj` / `ycode` in both walks | 0.77 s | 0.84 s (slower: more registers per thread) |
| `PRIV_LEVELS` 64 -> 0 (no per-warp level tables, atomics on the shared starts) | 0.76 s (about 300 levels), ~0.76 s (about 20 levels) | 0.76 s, 0.72 s (inside the run-to-run spread) |

None shipped. What is left of the counter reading is the gather itself: the only change that removes it is a coalesced pre-gather of the levels (and y codes) of each resample into a device scratch of 2 to 4 bytes per draw (205 to 410 MB for a 128-column block), which competes for the 2.58 GB pool with the concurrent stages; not tried.
