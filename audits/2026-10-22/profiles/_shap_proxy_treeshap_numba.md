# Profile record: CPU (numba) TreeSHAP, main effects and interactions

Host: 4 physical cores / 8 logical, numba tbb layer, 8 threads, 2808 MHz, no throttle on the CPU side; another agent used 30-95 percent of the CPU during the runs (load recorded with `hardware_state.snapshot()` before and after each run, all A/B pairs interleaved in one process, private `NUMBA_CACHE_DIR`, stale caches wiped). Model: LightGBM 100 trees depth 6, f = 100, trained on the rows given (about 2860-3160 leaves, 2760 inner nodes, mean 4.2 unique features per leaf, main table 2.75 MB, interaction table 5.7 MB). No fastmath anywhere; NaN test kept (`xv != xv`), pinned by NaN tests against the path kernel.

Commits: a68d0f9df (interaction table form), 210f9697f (main table preorder/blocked/4-abreast kernel), e148dec60 (blocked interaction kernel).

## cProfile ranking (tottime, one call)

| call | kernel | build of tables | rest |
|---|---|---|---|
| main phi, 6000 rows | `_table_batch_blocked` 48 ms | `_build_tables` 4 ms | python glue < 1 ms |
| interactions, 1000 rows | `_interaction_table_batch_blocked` 85 ms | `_build_itables` 7 ms + `_build_tables` 4 ms | `_per_tree` split-feature count 1 ms |

Before the change the interaction call at 200 rows was 785 ms in `_interaction_batch` alone (about 4 ms per row on 8 threads, 40 times the main phi cost).

## Cost model and ablation (main table form, 1 thread, 6000 rows unless noted)

Per row: 2760 inner-node visits (route), 2860 leaf visits, about 12000 table adds (mean ud 4.2).

| part (stack-free preorder kernel, unblocked, 3000 rows) | ms | per unit |
|---|---|---|
| full | 296 | |
| route only | 66 | 22 cycles per inner node |
| leaf adds only (mask faked) | 179 | 12-14 cycles per add (throughput would be 1-2) |
| route + leaf index, no adds | 103 | |

Reading: the leaf adds, not the routing, are 60 percent of the time, and they run at 12 cycles per add: every leaf of a tree adds into the features near its root (feature of the root split is on every path), so the adds form a load-add-store-forward chain through memory. Second effect: the ensemble metadata (about 400 KB) plus the table (2.75 MB) streams through L2/L3 once per row. Fixes in that order: preorder lists (branch-free route, no DFS stack), trees outer over blocks of rows (metadata read once per block), four rows abreast (four independent chains, records loaded once for four rows). Ablation after the change (6000 rows, 1 thread): blocked + 4 rows 228 ms against 442 ms for blocked alone and 798 ms for the DFS stack.

## Thread sweep and before/after, main phi (7 interleaved repeats, `final.py`)

| rows | threads | stack best / median ms | new best / median ms | speed-up best / median | max diff to stack |
|---|---|---|---|---|---|
| 6000 | 1 | 779.5 / 839.6 | 215.9 / 251.6 | 3.61x / 3.34x | 6.2e-15 |
| 6000 | 2 | 406.5 / 412.2 | 112.3 / 130.0 | 3.62x / 3.17x | 6.2e-15 |
| 6000 | 4 | 223.9 / 227.0 | 78.1 / 82.8 | 2.87x / 2.74x | 6.2e-15 |
| 6000 | 8 | 163.8 / 170.8 | 64.5 / 70.5 | 2.54x / 2.42x | 6.2e-15 |
| 20000 | 8 | 547.1 / 570.7 | 194.3 / 208.9 | 2.82x / 2.73x | 6.2e-15 |
| 100000 | 8 | 2728.9 / 2819.5 | 983.2 / 988.3 | 2.78x / 2.85x | 6.2e-15 |

Difference of the new kernel to the PATH kernel (the reference): 7.1e-15 (6000 rows), 5.3e-15 with 15 percent NaN. Hardware: CPU 2808/2808 MHz, load 44 percent before, 78 percent after (other agent), GPU idle (not used). The 1-thread to 8-thread efficiency of the new kernel is 215.9 / (8 * 64.5) = 0.42 (stack kernel 0.59): the kernel got faster per core, the remaining limit at 8 threads is the 4 physical cores sharing L2 plus the other agent's load.

## Interaction tensor (`interaction_tensor_numba`), f = 100

| rows | threads | conditioned reference best / median ms | table form best / median ms | speed-up | max diff Phi |
|---|---|---|---|---|---|
| 200 | 8 | 785 / 823 | 19 / 24 (first version, DFS stack) | 41x | 1.8e-15 |
| 1000 | 8 | 5457 / 5687 | 84 / 90 (final) | 65x | 4.4e-15 |
| 400, 15 % NaN | 8 | 2007 / 2066 | 30 / 36 (final) | 67x | 3.6e-15 |

Why it is exact and cheap: a leaf whose path does not contain feature j gives equal on and off scans, so only the `ud` features of its path matter; with j on the path, "forced on" is (bit of j) times the main contribution of the path without j and "forced off" is the zero fraction of j times the same, so `0.5*(on-off) = 0.5*(bit_j - z_j) * C[j][mask without j][q]`. The table holds ud * 2^(ud-1) * (ud-1) entries per leaf (5.7 MB here, build 11 ms). Blocking and four rows abreast on the interaction kernel (403 rows, 1 thread: 263 -> 93 ms; 2000 rows, 8 threads: 349 -> 206 ms; run under load). Final thread sweep at 1000 rows: 260 / 155 / 96 / 91 ms for 1 / 2 / 4 / 8 threads (efficiency 0.36 at 8): bound by the 80 KB Phi row each sample writes (four rows abreast = 320 KB, above L2) and by SMT, not by compute.

## What the limiter is now

Main phi: leaf adds (read-modify-write into phi) and L2 traffic of the table, with 8 threads on 4 cores. Interactions: scatter adds into the (P, P) tensor rows and the epilogue (symmetrise + diagonal, 22 ms per 1000 rows on 1 thread, about 8 percent now).

## Tried and rejected (this host, interleaved A/B)

| idea | result | verdict |
|---|---|---|
| unblocked preorder kernel (branch-free route only) vs DFS stack | 627 vs 793 ms best at 6000 rows 1 thread, but the median gap was 1-2 percent in an A/A run with the same kernel on both sides showing 7 percent best-of difference | kept only as part of the blocked kernel; alone it is inside noise |
| packed int32 node and leaf records (AoS, int32 mask) on the blocked kernel | 379 vs 442 ms best, 394 vs 463 median (single-row), then irrelevant once four rows abreast | not adopted: the memory-layout lever is dwarfed by the add chains; open for re-test on top of the 4-row kernel |
| skip zero-valued leaves | 0 of 3158 leaves are zero | nothing to skip |
| `set_parallel_chunksize(1 or 4)` with tbb (n = 20000, 8 threads, block 16) | 182.9 / 183.4 vs 194.1 ms best (6 percent), median 197 vs 210 | within noise of the 5 percent bar and process-global state: not adopted |
| block 16 / 32 / 64 / 128 (main) | 1 thread: 449 / 445 / 429 / 437 ms (single-row inner); 8 threads, 20000 rows: 32 beats 64 in one run, 64 beats 32 in the next | 64 default, fewer for small inputs (`_auto_block`) |

## Open ideas not reached

- float32 table with float64 accumulator (tolerance 1e-6, SHAP rank check): the table is 2.75 MB, the chain rather than the bandwidth dominated before the change, re-test now.
- eight rows abreast, and the packed-record layout on top of the 4-row kernel.
- table form for leaves with more than 12 unique features (splitting the mask table): the current ensemble never exceeds 7, the fallback to the path kernel covers it.
- interaction kernel: accumulate the tensor tile by tile (rows of 4 samples x feature blocks) to keep the Phi write set in L2; fuse the epilogue per row while the Phi row is hot.
- PAYOFF_SHARE re-sweep (the table kernel is cheaper per row now, the build is 4 ms for 100 trees, the rule is conservative).
