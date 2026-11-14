# _binned_numeric_agg_fe.py (binned numeric-aggregate FE) with its carved modules: profile record (2026-11-14)

Family: `_binned_numeric_agg_fe` (the host fit and recipes) with `_binned_numeric_agg_resident` (the device OOF candidate builder `build_binagg_oof_matrix_gpu`), `_binned_numeric_agg_batches`, `_binned_numeric_agg_cands`, `_binned_numeric_agg_redundancy`, `_binned_agg_cheap_mi` (quantile edges, cheap MI proxy; holds the tuner entry `binagg_quantile_edges_backend`) and `_binned_agg_linear_usability`. The per-cell moment kernels are in `_cell_moments_kernel`.

## cProfile

Whole default fit (100000 x 100, one warm pass): no function of this family reaches 0.05 s of cumulative time, the family is not exercised by the default fit. Its device builder measured alone (100000 rows, 20 numeric columns, 5 group columns x 10 aggregated columns x 2 statistics = 100 output columns, 5 folds): 64 ms per `build_binagg_oof_matrix_gpu` call, of which `_pair_fold_stat_tables` 21 ms (50 calls, `fold_cell_moments` 16 ms), `_stats_from_moments_gpu` 4 ms, the rest the per-column gathers and a synchronise.

## Kernels (nvprof, that call, two calls profiled)

Kernel launches per call: 100 `binagg_oof_gather_into_float32` (one per output column, 99 us each, 18.4 % of the GPU time), 50 `fold_cell_centred` (541 us each, 50.4 %), 50 `fold_cell_sums` (307 us each, 28.6 %), then about 25 small elementwise launches. Achieved occupancy: the gather kernel 0.82 to 0.90, the elementwise kernels 0.83 to 0.88, the two fold-cell kernels only 0.18 to 0.19. Throughput: the gather kernel reads 6.5 to 7.8 GB/s and writes 26 to 33 GB/s, SM efficiency 82 to 96 %; `fold_cell_sums` reads 9.3 to 9.8 GB/s at SM efficiency 93 to 99 %, `fold_cell_centred` reads 5.4 to 5.7 GB/s: both fold-cell kernels are at a small share of the card's ~112 GB/s with the SMs busy, so they are bound by their per-cell work and their low occupancy, not by memory bandwidth. Open lever (not taken, the family is cold in the default fit): the 0.18 occupancy of the fold-cell kernels and the 100 gather launches (one per output column, 31 us launch-bound at 100000 rows, tracker GPU11-61).

## Host-or-device choice

`quantile_edges` takes the device quantile (bit-identical to `np.quantile`, one device sort) by the per-host kernel-tuning entry `binagg_quantile_edges_backend` (`host_device_sweep` over the column length, tied values included): host up to 20000 rows, device from 100000 rows on the development card, max abs difference 0.0; the former fixed floor of 200000 rows is the pre-sweep fallback and sent the columns of 100000 to 200000 rows to the host.

## Streaming

The device builder keeps its operands in byte-bounded caches (`_BytesLRU`, a share of the free VRAM; an evicted entry is rebuilt from its column) and allocates the candidate matrix once, as the (columns, n) block or its row-major twin, writing the output columns into it one at a time as they are gathered, so the peak is the cached group and aggregated columns plus that one block (the earlier collect-and-stack form held about three times the block). The device quantile/maximum reductions elsewhere in the family run in column blocks of the matrix.
