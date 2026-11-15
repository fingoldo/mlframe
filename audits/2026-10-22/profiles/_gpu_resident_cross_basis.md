# Profile record: device-born orthogonal cross-basis candidates (`_gpu_resident_cross_basis`, host evaluator `hermite_fe`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: 5 numeric float32 columns, the pair cross-basis products of every unordered pair at degrees 1 and 2 (40 candidate columns), 100000 rows, 10 bins, binary target; the device path rebuilds the products from the raw operands and scores their plug-in MI.

## What the module is
The orthogonal cross-basis families (pair, triplet, quadruplet, adaptive arity, univariate uplift) multiply per-leg orthogonal-polynomial columns `h = basis(x)_deg` and score the products by `MI(col; y)`. The host path builds the `n x K` product matrix with the numba polynomial evaluators of `hermite_fe` and scores it (`mi_classif_batch_chunked`). The device path evaluates each distinct leg once with the batched Clenshaw evaluator, multiplies the legs of every candidate in one `mlframe_leg_products` launch and scores the matrix with the resident plug-in MI. The kernel-tuning entry `crossbasis_gate_backend` chooses the side by row count; when the host is chosen the resident scorer answers `None` and the caller's host scorer runs.

## Column blocks (streaming)
`_product_mi_blocked` builds a block of product columns, scores it and releases it; a block is a quarter of the room under this thread's pool limit over three copies of its `n x block` double columns, so the working set does not grow with the number of candidates. The pair, adaptive-arity and univariate-uplift scorers all go through it. A column's MI depends on that column alone, so the blocks give the single-matrix values exactly (a test with a room of three columns compares equal).

## cProfile (one device scoring, 100000 rows, 40 candidates)
0.011 s for 2201 function calls: 0.004 s in the one blocking copy of the MI vector (the host waiting for the queued kernels), 0.001 s for the five moment fingerprints that route each column to its basis. The Python side is a few launches, not a loop.

## nvprof (hardware counters)
Launches per scoring: one `mlframe_leg_products_cm` launch for all candidates, the leg evaluations (cached per fit), `radix_select_interp_f64_v3` for the edges of the block, `mi_hist_split` and `mi_from_counts_col` for the histograms and the MI. The leg-product kernel now writes the (K, n) block directly with consecutive threads on consecutive rows, and the MI kernel reads that block as it is (`x_is_cm`), so the `transpose_f64` launch (0.73 ms) of the first version is gone. GPU time 3.8 ms: `radix_select_interp_f64_v3` 58 percent (2.20 ms), `mlframe_leg_products_cm_float32` 21 percent (0.79 ms), `mi_hist_split` 16 percent (0.62 ms).

| kernel | achieved occupancy | other counters |
|---|---|---|
| `mlframe_leg_products_cm_float32` | 0.91 | DRAM read 26 GB/s, global load efficiency 82.5 percent (was 32.4 percent, 1.12 ms, when a thread walked the legs of its cell at a stride of the leg block), single-precision FLOP efficiency 0.5 percent: a gather-and-multiply, memory bound |
| `radix_select_interp_f64_v3` | 0.69 | DRAM read 18 GB/s, global load efficiency 99.99 percent |
| `mi_hist_split` | 0.87 | DRAM read 49 GB/s, global load efficiency 75 percent |
| `mi_from_counts_col` | 0.10 | 47 microseconds, a reduction over the histograms |

Whole-GPU utilization: 3.8 ms of GPU time in an 8.1 ms wall (47 percent busy); the call is bound by the edge selection of the binner, which is shared infrastructure.

## Wall clock (host product matrix + MI against the device-born scoring, same top candidates)
| rows | host | device |
|---|---|---|
| 5,000 | 7.7 ms | 2.6 ms (3.0x) |
| 30,000 | 24.8 ms | 4.9 ms (5.1x) |
| 100,000 | 82.3 ms | 8.1 ms (10x) |

The device wins at every swept size on this card (the resident legs are cached per fit and the products are one launch), so the entry changes nothing here; it keeps a card where the host wins from choosing the device. Per-leg values differ from the host evaluators at 1e-12 (forward recurrence against Clenshaw); the selection parity tests (pair, triplet, quadruplet, adaptive arity, univariate) agree with the host ranking.
