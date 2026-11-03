# _orth_auto_scorer_fe (bootstrap LCB tables of the auto-scorer family)

`hybrid_orth_mi_auto_scorer_fe` -> `select_best_scorer_per_column` -> `_bootstrap_lcb_tables` -> `_batched_bootstrap_values`: for each of the (about 10) bootstrap resamples every batched scorer runs on the whole column block. In the 100000 x 100 every-family fit the family sits at 22 sampler samples (about 4 s), process CPU 81%, GPU 3%: host-bound and already parallel, so only a device kernel or less work moves it.

Per-scorer cost for one resample, 24000 rows x 108 float32 columns, warm, GTX 1050 Ti host (8 cores, `D:/Temp/sc_bench.py`):

| scorer | ms | note |
|---|---|---|
| `hsic` (`_hsic_biased_columns_njit`, n_sample 1000) | 327 | median pairwise distance (500k pairs, partition) + 500k `exp` per column against the centred y Gram matrix |
| `ksg` (discrete target) | 170 | |
| `dcor` | 155 | |
| `xi` | 141 | |
| `copula` | 78 | |
| `tail_dep` | 73 | |
| `plug_in` | 12 | |

About 0.95 s per resample serial, 10 resamples per call, several families call it (auto-scorer, ensemble). Ranking of device-kernel candidates: HSIC first (one block per column, the bandwidth from a block-wide sort of the 500k distances, then the `exp` sum with `lc`; exactness of the median is the contract point), then dCor and xi (both sort/rank based, `cp.argsort(axis=1)` of the block). Constraint found in the earlier stage-concurrency study: these run on pool threads (one per resample), and several threads feeding one CUDA context was the suspected cause of the illegal-address corruption, so a device version should run the resamples as one batched launch from the main thread, not from the pool.

Not started; this note is the measured starting point.

## Done: exact O(m log m) median pairwise distance (2026-11-02)

The bandwidth of HSIC (and of the random-Fourier family) is the median of the m(m-1)/2 pairwise distances of a column. `_median_pairwise_distance` built all 500k distances of a 1000-row sample and quickselected: 6.8 ms per column. It now sorts the column and finds the k-th smallest distance by bisection on the distance with a two-pointer count (`_kth_pair_distance`); the quickselect stays as `_median_pairwise_distance_select` for non-finite columns and as the test reference. Rounding is monotone, so the two-pointer pass is valid in float32 as well, and the value is bit-identical (`test_median_pairwise_distance.py`: 10 sizes x 5 column shapes x 2 dtypes, plus the non-finite fallback).

108 columns x 1000 rows: median 737 ms -> 54 ms serial; the whole `_hsic_biased_columns_njit` batch 265 ms -> 133 ms on 8 threads. What remains of HSIC is the 500k `exp` sum per column against the centred y Gram matrix, which is where a device kernel would now start.

## Done: xi batch draws the next block's permutations while the kernel scores the current one

`xi_correlation_batch` drew one `rng.permutation(n)` per column (52 ms of 141 ms for 108 columns x 24000 rows) before each parallel kernel call. The draws must stay one generator in column order, but the shuffle releases the GIL, so a single helper thread now draws block i+1 while the kernel scores block i (the helper is the only user of the generator and blocks are drawn in order: the stream, hence every value, is unchanged; `test_xi_batch_stream.py` compares with a serial per-column draw). Block size 32 -> 12 columns so there is more than one block to overlap: 151 ms -> 114 ms (block 12), 118 ms (block 16), 130 ms (block 32), same process settings, values identical.

## 2026-11-03: auto-scorer bootstrap at 100000 x 100 (30000-row decision subsample, 5 resamples, 317 columns)

Warm family wall 4.57 s (in the fit 4.33 s). Thread-summed scorer time: KSG 3.41 s (10 calls, entered one thread at a time through `parallel_kernel_entry`), xi 2.46 s.
`_ksg_cd_fast._mi_cd_column`: the neighbour count within the k-th radius took two `np.searchsorted` over the whole sorted column per point (about 30 cache-missing steps);
the same bounds are now found by walking out from the point's own sorted position (a few steps). Bit-identical MI; 317 columns x 30000 rows 0.68 -> 0.45 s (2 classes),
0.60 -> 0.45 s (3 classes).
`_orthogonal_xi_fe._xi_columns_radix_njit` (float32 columns): `lexsort((perm, col))` as a stable 4-pass LSD radix sort of the order-preserving keys started from the
`perm` order, instead of a mergesort plus a re-sort of every tie block. Bit-identical Xi; 302 columns x 30000 rows 0.41 -> 0.28 s.
Per-scorer split after the KSG / xi changes (warm, thread-summed, 30000-row subsample, 5 resamples): KSG 2.16, copula 1.79, tail dependence 1.65, HSIC 1.54,
xi 1.30, dCor 0.98, plug-in 0.40 s. `_orthogonal_copula_mi_fe._copula_columns_radix_njit` (float32 rows): the tie runs from a 4-pass LSD radix sort instead
of a mergesort (the MI depends only on the runs); bit-identical, 300 columns x 30000 rows 0.202 -> 0.050 s.
