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
`_orthogonal_tail_dependence_fe._tail_dep_columns_njit` takes the same radix order for float32 rows (`_numba_utils_shared.radix_order_f32`, now shared with the
copula kernel): bit-identical, 300 x 30000 0.219 -> 0.046 s.
`_ksg_cd_fast._mi_cd_column`: the per-column `np.argsort` is an 8-pass LSD radix order of the float64 keys (`_numba_utils_shared.radix_order_f64`); the k-NN
walk reads the sorted values only, so the MI is bit-identical. 300 columns x 30000 rows: 0.459 -> 0.418 s (2 classes), 0.506 -> 0.371 s (3 classes).

## 2026-11-16: under the contract (owner: the CT11-58 exemption is replaced by the real contract)

The auto scorer draws the bootstrap resamples, builds the `(n, k)` block of the finite columns and its transposed device copy once per resample, and hands them to the seven batched scorers (plug-in MI, KSG, copula MI, xi, dCor, HSIC, tail dependence). The numba and CUDA kernels and the tuner entries are owned by the scorer modules (`GPU_BACKENDS` of `fe_gpu_coverage_inventory` lists them, the test pins that every one meets the whole contract itself: `_orthogonal_ksg_mi_fe` (this session, CT11-67), `_orthogonal_copula_mi_fe`, `_orthogonal_xi_fe`, `_orthogonal_dcor_fe`, `_orthogonal_hsic_fe`, `_orthogonal_tail_dependence_fe`, `_orthogonal_univariate_fe`); the host form is the numba KSG kernel of `_ksg_cd_fast` plus the host batches of the same modules.

Column blocks (new): `_batched_bootstrap_values_in_column_blocks` hands the scorers at most `_BOOTSTRAP_BLOCK_BYTES` (256 MB) of a resample at a time; the scorers are column-separable, so the blocks laid side by side equal one call (new test); a 1000-column frame of a million rows was a 4 GB block per resample.

### Per-scorer cost of one resample (100 columns, GTX 1050 Ti, strict-resident, warm, synchronized)

| rows | plug-in | KSG | copula | xi | dCor | HSIC | tail | device block |
|---|---|---|---|---|---|---|---|---|
| 30000 | 0.006 s | 0.088 | 0.039 | 0.039 | 0.009 | 0.010 | 0.009 | 0.004 |
| 100000 | 0.017 s | 0.284 | 0.081 | 0.112 | 0.010 | 0.011 | 0.028 | 0.014 |

KSG is half of a resample at 100000 rows; its record is `_orthogonal_ksg_mi_fe.md`.

### cProfile and timings (`hybrid_orth_mi_auto_scorer_fe`, 100 columns)

30000 rows: 1.51-1.64 s wall, 5.4-5.8 CPU-s (3.6 cores busy); 100000 rows: 5.77 s, 12.4 CPU-s. cProfile of the 30000-row call: 1.28 s of the 1.40 s is `lock.acquire`, the main thread waiting for the resample pool (the scorers run in the pool threads, which cProfile does not see).

### nvprof (30000 rows, 2 calls)

Kernel launches per call: about 100 radix-sort / merge-sort launches (DeviceMergeSortMerge 22 %, BlockSort 12.7 %, RadixSortOnesweep 4.8 %: the rank transforms of copula and xi and the order of the dCor / HSIC samples), 10 `dcor_cols`, 10 `ksg_psi_m_sums`, 10 `radix_select_interp`, 10 `hsic_biased_cols`, 10 `tail_counts`, 134 uploads (17 % of the device time). Device time about 0.33 s of the 1.5 s call (22 % busy): the call is bound by the host scorers and the order of the resamples, not by the kernels. Kernel metrics: `tail_counts` occupancy 0.65, SM activity 88 %, 20 GB/s; the radix histogram kernel occupancy 0.37, SM activity 98.8 %, 45 GB/s of the 112 GB/s card; the exclusive-sum kernel of the sort occupancy 0.17, SM activity 37 % (a small scan, 10 launches). Whole-GPU utilization of the call 22 % (device busy), CPU 3.6 of 8 cores.
