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
