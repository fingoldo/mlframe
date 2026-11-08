# Plan v4: cut the real transients first (binning, code matrix, GPU widening), exact full-n fit up to about 100M rows, subsample-fit mode only beyond

Revised after six independent reviews. v3 aimed at copies this environment no longer pays and at a mode (subsample-fit) that is needed only at about 1B rows. The two v3 reviews (Opus 5.5 medium) agree on what costs memory today. Numbers below are reviewer derivations from the code, to be confirmed by phase 0 measurements; file:line are from the reviews.

## Environment facts that change the picture

- Installed: pandas 3.0.3 (copy-on-write always on), polars 1.44.1, numpy 1.26.4. But `pyproject.toml:63` pins `pandas>=1.5,<3.0`, where CoW is off. Both must be supported and measured: on pandas 3 `pd.concat(axis=1)`, `df[cols]`, `iloc[a:b]` and a same-dtype single-block `to_numpy` share memory; `pd.DataFrame(dict)`, `pd.DataFrame(ndarray)`, `drop`, `copy()`, fancy `iloc` and multi-block `to_numpy` copy. On pandas 1.5 to 2.x `fe_append_columns` copies all of X, twice per family (about 0.8 GB per family at 1M x 100, 27 call sites of `fe_decide_on_subsample`).
- polars: `Series.to_numpy` is a read-only view for one chunk without nulls, else a copy; `DataFrame.to_numpy` always copies, F-order, and mixed Int64/Float32 columns come out float64. With FE on, a polars input is already a zero-copy pandas view (`_mrmr_class_fit_helpers.py:208-212`), so `fe_to_pandas` on it is a no-op.
- Codes are already compacted to int8/int16 (`_fit_impl_core_parts.py:303-305`); int32 is a transient. A CodeStore holding int32 would be 4x larger than today.

## Where memory actually goes (bytes per cell, 10% NaN)

Held for the fit: raw frame (4 or 8) + codes 1 + engineered floats (4 to 8 each). Transients: `categorize_dataset` float32 copy 4 (`_discretization_dataset.py:415`; on a mixed polars frame a float64 detour first, 12 at `:411`), NaN mask 1 (`:426`), int32 matrix 4 (`:328`), `np.where` over NaN cells about 16 per NaN cell (`:270`), categorical `concatenate` (`:581`); MDLP edge threads, one per logical core, about 24 B/row each (`_adaptive_nbins.py:654-666`); GPU path widens int8 to int32 on host (4) and device (4) (`_cmi_resident_cache.py:152,168`, `_batch_pair_mi_cuda_kernels.py:670`, `_batch_pair_mi_cuda_shared_fused.py:253`, `friend_graph_gpu.py:392,535`). Estimates: 1M x 100 about 2.2 GB peak; 10M x 80 about 15 GB plus 3.8 GB MDLP threads, and the int32 device matrix (3.2 GB) does not fit the 4 GB GPU.

## Ranked changes (largest effect per effort first)

1. **Bin column by column straight into the int8/int16 matrix** in `categorize_dataset`: bin count is known before allocation (`:319`), NaN code set per column, no global mask, no `np.where`, no int32 matrix and no compaction pass, column-major float32 filled one column at a time (kills the polars float64 detour and the transpose/subset copies at `:126,:152`), edge threads capped by a RAM budget. About -1.1 GB at 1M x 100, -7 to -11 GB at 10M x 80. Reuses `fe_column_blocks`.
2. **Upload codes as int8 and widen on the GPU** (`_pair_mi_perm_cuda.py:149` already does it): device 3.2 to 0.8 GB at 10M x 80, host -3.2 GB, 300 MB less on the device at 1M.
3. **One code-matrix concatenate per FE step** instead of one per family (`_step_score_parts2.py:539,662`, `polynom_pair_fe.py:720`, `_cluster_aggregate.py:697`, `_cat_interactions_step.py:428`). Each append breaks the id-keyed caches and costs an F-order host copy, an int32 host copy and a fresh 400 MB upload; reuse the `st._data_chunks` pattern (`_step_score.py:121`).
4. **`fe_append_columns`: `copy=False`** for `pd.DataFrame(cols, index=...)` and `pd.concat`, plus one merge per step (pandas 2 saves about 0.8 GB per family; no effect, no harm on pandas 3). Done for the composite group-agg builder (chunked, 51 MB extra peak against 240 MB).
5. **Honest null:** no per-permutation int64 aligned matrix on the fallback path (`_binning.py:222-247`: cache the float32 matrix, align as int8); the NaN mask only for columns with NaN (`_honest_spec.py:77`).
6. **Do not hold the raw frame after binning** where recipes can re-read columns on demand (scan or Arrow IPC memmap): removes the 4 to 8 B/cell term. Needs honest null and FE replay to read on demand.
7. **Closed-form engineered columns binned as they are replayed** (`_helpers.py:467-477`) instead of stored as full-n floats: 4 to 8 B/row down to 1 B/row for each.
8. Small: drop the redundant `X[reachable].copy()` in transform (`_mrmr_validate_transform.py:699`), `copy=False` in `:602-607`, `partial_fit` buffer as a list of batches (`_mrmr_partial_fit.py:255`, also keeps it small when pickled).

## Large-n correctness traps (new)

- Integers cast to float32 in `categorize_dataset` (`:411,415`) and nullable Int64 densify (`_inputs.py:169`): ids and epoch times above 2^24 merge into ties; rank target in float32 (`_inputs.py:35`) is not exact above 16.7M rows.
- CUDA kernels use `int n_rows` / int32 `atomicAdd` counts (`_gpu_kernel_sources.py:85-103`): fine at 1B rows per block, streamed totals need int64.
- polars default build caps a frame at 4.29B rows and `with_row_index` is UInt32: the split row id must be (file, row group, offset) as uint64.
- NaN-heavy frames: the about 16 B per NaN cell term (removed by change 1).

## Beyond about 100M rows: subsample-fit mode (deferred, opt-in)

Needed only at 1B rows and up; up to about 100M x 80 the int8 codes are about 8 GB and give the exact full-n selection. Design kept from v3 for later: target-only pass, hash split into A (choose), D (exact counts), C (certify); fit on A unchanged; frozen path certified on C by a G-test with Holm and an effect floor; class-aware weights (not through `_maybe_resample_for_sample_weight`, which does `np.asarray(X)[idx]`); group/time hashing; bias floors rescaled to n. Pass C is bound by parquet decoding on the CPU, not by GPU or PCIe: overlap decoding of the next row group with counting; collect the small three-way (f, s, y) tables for the path and the top runners-up in the same pass so a failed step does not cost another pass. Row-hash sampling cannot skip parquet row groups, so the choose pass reads all bytes. Full details of the statistical protocol are in the v3 text in git history (commit "Plan v3").

Dropped: the float `ColumnArena` wired into stages (kept as an unused, tested utility), streaming sketch edges (edges from a 2M-row subsample already meet the tolerance), uint8 codes before a cast audit, per-scenario RSS gates at 100k/300k (measure 1M and 10M, on pandas 3 and pandas 2).

## Phases

| # | Step | Gate |
|---|------|------|
| 0 | Safety net: golden snapshots captured from a reference commit in a subprocess (two package versions cannot be imported together) via `tests/feature_selection/golden/capture_baseline.py` and `_capture.py`, extended with polars input, `transform`, weights, categoricals, non-Range index, NaNs, pinned `MLFRAME_FE_STAGE_CONCURRENCY` and CPU vs GPU; golden dtype table per FE family; pickle/clone gate incl. after `partial_fit`; peak-RSS measurement script at 1M and 10M on pandas 3 (and 2 in a second environment); hook counting real full-frame conversions; rollback through `_fe_env_switch.py` | snapshots green on HEAD; baseline peak table saved in `profiles/` |
| 1 | Change 1 (bin into int8 column by column, thread cap) | codes equal after explicit cast; selection equal; 1M and 10M peak lower |
| 2 | Change 2 (int8 upload, widen on device) | same MI values; device bytes lower |
| 3 | Change 3 (one code concatenate per FE step) | same selection; GPU uploads per step counted |
| 4 | Changes 4, 5, 8 | peak lower on pandas 2 and 3 |
| 5 | Changes 6, 7 | peak lower; replay equality |
| 6 | Large-n traps (int ids, rank target, int64 totals) | tests with values above 2^24 and counts above 2^31 |
| 7 | Scale runs 1M x 100, 10M x 80 (and a parquet larger than RAM only if subsample-fit mode is built) | fit completes inside the RAM guard; timeline saved |
| 8 | Subsample-fit mode, only if a 1B-row use case is real | calibrated on a synthetic null set |

## Constraints at every step

pandas input keeps today's results; float64 only where a kernel contract or measured selection change requires it; every new path has a CPU twin, a column-block budget and a test against the in-memory path; slow paths log; the four cascades must give the same output whatever the thread timing.

## Phase 0 baseline (2026-11-07, `rss_scale_fit.py ... default`, GPU strict-resident, about 6 GB free RAM)

| size | wall | peak RSS growth |
|---|---|---|
| 1M x 100 | 448.4 s | 3304 MB (8.3x the 400 MB frame) |
| 10M x 80 | aborted at 46.6 s by the 1.5 GB free-RAM guard | 2375 MB (0.7x the 3.2 GB frame) at the abort |

10M x 80 does not get past the start of the fit on this machine: the frame plus its first copies exhaust the free RAM. Items 6 and 7 (do not hold the raw frame after binning; engineered columns as codes) are the ones that address it.

Done from the plan so far: item 1 (codes allocated int8/int16, MDLP edge workers capped by memory), item 2 (codes uploaded at their own width, widened on the device), large-n traps for integers above 2^24 (`categorize_dataset` shift, nullable Int64 densified to float64), the selection snapshot gate. Item 4 dropped (pandas 3 is the target). Item 5 low value (the int64 per-permutation matrix is on the fallback path only). Rank target in float32 above 16.7M rows: relative error about 6e-8 on an auxiliary FE target, left as is.

## 10M x 80 progress (2026-11-08, same machine, `default`, 1.5 GB free-RAM guard)

| change | abort time |
|---|---|
| phase-0 baseline | 46.6 s (degenerate audit: 128 cached unit columns, 5 GB at 10M) |
| degenerate unit cache bounded in bytes | 145.5 s (binned-agg null: all permutations stacked as int64) |
| permutation blocks, then labels gathered per permutation, then packbits | 144 to 154 s (permuted-label block on the host, then the uint32 pack) |
| permutations drawn on the device above 2M rows + packed labels cached across candidate blocks | 505.7 s once, 214 s once (CMIM / binned-agg codes as int64 vectors) |
| CMIM codes and binned-agg source-bin cache at int16, binned-agg device caches bounded in bytes | 229 to 233 s |

The remaining limit is not a spike: the RSS growth climbs from about 0.9 GB to 3.2 GB between 22 s and 31 s while the recipes of the engineered columns are replayed at full n and appended as float columns (`_apply_orth_fourier`, `_apply_hinge_basis`, `_apply_orth_triplet_cross`, `_apply_zscore`, `fe_append_columns`), and that state lives to the end of the fit; the binned-agg family adds about 2.5 GB on top. Item 7 (closed-form engineered columns kept as codes, recomputed from the recipe when a float is needed) is the change that addresses it.

### 10M x 80, continued (2026-11-08)

| change | abort time |
|---|---|
| full-n replayed engineered columns stored at float32 | 241 s (growth by 32 s: 3.19 GB to 2.16 GB) |
| resident-operand cache bounded in device bytes | 291.7 s |
| binned-agg usability: at most jobs + 1 pairs in flight, subsampled rows taken on the device | 393.2 s |
| bootstrap resample rows int32, two-class check without a sort (plus the tie-rule fix) | 456.5 s |

Each spike removed moves the abort by 60 to 100 s; the limit now is the resident growth of about 5.4 GB that the fit holds by 440 s on top of the 3.2 GB frame (16 GB machine, about 6 GB used by other applications). The bootstrap resample tables (about 1.6 GB at 10M rows, 10 resamples) are the last spike seen. The next step is structural: item 7 (engineered columns as codes) and building the resample tables per block of resamples instead of all at once.

### 10M x 80, state after the memory series (2026-11-08)

Abort time stays in 390 to 505 s across the last nine runs whatever spike is removed; the persistent growth is about 5 GB on a 16 GB machine with about 6 GB used by other applications. Measured facts:
- device allocations are not counted in the process RSS on this Windows host (2 GB on the device: +1 MB RSS); cupy's default pinned pool keeps a staging block per pageable upload until `free_all_blocks` (137 MB after a 300k fit);
- a 300000-row fit leaves the process 1.95 GB above its pre-fit RSS: 1.0 GB is the registered stage caches (now dropped at a family start when free RAM is below max(2 GB, 15%)), 0.9 GB is outside them (not identified);
- capping the FE pair-search buffer at 1 GB (`MLFRAME_FE_BUFFER_MAX_GB=1`) did not move the abort (392 s), so the plateau is not that buffer;
- the binned-agg family builds about half of the plateau; its aggregated-column cache and source warm-up were unbounded (fixed); a 1.5 GB step in one second at `resident_operand` remains (about 19 column uploads of 80 MB; no single call above 50 MB at 1M rows).

Next: identify the 0.9 GB held outside the registered caches after a fit (a snapshot diff before and after one fit at 300k), and run 10M x 80 on a host with more free memory to get its wall and true peak.

Resolved (2026-11-08): with a warm-up fit first and the registered caches released plus `gc` after it, one 300000-row fit leaves +98 MB RSS and 17 MB traced: no leak. The "0.9 GB outside the caches" of the earlier measurement was one-time process state (numba code, CUDA context, tuning cache) and uncollected garbage. The 10M x 80 plateau is live working data of the fit; on this machine (16 GB, about 6 GB used by other applications) it reaches the guard. The memory series ends here; a 10M wall and peak need a host with more free memory.
