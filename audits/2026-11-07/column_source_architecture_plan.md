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
