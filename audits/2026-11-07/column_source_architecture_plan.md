# Plan: block-wise column source, column arena and code store for MRMR

## Why

A fit at 1M x 100 peaks at 3.7 GB and 10M x 80 would need about 35 GB. A frame of a terabyte (mostly polars) cannot be fitted at all today. Measured causes:

- the frame is rebuilt many times: 68 `pd.concat([X, ...])` sites, 42 `fe_append_columns` sites, 15 `fe_to_pandas` full conversions, 529 `.to_numpy(` calls in `filters/`. Every site decides copy and dtype on its own;
- `pd.DataFrame(dict)` in `_composite_group_agg_fe` held the candidate dict next to pandas' stacked copy: 768 MB twice at 100000 rows (fixed, commit "Frame seam");
- `categorize_dataset` materialises a full `(n, p)` int16 code matrix in RAM (`discretization/_discretization_dataset.py:363`, `_maybe_collect_lazy` at the boundary);
- stages share one step-input frame and fold new columns by name (`_fe_stage_merge._fe_merge_new_columns`, `_fe_stage_cascade_run`), so any change to the frame type touches all four cascades.

## Target architecture

1. `ColumnSource` (reading seam, in `filters/_fe_frame_ops.py`): `fe_column_blocks(X, names, budget_bytes, dtype)` yields contiguous `(columns, rows)` blocks, one at a time. Done for pandas and polars. Extended to `pl.LazyFrame` (`scan_parquet`, `select(block).collect()`), never a whole-frame collect.
2. `ColumnArena` (engineered-column store): a preallocated `(capacity, rows)` float32 array, memmap on disk above a byte threshold. Candidates are written once; names map to row indices; `X` is not rebuilt. A pandas/polars view is created only on demand for the few places that still need a frame.
3. `CodeStore` (canonical form for MI): int8/int16 codes per column, filled block by block. Quantile edges come from one streaming pass (sketch, or the subsample the family already uses); the float column is read once. The existing MI kernels already work on codes.
4. Large-frame policy: when the code store does not fit in RAM it is a memmap; kernels read column blocks within the VRAM limit (`_fe_gpu_vram.pool_column_block`); family candidate generation runs on a row subsample only after column and pair streaming is exhausted; the final significance pass is a full block-wise pass.

## Phases (each lands as separate local commits, selection and wall checked before the next)

| # | Step | Main files | Gate |
|---|------|-----------|------|
| 0 | Measure: live-memory profile per stage at 100k/300k/1M (tracemalloc deep trace, `rss_scale_fit.py`) | `_benchmarks/profiling/` | peak and per-site table saved in `profiles/` |
| 1 | `fe_column_blocks` + `fe_n_rows`; `fe_numeric_float32_view` on it (done) | `_fe_frame_ops.py`, `tests/feature_selection/test_fe_column_blocks.py` | block shapes, budget, values equal for pandas and polars |
| 2 | Single-copy frame builders (`_frame_of_columns`, done) applied to the other `pd.DataFrame(dict)` sites found in step 0 | `_composite_group_agg_fe.py` and siblings | peak drops, frames equal, family tests green |
| 3 | `ColumnArena` with the `append(name, col)`, `block(names)`, `as_frame()` surface; unit tests incl. memmap mode | new `filters/_column_arena.py` | arena frame equals `pd.concat` result bitwise |
| 4 | Stage seam: `_fe_merge_new_columns` and `fe_append_columns` write into the arena; stages read through `fe_column_blocks`; pandas/polars frames stay as input only | `_fe_stage_merge.py`, `_fe_frame_ops.py`, the four cascades | same selection as before on the 100k all-families fit; peak lower |
| 5 | Replace the 15 `fe_to_pandas` full conversions by block reads or the numeric float32 view; keep the function only as an explicit fallback with a size guard (`fe_polars_exceeds`) | `_fe_stage_cascade_early_b.py` and siblings | polars-input fit at 300k makes zero `to_pandas` calls |
| 6 | `CodeStore`: `categorize_dataset` writes codes block by block into a preallocated (or memmap) matrix; `LazyFrame` is scanned per block, not collected | `discretization/_discretization_dataset.py`, `discretization/__init__.py` | codes bit-identical to the current matrix on the test sets |
| 7 | Streaming edges: one pass per column block for quantile/MDLP inputs, reusing `_adaptive_nbins` edge cache | `_adaptive_nbins.py`, `discretization/` | edges identical to the in-memory path where deterministic, MI/selection unchanged |
| 8 | Large-scale run: 1M x 100, then 5M x 80 with memmap codes, then a synthetic parquet larger than RAM through `scan_parquet` | `_benchmarks/profiling/rss_scale_fit.py` (+ parquet mode) | fit completes within the RAM guard; cpu/gpu timeline saved |

## Constraints kept at every step

- pandas input behaves exactly as now (selection, recipes, `transform`); float32 is the storage dtype, float64 only where the kernel contract needs it (checked in the float64 audit).
- engineered columns keep their names and recipes; `transform` replay does not read the arena.
- every new path has a CPU twin, a column-block budget and a test against the in-memory path; slow-path gates log.
- no row subsampling is introduced for the final pass; candidate generation subsampling stays as it is.

## Verification

- unit tests per phase (`tests/feature_selection/...`), selection-equality test on a fixed 100k x 100 all-families fit against the previous commit, same process;
- `rss_scale_fit.py` 100k / 300k / 1M with `RSS_TICK_S` for peak RSS and cpu/gpu timeline; tracemalloc deep trace for the per-site table;
- polars-input fit with `pl.DataFrame.to_pandas` hooked to count full conversions (target: 0);
- parquet larger than RAM through `scan_parquet` in phase 8.

## Risks

- the stage cascades run concurrently and share the step-input frame: arena appends need a lock or per-stage private regions merged in the fixed order;
- code-store memmap I/O can starve the GPU: needs a read-ahead block (the same one-worker prefetch used in `_modular_codes_gpu`);
- polars `LazyFrame` streaming engine support differs by version: a block read must fall back to `select(...).collect()` of that block only.
