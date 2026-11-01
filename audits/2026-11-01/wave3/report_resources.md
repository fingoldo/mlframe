# Wave 3 audit 2026-11-01: resource lifecycle and leaks

Scope: `src/mlframe` (read-only). Class: files, temp directories, processes and thread pools, shared memory, GPU memory pools and unbounded
module-level caches that an attacker or a long-running service could use to exhaust memory, disk, handles or GPU memory.

The auditor could not write this file itself; the findings below are its returned text. It was a grep sweep with targeted reads, so it is
shallow: `_GPU_POOL` in `feature_selection/filters/gpu.py:100` and `_CUDA_PROBE_CACHE` were not read, and only the wave-2 tracker was read
before auditing, so an item may already be recorded.

## Summary

8 findings: 0 High, 0 Medium, 1 Low, 7 Info.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| RES4-1 | Low | `estimators/pipelines.py:159` | FUTURE |
| RES4-2 | Info | `feature_selection/filters/_adaptive_nbins.py:632-649` | DOC |
| RES4-3 | Info | `_recipe_unary_binary.py:28, 38`, `_offset_product_fe.py:62` | DOC |
| RES4-4 | Info | `reporting/_shared_arrays.py:71-153` | RESOLVED |
| RES4-5 | Info | `reporting/_async_render.py:600-635` | RESOLVED |
| RES4-6 | Info | whole package (no `Popen`, no listening sockets) | REJECTED |
| RES4-7 | Info | cupy pools and the resident operand cache | RESOLVED |
| RES4-8 | Info | whole package (no `lru_cache(maxsize=None)`) | REJECTED |

## Findings

### RES4-1 CV dump directory is never removed (Low, FUTURE)
`tempfile.mkdtemp("mlframe-cv-")` creates the directory that receives the gridsearch CV dumps and nothing removes it. Any caller of
`optimize_pipeline_by_gridsearch` that omits `output_dir` leaves a directory behind per call. This is the private directory introduced for
audit item FS-02; the leak is the cost of keeping the dump replayable by path.

### RES4-2 `_EDGE_POOL` is never explicitly shut down (Info, DOC)
`_adaptive_nbins.py:632-649`: the global thread pool is bounded by its worker count and lives for the process.

### RES4-3 `functools.cache` keyed by a short preset string (Info, DOC)
`_recipe_unary_binary.py:28, 38` and `_offset_product_fe.py:62`: the cache key is a short preset name, so the cache is bounded by the number of presets.

### RES4-4, RES4-5, RES4-7 Release paths confirmed (Info, RESOLVED)
Shared memory is released and unlinked correctly (`_shared_arrays.py:71-153`); the async render executors are shut down
(`_async_render.py:600-635`); cupy pools are freed and the resident operand cache is capped.

### RES4-6, RES4-8 No such constructs (Info, REJECTED)
No `Popen` and no listening sockets in the package, and no `lru_cache(maxsize=None)`.
