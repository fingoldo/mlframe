# Wave 5 audit 2026-11-02: GPU context, device memory and thread/process safety

Scope: cupy and shared-memory code, read-only, Opus 5.5 low effort. Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| GPU5-1 | Low-Medium | `transformer/random_features.py:93-100`: device pool released only after a successful RFF call | RESOLVED |
| GPU5-2 | Low | `transformer/row_attention.py:179-207`: pool and pinned buffers released only after the k-scale loop completes | RESOLVED |
| GPU5-3 | Low | `transformer/_kernels_cupy.py:255-256`: the RFF path never cleared its pinned host buffers | RESOLVED |
| GPU5-4 | Info | `_batch_pair_mi_cuda_shared_fused.py:53, 183-188`: unlocked kernel compile; eager `import cupy` | REJECTED |

GPU5-1 to GPU5-3: the release now runs in a `finally` block on both paths, and the RFF path also clears the pinned buffers. Test: `tests/feature_engineering/transformer/test_gpu_memory_released_on_failure.py`. The docstring of `_get_pinned_buffer` ("grows but never shrinks") is out of date against the 32-entry LRU cap and is noted for a documentation pass.

GPU5-4: two threads compiling the same kernel is idempotent (the last assignment wins) and a RawKernel launch is stateless; importing cupy creates no CUDA context, so fork safety is not affected. Not changed.

Checked and safe: no CUDA context at import; pinned buffers are keyed per thread under a lock with a 32-entry cap; double-buffer streams are synchronised before reuse and on error; `_gpu_upload_fits` flushes and re-probes once, then rejects; shared-memory segments are unlinked on the failed-copy path, named with pid and uuid, and cleaned by the resource tracker at exit.
