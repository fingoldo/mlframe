"""Persistent per-hardware kernel-tuning cache for mlframe GPU dispatchers.

Why
---

The MRMR GPU stack ships several kernel-variant pairs that win on different
size axes (e.g. ``compute_joint_hist_batched_cuda`` global-atomic vs
``compute_joint_hist_batched_shared_cuda`` shared-mem atomic). The
"crossover" between them depends on the live hardware:

  * Shared atomic throughput / SM count varies 10-100x cc 6.x -> cc 9.x
  * cuRAND throughput tracks memory bandwidth
  * Per-block thread budget varies (cc 6.x = 1024; cc 7.x same; cc 8.0+ same)
  * shared memory budget varies (cc 6.x = 48 KB; cc 7.0 = 96 KB; cc 8.0 = 164 KB
    opt-in; cc 9.0 = 228 KB)

Hardcoded thresholds in the source code can be wrong on a different GPU.
This package keeps a per-host JSON cache of empirically-measured best
``(kernel_variant, block_size)`` decisions per ``(n_samples_bucket,
joint_size_bucket)``, so the dispatcher learns the right answer on first
run and uses it forever after.

Public API (lazy import everything; no side effects at module import)
---------------------------------------------------------------------

* :func:`hw_fingerprint` -- short, stable per-host key
* :func:`get_dispatch_table` -- returns the cached dict for a kernel name;
  triggers an auto-tune sweep on cache miss
* :func:`ensure_cache` -- explicit pre-warm hook (called from
  ``prewarm_fs_cupy_kernels`` so production callers get the auto-tune
  paid up-front instead of inside the first ``MRMR.fit``)

The cache lives under ``$MLFRAME_KERNEL_CACHE_DIR`` (default
``~/.mlframe/kernel_tuning/``). One JSON per ``hw_fingerprint``. Per-machine
re-runs (e.g. driver update) require deleting the file; the auto-tune is
idempotent.

Compatible-with-feedback rules
------------------------------

* ``feedback_keep_all_kernel_versions`` -- both kernel variants stay in the
  source; this module picks between them at runtime.
* ``feedback_fastest_default_with_dispatch`` -- the cache IS the dispatcher
  decision; production calls always go through it; opt-out via
  ``force_kernel=`` for tests.
"""
from __future__ import annotations
