# Two threads must never be inside one numba `parallel=True` kernel

numba's default threading layer is not safe to enter concurrently from several Python threads. Linux
mostly tolerates it. **macOS aborts the process.**

mlframe's first three-OS CI run (2026-09-08) crashed **92 xdist workers** with
`Fatal Python error: Aborted`. Every faulthandler dump showed two or more pool threads stopped at the same
`parallel=True` call site. No test failed; the workers died underneath them.

## Why this is easy to write by accident

The pattern reads as obviously good:

```python
with ThreadPoolExecutor(max_workers=n) as ex:      # spread columns over threads
    for result in ex.map(_score_one_column, columns):
        ...                                        # each worker calls the fast prange kernel
```

Nothing about it looks wrong, and on Linux nothing about it *behaves* wrong. The exposure here was created
by a correct optimisation: lowering `_PARALLEL_EDGES_MIN_COLS` from 128 to 2 (2026-08-02, measured
1.2x-7.2x) turned concurrent first entry from a width no real frame reached into the normal case.

## What to do

Hold the shared guard around the kernel call:

```python
from mlframe._numba_parallel_guard import parallel_kernel_entry

with parallel_kernel_entry():
    out = some_prange_kernel(...)
```

Guard the **outermost** kernel call, and prefer guarding at a shared dispatcher: one guard there covers
every caller, which is why `discretization.discretize_array` holds it rather than each of its callers.
It is a plain lock, so do not take it re-entrantly on one thread.

Serialising entry costs little. The kernel already parallelises across every core internally, so a second
thread waiting to enter is a thread that had no cores to run on anyway; everything outside the kernel --
sorting, binning, cache lookups -- keeps overlapping, which is where these fan-outs' speedups come from.

## Finding the sites

```bash
python -m mlframe._nested_parallel_scan                       # every fan-out in the package
python -m mlframe._nested_parallel_scan --from per_feature_edges   # one entry point
```

It walks the call graph four hops deep, because the motivating crash was four deep:
`per_feature_edges` -> `edges_fayyad_irani` -> `mdlp_bin_edges` -> `_mdlp_recurse_validated_bfs` -> kernel.
A same-module check sees none of that, which is worth knowing: the first version of the gate below only
looked at direct calls, and would not have caught the bug it was written for.

`tests/test_meta/test_no_unguarded_nested_parallel.py` is the gate that fails a build when a new unguarded
path appears. The scan prints raw paths; the gate additionally honours an allowlist of sites judged unable
to race, each with its reason.

## What the scan found in the first sweep

195 parallel kernels, 6 real thread fan-outs. Of the paths between them:

| Site | Verdict |
|---|---|
| `per_feature_edges` -> mdlp binning | Real, and actively crashing CI. Fixed: warm-up plus the guard. |
| FE pair sweep's double-buffered chunk pipeline | Real but latent (cupy-gated): a producer thread materialises chunk N+1 while the main thread scores chunk N. Four kernels reachable, all now guarded. |
| `_step_pairmi`'s single-worker pool | Not a race: it is a watchdog, and the calling thread blocks on `future.result(timeout=...)`, so one thread runs at a time. Allowlisted with that reason. |
| `_ice_metric` | Not a site at all -- `ThreadPoolExecutor` appears only in a comment explaining that the code is deliberately *not* threaded. |
