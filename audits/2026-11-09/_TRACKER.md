# MRMR GPU-path tracker (2026-11-09)

Statuses as in the other trackers: RESOLVED, PARTIAL, TODO, REJECTED, FUTURE (deferred with a trigger).

| ID | Item | Status | Note |
|---|---|---|---|
| GPU11-1 | CUDA graphs for chains of elementwise/raw kernels (Fourier detector phases, resident-MI candidate chains) | FUTURE | cupy launch path is about 20k ops/s per process, thread independent; graph capture works for elementwise/raw chains and fails for cuBLAS calls (einsum, matmul, solve). Trigger: after cross-family kernel fusion (GPU11-2) is exhausted, if the cupy op counter still shows launch-bound chains. Needs preallocated fixed-shape buffers and one graph per shape. |
| GPU11-2 | Fuse the per-family kernels that read the same columns into shared kernels / shared resident operand passes | TODO | Check what `_fe_resident_operands` and the shared leg prefetch already cover before adding. |
