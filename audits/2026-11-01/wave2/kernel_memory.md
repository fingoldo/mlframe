# Wave 2 audit: memory safety of kernels (MEM3)

Read-only pass over `src/mlframe`: numba `@njit` / `cuda.jit` kernels, `cupy.RawKernel` source strings, and `shared_mem=` launch sizing. No file was edited and no kernel was executed. Earlier rounds (2026-10-31, 2026-11-01 trackers) cover deserialization, exec, subprocess, network, filesystem, secrets, supply chain, untrusted input, concurrency and ReDoS; none of their IDs are re-reported here.

Coverage limit: the 354-file njit/cuda inventory was not read line by line. The pass traced every CUDA shared-memory launch in the GPU MI, CMI, moments, and DTW paths, every `RawKernel` string-interpolation site, and the GPU index-arithmetic paths in `signal/dtw.py` and `_fe_batched_mi*.py`. The njit-only integer-overflow sweep was sampled, not exhaustive.

## Findings

| ID | Severity | Location | Index or size expression | Input that exceeds the bound | Proposed disposition |
|---|---|---|---|---|---|
| MEM3-1 | Medium | `src/mlframe/signal/dtw.py:210, 217, 220` (`dtw_diagonal`); allocation at `:404` | `int stride = m + 1; cost[i * stride + j]`, all `int` (32-bit). Buffer `cp.full((n + 1, m + 1), 1e18)` float32, no size check. | Once `(n+1)*(m+1) >= 2^31` (about 8.6 GB of float32), `i*stride+j` wraps to a negative or out-of-range index in C. Those writes land outside the buffer. `n` and `m` are `len(x)` and `len(y)`, the caller's series lengths, so roughly 46k x 46k on a 24 GB card reaches it. The full-matrix path has no guard, and no `int64` index is used. | FUTURE: add a guard `(n+1)*(m+1) < 2**31` that raises or routes to the banded path, or switch the index to `long long` in the kernel. |
| MEM3-2 | Low | `src/mlframe/signal/dtw.py:246-254` (`dtw_banded_diagonal`); allocation `:351`, launch `:365` | `band[(i - 1) * bw + b]`, `band[i * bw + b]` with `int` indices, `bw = 2*w + 1`, buffer `(n+1, bw)`. | The same 2^31 wrap, reached when `(n+1)*(2w+1)` is large, e.g. a large user `window`. The band buffer is smaller than the full matrix, so this needs a bigger `n` or `w` than MEM3-1. | FUTURE: same guard or `long long` index as MEM3-1. |
| MEM3-3 | Low | `src/mlframe/feature_selection/filters/_fe_batched_mi.py:529-531` | `gcounts = cp.zeros(K * M, dtype=cp.int32)` with `M = nbins * Ky`; kernel index is 32-bit. | Needs `K * nbins * Ky >= 2^31`, which is more than 8 GB of counts. Not reachable under current `nbins` and `Ky` bounds (`_MI_FROM_CODES_MAX_SHARED` gate at `:496`). | REJECTED: unreachable in practice. The size is bounded by the shared-memory gate and the column count. Keep an eye on it if `K` grows past about 10^5. |
| MEM3-4 | Info | `src/mlframe/feature_selection/filters/_cell_moments_kernel.py:19, 90, 105, 110` | `shared_mem=2 * nc * 8` and `3 * nc * 8`, `nc <= MAX_CELLS = 2048`. | At `nc = 2048`, the `3*nc*8` request is 49152 B, exactly the 48 KB default with zero headroom. A device or driver with a smaller per-block limit would fail the launch. | RESOLVED: `nc > MAX_CELLS` returns `None` at `:90`, so the caller takes the scatter-add form. Headroom is thin but correct. |
| MEM3-5 | Info | `src/mlframe/feature_selection/filters/_batch_pair_mi_cuda_kernels.py:178-181, 293-320` | Static `shared.array(shape=(MAX_JOINT_BINS_CUDA, MAX_Y_BINS_CUDA))`; `cls_x = va*nb_b + vb` used as the row index. | `nbins[a]*nbins[b] > MAX_JOINT_BINS_CUDA` or `n_classes_y > MAX_Y_BINS_CUDA` would write past the static array. | RESOLVED: the caps are enforced before launch (error at `:798` for `n_classes_y`, `:817` for `max_joint`). The row-chunked variant at `:293-344` sizes its dynamic buffer from `max_joint*n_classes_y` and is gated by `_hist_kernel_shared_fits_budget`. |
| MEM3-6 | Info | `src/mlframe/feature_selection/filters/_fe_batched_mi.py:386, 420, 496, 531, 538`; `_fe_batched_mi_mm_cm.py:59` | `shared_mem=int(nbins) * Ky * 4` and `Kx * Ky * 4`. | `nbins * Ky * 4 > 44000` would exceed the per-block limit. | RESOLVED: each launch is preceded by `if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED: return None` (`:386`, `:496`, `_fe_batched_mi_mm_cm.py:59`) and `Kx*Ky*4` at `:673`. |
| MEM3-7 | Info | `src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py:290`; `_fe_batched_mi_cmi.py:473-486, 663-674, 747-757, 820-829`; `_hermite_fe_mi.py:652`; `_gpu_resident_basis.py:181-608`; `_fe_sufficient_summary_pairs.py:312`; `_family_maxt_cuda_bits.py:118`; `_shap_proxy_gpu.py:95` (about 25 `RawKernel` string-replace sites) | `src.replace("{REAL}", real_t)`, `.replace("{ACC}", ...)`. | Interpolated value is `real_t` from a fixed set (`"float"` / `"double"`), or `_ACC_TYPE` a constant. Not from data or config. | REJECTED: no untrusted value reaches the source template. Recorded for completeness; re-open if any template argument is ever built from config. |
| MEM3-8 | Info | `src/mlframe/feature_selection/filters/_fe_batched_mi_cmi.py:663-682, 747-757, 820-829` | `M = prod(cards)`, guarded by `M*4 + 4096 <= _JOINT_ENTROPY_SH_LIMIT`; fused kernels size block 0-2 histograms from `Kx`, `Ky`, `Kz`. | Joint cell count above the device limit. | RESOLVED: each fused path returns `None` (`:748`, `:821`) and the caller falls back to per-term launches. The `Mmax*4` shared request matches the gate. |

## Not found

- No `cp.RawKernel` source was built by f-string interpolation of data or config values; all `.replace` placeholders take constants.
- No `cuda.shared.array` with a size taken from input data and without a gate. Every data-sized dynamic `shared_mem` was checked against a gate.
- The `_batch_mi_noise_gate_kernels.py` dynamic shared buffers (`:579, 631, 692-694`) sit behind `_cuda_shared_mem_per_block()` at `:572`, not separately read.

## Disposition summary

| Disposition | Count |
|---|---|
| FUTURE | 2 (MEM3-1 Medium, MEM3-2 Low) |
| REJECTED | 2 (MEM3-3, MEM3-7) |
| RESOLVED | 4 (MEM3-4, MEM3-5, MEM3-6, MEM3-8) |
| DOC | 0 |

Severity counts: 0 high, 1 medium, 2 low, 5 info.
