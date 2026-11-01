# Wave 4 audit 2026-11-02: memory and index safety in native kernels

Scope: 11 numba and CUDA kernels of roughly 90 files, read-only, Opus 5.5 low effort. Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| MEM5-1 | Low | `_cat_target_encoding_and_weighted.py:256-271` `_group_aware_shuffle`: indexes `group_y[g]` with no range check; only caller passes `rng.integers(0, 4, n)` with `n_groups=4` | RESOLVED |
| MEM5-2 | Low/Info | `_cat_kway_materialize.py:114-118, 134-145`: combined codes and `freqs[codes[i]]` assume codes fit the tables | RESOLVED |
| MEM5-3 | Info | `shap_proxied_fs/_shap_proxy_treeshap_gpu.py:56-108`: node ids from the fitted model's arrays | DOC |

MEM5-1 and MEM5-2 are closed with a host-side range check that raises before the kernel runs.
MEM5-3: the tree structure comes from a fitted model (trusted unless a model file from someone else is loaded, which is already unpickling-level trust). Documented.
Checked and guarded: `_strat_clf_kernel`, `_parity_mi_grid_njit`, `cmi_assemble`, the TreeSHAP thread index and 64-bit row offsets, all `np.frombuffer` sites (internal buffers with explicit `count=`).
Not reviewed: the large RawKernel families (`_gpu_resident_select_kernels.py`, `_gpu_resident_basis.py`, `_fe_batched_mi.py`, `_batch_mi_noise_gate_kernels.py`, `_batch_pair_mi_cuda_*`, `_cmi_cuda.py`, `_hermite_fe_mi.py`, `transformer/_kernels_cupy.py`); int32 products such as `i * K` there were not checked. Next wave.
