# Wave 5 audit 2026-11-02: integer overflow and unchecked indexing in CUDA raw kernels

Scope: cupy RawKernel and numba.cuda kernels of the filters and transformer packages, read-only, Opus 5.5 low effort. Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| OVF5-1 | High | `feature_engineering/transformer/_kernels_cupy.py` `row_attention_stage4_fused`: `q * K`, `q * head_dim`, `nid * head_dim` as int products | RESOLVED |
| OVF5-2 | Low | same kernel: no host check that K fits one block (1024 threads) | RESOLVED |
| OVF5-3 | Low | `_batch_pair_mi_cuda_shared_fused.py:125-130`: warp shuffle assumes `threads_per_block` is a multiple of 32 | RESOLVED |
| OVF5-4 | Info | fixed `__shared__ double sh_h[256]` arrays indexed by `tid` (`_fe_batched_mi_cmi.py`, `_gpu_resident_basis.py`) | DOC |
| OVF5-5 | Info | int histogram counters, safe below 2^31 rows | REJECTED |

OVF5-1: the products wrapped once rows times head_dim passed 2^31 (about 3.4e7 queries at head_dim 64) and read or wrote out of bounds with no error. All offsets are now `long long`; the launcher refuses K above 1024 (OVF5-2). Test: `tests/feature_engineering/transformer/test_kernels_cupy_softmax_temp_zero_guard.py::test_stage4_refuses_a_top_k_above_the_block_limit`. The 2^31-element wrap itself cannot be exercised on a 4 GB card; the existing row-attention parity tests cover the unchanged results.

OVF5-3: the agent missed that `batch_pair_mi_cuda_shared_fused` already raises for a `threads_per_block` that is not a multiple of 32 in [32, 1024] (`_batch_pair_mi_cuda_shared_fused.py:219`).

OVF5-4: the launchers hard-code 256 threads; documented, no code change.

Checked and guarded: `_hermite_fe_mi.py` evaluators, `_gpu_resident_select_kernels.py`, `_gpu_resident_basis.py`, `_fe_batched_mi.py`, `_fe_batched_mi_cmi.py`, `_cmi_cuda.py`, `_batch_pair_mi_cuda_shared_fused.py` (`validate_pair_codes`), `_batch_mi_noise_gate_kernels.py`.

Not read: host launchers of `_gpu_resident_extval*`, `_gpu_resident_*_ktc`, `_fe_batched_mi_guards`.
