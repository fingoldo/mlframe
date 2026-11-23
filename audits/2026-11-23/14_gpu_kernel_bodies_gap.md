# Audit 14: MRMR GPU kernel bodies (gap audit, Haiku, read-only, nothing executed)

Scope: CUDA source strings (cupy RawKernel) and `@_nb_cuda.jit` bodies in `_batch_pair_mi_cuda_kernels.py`, `_cmi_cuda.py`, `_cmim_min_gpu.py`, `_fe_batched_mi_cmi.py` (+ `_fe_batched_mi_cmi_hist.py`), `_family_maxt_cuda_bits.py`. `_rff_gram_energy_gpu.py` has no kernel (host-side cupy). Kernels outside the listed files not checked.

Tracker rows: AUD23-65 to AUD23-70.

Headline: no confirmed int32 overflow reachable within the host guards, no unguarded race, no non-determinism across identical launches. Every shared-memory histogram write goes through atomicAdd / atomic.add; integer atomics are order-independent.

## Findings

1. Shared-memory overrun for the marginals, `_cmim_min_gpu.py:196` and `:228`. Severity low. Confirmed.
   - The guard checks only `kx*ky*kz_max <= MAX_CELLS (6144)`. The launch size at :228 is `4*(kx*ky*kz_max + kx*kz_max + ky*kz_max + kz_max)`; the marginals are not bounded.
   - Example: kx=ky=1, kz_max=6144: 4*24576 = 98,304 B, above the 48 KB per-block limit. The launch fails; the broad `except` at :238 returns None. Effect: silent fall back to the slower path, not corruption.

2. Docstring cap mismatch, `_batch_pair_mi_cuda_kernels.py:788`. Informational. Confirmed.
   - Says `MAX_JOINT_BINS_CUDA = 128`, but `_derive_max_joint_bins` (:131-139) yields 256 on a 48 KB device. Static `sm_hist` (:187-190) is (256,16) int64 plus 256 int64 = 34,816 B, which fits. Host guards at :822 (`n_classes_y <= 16`) and :836-841 (`nb_a*nb_b <= MAX_JOINT`) enforce the caps. No overflow.

3. Dynamic-shared sizing, `_batch_pair_mi_cuda_kernels.py:327-348`. Confirmed safe. Layout `max_joint*n_classes_y + max_joint` int32 matches the gate at :377; cell index stays small because the gate caps it.

4. Index width in the CMI kernel, `info_theory/_cmi_cuda.py:195, 208, 213`. Confirmed safe under guard. `joint_size` and `cy*(nbx*nbz)` are int32; `_cmi_cuda_shmem_fits` (:636-643) caps `joint_size*4 <= 48 KB`, so the index is at most about 12k. Host `n` passed as `np.int32(n)` (:455); overflow only at n >= 2^31, made implausible by the VRAM guard (:652-658). Hypothesis, very low.

5. Index truncation, `_fe_batched_mi_cmi.py:633-636`. Confirmed safe under guard. Product in `long long` then cast to `int`; safe only because `cmi_joint_entropies_gpu` (:661-663) requires `Kx*Ky*Kz*4 + 4096 <= SH_LIMIT`.

6. Float reduction order depends on thread count. Low. Deterministic per launch config; differs across configs.
   - `_cmim_min_gpu.py:58-85, 93-100`: per-thread strided partials plus warp shuffle, then serial sum over warps; order depends on `blockDim` chosen by `block_size_for` per device.
   - `info_theory/_cmi_cuda.py:264-277`: tree reduction over `red_threads` (power of two, :472); order depends on occupancy.
   - `_fe_batched_mi_cmi.py:93, 128, 154, 507, 523, 540, 644`: tree reductions; fixed 256 for the column kernel (:387-389), power-of-two `_tree_block` elsewhere (`_fe_batched_mi_cmi_hist.py:276-278`).
   - Results can differ at roughly 1e-14 to 1e-16 relative between GPUs or block widths. The `clipped < best` min at `_cmim_min_gpu.py:102` can flip on ties. Docstrings claiming "bit-identical" (`_fe_batched_mi_cmi.py:490-491`, `_cmi_cuda.py:228-229`) are not true across block widths.

7. Silent row drop if the tree width is not a power of two. Confirmed safe at current call sites. `_fe_batched_mi_cmi.py:93` uses `s = nt>>1`, dropping elements for non-power-of-two `nt`. Call sites use fixed 256 or `_tree_block` (pow2=True). `sh_h[256]` requires `nt <= 256`. Both hold today.

8. `rstride` zero. Hypothesis, low. `_fe_batched_mi_cmi.py:113`: if `blockDim < g` (g <= 8), `rstride = 0` and no rows are counted, with no error. Reachable only with block width below 8.

9. Out-of-range codes spill into the next column's histogram. Hypothesis, low. `_fe_batched_mi_cmi.py:85, 119, 148`, `_batch_pair_mi_cuda_kernels.py:206`, `_cmi_cuda.py:208` index `j*M + x*Kb + b` with no bounds check in the kernel; correctness relies on the host guaranteeing `x < Kx` and `b < Kb`. Host side not checked.

## Checked and clean
- `_family_maxt_cuda_bits.py`: v2 shared size guarded by `_v2_applies` (:310-319); v3 writes guarded by `b0+p < B` (:213); `__syncthreads` precedes the early return (:191-192); `sh_tot` zeroed before the first barrier (:154, :164); all `sh_tot` updates atomic (:173).
- Race check: every shared histogram update is atomic; zeroing loops write disjoint ranges before a barrier.
