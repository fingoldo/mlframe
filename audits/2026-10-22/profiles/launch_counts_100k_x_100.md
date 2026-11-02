# Launch counts of our compiled kernels in a 100000 x 100 every-family fit (2026-11-02)

Method: `D:/Temp/launch_count.py` (a Python subclass of `cupy.RawKernel` plus a counting proxy around `RawModule.get_function`, installed before mlframe imports; warm fit at 20000 x 30 first, then one counted fit at 100000 x 100; wall 145.9 s). It counts only compiled `RawKernel` / `RawModule` kernels: cupy's own elementwise kernels, ufuncs, sorts and reductions are cdef types whose `__call__` cannot be patched, and nsys exports of the whole fit fail on this host (see memory `nsys-export-disk-limits`).

Result: **15,856 launches of our own kernels in the whole fit**, at roughly 10 to 30 us of launch cost each that is under 0.5 s of the 146 s wall. Our compiled kernels are NOT the launch-bound part. The 14 million events nsys recorded for the same fit are therefore cupy's built-in ops (elementwise, sort, reduction, memset, copies), which is where per-column device loops and small temporaries live.

Top kernels: `radix_select_f32_v3` 1021, `radix_interp` 1021, `ent_nnz_1d` 902, `mi_hist_split_f32` 879, `mi_from_counts_col_f32` 879, `cmi_joint_entropies` 864, `periodogram_power` 760, `joint_hist_nnz2` 654, `joint_hist_nnz3` 643, `mi_hist_split` 530, `build_op_cands_cm` 480.

Top callers (launches): `_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident` 3024, `_hermite_fe_mi._resident_mi_candidates` 2411, `_fe_batched_mi_cmi.joint_entropy_gpu` 1804, `_mi_greedy_cmi_fe_gpu._nc` 1308, `_mi_greedy_cmi_fe_gpu._cmi_from_binned_cupy` 1047, `hermite_fe._als_fused_gpu.als_sweep_fused` 742, `_binned_numeric_agg_resident._pair_fold_stat_tables` 696.

Next: find where the cupy built-in ops are issued (a `sys.setprofile` or `cupy.cuda.runtime` hook, or an nsys window of one stage with `--capture-range=cudaProfilerApi`), then replace per-column elementwise sequences (like the one batched in `_qbin_block_gpu`) by one block operation. `joint_entropy_gpu` (1804 launches) and `_nc` / `joint_nnz_gpu` (1308 plus an allocation each) are the first of our own call sites worth batching per candidate block.
