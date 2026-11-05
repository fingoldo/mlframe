# nsys, all families, 100k x 100, warm fit (2026-11-05)

Command: nsys 2022.4.2 `profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx` around the second fit of `D:/Temp/nsys_fit.py` (one process). Reports: `gpukernsum`, `cudaapisum` (this nsys has no `cuda_gpu_kern_sum` names).

GPU kernels (top, share of kernel time): `tied_replicate_mi` 9.8 % (4 launches, 227 ms avg), `batched_joint_hist2_i32` 6.9 % (642 launches, 1 ms), `triple_mi` 5.1 % (1 launch, 469 ms), `lof_scores` 3.8 %, `null_mi_perms_bits` 2.9 %, `fold_cell_centred` 2.1 % (348 launches), `radix_select_interp_f64_v3` 1.8 %, `ent_nnz_1d` 1.8 % (415 launches); the rest is a long tail of kernels under 1.5 % each. Total kernel time is about 9 s of a ~90 s fit, so the device is idle most of the wall.

CUDA API (share of API time): `cudaMemcpyAsync` 42.9 % (6.8 s, 13.8k calls, max 469 ms: host-blocking readbacks), `cuLaunchKernel` 20.3 % (3.2 s over 69.9k launches, 46 us each), `cudaEventDestroy` 10.0 % (1.6 s over 816 calls, max 190 ms), `cudaMalloc` 7.4 % (1.2 s, 700 calls), `cudaLaunchKernel` 6.1 %, `cudaStreamSynchronize` 5.4 % (0.85 s), `cudaFree` 2.2 %.

Reading: the fit is launch- and readback-bound, not kernel-bound. Levers by size: (1) host readbacks (`cudaMemcpyAsync` 6.8 s) - batch them or keep results resident; (2) ~90k launches at ~46 us - fuse small kernels (`ent_nnz_1d`, `fold_cell_*`, `batched_joint_hist2_i32`); (3) `cudaEventDestroy`/`cudaMalloc` 2.8 s - reuse events and pool buffers. The single heavy kernels (`tied_replicate_mi`, `triple_mi`, `lof_scores`) total under 2 s.
