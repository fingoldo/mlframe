# `_orthogonal_ksg_mi_fe` (KSG mutual-information scorer of the orthogonal-basis candidates) with its carved children `_ksg_cd_fast` (numba) and `_ksg_cd_gpu` (CUDA)

2026-11-16. The scorer is the Ross continuous-versus-discrete k-nearest-neighbour MI (the sklearn `mutual_info_classif` estimator): per class the k-th neighbour distance of every point, then the count of points of all classes closer than that distance. `_ksg_cd_fast` holds the numba kernel `_mi_cd_columns` (parallel over columns) and the batched entry `ksg_mi_discrete_target_columns` (scaling, the per-column Kraskov noise, then the host/device choice); `_ksg_cd_gpu` holds the CUDA kernel `ksg_psi_m_sums` and the kernel-tuning entry that picks the backend by row count (`fe_ksg_cd_device`, fallback `DEVICE_FALLBACK_ROWS`).

This pass: the kernel source of `_ksg_cd_gpu` was a bare `double*` kernel (the host kernel works on noised 64-bit columns and the distance comparisons must match it exactly); it is now typed by `{REAL}` / `{ACC}` and instantiated at 64 bits (`render(_SRC, real=np.float64)`), same arithmetic, `kernel(cp, real)` is the compile point of the register-pressure registry and of the offline NVRTC->ptxas test (sm_61/86/120), so the untemplated-pointer review marker is gone.

## cProfile and timings (100 columns x 100000 rows, one class pair, k = 3, warm)

`ksg_mi_discrete_target_columns` end to end: 0.286 s on either route when the cache picks the device. The kernels alone: host numba `_mi_cd_columns` 0.352 s, device `mi_cd_columns_device` 0.142 s (2.5x); device and host agree to 7.8e-12 (the device sum of psi is formed in a fixed block order). cProfile of the end-to-end call: 100 `cupy.argsort` calls 0.099 s (the per-column device sort), the host preparation of the batch (copy to 64 bits, std, scaling, noise) 0.068 + 0.04 + 0.02 s, the device read 0.025 s.

## nvprof (GTX 1050 Ti, sm_61), 5 device calls

Kernel launches per call: 1 `ksg_psi_m_sums` plus about 800 radix-sort launches (100 columns x 8 passes), 100 `cupy_take` and 1 transpose. Per call: `ksg_psi_m_sums` 14.8 ms in 1 launch (18 % of the device time); the radix sorts of the columns 36 ms in 800 launches (43 %: one launch group per column, KS11-02 measured and rejected a launch-reduction attempt); the upload 12.8 ms (15 %); `cupy_take` 5 ms; `transpose_f64` 1.7 ms. Kernel metrics: achieved occupancy 0.996, multiprocessor activity 99.9 %, global load efficiency 53.8 % (a thread walks outward from its point through sorted memory: neighbouring threads read neighbouring addresses, the efficiency is the stride of the class-skipping walk), device read throughput 7.5 GB/s of the 112 GB/s card: latency bound by the dependent neighbour walk, not by bandwidth. Whole-GPU utilization of the call: the device is busy about 60 % of the 0.14 s; the sort is the lever (KS11-02), not the kernel.

## Equivalence

80 KSG tests (`test_ksg*`, host/device equality, discrete-target batches, the orthogonal KSG scorer) green after the typing change.
