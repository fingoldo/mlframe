# `_orth_extra_basis_fe_generate` (the extra-basis generator: spline, Fourier with the adaptive and chirp grids, wavelet) and its carved `_orth_extra_basis_fe_generate_doc`

The generator is the entry point of the extra-basis stage (`generate_extra_basis_features`, `hybrid_orth_extra_basis_fe_with_recipes`); its fits, njit coarse-basis kernels and helpers are `_orth_extra_basis_fe` and `_orth_extra_basis_fe_helpers` (record `_orth_extra_basis_fe.md`), its device-born path is `_extra_basis_resident` (record `_extra_basis_resident.md`). The measurement below is of this generator.

## 2026-11-16: the extra-basis stage at 100000 rows, host and device profile (family `_orth_extra_basis_fe`, `_orth_extra_basis_fe_helpers`, with `_orth_extra_basis_fe_generate`)

Host: GTX 1050 Ti (sm_61, 112 GB/s), 8 logical cores, GPU 78 C and SM clock 1695 of 1911 MHz at the start (no throttle bit). One `generate_extra_basis_features` call: 12 float64 source columns (4 periodic, 4 wide-range, 4 positive), bases spline + fourier (adaptive grid, chirp) + wavelet, a target that excites all of them; 216 columns emitted on every path.

| path | wall per call |
|---|---|
| host (`MLFRAME_FE_GPU_STRICT=0`) | 1.60 s |
| AUTO with the fit state of a 100000 x 100 fit | 0.70 s |
| STRICT-GPU | 0.69 s |

cProfile: on the host path 1.45 s of 1.62 s is the main thread waiting on the column pool (`_thread.lock.acquire`, 200 waits: the njit kernels run on the worker threads, which cProfile does not see); pandas block merging 0.07 s and 26 `_stack_arrays` 0.03 s are the output assembly. On the device path 0.49 s is the same wait (308 waits) with the `.get()` of the emitted matrix 0.04 s over 14 calls.

Launches per call and nvprof (STRICT, one call): `periodogram_power` 20.0 % of the GPU time (56 launches of 457 us), the double-precision `gemv2T` 10.2 % (24 launches), `mi_from_codes_tile` 7.6 % (24 launches of 407 us), `dgemm_largek` 6.3 % (12 launches), the host-to-device copies 4.7 % (156 copies), `leg_codes` 3.2 % (12 launches of 341 us). Metrics (nvprof, 20000-row run): `periodogram_power` achieved occupancy 0.145 on average (0.044 to 0.369), DRAM read 0.59 GB/s, multiprocessor activity 77 % - a latency / launch-shape bound kernel, not a bandwidth bound one, and the cheapest remaining lever of this stage (about 25 ms of a 128 ms GPU call at this size); `periodogram_power_b` occupancy 0.44, activity 97 %; `leg_codes` occupancy 0.67, write throughput 57 GB/s of 112 GB/s.

Backend choice and memory: STRICT / AUTO by fit size route the stage to the device-born path (`_extra_basis_resident`), whose host-versus-device choice and column-block streaming are those of that module's record; the host path emits one fit and one basis matrix per source column and basis.
