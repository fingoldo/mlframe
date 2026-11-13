# `batch_mi_noise_gate_gpu` and its carved modules (histogram permutation noise gate)

The family: `batch_mi_noise_gate_gpu` (dispatcher, resident and cupy entry points), `_batch_mi_noise_gate_kernels` (numba.cuda kernel factories, the njit reducers and gate rule), `_batch_mi_noise_gate_fused`, `_batch_mi_noise_gate_gpu_helpers`, `_batch_mi_noise_gate_column_tiles`, `_batch_mi_noise_gate_screen_v1`, `_batch_mi_noise_gate_tuning` (carved out of the parent for the file-size ceiling; judged as one unit, `CARVED_FAMILIES`). The bit-set backend `_batch_mi_noise_gate_bits` has its own record and is the default wherever its kernel applies.

Measured 2026-11-14, GTX 1050 Ti (sm_61, 112 GB/s). Shape: 20000 rows x 4000 int8 columns of 16 bins, 10 classes, 49 permutations, gate on the device, codes resident (`d_disc_resident`), backend forced to the histogram pair (`force_noise_gate_backend("hist")`).

## Launches per call and the GPU profile (nvprof, 3 calls)

Launches per call: 1 histogram kernel (all permutations in one launch), 1 MI-from-counts reduction, 1 `transpose_i8` of the codes to column-major, plus a few small cupy and cub operations; no launch per column or per permutation. GPU time per call 315 ms: histogram kernel 260 ms (82.5 %), MI reduction 38 ms (12.2 %), transpose 11.8 ms (3.8 %), uploads 4.6 ms. Against the bit-set gate on the same data (80 ms per call, record `_batch_mi_noise_gate_bits.md`) the histogram pair is 4x slower, which is why the backend entry sends every shape the bit-set kernel can take to it.

Metrics (nvprof): histogram kernel achieved occupancy 0.74, multiprocessor activity 99 %, device-memory read throughput 11.6 to 12.6 GB/s of 112, write 0.96 to 1.0 GB/s, global load efficiency 99 %, store efficiency 87 %, shared-memory efficiency 0.23 % (the bin counters are shared-memory atomics: bank conflicts and serialisation on the hot bins are the limiter, not memory); MI reduction kernel occupancy 0.49, activity 98 %, read 10 to 11 GB/s, load efficiency 25 % (it reads the counts table at a stride; 12 % of the path, left because the path is the fallback).

## Host profile (cProfile, 3 calls)

0.974 s for 3 calls (0.325 s each), 11677 function calls: 0.956 s is `safe_cuda_api_call` in numba's driver layer (194 calls: the blocking waits on the kernels above plus the argument preparation of the numba.cuda launches, 0.81 s of it in `from_cuda_array_interface`, which is the cost of handing cupy arrays to numba kernels); the njit gate rule 1 ms.

## Backend choice, streaming, host twin

The bit-set gate against the histogram pair is the tuner entry `noise_gate_bits` (swept over rows x columns x classes x bins), the CPU against the device backends is `_batch_mi_noise_gate_tuning`'s sweep; the host twin is the njit `batch_mi_with_noise_gate` this family mirrors (and the njit reducers in `_batch_mi_noise_gate_kernels`). Column blocks: the resident gate tiles its columns when the counts buffer does not fit the VRAM cushion (`_batch_mi_noise_gate_column_tiles`, equality of the tiled and untiled results pinned by its tests).
