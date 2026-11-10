# `_fe_batched_mi_cmi.py` — byte-code joint-histogram kernels for the batched CMI

Measured 2026-11-10, GTX 1050 Ti, `D:/Temp/cmi_bench.py` (one `batched_cmi_gpu` call, y 2 classes, z 10 classes, x 10 bins; 5-call mean).

| input | n x K | per call |
|---|---|---:|
| int64 matrix (before) | 30000 x 1000 | 14.0 ms |
| uint8 matrix, global atomics above 128 cells | 30000 x 1000 | 9.3 ms |
| uint8 matrix, shared histograms up to 256 cells | 30000 x 1000 | 7.7 ms |
| int64 -> uint8 | 100000 x 400 | 9.7 -> 9.2 ms |

Per-kernel (`nvprof`, 30 calls at 30000 x 1000 bytes): the (x,y,z) joint with 200 cells went through global atomics, 5.17 ms; the (x,z) joint through the shared-memory kernel, 2.76 ms; the entropy reduces 0.05 ms each. The int64 candidate matrix cost 8 bytes a code per pass; the CMIM scorer calls the CMI about 600 times on the same candidate block, so it now keeps the block as `uint8` (codes < 256) and `batched_cmi_gpu` leaves a byte matrix as bytes. New kernels `batched_joint_hist2_i32_u8`, `batched_joint_hist2_i32_smem_u8`, `batched_joint_hist1_i32_u8` (int32 counts as before); the shared-histogram band rises to 256 cells for bytes (`_HIST_SMEM_MAX_CELLS_U8`, the 128-cell cap was measured on the int64 kernel). The fused entropy kernels (rare path) widen the matrix first. Counts are exact integers: results are bit-equal (`tests/feature_selection/gpu/test_batched_cmi_byte_codes.py`, 48 related tests pass).

Remaining cost at 100000 rows is the atomic traffic of the (x,y,z) joint, not the code reads.
