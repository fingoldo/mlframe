# `_gate_rank_divergence` (gate rank of the conditional gate: divergence of the probe MI across a median split, for every candidate column)

Measured 2026-11-16, GTX 1050 Ti (sm_61, 112 GB/s). Shape: 100000 rows x 100 float32 gate columns, 3 probe operands, 3 classes, 10 bins.

## Host profile (cProfile, 5 calls of the numba walk)

1.02 s for 5 calls (204 ms per call), 326 function calls: `_divergence_njit` 0.80 s (the medians of the 100 columns, then the chunked walks: prange over 16-column chunks, one walk over each probe's global order per chunk), the three stable probe argsorts 0.16 s, the Python around them 0.04 s. The per-column loop it replaces took 1.6 s for the same call (2 batched-MI calls per gate column).

## Launches per call and the GPU profile (nvprof, 6 calls of the device walk)

Launches per call: about 7 per probe plus the medians: 3 stable `argsort` (cub merge sort, 28 % of the GPU time), 3 inclusive prefix scans of the half mask (`cupy_inclusive_batch_scan_kernel`, 21 %), 3 gathers (`cupy_take`, 5 %), 3 fused `gate_divergence_key` passes (3 %), 3 `bincount` launches, the vectorised plug-in reductions (about 60 small cupy launches per call in all), and `cp.median` of the matrix (0.37 s for 5 calls) when the caller has no medians (STRICT hands over the radix medians).

Metrics (nvprof): `gate_divergence_key` achieved occupancy 0.89, multiprocessor activity 98 %, device-memory read throughput 50 GB/s of 112 GB/s; the cub merge sort occupancy 0.68 to 0.80 at 16 to 45 GB/s; the scan occupancy is low (0.02 to 0.81, 23 % average activity). 244 ms per call from host arrays at 100000 rows (upload included), 166 ms from resident arrays with the radix medians.

## Backend choice and streaming

Backend choice: the tuner entry `gate_rank_divergence_backend` (`gate_divergence_on_device`, a sweep over the row count 5000 to 300000 at 100 columns, host-in / host-out so the upload counts, both walks equal to 1e-9; measured difference 1e-16): host up to 100000 rows, device from 300000. STRICT-GPU hands over the resident matrix and uses the device walk without asking. `MLFRAME_GATE_DIVERGENCE_DEVICE=0|1` forces a side. The per-column loop of `_rank_and_prune` stays as the fallback (`MLFRAME_GATE_RANK_LOOP=1`, a non-finite matrix, a device fault).

Streaming: the device walk takes the gate columns in column blocks of `_DEVICE_BLOCK_BYTES` (256 MB of temporaries, about 28 bytes per row and column) so any n x p fits; each block carries the probe columns as its first columns.
