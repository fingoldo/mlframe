# `_batch_mi_noise_gate_bits` (bit-set permutation noise gate)

Measured 2026-11-14, GTX 1050 Ti (sm_61, 112 GB/s). Shape of a gate call: 20000 rows x 4000 int8 columns of 16 bins, 10 classes, 50 permutations plus the observed target, gate rule `confidence 0.95` applied on the device (the (K,) gated vector is the only thing that crosses the bus). Codes and the (P, n) target matrix are resident when the call starts.

## Launches per call and the GPU profile (nvprof, 4 calls)

Launches per call: 1 `pack_cls` (class bit sets of every permutation), 1 `pack_bins` (the bin bit sets of one column block; one block here, `group` columns by the free-memory rule), 1 `family_null_bits_v2` (the popcount MI of every (permutation, column) pair), plus about a dozen small cupy operations for the xlogx table sums and the gate rule on the (P, K) matrix. GPU time per call 80 ms: `family_null_bits_v2` 65.6 ms (82.5 %), `pack_bins` 9.8 ms (12.4 %), the table upload 3.5 ms, `pack_cls` 0.2 ms, the cupy gate-rule operations about 1 ms in total.

Metrics (nvprof): `family_null_bits_v2` achieved occupancy 0.49, multiprocessor activity 98 to 99 %, device-memory read throughput 10 to 36 GB/s, global load efficiency 97 %, store efficiency 11 to 13 % (one MI value per block: a handful of stores); `pack_bins` occupancy 0.50 to 0.52, activity 98 to 99 %, read 10.7 to 18.5 GB/s, write 17.6 to 29.6 GB/s, store efficiency 98 to 99 %; `pack_cls` occupancy 0.88 to 0.90, store efficiency 25 % (negligible time, 0.2 ms). The gate is bound by the popcount arithmetic of the null kernel (its read throughput is a third of the card's), not by memory; the small cupy operations run at 4 to 30 % multiprocessor activity but add up to about 1 ms.

## Host profile (cProfile, 5 calls)

0.408 s for 5 calls, 1016 function calls: 0.395 s is the blocking `.get()` of the gated vector (it waits for the kernels above, so it is the GPU time surfacing), the Python around the launches 0.005 s, the xlogx table 0.002 s. No per-column or per-permutation Python loop: the column loop runs per block of `group` columns.

## Backend choice and streaming

The bit-set gate against the histogram gate is the tuner entry `noise_gate_bits` (rows x columns x classes x bins, swept for real; fallback: the bit-set gate wherever its kernel applies). The host side of the same decision is the njit permutation gate of `batch_mi_noise_gate_gpu`. The columns go through the kernels in blocks sized from the free device memory (at least 1024 columns per launch, at most 768 MB of bit sets), declared in the module docstring; the bit sets of the whole matrix never exist at once.

Data placement: the gate takes resident codes and a resident target matrix (the codes are device-born in the FE gate) and returns host vectors; the entry is the device-in / host-out case of the gate and nothing here makes the host-in case reachable.
