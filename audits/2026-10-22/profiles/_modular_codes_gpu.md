# `_modular_codes_gpu` (family-test codes of the modular candidate pool, built on the device)

Measured 2026-11-14, GTX 1050 Ti (sm_61, 112 GB/s, double rate 1:32). Shape: 16 float32 source columns x 400000 rows x 5 periods = 96 million candidate cells, 10 bins; sources and codes live on the host (the pool is built from a pandas frame and the int16 codes go back to host memory in both backends), so every wall below includes the upload and the copy back.

## Host or device: the kernel-tuning cache

The choice used a fixed element bound (`DEVICE_MIN_ELEMENTS` 1.5M, 20M under the strict-resident path). Measured per call, host pool against device pool (sources and codes on the host in both): 4 sources 5000 rows 0.008 / 0.010 s, 20000 rows 0.020 / 0.012, 100000 rows 0.057 / 0.025, 400000 rows 0.195 / 0.072; 16 sources 5000 rows 0.025 / 0.026, 20000 rows 0.058 / 0.039, 100000 rows 0.176 / 0.089, 400000 rows 0.639 / 0.294. The entry `fe_modular_codes_backend` (`_SPEC`, `host_device_sweep` on the element count, 8 sources x 5 periods) now decides outside the strict-resident path (which keeps its residency bound: a small pool must not pay a bulk device-to-host copy). The sweep was run for real after the kernel change below: host up to 1.5M elements, device from 6M (bin shares agree to 5e-7). The residency of the data is part of what is measured: the entry describes this call's host-in, host-out boundary, which is the only one this module has.

## Kernel (`modular_all_ops`, nvprof, replayed metrics)

Before: 38 ms per launch (3 launches, 114 ms, 18 % of the call's GPU time), achieved occupancy 0.51, multiprocessor activity 98 to 99 %, double FLOP efficiency 69 % of the 1:32 peak, device-memory throughput 3 to 9 GB/s: bound by the double-precision `sincos`, not by memory. After (sine and cosine of the double-formed angle taken at the stored float type): 15.2 ms per launch (45.6 ms in total, 2.5x), occupancy 0.71, double efficiency 59 %, write throughput 20 to 24 GB/s, store efficiency 99 %. In isolation 37.2 -> 17.6 ms. Stored values differ from the double route by at most 2.7e-7. The `sincospif` variant of the turn fraction was measured too: 19.8 ms and the same cell movement, so the plain `sincos` of the float angle stays. What is left in the double pipe is the residue (`v / p` and `floor`), which stays double because a float32 timestamp has resolution 128. **Rejected (measured):** the division replaced by `v * (1 / p)` with a division fallback for the rows whose product lies within 8 ulp of an integer (keeps the quotient the host's in every row; output bit-identical to the division kernel) is slower, 20.5 against 17.6 ms: on a 1:32 card the guard (`floor`, `fabs`, two compares, a multiply) costs as many double-rate operations as the division it saves.

Launches per call: one `modular_all_ops` launch per block of source columns (`_sources_per_block`, a column block sized from the free pool), then the resident quantile coder (radix histograms, sort, scan: about 20 launches per block). GPU time of the 16 x 400000 call: quantile coder 22 %, radix histograms 35 %, copy to host 20 %, `modular_all_ops` 7 % (45.6 of 640 ms) after the change.

## Host profile (cProfile, one call)

0.267 s for 4189 function calls: the device sort 0.127 s (waits for the queued kernels), the blocking `.get()` copies 0.093 s, the coder's own Python 0.061 s; the host prefetch thread builds the next source block while the device works.

## Equivalence

Residue rows code exactly as on the host. Sine / cosine rows: 3e-6 of cells move one bin on a continuous column, none on binary or heavy-tailed ones, 8 percent on a 500-level integer column where tied values split differently (arbitrary on either side). Tests pin the exact residue rows and these bounds.
