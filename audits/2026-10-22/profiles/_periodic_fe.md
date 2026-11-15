
## 2026-11-03: modular gate-first pool built as a row matrix

100000 x 100, periods (7, 12, 24, 30, 365): 1500 candidates. Phase walls of the warm `hybrid_modular_fe_with_recipes` (one process, second call): generate 0.71 s,
family codes 1.00 s (device coder 0.60 s, the DataFrame-to-array copy and transpose the rest), bootstrap baseline 0.71 s, family max-T 0.42 s, survivor scoring 0.25 s.

The gate now fills the `(K, n)` candidate matrix directly (`modular_candidate_block`, same names / order / values as `generate_modular_features`) and codes its rows in place
(`family_codes_of_rows`, same codes and counts as `_family_codes`). Warm wall 3.29 -> 2.90 s, cold 6.02 -> 5.66 s; same six columns appended.

Left: the device coder uploads 1500 float32 candidates (600 MB) and copies the int16 codes back for the max-T pass. Computing the residues on the device from the 100 uploaded
source columns would remove the upload; the max-T pass would have to take resident codes to remove the copy back.

## 2026-11-03: modular candidate codes computed on the device

`_modular_codes_gpu.modular_family_codes_gpu`: the source columns are uploaded once in double precision, one kernel writes mod / sin / cos of every period into the
float32 rows the resident coder reads (`_quantile_codes_gpu.quantile_codes_resident_f32`, same edges and code kernel), blocks of sources within 20% of the pool.
Only rows whose 21 order statistics hold at most `n_bins` distinct values come back to the host for the exact few-level check.

1500 candidates x 100000 rows: pool + codes 1.29 s (host pool, device coder) -> 0.83 s; codes equal to the host pool's in every row. Warm family wall 2.90 -> 2.54 s
(3.29 s before the row-matrix change), cold 5.66 -> 5.07 s, same six columns. `MLFRAME_MODULAR_CODES_DEVICE=0|1` forces the path; any device fault keeps the host pool.

## 2026-11-15: family record with hardware counters (device twin `_modular_codes_gpu`, full figures in its record)

The residue / sine / cosine family of `_periodic_fe` (host kernels `_modular_njit`, `_modular_all_ops_njit`, `_modular_all_ops_into`) runs on the device through `_modular_codes_gpu`, which owns the kernel `modular_all_ops`, the kernel-tuning entry choosing host or device by rows and sources, and the column-block streaming (a block of source columns sized from the free pool, one launch per block, the next block built by a host prefetch thread). Measured on the GTX 1050 Ti (sm_61, double rate 1:32), 16 sources x 400000 rows x 5 periods, elevated process, nvprof hardware counters:

- cProfile of one call: 0.267 s, 4189 function calls; the device sort waits 0.127 s for the queued kernels, the blocking copies back 0.093 s, the coder's own Python 0.061 s.
- Launches per call: one `modular_all_ops` launch per source column block, then the resident quantile coder (about 20 launches per block); the three candidates of every period come from that single launch, not from three.
- Occupancy: achieved 0.51 before the change below, multiprocessor activity 98 to 99 percent; double FLOP efficiency 69 percent of the 1:32 peak, i.e. the kernel was bound by double-precision sincos, device-memory bandwidth only 3 to 9 GB/s.
- Throughput after taking sine and cosine of the double-formed angle at the stored float type: 15.2 ms per launch (was 38 ms), 45.6 ms of the call's 640 ms GPU time (7 percent); the quantile coder (22 percent), radix histograms (35 percent) and the copy back to host (20 percent) dominate the whole-GPU utilization.
- Wall: host pool against device pool at 400000 rows 0.195 s / 0.072 s, at 100000 rows 0.057 s / 0.025 s; below 5000 rows the host wins and the tuner keeps it.
