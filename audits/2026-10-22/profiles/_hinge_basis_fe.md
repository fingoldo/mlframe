# `_hinge_basis_fe` (hinge / slope-change basis) with its carved children `_hinge_precheck_gpu`, `_hinge_detect_gpu_resident`, `_hinge_detect_gpu_resident_batch`

2026-11-15. The family was an exemption (FW11-02: isolated 2.11 s at 100000 x 100, 1.56 s of it the host `np.linalg.qr` precheck). It is under the contract now: a numba kernel, a fused CUDA kernel and a kernel-tuning entry for the precheck, the older device-resident detector kept for the STRICT path.

## What the precheck is

`_hinge_slope_change_plausible` fits the continuous two-segment hinge at three coarse quantiles and keeps the best fractional SSE drop over the plain line (bar 0.005). For `B = [1, x]` the projection is closed form (sums of `xc`, `yc` and of each cut's relu leg), so a column needs two reads instead of a QR and ~8 matrix-vector passes. `_hinge_precheck_gpu` holds the three forms: `_drops_njit` (`prange` over columns, `np.quantile` for the cuts), `hinge_precheck_drops` (CUDA, one block per column, the cuts from the sort-free radix select `_radix_quantiles`, bit-identical to `cp.percentile`) and `hinge_precheck_on_device` (kernel-tuning entry `hinge_precheck_backend`, a sweep over the row count; `MLFRAME_HINGE_PRECHECK_DEVICE=0|1` forces a side). `_precheck_survivors` of `_hinge_basis_fe` runs it once for the whole block of columns in column blocks of 256 MB and passes only the survivors to the per-column detector, whose own precheck and pinned QR are unchanged.

## cProfile

Host path, 100 columns x 100000 rows (`hybrid_hinge_fe_with_recipes`, no STRICT): before 1.47-1.51 s, after 0.42-0.66 s, the same two engineered columns. cProfile of the new path (0.41 s): the device read of the drops 36 ms, `ufunc.reduce` 29 ms, `np.cov` 60 ms cumulative (the downstream held-out checks of the survivors), `_hinge_cut_scan_njit` 20 ms over 3 survivors. The per-column `np.linalg.qr` of the old path (about 1.1 s of 100 columns) is gone.

The precheck alone, 100 columns x 100000 rows, warm: QR path 1.14-1.18 s, numba kernel 0.101 s, device 0.042 s (was 0.109 s with a column sort for the cuts; the radix select removed the sort that was 43 % of the device call). Equal to the QR reference to 1.8e-14 (host) and 1.4e-15 (device); the pure-line column, where the closed form reads a drop of about 0 and the QR path rejects outright, is below the bar in both.

## nvprof (GTX 1050 Ti, sm_61), 4 calls = 1 warm-up + 3

Kernel launches per call: 3 (`radix_select_interp_f64_v2` 15 ms, `transpose_f64` 1.9 ms, `hinge_precheck_drops` 13 ms; the first call includes the compile) plus the upload copy. `hinge_precheck_drops`: achieved occupancy 0.49 (100 blocks of 256 threads on 6 SMs, 17 doubles of accumulators per thread), multiprocessor activity 85 %, global load efficiency 99.97 % (the column is read coalesced, a row per block), device read throughput 11-18 GB/s of the 112 GB/s card (the kernel does 3 x 5 double sums per element, so it is arithmetic and latency bound, not bandwidth bound). Whole-GPU utilization of one call: HtoD copy 28 % of the device time (the 80 MB block), the select 32 %, the precheck kernel 28 %.

## Kernel-tuning entry

`hinge_precheck_backend` (axis: rows, 64 columns, host numba against device, equivalence 1e-9): swept for real on this host, 5 regions persisted, device chosen. The same pass found that the sweeps of `fleuret_confirm_backend` and `gain_stats_codes_backend` (my earlier modules) read `dims["n_rows"]` while the grid passes `n_samples`: both raised `KeyError` and persisted nothing, so those tuners had always fallen back to their row marks; fixed and swept (5 regions each).

## Equivalence and tests

`tests/feature_selection/filters/test_hinge_precheck_gpu.py`: the numba kernel against the QR precheck loop (drops to 1e-9, verdicts equal), the CUDA kernel against the numba kernel, the forced switch, survivors keep every column the detector finds a hinge in, a two-column block budget gives the same drops. The detector's pinned bit-identity tests are untouched (the detector keeps its QR).
