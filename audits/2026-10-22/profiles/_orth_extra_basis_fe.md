
## 2026-11-03: Fourier extra basis, py-spy of the worker threads

Fit stage sample: 14 samples on the main thread waiting for the column pool (low CPU). Standalone (100000 x 100, 30000-row decision subsample, fourier
basis): cold 2.62 s, warm 0.62 s. py-spy over all threads of the cold run: `np.var` from the `np.std(leg) > 1e-12` constancy checks (two per frequency per column) 361
of 1443 worker samples, `_is_int_as_cat_axis` waiting in `acquire` (the numba first-call dispatch lock, cold only) 219, the heavy-tail detector's partitions 260.
The constancy checks now go through `_std_vs`: one min / max pass, and the std only when `range / sqrt(2 n)` cannot decide (a zero range is not shortcut: the
float32 std of a constant column is a 9.5e-7 residue the threshold sees). Same verdicts; cold 2.62 -> 2.47 s, warm 0.62 -> 0.56 s.

## Coarse sweep as matrix-vector products (2026-11-03)
Where found: 50k x 100 GPU all-families fit, 4.6 s of main-thread wait on the edge-jobs pool; a wrapper on `map_on_edge_pool` gave
`generate_extra_basis_features` 2.42 s of it (one call, 108 columns, 30000-row decision subsample, Fourier + adaptive + chirp).
The pool does not parallelise this call (pool 5.73 s vs serial 5.90 s on the captured call): the per-column kernels are numba `parallel=True`.
Serial cProfile of the captured call: `_greedy_fourier_frequency_search` 1.20 s own time (2 dots per grid frequency per greedy step, in Python),
`_coarse_basis_njit` 1.19 s, `_power_centered_batch_njit` 0.57 s, argsort 0.56 s.
Change: the coarse sweep scores every grid frequency with one BLAS matrix-vector product per sin / cos block (`_best_coarse_freq`, first max on a tie,
NaN never wins). Captured call, warm, alternating: 4.26 -> 3.46 s; emitted columns, values and metadata identical (test_fourier_coarse_gemv).
`MLFRAME_FOURIER_COARSE_GEMV=0` keeps the per-frequency loop. Next here: the coarse basis build (1.19 s) and the per-column argsorts.

## Coarse basis by the Chebyshev recurrence (2026-11-03)
The coarse grids are uniform (0.5 k: 32 frequencies linear, 48 chirp), so `_coarse_basis_recur_njit` takes one sin / cos per row and the higher multiples by
`s_{k+1} = 2 cos(t) s_k - s_{k-1}`, parallel over rows; means and sums of squares are formed per frequency in the old row order. Basis agrees with the direct
build to 1e-9 (48 frequencies, z in [-4, 4]); captured 30000 x 108 fit call: emitted columns, values and metadata identical. Warm, alternating: 3.02-3.06 -> 2.54 s
(with the matrix-vector sweep above: 4.26 s -> 2.54 s for the call). Non-uniform grids keep `_coarse_basis_njit`; `MLFRAME_FOURIER_COARSE_RECUR=0` turns it off.

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
