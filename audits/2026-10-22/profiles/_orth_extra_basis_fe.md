
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
