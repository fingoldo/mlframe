# _hsic_gpu (device biased HSIC), 2026-11-03

Where found: 50k x 100 GPU all-families fit; wrapping the auto-scorer bootstrap scorers gave HSIC 1.94 s (13 calls, 15000-30000 rows x 105-210 columns,
500-row subsample), the largest of the seven, thread-summed in the bootstrap pool.
Kernel: one block per column, the column's subsample in shared memory, each thread sums its rows a over b > a of exp(-(x_a - x_b)^2 g) Lc[a, b], block reduction.
`{REAL}` (crit_float_dtype, float32 by default) for the values and the exponent; Lc, the products and the sums in double: Lc's rows sum to 0, and a flat
column's score is that sum (float32 Lc gave -6.8e-9 instead of 0 at m=1000). Bandwidths (median pairwise distance) on the host, `hsic_column_sigmas` (prange).
Accuracy vs `_hsic_biased_columns_njit`: float32 within 3.7e-6 of the largest score, double within 5e-15; same column ranking in every case (test_hsic_gpu).
Speed (GTX 1050 Ti, m=500, k=210, min of 7, host bandwidths included): 72 ms -> 15.5 ms.
Dispatch: KTC `hsic_biased_backend` over (m, k), async sweep, fallback device from k*m*m >= 1e6; `MLFRAME_HSIC_DEVICE=0` or no GPU keeps the host kernel.
Columns are uploaded in blocks bounded by a quarter of the free VRAM; any device fault returns None and the host kernel runs.
Fit check, 30000 x 40 all families, same process, fit cache cleared: same 8 selected features with the device path off and on (twice each); walls 42.7 / 41.6 s (noise).
