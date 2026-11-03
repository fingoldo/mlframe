# _dcor_gpu (device dCor moments), 2026-11-03

Where found: 50k x 100 GPU all-families fit, auto-scorer bootstrap: dCor 1.20 s (14 calls, 15000-30000 rows x 105-210 columns, 500-row subsample),
thread-summed; the host `_dcor_columns_kernel` is serial (nogil, no prange).
Kernel: one block per column, its subsample in shared memory; thread i owns rows i and accumulates the row sum, sum d^2, sum d B[i, q] and (row / m)^2;
the block reduces the four totals and the host forms dcov2 / dvar2_x as the host kernel does. All double: dvar2_x = sum d^2 - 2 m sum r^2 + m^2 g^2 cancels.
Accuracy vs the host kernel: within 2.4e-13 of the largest value (m=1000), 1e-14-1e-15 elsewhere (test_dcor_gpu).
Speed (GTX 1050 Ti, m=500, k=210, min of 7, uploads included): 63.3 ms -> 8.0 ms.
Dispatch: KTC `dcor_columns_backend` over (m, k), async sweep, fallback device from k*m*m >= 1e6; `MLFRAME_DCOR_DEVICE=0` or no GPU keeps the host kernel.
Fit check, 30000 x 40 all families, same process, fit cache cleared: same 8 selected features off / on (twice each); walls 35.4 / 35.3-35.8 s (noise).
Found on the way: `distance_correlation` rounded x (not y) to float32 before the double distance matrices -> asymmetric in the 9th digit; fixed (d43725fc5).
