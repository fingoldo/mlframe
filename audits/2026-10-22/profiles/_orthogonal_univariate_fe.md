# `_orthogonal_univariate_fe` (Fourier detector kernels: `_fourier_batch_gpu`, `_fourier_fused_gpu`)

Measured 2026-11-13, GTX 1050 Ti (sm_61, 112 GB/s, double rate 1:32). Shape of the hot kernel: 64 columns x 30000 rows (float32) x 96 trial frequencies per column = 1.84e8 sine-cosine evaluations per call.

## What the module does on the device

Per column block, the detector runs a coarse periodogram scan, two refinement scans, a held-out confirmation and a least-squares deflation as a handful of kernels over `(C, n)` blocks (`periodogram_power_b`, `sincos_gram_b`, `solve3_b`, `apply_deflate_b`; the single-column twin has the same kernels without the column axis). Launches per call: one `periodogram_power_b` launch per scan (grid `(F, C)`, one block per column and frequency), one gram launch, one solve and one deflate per accepted frequency; the host reads one small vector per phase of an iteration, not one per column.

## This pass: sample-type Kahan partials in the periodogram

The kernel took the sine and cosine in the sample type (`trig`) and then accumulated six sums per thread in double: with the card's 1:32 double rate the FP64 pipe was the limit (nvprof: double FLOP efficiency 42 percent). The per-thread partials are now held in the sample type with Kahan compensation and combined in double in the block reduction (`KAHAN_ADD`; for double samples the partials are double). Same input, kernel in isolation: 76.5 -> 35.5 ms per call (2.2x); under nvprof 4 launches 35.5 ms each, 96.9 percent of GPU time. Agreement with the double-partial kernel, power statistic: max relative error 2.4e-5 (64 x 30000), 2.9e-6 (32 x 100000), 1.1e-5 (128 x 5000), p99 about 4e-7; the argmax frequency of every column unchanged; uncompensated float partials were 2.1x faster too but reached 1.7e-4, so the compensation stays (35.6 against 33.4 ms). End to end the batch detector returns the same frequency lists as before on 360 random columns (6 seeds of 60 columns, 8000 to 50000 rows, signal amplitudes 0.4 to 1.5, one to three planted tones): 0 columns with a different list. The 72 Fourier tests (batch against single-column detector, fused kernels, recurrences, tail aliasing, adaptive grids, extra-basis blocks) and the offline ptxas checks of the sources pass.

## nvprof metrics after the change (`periodogram_power_b`, replayed)

Achieved occupancy 0.536, multiprocessor activity 99.4 %, double FLOP efficiency 27.4 %, single FLOP efficiency 10.9 %, device-memory read throughput 0.44 GB/s of 112. The kernel is bound by the arithmetic around the sine-cosine (the double argument reduction `w * z` minus a multiple of 2 pi, kept in double so large frequencies stay exact, and the conversions), not by memory; the next lever would be a float argument reduction below a frequency bound, which changes the phase precision and is not taken here.

## Not changed

`sincos_gram_b` and `apply_deflate_b` stay in double: they run once per accepted frequency, not per trial frequency, and their sums feed a 3x3 solve whose conditioning matters.
