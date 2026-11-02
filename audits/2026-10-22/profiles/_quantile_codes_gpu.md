# _quantile_codes_gpu (quantile codes of a block of finite columns)

Kernels `radix_hist1/2/3` (order-statistic selection) and `quantile_codes_f32/f64` (codes) in `src/mlframe/feature_selection/filters/_quantile_codes_gpu.py`; dispatched by `quantile_codes_block_auto` from `_family_maxt_gate._family_codes`.
Host twin: `quantile_codes_block` (parallel njit, selects the ~22 order statistics the edges read), kept for float64, small blocks, columns with at most `n_bins` distinct values and `MLFRAME_QUANTILE_CODES_DEVICE=0`.

Two device variants, both with codes equal to the host's (tests: `tests/feature_selection/fe/test_quantile_codes_gpu.py`, float32 and double, tied / few-level / binary / constant / heavy-tailed columns):

| 100000 rows | sort variant (`method="sort"`) | radix selection (default, float32) | host |
|---|---|---|---|
| 1944 columns | 1963 ms (float32), 2491 ms (double) | 871 ms | 2085 to 2550 ms |
| 256 columns | 285 ms | 130 ms | 258 to 278 ms |

The sort variant is REJECTED as a default: the batched sort costs 0.75 ms per column (191 ms per 256), the host's selection costs about 1 ms per column on 8 cores. The radix variant avoids the sort: three passes of 11 + 11 + 10 bits over the order-preserving integer key
of the float with a histogram per target rank, then the edges are read with numpy's linear-interpolation arithmetic in double and every value counts the kept edges below it.

GPU profile of the radix variant (nvprof, 256 columns x 100000 rows): `radix_hist1` occupancy 0.96, multiprocessor activity 97.5%, load efficiency 100%, warp efficiency 98.9%; `radix_hist2` and `radix_hist3` occupancy 0.93 to 0.95, activity 90 to 99%,
warp execution efficiency 36 to 37% (only the values whose bin holds a target take the 21-target loop; the shared `is_target` prefilter made them cheaper but leaves the divergence); `quantile_codes_f32` occupancy 0.69, activity 95%, warp efficiency 100%.
Per 256-column block: host-to-device 26 ms, order statistics 37 ms (the three histogram kernels about 20 ms, the rest the cumulative sums), codes kernel 11 ms, device-to-host 14 ms. Remaining headroom: the transfers (pinned buffers) and the divergence of the two target passes.

## Full-fit A/B (100000 x 100, every family, 2026-11-01)

Back to back in two processes: device quantile codes off 171.4 s, on 154.7 s; no OutOfMemoryError and no family raised in either. The call-level gain (about 1.2 s per `_family_codes` call, 2.4x on the call) cannot explain 17 s, so most of the gap is run-to-run noise;
the same code also produced a 228.8 s outlier in the run before. Do not read the wall difference as the gain of this change; the instrument for it is the call time and the stage seconds in the sampler.

## Crossover re-measured (2026-11-02): device from about 1M elements, `DEVICE_MIN_ELEMENTS` 20M -> 1.5M

The 20M-element bound kept the modular family's candidate coding on the host (30000-row subsample x 335 columns = 10M elements): `quantile_codes_block_auto` was the top sampled frame of `hybrid_modular_fe` (15 samples, about 2.7 s per fit, GPU 1%). Same-process sweep, float32, host kernel on 8 threads against the radix device path (codes and counts equal in every row; best of 3):

| rows | columns | host ms | device ms |
|---|---|---|---|
| 20000 | 32 / 64 / 128 / 335 | 5.9 / 12.8 / 25.8 / 69.3 | 7.6 / 10.1 / 18.2 / 46.3 |
| 30000 | 16 / 32 / 64 / 128 / 335 | 4.3 / 9.6 / 23.8 / 43.3 / 111 | 6.4 / 7.8 / 12.1 / 21.8 / 56.8 |
| 100000 | 8 / 16 / 32 / 64 / 128 / 335 | 8.1 / 18.3 / 34.9 / 64.5 / 148 / 404 | 9.0 / 9.1 / 14.8 / 24.4 / 47.5 / 124 |

The device wins from about 1.0 to 1.3M elements (1.9x at 30000 x 64, 3.3x at 100000 x 335). Float64 blocks stay on the host: the sort-based device path is not faster there (416 ms against 378 ms at 100000 x 300). The bound is now 1.5M; `MLFRAME_QUANTILE_CODES_DEVICE=0|1` still forces a path. A per-host tuner entry would keep a faster or slower card honest: the crossover belongs in the KTC of the new-GPU machine's checklist (CUDA.md section 14).

Fit-level check after the change (100000 x 100, every family, VRAM used by other processes 1.1 GB): wall 158.4 s, no OutOfMemoryError / "FE raised", same 6 selected features; sum of the top-30 family walls 78.1 s against 79.5 s before (inside the noise), `gate:fe_modular_enable` 6.7 s against 7.1 s. The coding calls of that family are a minor part of its time at this size (the baseline bootstrap and the family gate are the rest), so the fit-level gain is small; the per-call gain (1.9x to 3.3x) is the measured one.
