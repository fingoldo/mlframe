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
