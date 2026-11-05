
## 2026-11-03: per-group arrays instead of string-keyed dicts for the candidates

Stage sample (100000 x 100): 8 of 16 samples on the main thread waiting for the (key set, numeric column) pool, whose tasks built the key strings and a
`{key string: value}` dict per statistic and read them back per distinct key for the broadcasts and residuals (Python, GIL-bound). The key strings are now built
once per key set; when they are distinct the lookups are `_ArrayLookup` mappings (the dict is built only when a winner's recipe reads it) and the broadcasts /
residuals use the arrays. Keys that print alike keep the dict path. 12 key sets x 100 numeric columns (6000 candidates): generation 6.33 -> 5.02 s (first pair;
the standalone run pages with two 2.4 GB frames held), columns and recipe lookups equal to the dict path.

## 2026-11-05: GPU port considered, not built

Warm 100k x 100 fit: the family costs 4.5-4.9 s of wall (gate level), 1.2 s of it on the main thread, the rest in the edge-jobs pool. `_emit_num_col_features` computes
the per-group mean / std / count with the numba `group_mean_std_count` (one pass over the rows, memory-bound) and broadcasts through `_broadcast_lookup`; the worker
samples put the time in the Python layer around those kernels (the `{key string: value}` lookups the recipes keep, `_array_lookup`, the recipe dicts, the per
(key set, column, statistic) bookkeeping), not in the arithmetic. A device kernel (atomic group sums, broadcast, residuals) would replace the part that is
already a one-pass numba kernel and leave the Python layer in place, so it is not built; the lever is fewer Python objects per candidate, not a faster reduction.
