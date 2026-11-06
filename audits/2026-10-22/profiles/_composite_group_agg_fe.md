
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

## Inside the 100000 x 100 every-family GPU fit (module_walls.py, wall 83.9 s with the wrappers, 2026-11-06)

Inclusive seconds summed over threads: `hybrid_composite_group_agg_fe` 3.63 (1 call) = `generate_composite_group_agg_features` 2.35 + `auto_detect_key_sets` 0.65 + the rest 0.6. Inside the generator `_emit_num_col_features` 3.30 over 384 calls (summed over the edge pool threads, so it exceeds the 2.35 s of wall): `_broadcast_lookup` 1.24 over 1152 calls (about 1.1 ms each, a host table gather over 100000 rows), `_composite_residuals` 0.60, `_global_value_for_stat` 0.16, `build_composite_keys` 0.42 over 15 calls, `_unique_inverse` 0.10. A batched device gather of all the stat tables of a key set would remove at most about 1.2 thread-seconds, under 0.3 s of wall on 8 threads, and adds a host-to-device copy of every emitted column unless the consumer takes device columns. Not changed.
