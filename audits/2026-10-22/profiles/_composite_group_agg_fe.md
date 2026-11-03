
## 2026-11-03: per-group arrays instead of string-keyed dicts for the candidates

Stage sample (100000 x 100): 8 of 16 samples on the main thread waiting for the (key set, numeric column) pool, whose tasks built the key strings and a
`{key string: value}` dict per statistic and read them back per distinct key for the broadcasts and residuals (Python, GIL-bound). The key strings are now built
once per key set; when they are distinct the lookups are `_ArrayLookup` mappings (the dict is built only when a winner's recipe reads it) and the broadcasts /
residuals use the arrays. Keys that print alike keep the dict path. 12 key sets x 100 numeric columns (6000 candidates): generation 6.33 -> 5.02 s (first pair;
the standalone run pages with two 2.4 GB frames held), columns and recipe lookups equal to the dict path.
