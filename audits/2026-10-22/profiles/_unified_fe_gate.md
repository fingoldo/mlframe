
## 2026-11-03: raw-MI noise floor memo hits

Whole-fit py-spy (100000 x 100): `raw_mi_noise_floor` 1.7 s of the main thread; 14 calls, 13 of them memo hits at 0.155 s each. A hit still built the (k, n)
row matrix it hashes for the key, column by column from pandas columns that are strided views of the C-ordered source (one cache line per value).
`_numeric_rows` copies them in row tiles (`_numba_utils_shared.gather_columns_to_rows`): same matrix, same key; hit 0.155 -> 0.036 s (about 1.5 s a fit).
