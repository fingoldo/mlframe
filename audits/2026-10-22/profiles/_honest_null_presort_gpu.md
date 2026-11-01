# _honest_null_presort_gpu (sort order and tie flags of the honest-null columns on the device)

`presort_and_ties_gpu` in `src/mlframe/feature_selection/filters/_honest_null_presort_gpu.py`, called by `_honest_null_accept._order_and_ties` before the root-split acceptance. Host twin: `_presort_columns` + `_tie_flags` (kept for non-finite blocks, small blocks, `MLFRAME_HONEST_PRESORT_DEVICE=0` and any device fault).

`cupy.argsort` along the rows of a column-major copy returns the same permutation as the host's stable sort, ties included (tests: `tests/feature_selection/fe/test_honest_null_presort_gpu.py`, float32 and double, heavily tied, binary and constant columns, n = 997 / 20011 / 50000); the tie flag of sorted position `i` is `sorted[i] == sorted[i + 1]`.

Measured (GTX 1050 Ti, 250 columns x 100000 rows, machine under load): host presort 2230 to 2560 ms + tie flags 313 ms; device argsort 676 ms for the same columns. Whole `_continuous_accept_block` (250 columns x 999 permutations x 100000 rows, same process): 7.3 to 7.6 s with the host sort, 6.3 to 6.4 s with the device sort; order, tie flags and the accepted pairs identical.
Same call with the older float k5 kernel instead of the bitset acceptance: 10.8 to 12.8 s. Remaining in the 6.3 s: the acceptance kernel (about 4 s for 250k blocks), host-side packing of the permuted targets (0.16 s), uploads.

Memory: columns go through in blocks of at most 25% of what the pool can still hand out (an int64 argsort next to the float copy: 28 bytes per element), and the pool's unused cache is returned afterwards (see `_bootstrap_rank_mi_gpu.md` for the pool-pressure lesson).
