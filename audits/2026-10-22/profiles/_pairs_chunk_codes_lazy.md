# Pair-search chunk codes: copy back only the columns the family gate reads, 2026-10-29

Whole-fit profile at HEAD (n=20000): `ensure_host_codes_filled` 2.09 s over 17 chunks. `_compute_one_fe_chunk_part2` filled the whole (n, K) int8 codes block on the host for every chunk so the
pair family gate could later read one column per configuration, but the gate reached its exact test (the only reader of the codes) in 12 `family_pass_batches` calls over 7 pairs in the whole fit.

Change: the eager fill is gone. `ChunkCodes.capture` (before the dispatch clears the deferred-fill record) keeps the device array and the registry's filled flag; the pair state stores
`LazyCodeColumn` references; `_apply_pair_family_gate` calls `resolve_code_columns` after its table-floor early exit, which copies the needed columns of one chunk in a single transfer.
Items that are not lazy columns (arrays, `LazyHostCodes` of the extval path) are passed through untouched; a first version converted them with `np.asarray`, which stripped the device
array the device family test reads and raised the strict-fit traffic from 3.3 MB to 46.6 MB (caught by `test_strict_fit_copies_back_only_result_columns`, back at its 3,295,159 B).
`MLFRAME_PAIR_LAZY_CODES=0` restores the eager fill.

Result (all-flags warm fit): `ensure_host_codes_filled` removed from the profile; `resolve_code_columns` 1.4 s over 12 calls. Fit wall 75.5 -> 67.5 s (cProfile run); `fit_sel_gate.py` seeds 0 and 2:
68.5 -> 61.1 s and 83.9 -> 73.8 s, selection identical in both. Part of the wall difference is GPU load noise; the removed copies are about 1.5 s by the profile.

Not changed by this: the strict-fit test (pair-search audit, n=30000) measures 3,275,410 B with the switch on and off, because that path keeps `LazyHostCodes` and never copies chunk codes; its
overrun against the 2.4 MB budget comes from the sites listed in `strict_fit_d2h_traffic_2026_10_29.md` (score tables and the single-column reads). An earlier audit run showed 12.9 GB in one
tuning-cache state that I could not reproduce afterwards.
