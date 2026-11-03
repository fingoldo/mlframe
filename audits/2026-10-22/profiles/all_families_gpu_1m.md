# 1M x 100 all families, GPU (2026-11-03): what breaks

Run 1 (stage_sample, 9.1 GB host RAM free at start): crashed in `_rebuild_full_survivor_col` with numba `NotImplementedError: float16` (a float16 operand
column; fixed by promoting float16 to float32, 93b68ada7). Before that: 165 radix-select OutOfMemoryError fallbacks to cp.percentile (the (k, n) transpose of
1M-row candidate matrices, 232-400 MB each, against a pool already holding 1,810,129,920 of 1,976,985,191 bytes at every failure; fixed with column blocks,
7a7a367fb); device honest null out of memory (624 MB) -> host; binned-aggregate device gate declined -> host scoring; numeric_decompose dropped by a host
MemoryError (2.79 GiB for the 1M x 749 candidate frame; fixed with source-column blocks, 9421dacce).
Run 2 (vram_holders.py, the three fixes in): host RAM ran out before any device out-of-memory (3.4 GB available of 15.9 at the stop, the fit's commit ~7 GB):
count / frequency encoding, missingness indicator, grouped agg and composite group agg dropped on 4-8 MB allocations, modular dropped on 2.99 GiB
(the (1605, 1M) int16 candidate code matrix). Stopped by hand.
Binding limit at 1M on this box is host RAM (16 GB physical, ~5 GB held outside any process). Next: per-stage host peak memory, then stream the families
that build whole candidate matrices (modular codes first: 1605 x 1M int16 = 3 GB in one allocation).
