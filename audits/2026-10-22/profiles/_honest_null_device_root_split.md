# `root_split_stats` (honest null of plain MDLP, device), 2026-10-29

nsys over the warm n=20000 fit (`nvprof_fit.py 20000`): total kernel time is about 5 s of a 60 s wall, so the device is mostly idle and the `cupy .get` wait is launch and synchronisation latency, not kernel time.
Largest kernels: `root_split_stats` 658 ms (1 launch, 110 blocks) + 330 ms (12 launches, ONE block of 128 threads, 27.5 ms each), `radix_select_f32_v3` 644 ms (23 launches), `family_null_bits_v2` 368 ms,
`bin_codes_f32_int8` 294 ms, `fe_materialise_cm` 260 ms.

`root_split_stats` is one thread per (column, permutation) scanning n sorted rows sequentially, so a launch costs n times the latency of the dependent loads (ord -> label, count -> table), about 1.3 us
per row; with at most 128 pairs it uses one SM, and more blocks would not shorten it. Splitting one scan across threads would change the double rounding of the running sums, and the gains must round
as the host's (`--fmad=false`), so that is not available.

Applied: `_SOURCE_V2` reads the next row's label one step ahead (it is both the boundary test of step i and the label of step i + 1), taking the ord -> label load off the critical path. The original
`_SOURCE` is kept; `MLFRAME_HONEST_ROOT_SPLIT_V2=0` selects it. Bit-identical outputs (gain, candidate count, index) on every shape of `D:/Temp/root_split_bench.py`
(n=20000; K=2 and 8; pairs 104 .. 13986): 1.02x to 1.18x per launch (K=2, 104 pairs 30.4 -> 25.9 ms; K=8, 13986 pairs 350 -> 329 ms). Whole fit: within noise (58.3 / 67.3 s against 58.0 / 67.3 s), selection identical on seeds 0 and 2.

Not tried (needs a different structure, not a tweak): caching the table value of each class's next count in registers (the table load is the remaining dependent latency), a two-level scan with an exact
re-summation. The 12 single-block launches are 0.33 s in total, about 0.5 percent of the wall.
