# `_conditional_quantile_rank_fe` profile

## 2026-11-06: where the time is

Standalone, host path, 100000 rows, `generate_conditional_quantile_rank_features` over 6 numeric columns (30 ordered pairs): wall 1.16 s. Top lines: `ndarray.sort` 642 calls 0.37 s (the per-bin sorted reference values of the pair), `searchsorted` 306 calls 0.28 s (the row-level ranks against them), the pair loop's own lines 0.27 s. In the 100000 x 100 GPU fit the family's pair function takes 4.2 s of wall at 300000 rows and under 1.5 s at 100000 (per-family walls, `mid_b`).

Not changed: the sort and the search are the algorithm (an empirical rank of `x_i` inside the quantile bin of `x_j`), the sorts are of value arrays whose order is not needed beyond the sorted values, and a radix order would change nothing but the sign of a zero. A device version would need a segmented sort and a segmented search per bin for 30 pairs; the host version is 40 ms per pair. Not built.
