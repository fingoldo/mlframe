# `_conditional_quantile_rank_fe` profile

## 2026-11-06: where the time is

Standalone, host path, 100000 rows, `generate_conditional_quantile_rank_features` over 6 numeric columns (30 ordered pairs): wall 1.16 s. Top lines: `ndarray.sort` 642 calls 0.37 s (the per-bin sorted reference values of the pair), `searchsorted` 306 calls 0.28 s (the row-level ranks against them), the pair loop's own lines 0.27 s. In the 100000 x 100 GPU fit the family's stage function takes 3.06 s of wall at 100000 rows (first per-family run) and 4.2 s at 300000 rows (`mid_b`); both are sums over the pool threads and the stage's other work, not the standalone 1.16 s.

Not changed: the sort and the search are the algorithm (an empirical rank of `x_i` inside the quantile bin of `x_j`), the sorts are of value arrays whose order is not needed beyond the sorted values, and a radix order would change nothing but the sign of a zero. A device version would need a segmented sort and a segmented search per bin for 30 pairs; the host version is 40 ms per pair. Not built.
