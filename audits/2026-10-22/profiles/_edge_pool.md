# _edge_pool: per-stage share of the edge thread pool

`map_on_edge_pool` and about ten FE modules with their own `ThreadPoolExecutor` sized every pool to all cores through `_resolve_edge_jobs`. Under FE stage concurrency (`run_stages_concurrently`) three cascades each opened such a
pool, so up to 3 x cores threads competed for the cores. `concurrent_stages(n)` marks n cascades as running together and `_resolve_edge_jobs` gives an unset `n_jobs` `cores // n` workers
(explicit `n_jobs` and `MLFRAME_EDGE_JOBS` are kept). The oof_warp column scan now goes through `map_on_edge_pool` too.

Measurement: warm 100k x 100, all families, one process, alternating share/noshare fits on distinct data seeds (D:/Temp/ab_share.py; refits on
the same data hit the fit cache and take 2 s, so they cannot be compared).

| mode | walls (s) |
|---|---|
| share | 91.7, 80.8 (mean 86.3) |
| noshare | 93.0, 92.5 (mean 92.8) |

Order: one discarded warm 100k fit (76.3 s), then noshare, share, share, noshare (fits after the first in a process run 15-20 s slower,
so a plain alternation is biased). Gain about 6.5 s (7%).

Earlier runs of this A/B were void: the stage slot pools sat below their held bytes and kept cached blocks across fits (fixed in
12d7a716a and a5d91e8ae), so every fit after the first took 258-338 s.
