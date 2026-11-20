# MRMR performance audit wave 2026-11-20 (wall time and CPU / GPU occupancy) -- master tracker

Four read-only agents (E CPU / GIL, F GPU / kernels, G overlap / scheduling, H repeated work), 21 findings by reading only; reports in this folder. Their own estimates add up to about 5-9 s at best against a 9-14 s gap to 30 s, each item is below the run-to-run noise of the benchmark (3-5 s uncontrolled, about 1 s with the GPU temperature held), so every item is judged by an alternating A/B on the benchmark (record GPU temperature and clocks) before it counts.

| Finding | Estimate | Status | Measured |
|---|---|---|---|
| E-F01 live stage count for the edge workers | 1-2 s | IMPLEMENTED, NEUTRAL | old 39.95 s, new 40.05 s (4 pairs, GPU 64-74 C) |
| E-F02 edge pool sized to the cores | 0-5 s | IMPLEMENTED, NEUTRAL | same A/B |
| G-F01 honest null on the idle slot | 1.0-1.4 s | TODO | |
| G-F02, G-F03 | 0.5-0.8 s, 0.3-0.6 s | TODO | |
| E-F03..F07, F-F01..F06, H-F01..F05 | 0.02-0.5 s each | TODO | |
