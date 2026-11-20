# MRMR performance audit wave 2026-11-20 (wall time and CPU / GPU occupancy) -- master tracker

Four read-only agents (E CPU / GIL, F GPU / kernels, G overlap / scheduling, H repeated work), 21 findings by reading only; reports in this folder. Their own estimates add up to about 5-9 s at best against a 9-14 s gap to 30 s, each item is below the run-to-run noise of the benchmark (3-5 s uncontrolled, about 1 s with the GPU temperature held), so every item is judged by an alternating A/B on the benchmark (record GPU temperature and clocks) before it counts.

| Finding | Estimate | Status | Measured |
|---|---|---|---|
| E-F01 live stage count for the edge workers | 1-2 s | IMPLEMENTED, NEUTRAL | old 39.95 s, new 40.05 s (4 pairs, GPU 64-74 C) |
| E-F02 edge pool sized to the cores | 0-5 s | IMPLEMENTED, NEUTRAL | same A/B |
| G-F01 honest null on the idle slot | 1.0-1.4 s | TODO | |
| G-F02, G-F03 | 0.5-0.8 s, 0.3-0.6 s | TODO | |
| E-F03..F07, F-F01..F06, H-F01..F05 | 0.02-0.5 s each | TODO | |
| F-F05 `Device().attributes` per call | 0.05-0.3 s | REJECTED (measured) | 3.1 us a call, 30 ms for 10000 calls |
| G-F01 honest null on the idle slot | 1.0-1.4 s | DEFERRED | the permuted labels are drawn from one rng stream in call order (`_permuted_labels_cached`): a worker that fills the cache earlier changes the draws and the floors, so it is not selection-equivalent without restructuring the stream; the gain is also inside the noise |
| E-F03 honest spec serial prep | 0.1-0.3 s | REJECTED (measured) | 0.032 s for 100 columns |
| E-F05 frame fingerprint on the main thread | 0.2-0.4 s | REJECTED (measured) | 0.039 s a miss, two calls a fit |
| H-F04 repeated column hashes | up to 0.5 s | REJECTED (measured) | 22 us a column hash |
| E-F04 replay pool share | 0.2-0.5 s | IMPLEMENTED | `_fe_stability_vote._host_replays` |
Lesson: measure the cost of the operation in a micro-benchmark before implementing; four estimates by reading were 3 to 10 times too high.

Noise floor (2026-11-20, unchanged tree at 5a8519ed1, GPU 72-77 C, 1695-1708 MHz, 6 runs back to back): 41.3 / 40.9 / 38.9 / 39.3 / 40.0 / 40.4 s, mean 40.13 s, sd 0.92 s, range 2.4 s. A paired difference has sd about 1.3 s, so four alternating pairs resolve a gain of roughly 1.3 s or more; items estimated below that are judged by an isolated micro-benchmark of their own cost, not by the wall.
