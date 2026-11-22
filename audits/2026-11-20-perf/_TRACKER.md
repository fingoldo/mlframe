# MRMR performance audit wave 2026-11-20 (wall time and CPU / GPU occupancy) -- master tracker

Four read-only agents (E CPU / GIL, F GPU / kernels, G overlap / scheduling, H repeated work), 21 findings by reading only; reports in this folder. Their own estimates add up to about 5-9 s at best against a 9-14 s gap to 30 s, each item is below the run-to-run noise of the benchmark (3-5 s uncontrolled, about 1 s with the GPU temperature held), so every item is judged by an alternating A/B on the benchmark (record GPU temperature and clocks) before it counts.

| Status | Finding | Estimate | Measured |
| --- | --- | --- | --- |
| IMPLEMENTED, NEUTRAL | E-F01 live stage count for the edge workers | 1-2 s | old 39.95 s, new 40.05 s (4 pairs, GPU 64-74 C) |
| IMPLEMENTED, NEUTRAL | E-F02 edge pool sized to the cores | 0-5 s | same A/B |
| **TODO** | G-F01 honest null on the idle slot | 1.0-1.4 s |  |
| **TODO** | G-F02, G-F03 | 0.5-0.8 s, 0.3-0.6 s |  |
| **TODO** | E-F03..F07, F-F01..F06, H-F01..F05 | 0.02-0.5 s each |  |
| **REJECTED** (measured) | F-F05 `Device().attributes` per call | 0.05-0.3 s | 3.1 us a call, 30 ms for 10000 calls |
| **DEFERRED** | G-F01 honest null on the idle slot | 1.0-1.4 s | the permuted labels are drawn from one rng stream in call order (`_permuted_labels_cached`): a worker that fills the cache earlier changes the draws and the floors, so it is not selection-equivalent without restructuring the stream; the gain is also inside the noise |
| **REJECTED** (measured) | E-F03 honest spec serial prep | 0.1-0.3 s | 0.032 s for 100 columns |
| **REJECTED** (measured) | E-F05 frame fingerprint on the main thread | 0.2-0.4 s | 0.039 s a miss, two calls a fit |
| **REJECTED** (measured) | H-F04 repeated column hashes | up to 0.5 s | 22 us a column hash |
| **IMPLEMENTED** | E-F04 replay pool share | 0.2-0.5 s | `_fe_stability_vote._host_replays` |
Lesson: measure the cost of the operation in a micro-benchmark before implementing; four estimates by reading were 3 to 10 times too high.

Noise floor (2026-11-20, unchanged tree at 5a8519ed1, GPU 72-77 C, 1695-1708 MHz, 6 runs back to back): 41.3 / 40.9 / 38.9 / 39.3 / 40.0 / 40.4 s, mean 40.13 s, sd 0.92 s, range 2.4 s. A paired difference has sd about 1.3 s, so four alternating pairs resolve a gain of roughly 1.3 s or more; items estimated below that are judged by an isolated micro-benchmark of their own cost, not by the wall.

Implementation pass (the owner asked for the E-H findings to be implemented before free optimisation): IMPLEMENTED E-F01, E-F02, E-F04, E-F07, F-F03, F-F04, F-F06; REJECTED by measurement E-F03, E-F05, F-F05, H-F04; DEFERRED with a stated blocker E-F06, F-F01, F-F02, G-F01, G-F02, G-F03, H-F01, H-F02, H-F03, H-F05 (see each report). The implemented items together are well under 1 s and are not separable on the benchmark (noise sd 0.92 s); each carries a sensor.
