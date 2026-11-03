
## 2026-11-03: f32-X fused MI kernel

Float32 edge search (edges rounded up to float32; same codes): 1000 columns x 100000 rows 0.0195 -> 0.0166 s. Rejected after measuring: a shared histogram
per warp (8 copies of the 20-cell binary table, merged before the MI) 0.0165 s, no gain: the atomics are not the limit. The kernel reads 400 MB in 16.5 ms
(~24 GB/s of ~112 GB/s), so it is latency / occupancy bound, not compare- or atomic-bound.
