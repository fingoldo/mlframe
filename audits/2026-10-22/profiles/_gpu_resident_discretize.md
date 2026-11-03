
## Float32-edge tiled binning kernel (`bin_codes_f32e`, 2026-11-03)
Where found: 50k x 100 GPU all-families fit (stage_sample, 80.1 s): 4.0 s on the main thread waiting in the pair analytic gate's first `.get()`;
a synchronising wrapper split it into 0.44 s of gate work and 5.26 s of the producer's queued GPU work (58 calls, 30000 x up to 3888 int8 codes).
Producer at 30000 x 3888: materialise 42 ms, radix edges 72 ms, code assignment 37 ms, float candidate D2H 132 ms.
Change: float32 candidates are compared with the edges rounded up to float32 (exact: `e <= (double)x` iff `up(e) <= x`), in a 32-column x 8-row tile
with the tile's edges in shared memory (8 KB max, at most 64 interior edges) and no 64-bit modulo. Bit-identical codes (test_bin_codes_f32_edges).
GTX 1050 Ti, 30000 x 3888, nbins=10, int8, min of 7: 26-28 ms -> 11.8-12.4 ms. Off switch `MLFRAME_FE_BIN_CODES_F32EDGES=0`; the double-compare
kernel stays for float64 input and wider edge tables. The 2026-06-27 rejection (whole (ne, K) table in shared, 40 KB, 1 block/SM) does not apply: the
tile holds 32 columns only. Next in this producer: the radix edge select (72 ms) and the float D2H (132 ms, overlapped by the copy stream).

## Float32 radix select v4: candidate compaction (`radix_select_f32_v4`, 2026-11-03)
The f64 v3 compaction ported to 32-bit keys: the byte-1 full scan (top 16 bits fixed) appends every window-matching key to a (K, max(1024, n/4)) uint32
scratch, the byte-0 pass scans only those. Bit-identical edges to v3 (test_radix_select_f32_v4: normal, tied (scratch overflow -> full scans), constant,
tiny-magnitude, mixed-sign columns). GTX 1050 Ti, nbins=10, min of 5, alternating: 30000 x 3888 66-70 -> 58-60 ms; 100000 x 583 29.5-32 -> 24-27 ms.
Launched ahead of the KTC-dispatched v3/bsearch/linear kernels when its scratch fits the VRAM budget (`_scratch_fits`); `MLFRAME_RADIX_F32_V4=0` disables it.
Measured beforehand: 466 M element visits in 72 ms = 26 GB/s, a quarter of the card's bandwidth, so the passes are match/atomic-bound and fewer visits is the lever.
