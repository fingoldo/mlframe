# _family_maxt_cuda_bits (device bit-set family null)

## v3: permutation-blocked kernel with tiled class bit sets (2026-11-13)

`family_null_bits_v2` was the largest kernel of the 100000 x 100 fit survey (17.4%, 5.4 s over 671 calls). Bench `D:/Temp/exp/bench_family_bits.py` (100000 rows, 1000 columns of 8 bins, 2 classes, 40 permutations).

v2 counters (nvprof, 256 threads, 40 registers, 12.2 KB dynamic shared memory per block): 8.86 ms per launch, achieved occupancy 0.73, IPC 2.04, issue-slot utilisation 34.6%, global load efficiency 99.5%,
warp execution 97.7%, **78% of the issue stalls on memory dependency**, L2 read 190 GB/s, DRAM 23 GB/s. The popcount floor I first blamed (quarter-rate `popc`, about 3.7 of the 9 ms) was not the limiter: the
kernel re-reads each column's bin bit sets once per permutation (40 x 100 MB = 4 GB through L2). Unrolling the word loop gains 6% (the compiler already unrolls it).

Attempts, same bench, checksum bit-equal in every case:
- columns x permutations register block (2 x 2) with the whole class bit sets in shared memory: 9.7 ms against 9.5 (no change), 4x slower at 100 columns: PB = 2 doubled the shared tile to 25 KB, one block per SM.
- **v3 (kept)**: a block takes PB permutations; the class words are staged in tiles of 128 words (4 KB for four permutations), the tile loop is outermost and the bins inside, with the counters of every bin of the
  column in registers (`CAP * PB * NY1 <= 32`), so each bin word is loaded once for PB permutations: **5.98 ms against 9.69 (1.62x)**; PB = 2 8.1 ms, PB = 8 (64 counters) 7.2 ms (register pressure).
  Block width 64, 128, 256: 6.18 / 5.90 / 6.22 ms.
- 100 columns 1.72 against 2.07 ms; four classes (24 counters for one permutation) and more keep v2 (15.5 ms either way).

v3 counters: 5.32 ms per launch, 100 registers, 0 local bytes, 4 KB shared, grid (10, 16) of 64 threads, occupancy 0.22, IPC 2.20, issue-slot utilisation 34.3%, global load efficiency 98.1%, memory-dependency stalls 78 -> 51.6%,
execution-dependency stalls 17.7 -> 39.4%, L2 read 145 GB/s. It is now split between latency and the dependent popcount chain. Not tried: bin skipping (the bins partition the rows, so the last bin's counts follow from the class
totals: one of 8 bins fewer to count, about 12%), warps-per-column cooperation to lift the 0.22 occupancy.

Tests: `tests/feature_selection/gpu/test_family_null_bits_v3.py` (maxima and the full MI matrix equal v2's bit for bit over five shapes with short last words and tiles, odd permutation counts, columns with 2 to 8 levels, two
and three classes, two seeds; mutants that drop the tile offset of the bit words or shift the staged class words fail 4 and 10 of 12). v3 is in the register-pressure registry (`kernel_v3`).
