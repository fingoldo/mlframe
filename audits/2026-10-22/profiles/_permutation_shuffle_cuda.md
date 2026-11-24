# _permutation_shuffle_cuda (sort-free target shuffles of the order-1 maxT floor)

Module: `src/mlframe/feature_selection/filters/_permutation_shuffle_cuda.py`; caller: `_permutation_null_resident.gen_target_shuffles_cupy` (the device shuffle generator the strict-resident fit forces on for the order-1 floor,
`stage:fdr_gain_floor`; `MLFRAME_SHUFFLE_GEN=sort` keeps the argsort version, `gen_target_shuffles_cupy_sorted`). Measured 2026-11-24 on the development host (GTX 1050 Ti, 6 SMs; i7-7700HQ), GPU 66 C, SM clock 1721 of 1911 MHz, no
throttle, nothing else running. Tools: cProfile / wall timers for the host side, nvprof 12.8 (counters, trace), a counter thread for the GIL.

## Why

py-spy `--gil` over the 100000 x 100 all-families fit (24.3 s of GIL-held time in a 47 s fit) put `device_argsort` first among the mlframe functions (1.33 s innermost, 1.85 s with its callers), 0.87 s of it under
`gen_target_shuffles_cupy` on the serial path of the floor. The generator drew a `(nperm, n)` float64 key matrix and argsorted it (float64 keys: float32 uniforms would tie).

Versions kept: `gen_target_shuffles_cupy_sorted` (random keys + argsort + gather) and the Feistel kernel (`feistel_shuffle`, one launch for all rows, selected by default).

## Wall of one generator call (device idle after it, no other thread running; median of 7)

| shape (nperm, n) | argsort of keys | Feistel kernel |
|---|---:|---:|
| (25, 100000) | 43.5 ms | 4.5 ms |
| (50, 100000) | 80.9 ms | 10.1 ms |
| (100, 30000) | 33.6 ms | 4.1 ms |

Whole fit (`spy_fit2.py`, 100000 x 100 all families, quiet machine, same eleven selected columns): 40.4 to 40.7 s before, 39.8 s with the kernel. Always batching the device sorts (the safe wrapper sorts a vector at a time from 80000 elements)
on top of it: 39.8 s, so that threshold is left as it is.

## GIL

A bare `cp.argsort` / `cp.sort` of a 2-D matrix holds the GIL for most of its synchronous run: a Python counter thread keeps 8.9 % (argsort of 25 x 400000 float64, 124 ms), 16.1 % (sort of the same, 107 ms) and 34.4 % (argsort of 50 x 100000
float64, 84 ms) of its free pace, against 94 to 95 % while a thread sleeps; a 1-D `cp.sort` of 8 M float32 keeps it at 62 %. The Feistel launch is asynchronous (0.08 ms of Python for the launch, 0.26 ms for the key upload): the
counter thread keeps 96 to 111 % of its pace during the call (the sleep baseline of the same runs: 102 to 106 %).

## Launch and counters (`nvprof`, one call of (50, 100000): 5 million threads, int32 labels)

Launches per call: 1 kernel (`feistel_shuffle`), 1 host-to-device copy of the `(50, 8)` round keys (1.7 us). The kernel takes 6.67 to 6.80 ms of the 8.25 ms the call takes end to end (the rest is the key upload, the launch and the
synchronising read of the benchmark); share of the GPU time in the window 99.97 %.

Grid (391, 50, 1), block 256 threads, 22 registers per thread, no shared memory, no local memory (local load transactions 0).

| metric | value |
|---|---:|
| achieved occupancy | 0.721 |
| SM efficiency | 99.1 % |
| executed IPC | 3.65 (peak 4) |
| issue-slot utilization | 86.7 % |
| global store efficiency | 96.2 % (adjacent threads write adjacent elements of a row) |
| global load efficiency | 12.5 % (the round keys are one word read by the whole block, a broadcast, and the label gather is data dependent: 0.4 MB of labels stays in L2) |

Reading: issue-bound. The work is the round function: eight rounds of a murmur3-finalizer mix (two 32-bit multiplies, quarter rate on Pascal) per walk, 2.6 walks per element on average at n = 100000 (the domain is the
next even power of two, 2^18, over n), so about 1e8 round evaluations in 6.7 ms. Fewer rounds would be faster and weaker: eight is twice the four a Luby-Rackoff pseudorandom permutation needs.

## Quality (`tests/feature_selection/gpu/test_permutation_shuffle_cuda.py`, 20 tests)

- every row is a permutation of the labels for n = 1, 2, 3, 5, 31, 32, 33, 100, 1009, 4096, 4097, 100000 and for int8 / int16 / int32 / int64 labels;
- the image of every position is uniform: chi-square over the n x n table of `pi(i) = j` counts, chi-square per degree of freedom 0.9974 (n = 40, 300 seeds, standard deviation of the mean 0.0021), 0.9995 (n = 7, 300 seeds, 0.0126),
  0.9999 (n = 257, 100 seeds, 0.0006); the p-values of the 300 seeds against a uniform: Kolmogorov-Smirnov p = 0.046, 0.941, 0.959;
- the joint image of positions 0 and 1 (n = 12, 60000 rows) is uniform over the 132 ordered pairs of distinct values (p = 0.42) and never equal;
- fixed points, descents and successions have the means of a uniform permutation (1, (n - 1) / 2, (n - 1) / n) within five standard errors; two rows of one matrix have correlation below 0.1 over positions.
The rows are pseudorandom permutations, not draws from all of S_n; the floor is statistically equivalent to the sort-based one, as it already was to the host generators (a different stream).

## Not done

- The other users of the safe sort (`device_argsort` / `device_sort`: the KSG block sums, the OOF-warp block statistics, rank binning, the grouped quantiles): 1.0 s of GIL-held time together in the fit, flat across five call sites.
- Hoisting the round keys into registers (a broadcast load per round today): the kernel is multiply-bound, a few percent at best.
