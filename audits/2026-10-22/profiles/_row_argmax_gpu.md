# `_row_argmax_gpu` (row-wise argmax of the conditional gate's operand triples)

Measured 2026-11-16, GTX 1050 Ti (sm_61, 112 GB/s). Shape of a call: 3 float32 resident columns of 1,000,000 rows (the gate's operand triple; the columns are uploaded once and shared by every triple through `resident_operand`).

## Launches per call and the GPU profile (nvprof, 21 calls)

Launches per call: 1 `row_argmax` (plus one 24-byte host-to-device copy of the column pointers). The earlier form launched a second copy kernel, `cp.stack` of the three columns into an (m, n) matrix (12 MB read and written, as long as the argmax itself); reading the columns in place through a pointer array removed it: 2 M rows, resident, 1.2 -> 0.7 ms.

Metrics (nvprof): `row_argmax` achieved occupancy 0.91, multiprocessor activity 98.8 %, device-memory read throughput 29 to 59 GB/s and write 19 to 33 GB/s (about 230 microseconds for 12 MB read and 8 MB written, close to the card's 112 GB/s bandwidth), global store efficiency 100 %, global load efficiency 85 % (the m pointer loads of each thread are uniform broadcasts and count as partial transactions; the column reads are coalesced).

## Host profile (cProfile, 20 calls, 1M rows x 3)

Device form: 0.010 s for 20 calls, 904 function calls (the Python around the launch 1 ms per 20 calls, the rest is the final synchronize that surfaces the GPU time). Host twin: 0.030 s for 20 calls, all of it in the `prange` numba kernel (1.5 ms per call); the first version stacked the columns with `np.stack` first (63 % of its time), the twin now takes the tuple of columns directly: 2 M rows 10.5 -> 4.0 ms.

## Backend choice and streaming

Backend choice: the tuner entry `row_argmax_backend` (a sweep over the row count, 4 columns, device timed on resident columns because the operands are shared uploads, the equivalence checked on every 13th element, the unit test compares the whole vector). On this host: host up to 100000 rows, device from 500000; fallback: the row mark 200000. `MLFRAME_ROW_ARGMAX_DEVICE=0|1` forces a side. The gate asks it before building the resident handle and takes the numba twin (`_argmax_host`) on the host side; a non-finite operand keeps `np.argmax` (its NaN semantics).

Streaming: the launch is a grid-stride loop over row blocks (at most 4096 blocks); device memory is the (n,) float64 index and the pointer array, the columns are the caller's resident operands.
