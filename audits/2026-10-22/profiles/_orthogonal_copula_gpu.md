# `_orthogonal_copula_gpu` (device twin of `_copula_columns_njit` / `_copula_columns_radix_njit`)

Written 2026-11-09, tuned the same day.

Kernel `copula_joint`: one thread per sorted position; a value with no equal neighbour (the common case on a continuous column) is its own tie run and costs two comparisons, a tied value finds its run with two binary searches; the mean rank, the bin, and a shared-memory `nb x nb` histogram kept in 8 copies per block (the lane number picks the copy, so equal cells of neighbouring lanes do not serialise on one address) flushed with integer atomics. `{REAL}` / `{ACC}` placeholders. The batched device argsort sorts the columns. The joint tables are integers and equal the host's exactly; the MI of all tables is formed by one vectorised numpy expression (equal to the host's per-table loop up to the order of a floating-point sum, 1e-12 in the tests). Operands: numpy is uploaded, a resident cupy block is used as is (`_copula_mi_batch(..., device_block=...)`). Blocks of at most a quarter of the free device memory; a fault returns `None`.

nvprof metrics, 100000 rows x 40 columns: achieved occupancy 0.74 (dynamic shared memory 12.8 KB per block), shared-memory efficiency 5 % -> 35 % with the copies, global load efficiency 71 %, kernel time 4.1 ms before the lone-value fast path. Time per call of 40 columns at 100000 rows: host 42.8 ms, device 30.4 ms (5000 rows: 1.1 / 6.5 ms; 30000 rows: 7.6 / 15.2 ms). The cub radix sort is the rest of the device time.

Dispatch: kernel-tuning entry `fe_copula_columns_device`, sweep run on this host with `tune_spec` and read back in a fresh process: host at 5000 and 30000 rows, device at 100000. (A `choose()` call in the same process as `tune_spec` returned the fallback, host, for every size; read the choices in a new process.) No environment switch.

## nvprof breakdown, 2026-11-12 (GTX 1050 Ti, 30000 rows x 40 columns, 1 warm-up and 3 measured calls)

Command: `nvprof --csv` with the CUDA 12.8 CUPTI directory on `PATH` (without it nvprof does not load `cupti64_2025.1.0.dll`). Time column in ms.

| kernel | total ms | share | launches |
|---|---|---|---|
| CUB radix sort, 64-bit keys (the batched argsort of the blocks) | 25.7 + 13.2 + 1.3 + 0.5 = 40.7 | 84% | 44 + 4 + 44 + 4 |
| `copula_joint` (the joint tables) | 2.38 | 4.9% | 4 |
| other (indexing, take, memcpy HtoD) | about 5 | 10% | the rest |

GPU total 48.5 ms over four calls, about 12 ms per call. The batched sort is the device cost; the tie-run and bin kernel is 0.6 ms per call. The module is sort-bound, as the Xi wide path is (see `_orthogonal_xi_gpu.md`). The optimization point of contract point 5 is therefore the sort, not the joint kernel. Correction, measured the same day: the sort keys are float32 already, and an argsort of the same shape takes 10.4 ms for float32, int32 and uint32 keys alike (cupy returns int64 indices), so a narrower key does not help; the earlier `64-bit keys` wording was wrong. The remaining candidate is one value sort shared by the scorers that read the same resample block: the copula kernel (value order, ties taken as runs) and the Xi kernel (value order, then the tie seed) sort the same float32 columns, so one argsort could serve both. Not implemented here; it is a change to the resample block's contract and needs its own equivalence test.
