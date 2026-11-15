# Profile record: device operand table of the pair search (`_gpu_resident_operand_build`, host evaluator `feature_engineering`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s, double rate 1:32), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: 6 float64 raw operands, 9 plain unaries each (`identity`, `neg`, `abs`, `sqr`, `sin`, `cos`, `tanh`, `cbrt`, `exp`): a 54-column float32 operand table, 100000 rows unless stated.

## What the module is
The pair search keeps an `(n, n_operands)` float32 table of transformed operands. The host builds it with the numpy / numba unaries of `feature_engineering`; this module can instead rebuild its plain unary columns on the device from the small raw operand columns (one `batch_unary` launch per float group, an op-code switch per column, written at float32 straight into the table) and copy only the fitted columns from the host, so that the bulk of the table never crosses PCIe. The kernel-tuning entry `operand_table_backend` chooses between that rebuild and uploading the host table by row count.

## Column blocks (streaming)
The raw operands are uploaded a column block of specs at a time: `_spec_blocks` cuts the specs into consecutive blocks whose distinct raw operands hold at most a quarter of the room under this thread's pool limit, each block uploads its raws (one matrix per dtype group), runs its batch launch and releases them, and the table itself is written column by column in place. A test with a budget of one raw operand per block compares the table with the single-block one exactly.

## What was changed on this path (2026-11-15)
- The batch kernel's output type was a bare `float` (offline contract finding); it is now the `{EDGE}` placeholder, rendered `float`, and the kernel has its offline recipe.
- The columns of a batch are processed sorted by op code: the threads of a warp run consecutive columns of one row, so equal ops sit side by side and a warp takes one or two branches of the op switch instead of one per op. `batch_unary` 19.3 -> 9.4 ms per launch at 54 columns x 100000 rows; the rebuild wall 40.6 -> about 21 ms. The table is unchanged (a test compares every column with the cupy unary chain value for value).
- A 32-bit index division was tried and measured neutral, and left out.

## cProfile (one device rebuild, 100000 rows)
0.026 s for 743 function calls: 0.020 s is the blocking copy of the comparison statistics of the sweep, 0.003 s the `column_stack` of the raw group on the host, 0.005 s the build itself (`_build_resident_operand_table_part1` and the prepass). The build is one kernel launch plus a handful of uploads.

## nvprof (hardware counters)
Launches per rebuild: 1 `batch_unary` (all 54 columns), the uploads of the raw group and of the specs. GPU time of the rebuild 10.3 ms: `batch_unary` 9.4 ms; achieved occupancy 0.74, global load efficiency 79.6 percent, DRAM read 2.5 GB/s, double-precision FLOP efficiency 9.7 percent of the 1:32 peak: the kernel is bound by the double-precision transcendentals (`sin`, `cos`, `tanh`, `exp`, `cbrt`) that the group type requires, not by memory. Whole-GPU utilization of the rebuild: 9.4 ms of kernel time in a 21 ms wall.

## Wall clock (upload of the host table against the device rebuild, same table)
| rows | upload | rebuild |
|---|---|---|
| 5,000 | 1.0 ms | 1.9 ms |
| 30,000 | 4.2 ms | 7.5 ms |
| 100,000 | 13.6 ms | 21-26 ms |

On this card the upload of the host table is faster than the rebuild at every swept size: PCIe moves the 54-column float32 table (21.6 MB) in 13.6 ms, while the rebuild computes the transcendentals in double on a 1:32 double pipe. The entry therefore picks the upload here, and the builder returns the uploaded table; a card with a faster double pipe (or a slower link) picks the rebuild. The rebuild computes each column in its group's type, as the CPU path does, so it equals the host columns to float32 rounding.
