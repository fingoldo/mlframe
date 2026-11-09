# `_orthogonal_copula_gpu` (device twin of `_copula_columns_njit` / `_copula_columns_radix_njit`)

Written 2026-11-09, same reason as `_orthogonal_xi_gpu.md`.

Kernel `copula_joint`: one thread per sorted position, two binary searches for the tie run, the run's mean rank, the bin, a shared-memory `nb x nb` histogram per block flushed with integer atomics; the sort is the batched device argsort. The joint tables are integers and equal the host's exactly; the MI is formed on the host from them with the host's loops, so the value equals the host's (7 tests: float32 and float64, 1500 to 40001 rows, continuous, tied, four-valued, constant and signed-zero columns; the public batch entry with the device forced on equals the host path). Blocks of at most a quarter of the free device memory; a fault returns `None` and the host kernel runs. Dispatch `use_device` (`MLFRAME_COPULA_DEVICE=0|1`, else the kernel-tuning cache `fe_copula_columns_device`, fallback host).

Timing (GTX 1050 Ti, 40 columns, synchronised, mean of 3, `bench_copula.py` in D:/Temp): 5000 rows host 1.1 ms / device 24.2 ms; 30000 rows 7.4 / 38.3 ms; 100000 rows 42.2 / 55.3 ms. The host radix kernel with one parallel task per column beats the device at every size; the device pays an upload, a thrust argsort of 22 ms at 100000 x 40 and a per-column MI loop on the host. The twin stays registered with the tuner, the fallback is the host.

Conclusion for the sort-bound scorers (xi, copula): on this card a host `prange` radix kernel is the faster path, so a device twin does not shorten the fit here; the lever for those families is fewer calls (the bootstrap resamples) or overlapping the host kernels with device work, not a device sort.
