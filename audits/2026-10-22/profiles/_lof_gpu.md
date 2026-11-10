# `_lof_gpu.py` (fused device LOF scoring against a frozen reference)

Version 2 of `_lof_fe._lof_transform` (the NumPy version stays as `_lof_transform_host`, the CPU twin and the oracle of the tests). One thread scores one row: it walks the reference points, forms the squared distance
from the coordinates in double, keeps the `k` nearest in an insertion-sorted list and finishes reach-distance and density ratio in the same kernel, so no `(rows, n_ref)` matrix exists. `{REAL}` module per element type
(`_MODULES_BY_REAL`), distances and sums in double. Limits: at most 16 coordinates and 64 neighbours (else the host runs). Row blocks of 4 million rows per launch bound the upload.

## Contract points (CUDA.md section 2)

| point | status |
|---|---|
| CPU twin kept | `_lof_transform_host` |
| fused GPU kernel | yes, no intermediate table |
| streaming with a VRAM bound | row blocks of 4 million rows; the reference (2000 points) is resident |
| type-generic | `{REAL}` float / double, both tested |
| KTC dispatch | tuner `fe_lof_device` (sweep over 2000 / 20000 / 100000 rows, 18 s, picks the device everywhere on this GPU; fallback 20000 rows; `MLFRAME_LOF_DEVICE=0\|1` forces a path) |
| tests | `tests/feature_selection/fe/test_lof_gpu.py`: sizes 31 to 20011, 1 to 16 coordinates, both dtypes, unsupported shapes, dispatcher |

## Measured (GTX 1050 Ti, warm, 100000 rows x 3 coordinates, 2000 reference points, k = 20)

| path | seconds |
|---|---|
| host `_lof_transform_host` | 3.62 |
| device `lof_transform_gpu` | 0.33 (11x) |

nvprof counters of `lof_scores`: achieved occupancy 0.98, SM efficiency 96.6%, device-memory utilisation mid, warp execution efficiency 47%, global-load efficiency 12.7%, IPC 0.36, local-load transactions 647 million.
The kernel is bound by its per-thread insertion list (the sorted `best_d` / `best_j` arrays live in local memory) and by divergence of the insertion branch, not by DRAM.

Agreement with the host: float64 to 1e-9; float32 within 2e-3 relative on 99.8% of rows, the rest within 10% (the host forms float32 distances as `|a|^2 + |b|^2 - 2ab`, the device from coordinate differences in
double, so a row whose k-th and (k+1)-th neighbours are almost equidistant can take the other one).

Candidates not done: keep the insertion list in shared memory (one list of 20 doubles per thread = 160 B x 128 threads = 20 KB per block) to remove the local-memory traffic; stage the reference in shared memory
to cut the repeated global reads. Measure before restructuring (CUDA.md section 6).

## 2026-11-10: the tuner stored "host" for every size (tolerance bug)

`fe_lof_device` chose `host` at 5000, 20000 and 100000 rows although the device scorer is 10x faster (100000 rows against a 2000-point reference: host 3.30 s, device 0.31 s, `D:/Temp/lof_bench.py`). The sweep required the device scores to equal the host's to 1e-9 relative, but the scores are float32 and differ by about 3e-7, so the device variant failed the equivalence check in every region and the tuner stored the host. Tolerance now 1e-5 / 1e-6 (the float32 level), salt 3 so the regions are measured again; after the re-sweep the choice is `device` at 2000, 20000 and 100000 rows. `lof` is 3.7 s of the critical `mid_b` lane of the 100k x 100 fit.
