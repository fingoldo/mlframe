# `hermite_fe/_nonpoly_cells_gpu.py` (device cells of fourier, rbf, sigmoid, pade)

Device twin of `hermite_fe/_nonpoly_cells.py` (the host cell evaluators). One RawKernel `nonpoly_cells`, one thread per output element, grid-stride loop; module compiled once per element type from the
`{REAL}` placeholder (`_MODULES_BY_REAL`, reference pattern `_als_kernels_gpu.py`). Cell values are stored as `{REAL}` (the criterion dtype), arguments and exponentials are evaluated in double.
The per-column fit statistics (range, mean / std, the nine quantile centres from one column sort, Silverman bandwidth, interquantile slope) are computed on the device from the block, so nothing but the cells is built.

## Contract points (CUDA.md section 2)

| point | status |
|---|---|
| CPU twin kept | yes: `nonpoly_cell` / `eval_basis_cell` on the host, the reference of the tests |
| fused GPU kernel | one launch per (basis, column block); no intermediate table in global memory |
| column-block streaming, VRAM bound | the caller (`_gpu_evaluate_basis_matrix` users) streams column blocks sized by `_gpu_k_chunk` |
| type-generic | `{REAL}` float / double, tested for both against the host cells |
| KTC dispatch | the host-versus-device choice of the callers is the `fe_basis_cells_device` tuner (`_basis_cells_ktc.py`) |
| tests | `tests/feature_selection/fe/test_nonpoly_cells_gpu.py`: sizes 129 / 1009 / 20000, both dtypes, all four bases, degrees without a cell, the device router on data built for each of the eight bases |

## Measured (GTX 1050 Ti, nvprof, warm, 20 columns x 100000 rows x degrees 1..9 per basis)

| kernel | launches | avg ms | note |
|---|---|---|---|
| `nonpoly_cells` | 4 (one per basis) | 17.8 (10.7 to 24.3) | 63% of the GPU time of the call; 180 cells x 100000 rows = 18 million cells per launch |
| cub sort for the quantile centres (rbf, sigmoid) | 22 + 2 | 0.76 / 5.1 | 25% of the GPU time |
| cupy mean / var / max / copy | few | 0.4 to 1.7 | statistics of fourier and pade |

`nonpoly_cells` counters (`nvprof --metrics`): achieved occupancy 0.81, SM efficiency 98.7%, device-memory utilisation low, warp execution efficiency 91%, global load efficiency 38%, store efficiency 95%,
IPC 0.79. The kernel is bound by double-precision `sin` / `cos` / `exp` on a Pascal part (1/32 double rate), not by memory.

Candidate not done: evaluate `exp` / the sigmoid in float with the argument formed in double (the stored value is float anyway; error about 1e-7 relative) - a trade of accuracy for speed on this GPU class, to be taken
only after the equivalence test passes at that tolerance; the load efficiency of 38% comes from 20-float rows read by a warp that straddles two cells' columns and is irrelevant while the kernel is compute-bound.
