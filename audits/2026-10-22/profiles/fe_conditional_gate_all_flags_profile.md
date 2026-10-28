# fe_conditional_gate_enable: profile inside the all-flags fit (n=20000, default GPU mode)

Measured 2026-10-28 with `D:/Temp/cg_allflags.py` (all FE flags on, warm fit on other data, default device mode; GTX 1050 Ti).

The gate's scan (`cheap_conditional_gate_scan`) takes 4.56 s of cProfile time in this fit:

| part | cum s | calls | note |
|---|---:|---:|---|
| `_gate_device_born_all_mi` | 3.81 | 24 | device-born gate MI for every pending tau grid |
| `gate_grid_mi_resident` (inside it) | 3.80 | 24 | 0.16 s per call: the resident grid MI kernel on the device |
| `_scan_gate_column` (whole scan) | 4.24 | 1 | the rest is host work and bookkeeping |

Reading: the family is already GPU-resident (the GPU profile in `fe_family_walls_gpu_vs_cpu_n20000.md` shows 4.68 s on GPU, 7.74 s on CPU). Its remaining cost is the device grid-MI kernel, 0.16 s per flush, so the next lever is that kernel (batch size per flush, or fewer flushes), which is a GPU change with its own equivalence check. Not changed here.

Separate heavy part in the same fit, not in this family: `_gate_engineered_accuracy` (`measure_feature_uplift`, 4.21 s, a different gate). Recorded for the next profile round.
