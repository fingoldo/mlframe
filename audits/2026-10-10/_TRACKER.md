# follow-up round 2026-10-10 -- master tracker

Low-hanging improvements after the offset-product operator and the FE operator factory landed (see [01_fe_followups_low_hanging.md](01_fe_followups_low_hanging.md)); the wave-2 operator evaluation (M, B, C, G) is tracked in `audits/2026-10-10/fe_operators_wave2/REPORT.md` [fe_operators_wave2/REPORT.md](fe_operators_wave2/REPORT.md).

Statuses: **RESOLVED** (done; the note names the test, file or commit that pins it), **PARTIAL** (part done; the note says what remains), **TODO** (open), **REJECTED** (measured or decided and declined, with the reason), **NOT A DEFECT** (investigated, behaves correctly), **DOC** (documented), **FUTURE** (deferred with a trigger).

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT | DOC | FUTURE |
|---|---|---|---|---|---|---|---|---|
| `01_fe_followups_low_hanging.md` | 10 | 5 | 3 | 2 | 0 | 0 | 0 | 0 |
| **Total** | **10** | **5** | **3** | **2** | **0** | **0** | **0** | **0** |

## Per-report status

| Status | Report | Findings | Area |
|---|---|---|---|
| **OPEN** | [01_fe_followups_low_hanging.md](01_fe_followups_low_hanging.md) | 10 | offset-product stage cost, dispatch, warm-up, fuzz coverage, feature scale, float64 copies |

### `01_fe_followups_low_hanging.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P3 | `L-01` | pair pre-filter before the offset-product scan | raw-MI synergy rule (joint - max marginal > 0.02 nats): recall 25/25 planted pairs, 31% of the others kept; tests pin recall and selection parity; GPU stage time unchanged |
| **RESOLVED** | P3 | `L-02` | device-born scan inputs in strict-resident mode | unary outputs, fills and clips built on the device from the uploaded scan rows (4.8 MB, was 34 MB); stage 0.306 -> 0.194 s at 100k, 0.743 -> 0.442 s at 1M; parity tests |
| **PARTIAL** | P3 | `L-03` | CPU/device dispatch of the scan through the kernel tuning cache | lookup and registered sweep done; remains: run `ensure` on a quiet box, test of the lookup |
| **PARTIAL** | P3 | `L-04` | warm the offset-product njit kernels in `mlframe-tune-kernels ensure` | warm-up specs registered; remains: run them and confirm a cold first fit no longer pays 10 s |
| **RESOLVED** | P3 | `L-05` | fuzz-combo axis for `fe_offset_product_enable` | axis, combo field, builder, enumerator and a test; 19 axis tests pass with --run-fuzz |
| **TODO** | P2 | `L-06` | scale audit of engineered columns for linear models | ridge MAE 39x worse with one unscaled feature in a smoke run; remains: per-family audit and decision |
| **TODO** | P3 | `L-07` | float64 -> float64 device copies (~1590, ~80 ms) and float64-producing families | remains: call-site map from the nvprof auto-tag, `copy=False`, per-family float32 decision |
| **PARTIAL** | P3 | `L-08` | host work around the scan (rank, quantiles, class coding) | 1M stage 0.934 -> 0.743 s (binary-search ranks, quantiles on scan rows); remains: unary maps and refit over the full column (~0.15 s) |
| **RESOLVED** | P2 | `L-09` | acceptance rule too lax at large n | relative-gain floor of 3% of the best baseline MI; only (c, d) accepted at 20k/100k/1M |
| **RESOLVED** | P1 | `L-10` | `log` unary replayed with a batch-dependent shift | frozen `log_shift_u/v` anchors in the recipe; two tests, the first fails against the registry replay |

## Wave 2 operators ([fe_operators_wave2/REPORT.md](fe_operators_wave2/REPORT.md))

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **TODO** | P2 | `W2-B1` | build the out-of-fold warp service and operator C (`oof_warp1d`) | verdict ADOPT: ridge MAE +73% / +38%, nothing accepted on N and 0, 0.07 s per column at 100k |
| **TODO** | P2 | `W2-B2` | build operator G (`row_stat`) | verdict ADOPT: helps the gradient-boosting model too (+2.6 to +9.3% MAE), exact subset in 8/8 seeds |
| **TODO** | P2 | `W2-B3` | build the pair screen M (`_pair_residual_screen`) | verdict ADOPT as a selection step: true pair ranked first 8/8; two fixes (bins rule, threshold 60 plus permutation floor) |
| **TODO** | P3 | `W2-B4` | build operator B (`oof_cell2d`), gated for linear and neural consumers | ridge MAE +45 to +55%, gradient-boosting model -0.8 to -1.7% |
| **OPEN** | P2 | `W2-02` | re-test the 2-D shrinkage default (m = 3) at n = 20000 | tested at n = 5000 only |
| **OPEN** | P2 | `W2-04` | re-check the acceptance rule at n = 100k and 1M for B, C and G | the offset product needed a relative-gain floor at large n |
