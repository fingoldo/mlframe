# follow-up round 2026-10-10 -- master tracker

Low-hanging improvements after the offset-product operator and the FE operator factory landed (see [01_fe_followups_low_hanging.md](01_fe_followups_low_hanging.md)); the wave-2 operator evaluation (M, B, C, G) is tracked in `audits/2026-10-10/fe_operators_wave2/REPORT.md` [fe_operators_wave2/REPORT.md](fe_operators_wave2/REPORT.md).

Statuses: **RESOLVED** (done; the note names the test, file or commit that pins it), **PARTIAL** (part done; the note says what remains), **TODO** (open), **REJECTED** (measured or decided and declined, with the reason), **NOT A DEFECT** (investigated, behaves correctly), **DOC** (documented), **FUTURE** (deferred with a trigger).

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT | DOC | FUTURE |
|---|---|---|---|---|---|---|---|---|
| `01_fe_followups_low_hanging.md` | 14 | 10 | 2 | 0 | 0 | 1 | 1 | 0 |
| **Total** | **14** | **10** | **2** | **0** | **0** | **1** | **1** | **0** |

## Per-report status

| Status | Report | Findings | Area |
|---|---|---|---|
| **OPEN** | [01_fe_followups_low_hanging.md](01_fe_followups_low_hanging.md) | 14 | offset-product stage cost, dispatch, warm-up, fuzz coverage, feature scale, float64 copies |

### `01_fe_followups_low_hanging.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P3 | `L-01` | pair pre-filter before the offset-product scan | raw-MI synergy rule (joint - max marginal > 0.02 nats): recall 25/25 planted pairs, 31% of the others kept; tests pin recall and selection parity; GPU stage time unchanged |
| **RESOLVED** | P3 | `L-02` | device-born scan inputs in strict-resident mode | unary outputs, fills and clips built on the device from the uploaded scan rows (4.8 MB, was 34 MB); stage 0.306 -> 0.194 s at 100k, 0.743 -> 0.442 s at 1M; parity tests |
| **RESOLVED** | P3 | `L-03` | CPU/device dispatch of the scan through the kernel tuning cache | sweep run on a quiet box: device 3.5x to 9.2x faster in all 9 cells; lookup test |
| **RESOLVED** | P3 | `L-04` | warm the offset-product njit kernels in `mlframe-tune-kernels ensure` | warm-up specs ran; fresh-process first call 3.1 s (2.66 s CUDA context), the 10 s compilation is gone |
| **RESOLVED** | P3 | `L-05` | fuzz-combo axis for `fe_offset_product_enable` | axis, combo field, builder, enumerator and a test; 19 axis tests pass with --run-fuzz |
| **DOC** | P2 | `L-06` | scale audit of engineered columns for linear models | closed as designed: linear models use the usability list (`transform_usability("linear")`), the MI list is for trees; on the linear list ridge is identical with the family on and off |
| **PARTIAL** | P3 | `L-07` | float64 -> float64 device copies (~1590, ~80 ms) | attributed (assembly copies of `cp.stack`, not redundant `astype`); binagg fixed: stage launches 1990 -> 650, OOF build 3.5x at 100k / 1.6x at 1M; remains: retention (~206), conditional gate (77), bases (35) |
| **PARTIAL** | P3 | `L-08` | host work around the scan (rank, quantiles, class coding) | 1M stage 0.934 -> 0.743 s (binary-search ranks, quantiles on scan rows); remains: unary maps and refit over the full column (~0.15 s) |
| **RESOLVED** | P2 | `L-09` | acceptance rule too lax at large n | relative-gain floor of 3% of the best baseline MI; only (c, d) accepted at 20k/100k/1M |
| **RESOLVED** | P1 | `L-10` | `log` unary replayed with a batch-dependent shift | frozen `log_shift_u/v` anchors in the recipe; two tests, the first fails against the registry replay |
| **RESOLVED** | P1 | `L-11` | unrelated pairs accepted (a weighted sum with better weights credited as an interaction) | 7/20 and 4/8 runs before; 0 of 68 after (weighted-sum baseline of the winner's own factors, 2-standard-error bar, practical-effect knob `fe_offset_product_min_relative_gain`); real pair kept 28/28; the surrogate-null and free-form-additive attempts were rejected and are documented; 4 tests incl. a teeth test |
| **NOT A DEFECT** | P2 | `L-12` | ridge error on one seed in three with the family on | ridge was scored on the MI list (trees); on the designed linear list it is unchanged, and L-13 gives it the family's gain |
| **RESOLVED** | P2 | `L-13` | offset products missing from the usability pool of the linear list | candidates added to the pool; ridge MAE +6.8%, +5.2%, +4.2% on three seeds (linear list), gradient boosting +1.6%, +5.4%, +0.9%; 2 tests |
| **RESOLVED** | P2 | `L-14` | one Fourier detection per column (3960 launches) | columns of a request run as one batch; frequency lists equal; `_propose_fourier_both_warps` 0.435 -> 0.193 s, escalation 0.875 -> 0.584 s (cProfile); 3 parity tests |

## Wave 2 operators ([fe_operators_wave2/REPORT.md](fe_operators_wave2/REPORT.md))

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **TODO** | P2 | `W2-B1` | build the out-of-fold warp service and operator C (`oof_warp1d`) | verdict ADOPT: ridge MAE +73% / +38%, nothing accepted on N and 0, 0.07 s per column at 100k |
| **TODO** | P2 | `W2-B2` | build operator G (`row_stat`) | verdict ADOPT: helps the gradient-boosting model too (+2.6 to +9.3% MAE), exact subset in 8/8 seeds |
| **TODO** | P2 | `W2-B3` | build the pair screen M (`_pair_residual_screen`) | verdict ADOPT as a selection step: true pair ranked first 8/8; two fixes (bins rule, threshold 60 plus permutation floor) |
| **TODO** | P3 | `W2-B4` | build operator B (`oof_cell2d`), gated for linear and neural consumers | ridge MAE +45 to +55%, gradient-boosting model -0.8 to -1.7% |
| **OPEN** | P2 | `W2-02` | re-test the 2-D shrinkage default (m = 3) at n = 20000 | tested at n = 5000 only |
| **OPEN** | P2 | `W2-04` | re-check the acceptance rule at n = 100k and 1M for B, C and G | the offset product needed a relative-gain floor at large n |
