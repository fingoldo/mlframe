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
| **RESOLVED** | P2 | `W2-B1` | out-of-fold warp service and operator C (`oof_warp1d`) | `_oof_warp_service.py` (njit cross-fit), `_oof_warp_fe.py`, stage, 11-point wiring, usability-pool offer, fuzz axis; acceptance by the standard-error bar (no fixed margin); 13 + 1 tests; 0.04 s for five columns at 20k, 0.2 s at 1M; remains: device-born fit for the strict mode (`W2-B1b`) |
| **RESOLVED** | P2 | `W2-B2` | operator G (`row_stat`) | `_row_stat_kernels.py` (parallel candidate kernel, sample-based bin edges), `_row_stat_fe.py`, stage, wiring, usability-pool offer, fuzz axis; baseline = best raw column and best linear mix of the subset, bar = 2 standard errors + practical-effect knob, subsets of at least 3 columns; 17 tests; exact subset recovered 12/12 and 5/5, 0 false accepts on 4 null/additive/pair layouts; 0.2-0.4 s per fit |
| **RESOLVED** | P2 | `W2-B3` | pair screen M (`_pair_residual_screen`) | built as the pair selector of B (the offset stage needs none: its pair pool recalls weak-marginal pairs 12/12); family-wise 5% Bonferroni level instead of a fixed threshold; main-effect bins `clip(n_even/250, 15, 60)`; winsorised standardised target (a rank target squashes an additive sum into false interactions - found and fixed); exact pair on bump / product / XOR, none on noise and on a strong additive target, 4 seeds each |
| **RESOLVED** | P3 | `W2-B4` | operator B (`oof_cell2d`), linear consumers only | `_oof_cell2d_fe.py`; offered to the usability pool for the pairs M reports; recipe, flag, fuzz axis, 13 tests with M; bump example ridge MAE -40% or better |
| **DOC** | P2 | `W2-02` | 2-D shrinkage default (m = 3) | re-tested at n = 20000 against 10 and 20: ordering consistent, small; keep 3 (an earlier fit bug, W2-10, had invalidated the first comparison) |
| **DECIDED** | P2 | `W2-04` | acceptance for B, C, G at large n | no false accepts with c = 40 up to n = 100000 (null gain x n at most 30); the fixed margins are replaced by the standard-error bar of L-11 (W2-11) |
| **RESOLVED** | P2 | `W2-10` | shrinkage `m` shadowed in the prototype `fit_B` | fixed in `stress.fit_B2`; verdict unchanged |
| **DECIDED** | P2 | `W2-11` | acceptance rule for B, C, G | standard-error bar + practical-effect constructor knob (supersedes c = 40 + 3%) |
| **RESOLVED** | P3 | `W2-12` | C on low-cardinality integer columns acts like a target encoding | columns with at most `FEW_CLASSES_MAX` distinct values are skipped (`test_nominal_like_columns_are_skipped`) |
| **OPEN** | P3 | `W2-13` | non-monotone multiclass needs one-vs-rest warps | |
| **RESOLVED** | P3 | `W2-14` | G dominated by `argsort` in `qbin` at 100k | the production kernel bins by edges from a 2048-row sample of the even rows (no full sort): 0.2-0.4 s per fit against 1.1 s in the prototype |
| **TODO** | P3 | `W2-B1b` | device-born warp fit for the strict-resident mode | the fit is njit on at most 100k rows (0.2 s at 1M); a fused kernel that builds bin sums per fold on the device would remove the host work |
