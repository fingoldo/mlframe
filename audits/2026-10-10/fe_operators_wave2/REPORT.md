# Wave 2 of the FE operator factory: operators M, B, C, G (2026-10-10)

Scope: the four best ideas of the brainstorm (`audits/2026-10-09/offset_product_fe/FOLLOWUP.md`) run through the factory protocol (`src/mlframe/feature_selection/_benchmarks/fe_operator_factory/README.md`).
Method: scripts in `src/mlframe/feature_selection/_benchmarks/fe_operator_factory/wave2/` (`W2/` below), 8 seeds, n of 5000 and 20000, held-out half, targets W (should win), H (a harder, noisier W with irrelevant columns), N (should not help), HN (hard no-help) and 0 (pure noise). Decision metrics: held-out MI and the relative improvement of MAE and RMSE of a ridge model and of a gradient-boosting model over the raw columns; R^2 is not used. Every number below is in `W2/results/summary.txt` and `summary.json` (per-run rows in `rows_*.jsonl`, logs in `log_*.txt`).
Written by the lead from the agent's final report because the agent could not write this file; numbers not present in `W2/results/` were not carried over.
Not run: the repository's own gating, dedup and overlap with `orth_spline` / `kfold_target_encoded`; the real `hybrid_offset_product_fe`; B against an additive baseline of two C warps; hard pure-noise layouts for B, C and G; classification targets; n of 100k or more; anything on the GPU.

## Verdicts

| Operator | Verdict | Evidence |
|---|---|---|
| C: cross-fitted 1-D binned warp | ADOPT | Held-out MI 0.79 -> 1.00 (oracle 1.01) on the idealised target; ridge MAE improves 73% (idealised) and 38% (hard); nothing accepted on N or 0; the gradient-boosting model gains nothing; 0.07 s per column at 100k rows. |
| G: row statistics over a learned column subset | ADOPT | The only operator that also helps the gradient-boosting model: MAE +6.6 to +9.3% (idealised) and +2.6 to +4.6% (hard); ridge +72% / +29%; the exact subset is learned in 8 of 8 seeds; nothing accepted on N, HN or 0. |
| M: additive-residual pair screen | ADOPT as a pair-selection step (no recipe) | Ranks the true pair first in 8 of 8 seeds on the ideal and hard targets; the y-screens rank it 4-5 (ideal) and 15-18 and 36 of 45 (the two hard pairs). Standalone gain is small: on the hard target the residual pick plus a numpy stand-in for the offset product gives ridge MAE +6.6% over raw plus warps. |
| B: cross-fitted 2-D cell table | ADOPT, gated, for linear or neural consumers only | Ridge MAE +45 to +55% on W (MI 0.31 -> 0.43-0.51, truth 0.64); the gradient-boosting model loses 0.8 to 1.7% on W. |

## Findings

| ID | sev | finding | evidence | disposition |
|---|---|---|---|---|
| W2-01 | P2 | Build order: warp service plus C first, then G, then M, then B. This departs from the README order because M's value (pair choice and cost) cannot be verified without the repository's pair stage. | the agent's cost and gain tables in `summary.txt` | DECIDED |
| W2-02 | P2 | C and B share one out-of-fold nonparametric service; its API: 5 folds, bin edges refitted inside each fold, defaults nb = 20 and shrink 10 (1-D), shrink 3 (2-D, shrunk toward the additive prior). The first 2-D default (m = 20) over-smoothed; m = 3 is better but was tested at n = 5000 only. | `log_B_5000_m3_*.txt`, `rows_B_5000_evalm3.jsonl` | OPEN - re-test m = 3 at n = 20000 before fixing the default |
| W2-03 | P2 | M needs two fixes before it is built: bins = clip(n_fit / 250, 15, 60) (15 bins leak lack-of-fit at n_fit = 10000) and a pair threshold of 60 plus a permutation floor, because the null maximum is 39.2 x n. | `m_bins.json`, `m_bins.txt`, `ablation_oof.txt` | DECIDED |
| W2-04 | P2 | Acceptance rule c = 40 (gain x n over the best existing candidate) holds for B, C and G: the calibrated null maximum is 23.2 x n_fit (C) and the evaluation null maximum is 18.4. The offset product needs the relative-gain floor as well (finding L-09 of `01_fe_followups_low_hanging.md`); B, C and G must be re-checked for the same large-n weakness. | calibration line of `summary.txt`, `log_calib*.txt` | OPEN - re-check at n = 100k and 1M |
| W2-05 | P3 | Cross-fitting protects the selection MI but did not improve downstream error; a fold-average replay is not consistently better than the full table, so the recipe stores the full table (`mode: "full"`). | `ablation_oof.txt`, the `leak\|` columns of `summary.txt` | DECIDED |
| W2-06 | P3 | Against a raw-only baseline the y-screen's additive pair looks best (ridge +17%); the order reverses once per-column warps are in the baseline, so every pair operator must be evaluated against raw plus C warps. | M logs | DOC |
| W2-07 | P3 | `fit_cell2d` returned NaN for empty cells at m = 0. | agent's run | RESOLVED - fixed in the wave-2 script |
| W2-08 | P2 | Replay is a pure function of the recipe and the source columns: 280 of 280 rows pass all five checks (JSON round trip, DataFrame by name, row permutation, single row, equality with the fit path). | `rows_*.jsonl` (`replay_all_pass`) | RESOLVED for the numpy replay of the experiments; the production recipes still need their own tests |
| W2-09 | P3 | Downstream CV used recipes frozen from the fit half, not selection inside each training fold; the gradient-boosting model ran 100 iterations at n = 20000 for B, C and G; timings were taken on a loaded box. | the agent's deviation list | DOC |

## Implementation specs (to be built in this order)

1. **Warp service and C.** Modules `_oof_warp_service.py`, `_oof_warp_kernels.py`, `_oof_warp_fe.py` under `src/mlframe/feature_selection/filters/`; recipe kind `oof_warp1d` with `extra = {full: {table}, mode: "full", lo, hi, nb, shrink}`; njit/prange per-column kernels, one device launch for all columns; acceptance c = 40 plus the relative-gain floor; wiring through the 11 touchpoints of the factory checklist.
2. **G.** `_row_stat_fe.py`, `_row_stat_kernels.py`; recipe kind `row_stat` with `extra = {stat, mu, sd, beta, clip}`; candidate columns capped at 16, scan on 20000 rows; statistics min, max, median, range, std, mean and log-sum-exp.
3. **M.** `_pair_residual_screen.py`, `_pair_residual_kernels.py`; returns a ranked pair list and stores nothing; feeds the pair pool of the existing pair stage and of the offset product (replacing the top-MI column pool).
4. **B.** The 2-D table on the same service, recipe kind `oof_cell2d`; gated so that it is offered only to linear and neural consumers (the gradient-boosting model does not gain).

Each step follows the standing rules: unit, business-value, noise-control, replay/pickle and cProfile tests, a fused device kernel that regenerates values where it applies, a kernel-tuning-cache dispatch, CHANGELOG entry, tracker rows.
