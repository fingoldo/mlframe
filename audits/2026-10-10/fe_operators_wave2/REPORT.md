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

## Follow-up: the items the first run did not test (second agent run, numbers in `W2/results/`)

Scripts: `b_vs_additive.py`, `null_layouts.py`, `classif.py`, `large_n.py`, `stress.py` (outputs `b_vs_additive.txt`, `null_layouts.txt`, `classif.txt`, `large_n.txt`, `profile_top3.txt`, `rows_*.jsonl`). Seeds: 6 (B against the warps, n = 20000), 20 and 8 (null layouts, n = 10000 and 100000), 3-4 (classification, n = 10000), 4 / 3 / 2 (large n: 20000 / 100000 / 300000). Not run: C on discrete columns downstream, one-vs-rest warps for 4 classes, n = 1M, any layout at n = 300k. The null-layout and large-n runs used a cheaper baseline than the full candidate pool, so their gains are upper bounds.

- **B against two C warps (n = 20000, m = 3).** Held-out MI on W: warp sum 0.268, B 0.504 (truth 0.640); on H 0.105 against 0.209. Ridge MAE gain of B over the warps: +51% (RMSE +52%) on W and +21% on H; gradient boosting +2.4% and +0.9%. On N, HN and 0 B adds exactly 0.000 for ridge (its MI is below the warp sum on N, -0.04, and HN, -0.009). B alone and warps plus B give nearly the same result. Shrinkage m = 3, 10, 20: ridge gain over the warps on W 0.510, 0.505, 0.497, on H 0.209, 0.208, 0.205 - consistent order, about 1-3 sd, keep m = 3.
- **Hard null layouts (10 layouts x B, C, G).** Zero false accepts with c = 40 in all 26 pure-noise cells at both sizes; the largest null gain x n was 30 at n = 10000 and 27 at n = 100000, so the noise level does not grow with n. The 3% floor alone accepts 25-50% of pure-noise seeds (the baseline MI is near zero there). One real exception: C on `discrete_sig` (a step signal on a 10-level integer column) at n = 100000 is accepted in 5 of 8 seeds (gain x n up to 159): it relabels a nominal column, like a target encoding; its downstream value was not checked.
- **Classification (n = 10000).** Relative log-loss gain over raw columns, logistic regression: binary B +17%, C +33%, G +32% (AUC +0.21 to +0.36) on W; nothing on N and 0. Gradient boosting: G +1.4% (Brier +2.4%), B and C 0 or slightly negative. The 4-class case has the same pattern. With a class order that is not monotone in the signal, C and B lose their MI edge (gain x n -71 and -41, rejected) though they still gain +20% and +7% in logistic log-loss; G is unaffected (+24.5%).
- **Large n.** Acceptance is identical at 20000, 100000 and 300000; gain x n grows linearly on the winning layouts (C_W 2042 -> 10147 -> 30577); N, HN and 0 are rejected at every n. Fit seconds: C 0.03 s per column at 100k and 0.11 s at 300k; B 0.05 s per pair and K at 100k, 0.17-0.28 s at 300k; G 0.2-1.4 s whatever n (it scans 20000 rows); replay at most 0.015 s. cProfile top self-time at 100k: B `searchsorted` 0.36 s, `partition` 0.33 s, `fit_cell2d` 0.18 s of 1.39 s; C `searchsorted` 0.04 s of 0.18 s; G `argsort` 0.72 s over 461 calls of 1.10 s.

| ID | sev | finding | disposition |
|---|---|---|---|
| W2-10 | P2 | In `run_ops.fit_B` the shrinkage `m` is overwritten by the train MI after the first pair, so every later pair was fitted with m of about 0.1: all earlier B numbers, including the m = 3 against m = 20 comparison, are invalid. `stress.fit_B2` is the fixed version; the B verdict is unchanged (the new numbers show the same ridge gain). | RESOLVED in `stress.py`; `run_ops.py` left as the historical record |
| W2-11 | P2 | Acceptance for B, C, G. The agent recommends c = 40 AND a 3% relative floor. Superseded by the finding L-11 of `01_fe_followups_low_hanging.md`: fixed margins are replaced by the gain bar of 2 standard errors of the paired MI difference (`_fe_gain_stats.paired_gain_se`) plus a practical-effect floor exposed as a constructor parameter; the agent's measurements (null gain x n at most 30 up to n = 100000) remain the evidence that the bar is not exceeded by noise. | DECIDED - build B, C, G on the shared standard-error bar |
| W2-12 | P3 | C on a low-cardinality integer column is accepted at large n and behaves like a target encoding. | OPEN - downstream check, possibly skip C for columns with at most `FEW_CLASSES_MAX` levels |
| W2-13 | P3 | A multiclass target whose class order is not monotone in the signal needs one-vs-rest warps, otherwise C and B lose their MI edge. | OPEN |
| W2-14 | P3 | G is dominated by `argsort` inside `qbin` at 100k (461 calls, 0.72 s). | OPEN - rank reuse or njit when G is built |
