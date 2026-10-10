# Offset-product FE: follow-up reports (stat agent, integration agent, brainstorm agent)

Evidence scripts: D:/Temp/agent_stat/ (exp6-exp11, time_cost.py), D:/Temp/agent_integ/, D:/Temp/agent_brain/ (to be cleaned and moved into the package before commit).

## Stat agent follow-up (12 seeds, n=30000, 10 bins)
| ID | Finding | Status |
|---|---|---|
| S1 | 2-parameter form (u+s)(v+t) from one 4x4 OLS on rank(y) (`y~a+bu+cv+d*uv`, s=c/d, t=b/d) reaches the truth MI on all six two-sided targets; every 1-parameter form fails there | DECIDED: ship `ols2` as the primary closed form |
| S2 | Closed forms (zero-crossing, OLS) match the 1-D grid within 0.002 on 10/13 targets; ols2 costs ~1 MI/pair (0.50 s vs 3.4 s 1-D grid, 30.4 s 2-D grid, 578 pairs) | grid demoted to optional refinement |
| S3 | Shift family loses to preset on additive/no-shift targets: must always be a union with the preset, and must beat the t=0 product | TODO in implementation |
| S4 | Fallback to the median when no crossing exists costs 0.10 MI; fallback must be t=0. With multiple crossings OLS is worse than no shift (+0.10 regret) | TODO: guard + test |
| S5 | Conditional-permutation null is INVALID (destroys sub-bin signal the shifted forms use) | DROPPED (supersedes O4 variant) |
| S6 | Acceptance: held-out half, margin about 26/n (0.0136 at n=2000, 0.0052 at n=5000), plus Miller-Madow ratio to MM joint >= 0.9; refit (s,t) on all data after acceptance. In-sample margin (4/n 1-D, 7/n 2-D) fails the no-shift control (0.30 false accept) | TODO |
| S7 | Small n (< ~20*bins^2): raw ceiling rejects the truth; use 6 bins or MM-corrected ratio | TODO (relates to 0.9 prevalence ceiling item) |
| S8 | Ridge R2 +0.15..0.19 with the 2-parameter feature; HistGradientBoosting unchanged (within 0.002). Engineered features must be rank-scaled/winsorised (single-pick ridge R2 down to -1.6) | TODO: rank-scale in recipe |
| S9 | Pair pre-filter window (MM ratio preset-best/joint in [0.3,0.9), joint MI > 0.02): recall 1.00, keeps 1-5.7 of 10 pairs; optional because ols2 is ~1 MI/pair | optional |
| S10 | sqrt(x+k) family not covered (ZC .222, OLS .241, truth .330) | OPEN, won't fix now |
| S11 | Small replicate counts (R=16-76, B=16-20), minimal preset only for the null study, 2-3 seeds downstream | documented limitation |

## Integration agent follow-up
CPU kernels for the fused grid (no matrix materialisation), closed-form estimates and null prototyped in D:/Temp/agent_integ/; CUDA fused kernel designed (recompute instead of store, dedup/mm flags) and validated by numpy emulation only; CUDA never executed. Status: TODO port into package modules, run on GPU.

## Brainstorm agent (orthogonal operators, CPU, n=12000, 8 seeds, held-out MI)
Ranking: M additive-residual interaction screen (ADOPT; true pair rank 4.75 -> 1.0, false interaction MI 0.399 -> 0.009 on additive target); G row statistics over top-k columns (ADOPT; 0.147 -> 1.081); B cross-fitted 2-D binned E[y|x,z] (ADOPT; 0.294 -> 0.503, linear R2 0.006 -> 0.815); C cross-fitted 1-D warp (ADOPT; linear R2 0 -> 0.92, MI gain on sin is a 10-bin artefact); PROTOTYPE: S bit features, AA shares/entropy, J log-sum-exp (into G), L parity (check vs categorical triples), I circular phase, D prototype distance, A rank copula, F count above median; SKIP: K tropical, H oblique angle (SIR), E kNN (teacher only), B2 tree, O symbolic search (26x cost, +0.017).
Shared mechanisms: fused parameter-grid MI scan kernel, out-of-fold nonparametric warp service, whole-family permutation null service, run pair screens on additive-model residual too.
Caveats: reimplemented preset, idealised targets, no GPU, no replay test, no interaction with repo gating.
Status of adoption items: all TODO, to be scheduled after the offset operator (order: M, B+C, G).

## Where the scripts live

The evidence scripts named in the tables above were cleaned and moved into the package as `src/mlframe/feature_selection/_benchmarks/fe_operator_factory/` (protocol, script table and operator backlog in its `README.md`); run them as `python -m mlframe.feature_selection._benchmarks.fe_operator_factory.<subpackage>.<module>`.

| Old location | New location |
|---|---|
| `D:/Temp/agent_stat/` (core, core2, lib, exp1-exp11, agg6-agg11, pf_headroom, time_cost, smoke, `out/`) | `fe_operator_factory/stat_study/` (`lib.py` became `common/binning.py`; raw results in `stat_study/results/`) |
| `scripts_stat/` (exp1-exp5, lib), `case2_mi.py`, `case2_missing.py`, `dbg_case2.py` | `stat_study/` (earlier copies were identical to the agent_stat versions, so they were not duplicated) |
| `D:/Temp/agent_integ/` (offset_kernels, offset_closed_form, offset_fused_cuda, offset_null, cost_bench, t_curve) | `fe_operator_factory/kernel_prototypes/` (`null_out.txt` in `kernel_prototypes/results/`) |
| `D:/Temp/agent_integ/test_*.py` | `tests/feature_selection/fe/factory/` (converted to pytest with assertions at small n) |
| `D:/Temp/agent_brain/` (h, ops_pair, ops_multi, ops_M, ops_O, cost, results, logs) | `fe_operator_factory/brainstorm/` (results and logs in `brainstorm/results/`) |

Changes made while moving: the decision metric of the downstream scripts (`exp10_downstream.py`, `agg10.py`, brainstorm `h.run_case`) is now MAE and RMSE, reported as relative improvement over the raw-columns baseline, for a ridge and a HistGradientBoosting model. R^2 is no longer computed; the R^2 figures quoted in S8 and in the brainstorm paragraph above, and the `ds_*.json` / `results_*.jsonl` columns that hold R^2, are historical and legacy. The MAE / RMSE code was smoke-tested at tiny n only; the full experiments were not rerun, so no MAE / RMSE table exists yet for these findings.
