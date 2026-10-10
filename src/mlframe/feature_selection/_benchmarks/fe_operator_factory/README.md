# FE operator factory

The protocol, helpers and evidence scripts for turning a feature-engineering idea into an MRMR operator. It was built from three sets of throwaway scripts written while
fixing the red test `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_gate_vs_elementary.py::test_case2_warped_cd_interaction_still_captured_with_gate_on`
(report: `audits/2026-10-09/offset_product_fe/REPORT.md` and `FOLLOWUP.md`). The offset-product operator `(u + s) * (v + t)` is the worked example throughout.

```
fe_operator_factory/
  README.md                this protocol
  common/                  _paths.py (scratch / results folders), binning.py (qbin, plug-in MI), downstream.py (ridge / HGB MAE + RMSE harness)
  stat_study/              statistics of the offset-product operator: exp1..exp11, aggregators agg*, core/core2 numba library, results/
  kernel_prototypes/       njit fused offset-grid scorer, closed-form shifts, permutation null, CUDA source + numpy emulation, results/
  brainstorm/              17 orthogonal operator classes screened with one harness (h.run_case), results/
tests/feature_selection/fe/factory/   tests of the kernels, the emulation and the helpers
```

Run any script as `python -m mlframe.feature_selection._benchmarks.fe_operator_factory.<subpackage>.<module> [args]` (set `PYTHONPATH=src` in a checkout).
Fresh outputs go to `<system temp>/mlframe_fe_operator_factory/<subpackage>/` (`common/_paths.py`: `scratch_dir`); nothing is written into the repository. The `agg*` scripts read the committed
`<subpackage>/results/`; put `--fresh` on the command line to make them read the fresh scratch outputs instead (for `agg6` keep the preset as the first argument, e.g. `agg6 minimal --fresh`).

## 1. Protocol for a new FE operator

### (a) Idea card (write it before any code)

| Field | Content |
|---|---|
| Name and one-line definition | e.g. `offset_product`: `(u + s) * (v + t)` with `u = f(x)`, `v = g(z)` from the unary preset |
| Dependency structure it captures | what the existing pair preset (`create_unary_transformations('medium')` x `create_binary_transformations('minimal')`, 1734 combos per pair) cannot express and why (here: a factor that changes sign inside the data range, so the fixed-zero product has the wrong sign structure on one side) |
| Parameters | names, ranges, who fits them (train half only), what is frozen into the recipe |
| Closed form or search | closed form (one pass, e.g. a 4x4 OLS on `rank(y)`), single scan, or a search over a grid / trees. Closed form is preferred; a search needs a whole-family null (section d) |
| Where it plugs in | pair stage, multi-column stage, residual screen, or a shared out-of-fold service |
| Replay | the exact function the recipe applies at `transform` time; it must not reference `y` |
| Known failure modes | the cases it is expected to lose (here: additive targets, `sqrt(x+k)`, XOR sign) |

### (b) Three synthetic targets per operator: W, N, 0

* **W** (should win): the truth is exactly the structure the operator captures, plus noise at a fixed fraction of the signal std (the harness uses 0.2-0.3 * std; the statistics study used 1.0 * std).
* **N** (should NOT help): a target the existing preset already captures (a plain product, an additive target, the no-shift form). The operator must not beat the best existing candidate and must not be accepted.
* **0** (pure noise): `y` independent of `X`. Held-out MI of any candidate must be at the null level and nothing may be accepted.
* At least 8 seeds per case; the data are split in two halves, **all parameters, the best existing candidate and every selection are fitted on one half, and every reported number is measured on the other half**.
* Where to copy from: `brainstorm/ops_pair.py` (`mk` builds a W/N/0 generator, `CASES` lists the 3 cases per operator), `brainstorm/ops_multi.py` (`add`), `stat_study/exp6_tables.py` (13 targets) and `stat_study/exp7_twoparam.py`.
* Rare-class or tiny-n variants need larger n (see the small-n ceiling bias, finding O5).

### (c) Metrics

1. **Held-out MI**: held-out-half MI (10 quantile bins, plug-in, nats) of the best existing candidate (all pair preset combos plus the raw columns, chosen by MI on the fit half)
   versus the new operator's feature. `brainstorm/h.py` (`existing`, `MI`, `mi_pair`) does this; "wins" counts seeds where the new feature's held-out MI beats the best existing candidate.
2. **Downstream error**: 5-fold CV **MAE** and **RMSE** of a ridge model (standardised, alpha 1) and of `sklearn.ensemble.HistGradientBoostingRegressor` (150 iterations, no early stopping),
   each reported as **relative improvement over the raw-columns-only baseline of the same model**, `(err_raw - err) / err_raw` (positive = better). Features are chosen **inside each training fold**.
   * Full tier: `stat_study/exp10_downstream.py` (5-fold, selection inside each fold) aggregated by `stat_study/agg10.py`.
   * Screening tier: `brainstorm/h.run_case` (single split: fit half / score half; same models, same relative-improvement definition) prints and stores `rel_mae|<model>|<set>` and `rel_rmse|<model>|<set>` for the sets `ex` (best existing), `new` and `truth`.
   * Shared code: `common/downstream.py` (`make_models`, `fit_errors`, `errors_by_feature_set`, `relative_table`, `rel_improvement`).
3. **R^2 is not a decision metric** anywhere in the factory (the project owner finds it misleading): verdicts and aggregates use MAE and RMSE only. It is still computed and printed as a clearly
   labelled REFERENCE column (absolute difference to the raw baseline) so tables stay comparable with older reports. The pre-2026-10-10 result files hold only R^2:
   `stat_study/results/ds_*.json` (key `r2`) and `brainstorm/results/results_*.jsonl` (keys `lin0`, `lin_ex`, `lin_new`, `lin_truth`, `hgb`); they are legacy, labelled so, never used by an
   aggregate or verdict, and `agg10` skips and counts them. The MI numbers in those files are unaffected. The R^2 figures quoted in `FOLLOWUP.md` (S8 and the brainstorm line) are historical.
4. **State of the metric change.** The MAE / RMSE code was verified only by smoke runs at tiny size (`exp10_downstream --quick`: n = 1500, 1 seed, 2 targets; `run_case`: 1 seed, n = 1500-2000; unit tests in
   `tests/feature_selection/fe/factory/test_common_helpers.py`). **The full experiments were not rerun**, so no MAE / RMSE table exists yet for the committed runs; rerun `exp10_downstream` (about 100-170 s per target, n = 30000, seed on this box) and the `brainstorm` cases to produce them.
5. Scale warning (finding S8): a raw engineered feature with a heavy tail (a reciprocal) can destroy a ridge fit. In a smoke run of `ops_O O_W2` the best existing candidate gave a ridge MAE change of -39x versus raw. Engineered features must be rank-scaled / winsorised in the recipe before they reach any linear model; report that, do not hide it.

### (d) False-positive audit

* **Whole-family permutation null**: permute `y`, rerun the **entire** selection (base-form choice, every grid point, every pair) and record the maximum. A null that permutes `y` and rescores only the selected candidate has a 23-60 % false-accept rate at G >= 5 (finding O4). A conditional permutation inside strata of the preset-best bin is invalid because it destroys the sub-bin signal the operator uses (finding S5). `kernel_prototypes/offset_null.py` is the fast version (bin once, B shuffled-y histogram passes; deterministic across thread counts); `stat_study/exp9_fa.py` + `agg9.py` is the statistical version.
* **Noise inflation**: mean and q95 of `(family-max MI) - (preset-best MI)` on the 0 target (`stat_study/exp1_bias.py`, `exp3_offmedian_family_null.py null ...`, `agg9.py` header lines). Reported per n and per number of bins.
* **Acceptance rule**: a candidate is accepted only if its **held-out-half MI gain over the best shift-free / existing baseline exceeds `c / n`** (nats, `n` = rows scanned), then the parameters are refitted on all rows. The statistics study measured about 26 / n; the code uses **c = 40**
  (`ACCEPT_MARGIN_C` in `filters/_offset_product_fe.py`), calibrated by the largest null gain seen over 6 seeds x 15 pairs x n in 5k..300k, which was 24 / n.
* **Calibration script to rerun** whenever the family, the presets or the scan size change: `python -m mlframe.feature_selection._benchmarks.offset_product.null_gain` (prints, per n, the per-seed maximum null gain times n for a noise target and a ratio target; `c` must sit above the largest printed value).
  A new operator needs its own copy of that script pointing at its `hybrid_*_fe` function (copy it next to the operator; do not reuse `c = 40` blindly).
* **Pass criteria**: on the 0 and N targets, over >= 8 seeds (more for rates), the acceptance rate must not exceed the nominal level (0.05 here), and on N the operator must not lower the downstream error relative to the raw baseline.

### (e) Cost

Report wall time relative to the existing 1734-combo pair table at **n = 100k and n = 1M** (the table: 3.5 s / 40 s in `kernel_prototypes/cost_bench.py`; 6.2 s at n = 100k in the njit harness variant of `brainstorm/cost.py`).

* Classify: closed form (one pass, about 1 MI-equivalent per pair), single scan, or search (grid / trees / many restarts).
* Offset product measured: 80 columns (10 forms x 8 offsets) score in 0.19 s at 100k and 2.4 s at 1M (about 10 % of the table); on a 20k subsample about 1 %. For 578 pairs at n = 30000: closed-form 2-parameter 0.50 s, 1-D grid 3.4 s, 2-D grid 30.4 s (`stat_study/time_cost.py`).
* Brainstorm cost at n = 100k (`brainstorm/results/results_cost.json`, seconds): existing table 6.2; C warp 0.20; D 0.09; B cell table 0.30; G row stats 0.59; A 0.28; J 1.63; E kNN 3.2; B2 tree 1.7; O symbolic search 164.6 (26x the table); M residual screen (p = 8) 24.3.
* Fusable? An operator whose candidate is an element-wise function of at most two columns plus a few scalars can run in a **fused njit / prange / CUDA kernel that regenerates the candidate values instead of storing the `(n, K*G)` matrix**:
  `kernel_prototypes/offset_kernels.py` (CPU, exact parity with the repo binning, 1 thread per candidate so the result does not depend on the thread count) and `offset_fused_cuda.py` (CUDA design, **never run on a GPU**, numpy emulation only). Operators that need a global fit per candidate (trees, kNN, ALS) do not fit this pattern and belong on a shared out-of-fold service.

### (f) Verdict and report

* **ADOPT**: wins on W, no gain and no acceptance on N and 0, false-accept within nominal, downstream MAE / RMSE improvement on W for at least one model family and no loss on N, cost within a small fraction of the pair table, replay-safe.
* **PROTOTYPE**: wins on W but one of: cost unclear, gain mostly a binning artefact, overlaps an existing family, replay not yet specified.
* **SKIP**: no W gain over the best existing candidate, or the gain does not survive held-out, or the cost is out of proportion.
* Write the report to `audits/YYYY-MM-DD/<topic>/REPORT.md` in the format of `audits/2026-10-09/offset_product_fe/REPORT.md`: scope, a finding table `| ID | Finding | Evidence | Status |` with the evidence column naming the script, status one of
  OPEN / IN PROGRESS / RESOLVED / NOT A DEFECT / FUTURE, a decision record and the verbatim review summaries. Move finished audit files to `audits/implemented/` per the audit-wave procedure.

### (g) Implementation checklist (mirror the offset_product wiring)

1. New module(s) under `src/mlframe/feature_selection/filters/`, **each < 1000 LOC**, with njit kernels in a sibling (`_offset_product_fe.py` + `_offset_product_kernels.py`).
2. Recipe kind: add it to the `kind` `Literal` in `filters/engineered_recipes/_recipe_core.py` and a handler in `filters/engineered_recipes/_recipe_dispatch.py` (`"offset_product": _routed(".._offset_product_fe", "apply_offset_product_recipe")`).
3. Recipe registry: `recipes.<kind> = {}` in `filters/_mrmr_fit_impl/_fit_impl_core.py`.
4. Cascade params: the `_<kind>_pre_recipes=recipes.<kind>` entry in `filters/_mrmr_fit_impl/_fe_stage_cascade_run.py`.
5. A stage function called from a cascade module: `filters/_mrmr_fit_impl/_fe_stage_offset_product.py` (`_stage_offset_product`), imported and called from `_fe_stage_cascade_mid_b.py`, plus the `self.<kind>_features_ = []` reset there.
6. Roster attribute `<kind>_features_` in `filters/_mrmr_fit_impl/_fe_roster_attrs.py` **and** in `filters/_mrmr_fe_provenance.py` (**two places**: the kind-to-family map and the (attribute, family) list).
7. `filters/_mrmr_fit_impl/_fit_impl_stages/_engineered_dedup.py` (**two lists**), `_engineered_gates.py` (the recipes dict passed to the gates) and `_state.py` (`ROUTED_RECIPE_FAMILIES`).
8. MRMR `__init__` parameters in `filters/mrmr/_mrmr_class.py` (enable flag default ON when the operator corrects a measured gap, columns, max pair columns, top-k, scan rows), the legacy defaults in `filters/mrmr/_mrmr_setstate_defaults.py` (old pickles default OFF) and the enable flag's entry in `_SETSTATE_LEGACY_OVERRIDES` (also in `_mrmr_class.py`).
9. Regenerate `training/fs_params/mrmr.py`: `python -m mlframe.training.fs_params._generate`.
10. Tests: unit (closed form recovers known parameters, degenerate input returns NaN), **business value** (W target beats the best preset form), **noise control** (0 and N accept nothing), **replay / pickle** (transform reproduces the fit columns, survives pickle and NaN input), **wiring** (fit/transform round trip through MRMR, opt-out flag), **cProfile** (profile the stage at realistic n and record the top frames). See `tests/feature_selection/fe/test_offset_product_fe.py`.
11. CHANGELOG and README entry (and a note in the audit report with per-finding status).

### (h) Every script

Runtimes: "measured" = timed on this box in the session that wrote this file (1 cold numba compile included where marked); "log" = from the committed run logs; "not rerun" = original runs only, no timing available.

| Script | Purpose | Inputs / args | Runtime | Backs finding |
|---|---|---|---|---|
| `common/_paths.py`, `binning.py`, `downstream.py` | output paths, qbin / plug-in MI, ridge + HGB MAE/RMSE harness | `--fresh` flag of the aggregators | library | all |
| `stat_study/core.py`, `core2.py` | numba library: rank-bin MI, shift estimators (median, zero-crossing, OLS 1- and 2-parameter, quantile grid), preset scorer, rank-1 ALS | none | library (cold compile about 60-100 s) | S1, S2, S4 |
| `stat_study/smoke.py` | smoke run of `core` on one case-2 draw | none | about 1-2 min incl. compile (not rerun) | S2 |
| `stat_study/case2_mi.py` | MI of candidate forms of `log(2c) sin(d/3)` vs raw and joint | none | seconds (measured) | O1 |
| `stat_study/case2_missing.py` | best preset forms (minimal / medium), effect of each missing ingredient, ALS polynomial route | none | minutes (not rerun) | O1, O2, O3 |
| `stat_study/dbg_case2.py` | MRMR fit on case 2 and the rejection records for `(c, d)` | none; reads N from the biz-value test | about 1 min (not rerun) | O1 (why the pair is rejected) |
| `stat_study/exp1_bias.py` | selection bias of a quantile-offset grid, whole-grid vs selected-only null | none | minutes (not rerun) | O4 |
| `stat_study/exp2_general.py` | 9 targets: preset vs shift grid / input shift / centering / zero-crossing / affine / ALS vs 0.9 x joint ceiling | none | minutes (not rerun) | O6 |
| `stat_study/exp3_offmedian_family_null.py` | `off`: off-median crossings; `null n bins reps`: family-level null and right-target gain | `off` or `null <n> <bins> <reps>` | minutes (not rerun) | O4, O6 |
| `stat_study/exp4_prevalence.py` | 0.9 x joint ceiling bias vs n and bins (raw, Miller-Madow, debiased) | none | minutes (not rerun) | O5 |
| `stat_study/exp5_actual_case.py` | all estimators on the actual case-2 data, 5-fold OOS ALS | none | about 1 min (not rerun) | O1, O6 |
| `stat_study/exp6_tables.py` | main table: 13 targets x estimators at n = 30000 | `<minimal|medium> <seed>` -> `t_*.json` | minutes per call (not rerun) | S2, S3, S4, O6 |
| `stat_study/exp7_twoparam.py` | 1- vs 2-parameter shift families on two-sided targets | `<seed> [n]` -> `p2_*.json` | n = 1500: 82 s incl. cold compile (measured); n = 30000 minutes | S1 |
| `stat_study/exp8_zc_stress.py` | zero-crossing / OLS / grid stress across scenarios and SNR | `<reps> <A|B|C>` | minutes (not rerun; outputs in `results/stress_*.txt`) | S4 |
| `stat_study/exp9_fa.py` | false-accept / power study with whole-family and conditional nulls | `<n> <target> <R> <B> [seed0]` -> `fa_*.json` | about 11-21 s per replicate at n = 2000-5000 (log) | S5, S6, S7, S11 |
| `stat_study/exp10_downstream.py` | 5-fold CV MAE / RMSE of ridge and HGB with the preset / 1-parameter / 2-parameter feature | `<target idx...> [--quick]` -> `ds_*.json` | about 100-170 s per target, size, seed at n = 30000 (log); `--quick` 2 targets 130 s incl. cold compile (measured) | S8 |
| `stat_study/exp11_prefilter.py` | pair pre-filter study: joint vs preset-best MM MI per pair of 6 datasets | `<seed> <n>` -> `pf_*.json` | 17 s at n = 2000 (measured) | S9 |
| `stat_study/agg6.py`, `agg7.py`, `agg9.py`, `agg10.py`, `agg11.py`, `pf_headroom.py` | aggregate the runs above into the report tables (agg10: relative MAE / RMSE) | read `results/`, or the scratch outputs with `--fresh` | seconds (measured) | S2 / S1 / S6 / S8 / S9 |
| `stat_study/time_cost.py` | wall time of each shift family over 578 pairs, n = 30000 | none | about 1 min incl. compile (not rerun) | S2 (cost) |
| `kernel_prototypes/offset_kernels.py` | fused offset-grid scorer, variants a (partition) and b (histogram refine), replay kernels | library | tests about 30-60 s incl. compile (measured) | O7, integration follow-up |
| `kernel_prototypes/offset_closed_form.py` | OLS and zero-crossing shifts in one deterministic pass; CUDA twin source (never run) + emulation | library | tests as above | S1, S2 |
| `kernel_prototypes/offset_fused_cuda.py` | fused recompute-instead-of-store CUDA design + numpy emulation (**never run on a GPU**) | library | emulation seconds | O7 (cost design) |
| `kernel_prototypes/offset_null.py` | whole-family permutation null, bin once + B shuffled passes | `python -m ...offset_null` | seconds (log `results/null_out.txt`) | O4 |
| `kernel_prototypes/cost_bench.py` | 80 columns vs the 1734-combo table at 100k and 1M | `python -m ...cost_bench` | about 1-2 min (not rerun) | O7 |
| `kernel_prototypes/t_curve.py` | MI(t), OLS shift on raw vs rank y, null inflation, subsample choice of t | none | seconds (not rerun) | O7 |
| `kernel_prototypes/_synthetic.py` | `make`, `ref`: case-2 data and the materialise-then-score reference | library | library | tests |
| `brainstorm/h.py` | harness: existing-candidate baseline, `run_case`, out-of-fold helpers | library | library | brainstorm ranking |
| `brainstorm/ops_pair.py` | operators A, B, B2, E, C, D, H, I (+ W/N/0 cases) | `<cases> [--seeds S] [--n N]` -> `results_pair.jsonl` | about 20 s for 2 cases at n = 2000, 1 seed (measured) | brainstorm ranking |
| `brainstorm/ops_multi.py` | operators F, F2, G, J, K, L, AA, S, JG | as above -> `results_multi.jsonl` | under 100 s for one case at n = 1500, 1 seed (measured) | brainstorm ranking |
| `brainstorm/ops_M.py` | operator M: residual interaction screen | `[--seeds S] [--n N]` -> `results_M.jsonl` | seconds at n = 1500, 1 seed (measured) | brainstorm ranking (M) |
| `brainstorm/ops_O.py` | operator O: symbolic search (cost reference) | as ops_pair -> `results_O.jsonl` | under 100 s at n = 1500, 1 seed (measured) | brainstorm ranking (O: SKIP) |
| `brainstorm/cost.py` | wall time of every operator at n = 100000 vs the pair table | none -> `results_cost.json` | about 4 min (O alone 165 s) (log) | brainstorm cost column |
| `tests/feature_selection/fe/factory/*` | kernel parity, closed forms, CUDA emulation, helpers, MAE/RMSE harness | `pytest --no-cov -p no:randomly -q` | about 30-60 s incl. compile (measured) | regression guard |

Committed raw results (`results/`): `stat_study/results/*` backs REPORT O4-O6 and FOLLOWUP S1-S9 (`t_*`/`p2_*`/`fa_*`/`pf_*` json, `*_log_*`/`stress_*` txt, `ds_*` legacy R^2); `kernel_prototypes/results/null_out.txt`; `brainstorm/results/results_*.jsonl|json` and `log_*.txt` back the operator backlog below.
`results_multi.jsonl` holds two `K_W` rows (a first run with 0/8 wins and a rerun with 8/8); cases run with `pairs=False` (`F_0`, `F2_*`, `L_N`, ...) use the raw columns alone as the existing baseline, so their "wins" are not comparable with the pair cases.

## 2. Operator backlog (verdicts from `FOLLOWUP.md`)

Held-out MI, n = 12000, 8 seeds, existing = best of raw + 1734 pair combos (`brainstorm/results/*`). Build order: **M, then B + C on one shared out-of-fold service, then G**, then the PROTOTYPE list.

| Order | Operator | Verdict | Evidence (held-out MI existing -> new, truth) | Notes |
|---|---|---|---|---|
| 1 | **M** residual interaction screen | ADOPT | true pair rank 4.75 -> 1.0; false interaction MI on the additive N target 0.399 -> 0.009 (null top 0.008) | run the pair screens on the residual of an out-of-fold additive model; cost 24 s at p = 8, n = 100k, so restrict to a pre-screened set |
| 2 | **B** cross-fitted 2-D binned `E[y|x,z]` | ADOPT | 0.294 -> 0.503 (truth 0.638); ridge R^2 0.006 -> 0.815 (legacy metric, re-measure with MAE/RMSE) | 0.3 s at n = 100k; needs the shared OOF service; N target: new 1.03 < existing 1.08 so never preferred |
| 2 | **C** cross-fitted 1-D warp | ADOPT | linear R^2 0 -> 0.92 (legacy); the MI gain on `sin` (0.797 -> 0.997) is partly a 10-bin artefact | 0.2 s; same OOF service as B |
| 3 | **G** row statistics over the top-k columns | ADOPT | 0.147 -> 1.081 (= truth) | 0.6 s; J (log-sum-exp) folds into G |
| 4 | S bit features (popcount, trailing zeros, digit sum) | PROTOTYPE | 0.222 -> 1.089 | defined for integer-valued columns only |
| 4 | AA compositional shares / entropy | PROTOTYPE | 0.147 -> 0.987 | shares are defined for positive columns only |
| 4 | J log-sum-exp | PROTOTYPE (into G) | 0.331 -> 0.976 | |
| 4 | L parity | PROTOTYPE | 0.006 -> 0.602 (truth 0.677) | check against the categorical-triple machinery first |
| 4 | I circular phase | PROTOTYPE | 0.637 -> 0.727 | the period is a parameter; see `_benchmarks/bench_modular_period_detection.py` |
| 4 | D distance to a target-extreme prototype | PROTOTYPE | 0.463 -> 0.713 | |
| 4 | A rank copula product | PROTOTYPE | 1.197 -> 1.211 | small gain |
| 4 | F count above median | PROTOTYPE | 0.090 -> 0.333 (exhaustive F2 0.546, truth 0.728) | |
| - | K tropical min/max of sums | SKIP | first run 0/8 wins; rerun 0.417 -> 1.082 on a K-shaped W only | narrow family |
| - | H oblique angle (SIR) | SKIP | 0.698 -> 0.781 | labelled SIR in `FOLLOWUP.md`; no gain over the pair preset worth a new family |
| - | E kNN | SKIP as a feature (teacher only) | 0.294 -> 0.631 | slowest nonparametric (3.2 s) |
| - | B2 tree | SKIP | 0.294 -> 0.454 | dominated by B |
| - | O symbolic search | SKIP | +0.017 MI for 26x the pair table | |

Shared mechanisms the backlog needs: a fused parameter-grid MI scan kernel, an out-of-fold nonparametric warp service, a whole-family permutation null service, and running the pair screens on the additive-model residual too.
Caveats from the brainstorm: reimplemented preset, idealised targets, CPU only, no replay test and no interaction with the repo gating; the ridge / HGB numbers there were R^2 and must be redone as MAE / RMSE before an operator is adopted.

## Lessons from building the first five operators (offset product, out-of-fold warp, row statistics, pair screen, 2-D cell table)

Each of these cost a bug or a false start that the protocol above did not catch until a whole-MRMR check or a many-seed rate measurement; build them into the next operator from the start.

1. **Measure false-accept RATES over many seeds on null AND on structured-but-uninteresting layouts**, not one seed: pure noise, a weighted additive target, a nonlinear additive target, a pair product (the operator must not claim what a cheaper existing form owns). The offset product accepted an unrelated pair in 35-50% of the runs of a layout whose single-seed check looked fine.
2. **A candidate needs a baseline that contains the cheap degenerate version of itself.** A product with large shifts is a weighted sum; a median of two columns is their mean; a warp of a monotone column bins like the column. Compare against the best weighted sum of the candidate's own factors / the best linear mix of its own columns, with weights chosen by MI or least squares on the selection half and scored on the held-out half.
3. **No hand-set margins.** The acceptance bar is `Z` standard errors of the PAIRED difference of two plug-in MIs (`filters/_fe_gain_stats.paired_gain_se`: the spread of the per-row difference of the pointwise MIs over sqrt(n)), `Z` a normal critical value; a practical-effect floor is a constructor parameter with a documented default, never a literal in a function. Pair screens use a family-wise level (Bonferroni over the pairs).
4. **Do not rank-scale a target you intend to test for additivity.** The CDF squashing of an additive sum interacts (it made every strongly additive pair look like an interaction). Winsorise and standardise instead; keep the rank transform for calibrated-mean tables where only the order matters.
5. **The additive model of an interaction screen must be fine enough** (`clip(n_even / 250, 15, 60)` main-effect levels), or the lack of fit of a steep effect is called an interaction.
6. **Replay must not depend on the batch.** The registry `log` shifts by the batch minimum; a recipe freezes the anchor at fit time (`log_shift_u/v`). Test replay on a batch whose statistics differ from the fit's.
7. **Evaluate each consumer on its own list.** Trees use `transform` (the MI list), linear models `transform_usability("linear")` with `usability_aware_lists=True`; scoring ridge on the MI list reported a regression that was not one. Offer an operator's columns to the usability pool when a linear model is the one that gains (`_usability_*_pool.py`).
8. **The MRMR fit memo is keyed by data and constructor parameters.** An A/B that changes only a patch must call `MRMR.clear_fit_cache()` between arms, or the second arm returns the first one's fit.
9. **Profile with nvprof ranges before fusing** (`_benchmarks/profiling/`: `nvprof_autotag.py`, `nvprof_range_summary.py out.txt <kernel-substring>`): the copy kernels were `cp.stack` assembly copies, not `astype`, and the biggest launch consumer was a per-column sequential loop (batched, 3960 launches -> a handful).
10. **Wiring** is mechanical and long (recipe kind, dispatch, registries, constructor, setstate defaults, `fs_params` regeneration, fuzz axis, usability hook): `python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wiring --kind K --family F --flag fe_F_enable --fn apply_K_recipe --module _F_fe --prev row_stat` does a dry run of the ~25 edits (anchored on the lines of the previous operator, `--apply` writes them; `tests/feature_selection/fe/factory/test_wiring.py` pins it); the stage module (copy `_mrmr_fit_impl/_fe_stage_row_stat.py`), the usability hook, `fs_params._generate`, tests and CHANGELOG / audit rows stay manual.
