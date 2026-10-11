# Target transforms inside the MRMR scan: design and audit, 2026-10-11

Scope: the idea "try not only feature transforms but also TARGET transforms in the MRMR scan, cache the bins of every T_k(y), and score each candidate against the best of the K targets". Nothing was implemented. Every claim below was checked against the code at `D:/Temp/mlf_merge` (paths relative to `src/mlframe/`); where a check was not possible the text says so. Measurements are tiny read-only snippets (a few seconds, noisy box) and are labelled; everything else sized by reasoning is labelled ESTIMATE.

Status values for findings: OPEN, DECIDED (a recommendation the owner can overrule), NOT A DEFECT, FUTURE.

## 0. Verdict

| Question | Answer |
|---|---|
| Worth doing as a general "best of K targets" scan inside MRMR? | No. The headline benefit does not survive: y is already quantile-binned, so every monotone T_k gives bit-identical codes (section 2). The non-monotone transforms that do add information measure a different quantity (dispersion, a tail event, a residual), and an MI gain against T(y) says nothing about MAE/RMSE of a model of y (section 4). The repo's own composite-target code hit exactly this: `eps_mi_gain = -10.0` disables its MI pre-filter because the correct composite has structurally lower MI than raw y (`training/_composite_target_discovery_config_base.py:405-424`). |
| Is part of it already implemented? | Yes, three times. (1) Composite-target discovery in `training/composite/` screens a registry of ~25 target transforms by MI, re-ranks by tiny-model CV RMSE and an honest holdout, then trains one real model per accepted spec (section 1). (2) The residual rescue (`_residual_rescue_fe.py`) and the pair residual screen (`_pair_residual_screen.py`) already ask "what is left of y after the selected design" on the target's own scale with held-out evidence. (3) The sufficient-summary stop (`_fe_sufficient_summary.py`) already tests MI of the residual `r = y - E_hat[y | selected]` against a maxT permutation null. |
| What is genuinely missing? | A small family of NON-monotone target views used as extra scoring lenses for FE candidates that fail the identity gate: the absolute centred residual (dispersion), tail indicators, and the residual of the selection used as the target of the PAIR search (the pair screen has this for pairs; the unary/binary operator search and the offset product do not). |
| Expected gain | Small on selection quality, real only on targets with signal in the variance or in a regime that the mean-scale residual rescue cannot see. For a model whose metric is MAE/RMSE on y, a feature selected "for |y - m|" helps only if the model can use it (trees yes through splits; linear no). Not measured; section 8 gives the experiment. |
| Main risk | Selection bias from max over K x M (section 4) and a second, silent definition of "relevance" that disagrees with the downstream metric. Secondary: touching ~25 registration points and the fit memo for an operator that changes what `y` means to every stage (section 5). |
| Interference with the composite approach | None if the lens is internal to one MRMR fit and never changes the y the training suite passes in. The training-level composite path calls MRMR per composite target (evidence: `mrmr_skip_when_prior_was_identity`, `mrmr/_mrmr_class.py:1288-1299`); a lens inside MRMR would double count with it unless the lens family excludes everything composite discovery already owns (section 5). |
| Recommendation | Variant (a)+(d) of section 6: no K-target scan. Add a K = 2 optional "residual and dispersion lens" restricted to the pair/offset search, confirmed on held-out rows, behind a flag that defaults OFF until the section 8 experiment shows a downstream MAE win. K = 1 is byte-identical by construction. |
| Single first experiment | `exp_target_lens.py` in `_benchmarks/fe_operator_factory/wave2/`: on 5 synthetic targets (section 7.3) compare, over 20 seeds, the best existing candidate by held-out MI against identity y versus against the lens set, then score ridge and HistGB MAE/RMSE with the selected sets. Decision rule: ship only if the downstream MAE improves by more than 2 paired standard errors on the heteroscedastic and regime targets and the false-lens rate on noise is at the nominal level. |

## 1. What "composite" means in this repo

`grep -il composite` over `src` and `tests` returns more than 40 source files. They split into four unrelated meanings. The owner's question ("is it fitting a separate model for each feasible target transform?") only matches meaning A.

| # | Meaning | Where | Does it fit a model per transform? |
|---|---|---|---|
| A | Composite TARGET: `T = f(y, base)` (diff, ratio, logratio, linear/robust/Theil-Sen/monotonic/quantile residual, ewma residual, frac_diff, cbrt, signed power, log, Yeo-Johnson, quantile-normal, chains such as `chain_linres_cbrt`); a wrapper learns T from X and inverts at predict time | `training/composite/` (`__init__.py:1-60` docstring, registry `transforms/registry.py:148-560`, `CompositeTargetEstimator` in `estimator/_estimator.py`, `CompositeTargetDiscovery` in `discovery/__init__.py:96`) | Screens cheaply, then yes for the accepted few (below) |
| B | Composite FEATURE: an engineered column that composes operators, e.g. `add(div(sqr(a),neg(b)), mul(log(c),sin(d)))`; "C2" in the audits is the additive-fusion compound | `_fe_additive_fusion.py`, `_fe_additive_fusion_gpu_resident.py`, ctor flag `fe_additive_fusion_enable` default True (`mrmr/_mrmr_class.py:1744-1750`); feed-forward of engineered columns into the next pair search (`mrmr/_mrmr_class.py:556-600`, `:833-862`) | No, one MRMR fit |
| C | Composite GROUP KEY aggregates (multi-column group keys) | `_composite_group_agg_fe.py`, `fe_composite_group_agg_*` (`mrmr/_mrmr_class.py:2668-2673`) | No |
| D | Test-suite vocabulary: "composite-all-FE-on" means the mega fixture with every FE family enabled; `test_composite_fe_retains_strongest_signal` (`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_underselection.py:103`) is about that, not about targets | tests | No |

The owner almost certainly means A (the phrase "a separate model for each feasible target transform" is A's vocabulary), and the question "will it interfere with the Composite approach" is about A versus a new MRMR feature. Meaning B is the one that overlaps MRMR internals (section 5).

### 1.1 How meaning A actually works (answer to question 1)

It is a three-stage funnel, not one model per transform:

1. Screening by binned MI. For each (base, transform) pair, transform parameters are fitted on `train_idx` only, T is computed, and `mi_gain = MI(T, X_without_base) - MI(y, X_without_base)` is formed (`discovery/__init__.py:104-135`; `mi_estimator = "bin"`, `mi_nbins = 16`, `training/_composite_target_discovery_config_base.py:448-449`). Specs are cut to `top_k_after_mi = 32` (`:409`). The gain threshold `eps_mi_gain = -10.0` effectively disables the pre-filter (`:405-424`).
2. Tiny-model rerank. `discovery/_tiny_rerank*.py`, `_eval_waic.py`, `_yscale_scoring.py` rescore survivors by a small model's CV RMSE on the y scale, with WAIC and an honest holdout (`_honest_holdout.py`, `_honest_rmse_gate.py`), and the `min_honest_gain_to_train = 0.001` / `min_honest_gain_z = 2.0` floors (`:529`, `:535`).
3. Real training. Only the specs that pass are trained as full models (`CompositeTargetEstimator` fits the inner regressor on T and applies the inverse with smearing and clipping, `estimator/_smearing.py`, `_predict.py`).

So the cheap part (1) is already the "scan T_k with cached bins" the owner describes, but it is NOT a feature-selection step; it picks which target the model is trained on, and its output is judged on the y scale by steps 2 and 3. The record in the config comment is the important lesson: MI against T is not comparable with MI against y, and the pure-lag composite `T = y - y_prev` has MI close to 0 while being the best target.

Evidence that MRMR is run once per target in the suite: the cross-target identity cache (`mrmr_skip_when_prior_was_identity`, `mrmr_identity_cache_include_y`, `mrmr_identity_cache_ycorr_threshold`, `mrmr/_mrmr_class.py:1280-1299`) exists because "a second MRMR call on the same X for a composite target" wasted 88 minutes in a production log. I did not trace the suite code that calls MRMR for each composite spec (the `training/` tree is large); the comment is the evidence, so treat the call pattern as stated by the code's own documentation, not independently verified.

### 1.2 How y is encoded and binned in MRMR today

| Item | Fact | Citation |
|---|---|---|
| Y enters the data matrix as a column | `_inject_targets` appends the target(s) to `X`, then `categorize_dataset` discretises every column including targets | `_mrmr_fit_impl/_fit_impl_stages/_inputs.py` (`_inject_targets`), `_binning.py:78-146`, `_fit_impl_core.py:425-461` |
| Default target codes | Equal-frequency (`quantization_method="quantile"`), `quantization_nbins = 10` | `mrmr/_mrmr_class.py:351-352` |
| Target stays on quantile codes even under an adaptive feature binning strategy | The "target-rebin guard" re-bins continuous targets with the legacy quantile path; targets with at most `quantization_nbins` distinct values keep their native classes | `_binning.py:34-76` |
| Joint class and marginal of y | `merge_vars(factors_data, vars_indices=y, ...)` returns `classes_y`, `freqs_y`; several target columns are MERGED into one joint class, not scanned as separate targets | `_screen_predictors.py:448-449`, `:653-654` |
| Device copies | `classes_y_safe` and `freqs_y_safe` are made resident once (`resident_operand`, int32 / float64) and permuted in place per null draw | `_screen_predictors.py:495-523` |
| FE step consumers | `classes_y`, `classes_y_safe`, `freqs_y` are returned by `screen_predictors` and passed to the pair search, noise gate and every FE stage | `_fit_impl_core.py:620-631`, `:766-767`, `_screen_predictors.py:659` |
| Plug-in MI kernel | `compute_mi_from_classes` builds a (K_x, K_y) int64 joint table in one row pass then sums `jf * log(jf / (px * py))`; the y marginal `freqs_y` is precomputed and passed in | `info_theory/_class_mi_kernels.py:118-172` |
| Pair and batch scoring | `batch_mi_with_noise_gate` (CPU njit, GPU twins `batch_mi_noise_gate_gpu.py`, `batch_pair_mi_gpu.py`) score many candidate columns against `classes_y` with `fe_npermutations` null draws | `_feature_engineering_pairs/_pairs_score_steps.py:59-92`, `batch_pair_mi_gpu.py:109-117` |
| Continuous y copies kept for non-MI stages | `_fe_prewarp_y_continuous_` (float64 raw y) for ALS warp, offset product, residual rescue, usability; `_fe_escalation_y_rank_` (rank in [0,1]) for escalation proposers. Both are deleted at fit end | `_inputs.py:30-58`, `_finalise.py:482-484` |
| Held-out / FE-family encoder | `encode_y_for_classif_mi`: ints densified; floats with more than 32 distinct values `pd.qcut(q=10, duplicates="drop")`; content-hash memo of 2 entries for n >= 4096 | `_y_encoding.py:30-37`, `:73-95`, `:116-123` |
| Held-out gain statistics | `quantile_codes` (10 bins), `pointwise_mi`, `held_out_codes` (edges from the other half), `paired_gain_se` | `_fe_gain_stats.py:19-50` |

Consequence for the proposal: there is exactly one `classes_y` per fit and every stage reads it as a global. Two different binnings of y coexist (the screen's 10 quantile codes and `encode_y_for_classif_mi`'s 10 qcut codes); both are rank-based.

## 2. The proposal made precise, and what survives

Target family T = {T_1 = identity, ..., T_K}. For each k, `codes_k = quantile_codes(T_k(y), B)` once; `freqs_k` once; a candidate column c with feature codes `x_c` is scored as `MI_k(c) = I(x_c; codes_k)` and the winner is `max_k MI_k(c)` (possibly `max_k (MI_k(c) - null_k)`).

### 2.1 Rank invariance: most of the family is a no-op

Quantile bins depend on y only through its order. For any strictly monotone T the sorted order is unchanged, so the equal-frequency codes are identical (up to tie handling at the edges, which differs only when y has many repeated values).

| Transform | Monotone in y? | Changes the 10 quantile codes? |
|---|---|---|
| identity, standardise (affine, positive scale) | yes | No, bit-identical |
| rank / empirical CDF / quantile-normal (`quantile_normal_y`) | yes | No |
| log, sqrt, cbrt, signed power, Box-Cox, Yeo-Johnson (`log_y`, `cbrt_y`, `yeo_johnson_y` in the composite registry) | yes (on their domain) | No |
| winsorise at fences (Tukey) | weakly monotone | Only the extreme bins' membership of clipped ties; practically No |
| `y - base` with a base that is a function of X only | no | Yes, but this is a different question (residual) |
| `abs(y - m)`, `(y - m)**2`, sign, threshold `1[y > q]`, class-vs-rest | no | Yes |
| per-group normalisation `(y - mu_g)/sd_g` | no | Yes |
| difference or ratio of two targets (multi-output) | no | Yes |

Measured/verified claim: the default pipeline never sees a monotone transform. I did not run a numerical A/B of "identity versus log codes" because it follows from `np.quantile` plus `searchsorted` being order statistics; the check I did run is the cost of the encoding step (section 3).

### 2.2 Which of the owner's expected benefits survive

| Expected benefit | Survives? | Why |
|---|---|---|
| "Try log / rank / Box-Cox y and hope the best signal shines" | No | Identical codes, identical MI, zero gain, only an inflated K in the multiple-testing correction if they are included. Not a heuristic loss: provably nothing to find at the MI level. (Such transforms do matter for the downstream MODEL, which is what meaning A addresses.) |
| "Scanning can be much faster than composite" | Partly | True as a statement about cost (section 3), but the composite funnel is not slow because of step 1; it is slow because of steps 2 and 3, which MI cannot replace (section 4). |
| "Best signal shines" for non-monotone T | Conditionally | Real for dispersion, regime and residual targets, where identity-y MI is blind to the structure. But what "shines" is information about T(y), not about y. |
| "A feature selected for T(y) helps predict y" | Only if the model can use it | See 4.4. For `|y - m|` a tree model can route on it; a mean model gains nothing for MAE of y unless the feature also shifts the conditional mean. |
| Cache bins once | Yes | Genuine and cheap; the encoder already memoises (`_y_encoding.py:36-37`). |

Plain statement: the family worth scanning is NOT the one in the owner's list (transform of y), it is the non-monotone, y-dependent one (residuals, dispersion, tails, regimes). The latter is a small K (2 to 4), which also shrinks the multiple-testing price.

## 3. Speed

### 3.1 Cost of the current scan

| Quantity | Value | Source |
|---|---|---|
| Combinations per pair with defaults | 1734 (unary medium 17 x unary medium 17 x binary minimal 6) | `audits/2026-10-09/offset_product_fe/REPORT.md` Review 2 (`mrmr/_mrmr_class.py:868`) |
| Candidate evaluation | one row pass building a (K_x, K_y) table per candidate, then `K_x * K_y` log terms; 10 x 10 cells | `info_theory/_class_mi_kernels.py:118-172` |
| Null | `fe_npermutations` permutations of `classes_y_safe`, so a candidate that survives the first screen costs (1 + nperm) table builds | `_pairs_score_steps.py:59-92`, `_screen_predictors.py:495-523` |
| Offset product, whole stage | 0.316 s at 100k rows, 0.934 s at 1M (before the later reduction to 0.194 and 0.442 s on a quieter box) | `audits/2026-10-10/01_fe_followups_low_hanging.md` L-02, L-08 |
| Pair table versus K x G offset scoring | 80 columns on CPU 0.19 s at 100k, 2.4 s at 1M, about 10% of the 1734-combo pair table (3.5 s / 40 s) | offset product REPORT O7 |
| Whole fit | 5 to 11 s (stage 0.3 to 0.9 s) | L-08 |

Number of candidates M per fit is not a constant: it is (pairs kept by the pre-rank) x 1734 plus the unary families. I did not count M for a default fit (it needs a fit run and the box is busy). ESTIMATE: tens of thousands to a few hundred thousand tables per fit on the subsampled FE rows (`_fe_subsample.py`; at n above the scan size the FE stages work on a shared row draw, `_screen_predictors.py:97`), which is consistent with the 3.5 s pair-table figure at 100k.

### 3.2 What is cacheable across K targets

| Item | Per target or shared | Note |
|---|---|---|
| Y codes `codes_k` | per target, once per fit | measured below: 0.008 s (100k) / 0.088 s (1M) per `pd.qcut` of a heavy-tailed y; quantile of a fixed set of T_k is shared by sorting once if T_k is a function of rank (only non-monotone T_k need a fresh sort) |
| `freqs_k` / marginal entropy `H(Y_k)` | per target, once | O(K_y) |
| Feature codes `x_c` | shared across all K | the expensive part of a candidate (build unary/binary column, discretise) is paid once |
| Joint table `(x_c, codes_k)` | per target | K small scatters over the same rows |
| Permutation null | one y permutation `pi`; apply to every T_k simultaneously | preserves the dependence between the T_k, which is what the max statistic needs (section 4) |

Kernel shape for reuse: `joint_k[x_c[i], codes_k[i]] += 1` for k in 0..K-1 in the same row loop reads `x_c[i]` once and `K` code bytes (int8 codes at 10 bins). Memory-bound CPU kernel: ESTIMATE of cost for the table build is `(1 + 0.6 K)` of a single-target pass at K <= 4, because the feature read and loop control are shared and the K y reads are one cache line; on the GPU the shared-memory histogram needs K x 100 counters per block (K <= 4 is 400 int32, within the existing shared-histogram tile in `_batch_pair_mi_cuda_shared_fused.py`; I did not read the tile size, so this is unverified). The per-candidate log-sum is K x 100 cells.

### 3.3 Measured encoding costs (tiny snippet, busy box, 1 run)

| n | `pd.qcut(y, 10)` | stable `argsort(y)` | `np.bincount` joint of one 10 x 10 table |
|---|---|---|---|
| 100,000 | 0.0079 s | 0.0105 s | 0.00095 s |
| 1,000,000 | 0.0881 s | 0.2192 s | 0.0109 s |

Reading: per-target preparation is negligible (K = 4 at 1M adds under 0.4 s once per fit); the per-candidate table is ~1 ms per 100k rows single-threaded, so a K-fold accumulation cost is dominated by the number of candidates, not by the target preparation.

### 3.4 Expected overhead per fit (ESTIMATE, not measured)

| Design | 100k rows | 1M rows |
|---|---|---|
| Full lens on every candidate of the pair table (K = 3) | +1.5 to +2 x the pair-table time (3.5 s -> ~6 to 7 s) | the same factor on 40 s -> ~70 to 80 s: unacceptable on the full table |
| Lens only for the top-T candidates by identity MI plus candidates failing the identity gate (variant c) | +5 to 10% of the pair table | +5 to 10% |
| Lens inside the offset/pair-residual stage only (variant b) | +0.05 to 0.2 s | +0.4 to 1 s (the stage is already 0.2 to 0.9 s; K = 3 triples the table part, not the unary build) |

These follow the scaling of the offset-product measurements (O7, L-02, L-08). They must be replaced by a cProfile run before any decision (section 7.7).

## 4. Statistics and the danger of overfitting

### 4.1 Selection inflation

The winner statistic is `max_{k <= K, m <= M} MI_k(c_m)`. The existing machinery in the repo already treats this class of problem:

| Mechanism | Where | What it controls |
|---|---|---|
| Westfall-Young maxT permutation floors (order 2 / 3) | `mrmr/_mrmr_class.py:659`, `:717-733`, `_feature_engineering_pairs/_pairs_gates.py:31`, `info_theory/_batch_kernels.py` | best-of-pool chance ceiling over M raw columns |
| Whole-grid null (rerun the entire grid under permuted y), never "rescore only the selected t" | offset product REPORT O4: the selected-only null has 23-60% false accepts at G >= 5, the whole-grid null is nominal | exactly the K-target situation |
| Held-out half: edges and tables from even rows, evidence on odd rows | `_fe_gain_stats.py:19-50` (`held_out_codes`), `_residual_rescue_fe.py:7-15` | optimism of the plug-in MI |
| Paired gain standard error | `_fe_gain_stats.paired_gain_se` (`:42-50`) | acceptance by Z standard errors, no hand margin |
| Bonferroni family over the pool | `_residual_rescue_fe.py:11`, `_pair_residual_screen.py` `FAMILY_ALPHA` | family-wise level |
| Practical-effect floor as a constructor parameter | `fe_residual_rescue_min_relative_gain` (`mrmr/_mrmr_class.py:2991`) | no buried constants |

Pure-noise bias of the grid max in the offset audit was 0.0003 to 0.001 MI at n = 5k to 10k with 10 bins (O4); the preset search itself is the larger bias (best of 512 preset candidates on noise at n = 5k: 0.0115 versus 0.004 for a single candidate). A K = 3 lens multiplies the family by 3 but the members are strongly dependent (all functions of the same y and of the same rows), so the effective family is below 3M; do not assume it, calibrate by the null in 4.2.

### 4.2 Calibrated design (proposal)

1. Family is `K x M`. The null is built once per fit: one random permutation of the ROWS of y (so `T_k(y)` for all k are permuted together; compute T_k on the original y and permute the codes, or recompute T_k on the permuted y, both give the same joint structure for functions of y alone; for residual lenses the residual must be recomputed because it depends on the selected design, and the design is fixed, so permute y first then recompute residuals). Evaluate the whole family on the permuted copy and record `max_{k,m}`. Repeat `P` times (the repo's maxT stop uses 25 permutations, `fe_sufficient_summary_maxt_permutations: int = 25`, `mrmr/_mrmr_class.py:732`) and take the `1 - alpha` quantile. The rejected design is the null that permutes y and rescores only the selected (k, c).
2. A candidate is accepted for lens k only if (a) its held-out MI gain over the best IDENTITY score of the same candidate exceeds `Z` paired standard errors (`paired_gain_se`, Z a normal critical value for the family-wise level over the candidates tested), and (b) the transform parameters (the centre m in `|y - m|`, the threshold in `1[y > q]`, the residual design) were fitted on even rows and the MI evidence is computed on odd rows. This is "a transform may enter only if its gain over identity is significant".
3. Cap: K <= 4 including identity. The cap is a constructor parameter (`fe_target_lens_max`), not a threshold on evidence.
4. Practical-effect floor on the relative gain as a constructor parameter (same convention as `fe_residual_rescue_min_relative_gain`), default chosen from the section 8 null study, never a literal in a function.
5. Report the lens that fired in the provenance (`_mrmr_fe_provenance.py`) so a user can see "this column was selected for dispersion, not for the mean".

### 4.3 Target leakage into transform parameters

A transform with fitted parameters (the centre `m`, the quantile of a threshold, the residual model) is a function of y. If it is fitted on all rows and evaluated on all rows, the evidence is optimistic. The repo's pattern is the answer: fit on even rows, evaluate on odd rows (`_residual_rescue_fe.py:7-15`, `_pair_residual_screen.py`, `_fe_gain_stats.held_out_codes`), and the composite discovery's `train_idx`-only rule (`discovery/__init__.py:111-124`, test `test_alpha_train_only_changes_with_train_idx`). A transform without parameters (sign of `y - median`, with the median taken from the even rows) is fine under the same split. Recipes must never store y-dependent state needed at replay (see section 5).

### 4.4 What a selected feature "for T(y)" means downstream

| Lens | What the feature tells you | Does a model of y benefit? |
|---|---|---|
| residual `y - E_hat[y|selected]` | leftover mean structure | Yes, for MAE/RMSE of y if the model can fit the leftover; this is boosting-like and the residual rescue already optimises it directly on the y scale |
| `abs(y - m)` dispersion | where the noise is large | Not for the conditional mean. Useful for a quantile/interval model, for sample weights, or for a tree that splits on regime. A MAE-trained linear model gains nothing |
| tail indicator `1[y > q]`, class vs rest | the tail event | Helps classification of the tail; for regression only through a gating model |
| difference or ratio of two targets (multi-output) | the target contrast | Only if the deployed model predicts the contrast and inverts it |

The decision metric is MAE and RMSE of the final model on y, with R2 as a reference column only (`fe_operator_factory/README.md` section on metrics). A lens passes only if the downstream MAE does not worsen and improves on its designed target. MI against T(y) is a proxy that the repo has already seen fail (section 1.1, `eps_mi_gain`).

## 5. Interference with the existing code

### 5.1 Stage-by-stage

Fit order and files from `_mrmr_fit_impl/_fit_impl_core.py` and `_fe_stage_cascade_*.py`.

| Stage / component | Reads y how | Impact of a K-target lens | Change needed |
|---|---|---|---|
| Raw screen (`screen_predictors`) | `classes_y`, `freqs_y` global | Must stay identity: selection of raw features is the contract that every test pins | None; lens never touches the raw screen |
| Pair search (`check_prospective_fe_pairs`, unary x binary) | `classes_y`, `classes_y_safe`, `freqs_y`, GPU resident copies | Lens would be an extra `classes_y_k` passed alongside | New optional arg (default None); CPU and GPU kernels need a K-axis (loop over targets sharing the feature codes). ~10 files (the pair-phase estimate in the offset REPORT was 600 LOC across ~10 files) |
| Prewarp / ALS (`_fe_prewarp_y_continuous_`) | raw float y | Unchanged | none; it already uses raw y, not codes |
| FE auto-escalation (`_fe_escalation_y_rank_`) | rank of y | Rank view is the identity lens; no change | none |
| Offset product (`_fe_stage_offset_product.py`) | scan-row y codes and rank | Natural host for a lens (variant b); its held-out evidence and baselines (weighted-sum, `paired_gain_se`) already exist | add the lens as an extra target column in its scan; its null (whole-grid) must widen to K |
| OOF warp, row statistics, 2-D cell table, pair screen | held-out y codes | Same pattern, optional | per stage; not in the first version |
| Additive fusion (C2) (`_fe_additive_fusion.py:201-204`, `:361`) | `encode_y_for_classif_mi(classes_y)`, continuous y for tail test | The fused compound is selected by identity MI; a lens that scores `add(a**2/b, log(c)*sin(d))` against a residual target would credit a PART of the compound as new | exclude compounds and their operands from the lens set, or the lens target must be the residual of the design that already contains the compound |
| Residual rescue (`_residual_rescue_stage.py:122`, `_residual_rescue_fe.py`) | `_fe_prewarp_y_continuous_` on the target's own scale | Direct overlap with the "residual of the current selection" lens: same question, stronger test (squared residual, held-out t, Bonferroni, `fe_residual_rescue_enable` default True, `:2990`) | do NOT add a residual lens to the MI scan; call the rescue's residual machinery instead, or run the lens after the rescue on the post-rescue design |
| Sufficient-summary stop (`_fe_sufficient_summary.py`) | residual `r = y - E_hat` against a maxT null | If the stop declares the selection sufficient the FE search ends; a lens would be skipped, which is right. If the lens is residual-based it must see the same stop decision | order the lens stage after the stop test |
| Usability lists (`_usability_lists.py:97`, `build_usability_lists(self, X, y_cont)`) | continuous y, linear-gain based | A dispersion-selected column is of no use to the linear list | exclude lens-only columns from the linear list; the MI list (trees) keeps them |
| Recipes / replay | closed-form functions of X, no y reference | A lens changes how a candidate is chosen, not the recipe, so replay is unaffected, PROVIDED nothing y-derived is stored in `extra` | enforce with a test that `transform()` of an unseen batch has no y dependence |
| Fit memo `MRMR.clear_fit_cache` | keyed by data and constructor parameters (README lesson 8) | New ctor parameters change the key automatically; an A/B between arms still needs `clear_fit_cache()` | document in the test helpers |
| Cross-target identity cache | X fingerprint plus a y sample | Not affected by an internal lens | none |
| Setstate defaults (`_mrmr_setstate_defaults.py`) | legacy pickles default OFF | New flags need a legacy default of OFF and an entry in `_SETSTATE_LEGACY_OVERRIDES` | standard wiring (README section 1 item 8) |
| `fs_params` generation, fuzz axes | enumerates ctor params | A new ctor param appears in `training/fs_params/mrmr.py` after `python -m mlframe.training.fs_params._generate` | regenerate; add a fuzz axis for the lens flag |
| Per-fit caches | big per-K arrays | must be excluded from pickle (`__getstate__`) | add to the exclusion list in the same change |
| Multi-target `y` (several columns) | merged into ONE joint class by `merge_vars` (`_screen_predictors.py:448`) | "Difference/ratio of targets" is not a view the code has today; a joint class is not a contrast | a contrast lens needs the raw target columns, which `_fe_prewarp_y_continuous_` does not keep (multi-output y gives None, `_inputs.py:52`) |

### 5.2 K = 1 must stay byte-identical

True by construction if (1) the lens flag defaults to OFF and the K = 1 path does not construct any lens arrays, (2) the pair and batch kernels gain a default-None argument and the K = 1 branch calls the existing code unchanged (the repo's own convention for the fastest-default dispatch is a separate kernel version kept under `_v2`, never a modified one), (3) the encoder memo is not changed. Test: `fit` with the flag at the default versus a build without the module, compare `support_`, `engineered_recipes`, `cached_MIs` byte-for-byte, with `MRMR.clear_fit_cache()` between arms.

### 5.3 Interference with composite targets (meaning A)

| Risk | Detail | Mitigation |
|---|---|---|
| Double counting | the suite may already train on `y - base`; a lens `y - E_hat` inside MRMR on the residual target would ask the same question one level down | the lens set excludes anything the composite registry owns (diff, ratio, logratio, linear/robust/quantile/monotonic residual, ewma residual, monotone transforms). Document the split in the README: composite = model-level target choice, lens = feature-level extra evidence |
| The suite passes a composite T as `y` to MRMR | the lens then operates on T, with the "residual of the selection" being a residual of T. Valid, but the identity-cache correlation floor (`mrmr_identity_cache_ycorr_threshold`, 0.5) assumes related targets | no change; call out in the docs that lens columns are chosen per the y MRMR was given |
| Different notion of "relevance" in two layers | MRMR picks a column for T(y)-information, then the composite wrapper trains on T | stays consistent only if the lens is internal and the final judge is the y-scale model metric |
| Two competing selectors of the target | the lens must never change `y`, `classes_y` or the returned `support_` semantic for K = 1 | internal arrays only |

## 6. Alternatives

| Variant | Description | Cost | Gain | Verdict |
|---|---|---|---|---|
| (a) Iterated residual targets | boosting-like: select, form residual, rescan with the residual as target, repeat | the residual rescue and the pair residual screen already implement one pass with held-out evidence on the y scale (`_residual_rescue_fe.py`, `_pair_residual_screen.py`) | Already available; what is missing is feeding the residual into the unary/binary operator search | Partially implemented. Extend, do not duplicate |
| (b) Lens only for the pair/offset search | extra target columns in the offset product scan and the pair residual screen | low (+0.05 to 1 s) | targets the cases where a dominant additive term hides an interaction | Recommended host |
| (c) Per-candidate best-of-K only for candidates that failed the identity gate | rescue lane for rejected candidates | +5 to 10% | catches variance- and regime-type signals without inflating the family for candidates already accepted | Good, but needs the whole-family null with the failed set as M |
| (d) Tiny fixed K | residual (rank-free, from the design), `abs(y - median)` (even rows), one tail indicator, optionally log-ratio of two targets for multi-output | smallest family and null | smallest multiple-testing price | Recommended, combined with (b) and (c) |
| (e) Full registry scan of ~25 transforms | the owner's literal proposal | high, and 20+ of the 25 are monotone or composite-owned | none at the MI level | Rejected: provable no-op for the monotone members, double counting for the rest |

Recommendation: (d) in the host of (b), with the lane rule of (c). Concretely a K = 3 lens {residual of the current design, `abs(centred residual)`, upper-tail indicator at the even-row 90th percentile}, applied to (i) the offset-product scan and (ii) candidates that failed the identity gate, accepted by the section 4.2 rule, flag default OFF until the section 8 experiment passes. Reason: it is the only subset where the information is NOT already in the code and the identity MI is provably blind, while the cost and the multiple-testing price stay small, and the composite layer is untouched.

## 7. Implementation plan (if the section 8 experiment says go)

Standing rules: module under 1000 lines in a NEW submodule, one test of each kind with teeth, Edit not rewrite, no relaxed assertions, nothing hand-set margins, decision metrics MAE/RMSE.

### 7.1 Phases

| Phase | Work | Exit criterion |
|---|---|---|
| P0 | Offline experiment (section 8), no source change | verdict recorded in this audit |
| P1 | New module `filters/_target_lens/` with `lens_family.py` (builds the K views from y and the design on even rows), `lens_codes.py` (cached codes, freqs), `lens_null.py` (one-permutation joint null over K x M), `lens_kernels.py` (njit, `parallel=True` K-axis table build; CUDA twin later) | each file < 1000 lines; unit tests green |
| P2 | Host in the offset product scan (variant b): extra target columns, widened whole-grid null, `paired_gain_se` acceptance | biz-value test passes; K = 1 byte-identical |
| P3 | Failed-gate lane (variant c) in the pair phase behind the same flag | null FPR at nominal on pure noise |
| P4 | Wiring with `fe_operator_factory/wiring.py` (the README lists ~25 edits: recipe kind not needed here since no new column kind is created; ctor parameters, setstate defaults and `_SETSTATE_LEGACY_OVERRIDES`, `fs_params._generate`, fuzz axis, provenance, `__getstate__` exclusion). The wiring tool assumes a new recipe kind, so use it as a checklist, not blindly | `tests/feature_selection/fe/factory/test_wiring.py` style pin green |
| P5 | GPU strict-resident path: y codes for K views uploaded once with `resident_operand`, K-axis in the shared-memory histogram kernel, KTC (kernel tuning cache) for the block size, no hardcoded thresholds | strict-resident test: no unexpected host round trip; kernel-launch accounting not above the offset-product stage's |
| P6 | Docs: README section and CHANGELOG entry; audit status table updated per finding | rows RESOLVED / REJECTED with evidence |

### 7.2 Constructor parameters (names are proposals)

`fe_target_lens_enable` (default False initially; flip only after the downstream win and null are established, per the repo's "accuracy over legacy" convention, not before), `fe_target_lens_max` (cap on K, default 3), `fe_target_lens_family_alpha` (family-wise level, default the same as `FAMILY_ALPHA` of the pair screen), `fe_target_lens_min_relative_gain` (practical-effect floor), `fe_target_lens_scan_rows` (default aligned with `DEFAULT_SCAN_ROWS`, 50,000 in `_residual_rescue_fe.py:31`).

### 7.3 Tests (each must fail before, pass after)

| Test | Design |
|---|---|
| Unit: rank invariance | for 6 monotone transforms the 10-bin codes equal the identity codes (this pins the central claim of section 2 so nobody re-adds them) |
| Unit: non-monotone lens differs | `abs(y - m)` codes differ from identity codes on a symmetric y |
| Unit: parameters fitted on even rows only | fit on two different even/odd splits gives different `m`; evaluation rows never enter the parameter |
| Unit: joint null | one permutation applied to all K views preserves their pairwise dependence (rank correlation of codes unchanged) |
| Unit: kernel parity | K-axis table build equals K separate single-target builds, CPU versus GPU to 1e-9 |
| Null: false-positive rate | pure noise features and pure noise y, n = 5000 / 20000 / 100000, 50+ seeds each (the offset audit noted 15 to 30 replicates is too few), count the fraction of fits that accept any lens column; must be at or below the family-wise alpha plus 2 binomial standard errors |
| Null: structured but uninteresting | additive target, weighted-sum target, pair product (the lessons in the factory README, item 1): no lens column accepted, because the baseline already owns them |
| Biz value 1: heteroscedastic | `y = a*b + noise * |c|`, a, b, c, d..., U(0,1) and noise N(0,1) columns; identity y MI sees `a*b`; the lens target `abs(residual)` reveals `c`; downstream: HistGB MAE improves versus the identity-only arm (quantile or weighted model scores better on pinball loss) |
| Biz value 2: regime switch | `y = 1[z > 0.5] * f(a) + 1[z <= 0.5] * g(b) + noise`; with `z` a column, the residual lens on the first-stage design shows b's regime effect hidden by the dominant branch |
| Biz value 3: signal in a log-difference of two targets | multi-output `(y1, y2)` with `y1 - y2 = h(x) + noise` and `y1 + y2` dominated by an unrelated large term; the contrast lens finds `h`; this needs the raw target columns kept (section 5.1, last row), so it is only in scope if multi-output is in scope |
| Negative biz value | `log(y)` of a lognormal y: assert that no lens column is accepted, all the transforms are monotone and the lens adds nothing |
| K = 1 byte-identity | section 5.2 |
| Replay / pickle | `transform()` on an unseen batch with different statistics reproduces the fit columns; MRMR survives pickle; no per-K cache in `__getstate__` |
| cProfile | stage-level profile at 100k and 1M rows, top frames recorded; `nvprof_autotag.py` / `nvprof_range_summary.py` (README lesson 9) for the kernel-launch count on GPU |

### 7.4 Acceptance by statistics, not magic thresholds

Acceptance uses `Z` paired standard errors of the held-out MI difference to the best identity score, a family-wise level over the actual number of candidates tested, the whole-family permutation null of 4.2, and the practical-effect floor as a constructor parameter. No literal margin over n in a function.

### 7.5 Module size

Proposed: `_target_lens/` with four files under 400 lines each; hosts (`_fe_stage_offset_product.py`, `_pairs_core.py` at 1034 lines already) get only a call site, never new logic. Per the module-size rule `_pairs_core.py` must not grow.

### 7.6 Documentation

README section "Target lens", CHANGELOG entry, a note in the MRMR constructor docstring on the difference to composite targets (section 5.3), and this audit's status table.

### 7.7 Findings table for the implementation audit

| ID | Finding | Evidence | Status |
|---|---|---|---|
| T1 | Quantile bins make every monotone T_k a no-op for MI | `_binning.py:34-76`, `_y_encoding.py:116-123`, `_fe_gain_stats.py:19-25` | DECIDED (proved by construction; pinned by the proposed unit test) |
| T2 | The composite-target funnel already scans transforms by binned MI, re-ranks by y-scale tiny-model RMSE and trains a model per accepted spec | `training/composite/discovery/__init__.py:96-135`, config `:405-449`, `:529-535` | NOT A DEFECT |
| T3 | MI against T(y) is not comparable with MI against y; the repo disabled its MI pre-filter for this reason | `training/_composite_target_discovery_config_base.py:405-424` | NOT A DEFECT (design input) |
| T4 | Residual of the selection as a target already exists in the residual rescue, the pair residual screen and the sufficient-summary stop | `_residual_rescue_fe.py`, `_pair_residual_screen.py`, `_fe_sufficient_summary.py` | NOT A DEFECT |
| T5 | Unary/binary operator search and offset product do not see a residual target | `_pairs_score_steps.py:59-92`, `_fe_stage_offset_product.py` | OPEN (the one real gap; experiment first) |
| T6 | A K-target scan needs a joint null over K x M with one shared y permutation; selected-only nulls are miscalibrated | offset REPORT O4 | OPEN (design input, section 4.2) |
| T7 | Multi-output y is merged into one joint class; contrasts of targets are not representable and `_fe_prewarp_y_continuous_` is None for multi-output | `_screen_predictors.py:448`, `_inputs.py:52` | FUTURE |
| T8 | Per-candidate cost of K views is ESTIMATED, not measured | section 3.4 | OPEN |

## 8. The first experiment (cheap, offline, no source change)

Script: `src/mlframe/feature_selection/_benchmarks/fe_operator_factory/wave2/exp_target_lens.py` (new, not written in this task), built on `wave2/harness.py`, `stat_study/core.py` and `stat_study/exp10_downstream.py` (the downstream MAE/RMSE of ridge and HistGB). I checked that these exist (`ls` of the folders); I did not read their APIs in detail.

Design:

1. Five synthetic targets at n = 20000, 20 seeds: (i) heteroscedastic `a*b + N(0,1)*|c|`; (ii) regime switch; (iii) lognormal y with additive signal (negative control, all transforms monotone); (iv) additive nonlinear (negative control); (v) pure noise.
2. For each: best existing candidate by held-out 10-bin MI against identity y (the baseline the factory already computes), versus the best candidate by max over the lens set {residual of a ridge design, abs(centred residual), upper tail indicator}, with parameters from even rows and MI on odd rows, and the whole-family permutation null (200 permutations of y, each applies all K views).
3. Report: acceptance rate per target with Monte Carlo standard errors; downstream HistGB and ridge MAE/RMSE for the set selected with and without the lens (R2 as a reference column only); selection overlap with what the residual rescue already finds (to measure T4 overlap).
4. Decision: build P1 to P3 only if (a) the noise and the monotone control accept at the nominal rate, (b) on (i) and (ii) the lens arm improves HistGB MAE by more than 2 paired standard errors, and (c) the lens finds something the residual rescue (run on the same data) did not. If (c) fails, close as REJECTED with the evidence, keep the benchmark in the package (bench-attempt-rejected note), and record that the residual rescue covers the case.

Cost: a few minutes of CPU per target on a quiet box. Not run now (the machine is busy).

## 9. Open questions for the owner

1. Which "composite" is meant: A (target transforms in `training/composite/`) as assumed here, or B (the fused additive compound)? The answer to "separate model per transform?" is "screened cheaply, then a model only for accepted specs" for A, and "not applicable" for B.
2. Is variance/dispersion information a goal? If the deployed models only predict the conditional mean for MAE/RMSE, a dispersion lens selects columns the metric cannot reward, and the only worthwhile lens is the residual one (which the rescue covers).
3. Is multi-output with target contrasts in scope? It needs the raw target columns kept in the fit (today `_fe_prewarp_y_continuous_` is None for multi-output), which is a larger change than the lens.
4. Should the lens be allowed to add columns to the linear list, or only to the MI (tree) list? The recommendation is MI list only.
5. Default policy: stay OFF until the section 8 experiment passes, or ON at once as the repo does for corrective operators? The recommendation is OFF until a measured MAE win exists, because unlike the offset product this operator has no gap measured yet.
6. Is a registry-wide scan of the 25 composite transforms at the MRMR level wanted anyway as a diagnostic (a report of "which monotone and composite transforms would change the model")? It would be cheap but would not feed selection; I would put it in the composite discovery report, not MRMR.
