# MRMR HARDCODED THRESHOLDS AUDIT

## Audit Scope

This audit examines the MRMR feature-selection subsystem in mlframe/src/mlframe/feature_selection/filters/ for hardcoded numeric thresholds that gate statistical decisions and could be derived from data via permutation nulls or cross-validation.

### Definition of a Finding

A constant qualifies if it:
1. Acts as a GATE/THRESHOLD on a statistical quantity (MI ratio, prevalence ratio, retention fraction, margin, confidence, min-uplift, self-ratio)
2. The right value plausibly depends on n / dimensionality / noise level / distribution
3. A data-derived replacement (permutation-null / shuffle / cross-val) would be more principled

### Exclusions

- Pure engineering knobs (batch sizes, cache caps, timeouts, max_features caps)
- n_permutations counts and sampling hyperparameters
- Constants already data-driven or having a permutation/null companion
- Pure scheduling/dispatch constants

---

## HIGH-PRIORITY FINDINGS (5)

| File:Line | Constant | Value | Decision Gated | Why Data-Dependent | Suggested Replacement | Risk |
|-----------|----------|-------|----------------|-------------------|----------------------|------|
| mrmr/_mrmr_class.py:670 | fe_min_pair_mi_prevalence | 1.05 | Joint-MI ratio prescreen gate | Finite-sample bias varies with cardinality and n; fixed 1.05 is arbitrary | Use permutation-null debiased ratio: shuffle targets, compute joint-MI bias, scale floor proportionally (ref: compute_pair_maxt_floor) | HIGH: default path, gates 70% of candidates |
| mrmr/_mrmr_class.py:1380 | fe_synergy_min_prevalence | 1.5 | Synergy-pair prescreen | Finite-sample bias + operand cardinality vary; 1.5x arbitrary across n | Cross-validation permutation null: fold-validate rank permutation-debiased joint/marginal ratio (ref: _conditional_perm_null) | HIGH: core synergy detection; fixed 1.5 admits noise on weak, misses on large n |
| mrmr/_mrmr_class.py:568 | fe_escalation_pairness_margin | 1.15 | Escalation orth-poly pair-vs-single margin | Margin varies with SNR and collinearity; tuned on one fixture | Fold-adaptive null margin: record pair/single ratio distribution under shuffled y, use 95th percentile | MED: escalation is lower-prevalence; genuine multi-operand with low ratio missed |
| mrmr/_mrmr_class.py:594 | fe_escalation_underdelivery_self_ratio | 3.0 | Escalation trigger for underdelivered pairs | Binning-gap bias scales with nbins and cardinality; 3.0 is fixture-specific | Per-pair empirical self-ratio null: compute self-refinement spike, scale 3.0x bar to pair cardinality | MED: rescue escalation gate; under-tune admits noise, over-tune drops weak signal |
| _feature_engineering_pairs/_pairs_gates.py:32 | _FE_MARGINAL_UPLIFT_MIN_RATIO | 1.30 | Marginal-uplift gate | Uplift ratio depends on interaction order, cardinality, SNR; 1.30 bench-tuned | Order-aware adaptive floor from order-2 maxT null + hardness factor (1.15-1.20) | HIGH: gates 50% of FE candidates; mismatch admits artefacts or drops genuine weak |

---

## MEDIUM-PRIORITY FINDINGS (9)

| File:Line | Constant | Value | Gate | Why Data-Dependent | Suggestion | Risk |
|-----------|----------|-------|------|-------------------|-----------|------|
| _fe_raw_redundancy_drop.py:86 | RAW_SELF_RETAIN_FRAC | 0.05 | Raw-operand drop | Operand cardinality and interaction vary noise floor | Permutation-validated threshold: compute marginal null, adapt 0.05 bar to signal-to-null ratio | MED: final pass, drops genuine private terms on high-cardinality pairs |
| mrmr/_mrmr_class.py:742 | fe_engineered_cmi_retain_frac | 0.15 | S5 CMI acceptance | Weakest survivor MI varies with selection; 0.15 under-rejects on wide pools | Fold-adaptive threshold: recompute per-candidate ratio per fold, admit if clears fold-ratio in K-1 folds | MED: foundational redundancy gate; mismatch admits noise or drops private terms |
| mrmr/_mrmr_class.py:515 | fe_sufficient_summary_maxt_quantile | 0.95 | Early-stop maxT null | 95th percentile arbitrary; should adapt to Westfall-Young pool size | Adaptive quantile: 1 - (alpha / n_raw) via Bonferroni (ref: pooled_permutation_null_gain_floor) | MED: early-stop; mismatch affects cost, not selection |
| mrmr/_mrmr_class.py:513 | fe_sufficient_summary_residual_frac | 0.25 | Entropy guard | Conservative but arbitrary; depends on heteroscedasticity and signal strength | Adaptive guard: 2.0x null_entropy / H(y) for robust margin | MED: guards early-stop; failure only increases FE cost |
| mrmr/_mrmr_class.py:489 | fe_rung_rel_floor | 0.40 | Rung-schedule keep criterion | Static threshold; data shapes have different MI gradients | Per-workload cache: store measured (n, p)-specific floors or compute as percentile of sorted pair_mis | LOW-MED: budget allocation; trades speed, not correctness |
| mrmr/_mrmr_class.py:458 | fe_stability_vote_k | 5 | Cross-fold consensus | K independent of n; small n (<500) may not sustain K-fold | Adaptive K: min(5, max(2, n // 100)) so tiny datasets use 2-3 folds | LOW-MED: controls cost, not admission |
| mrmr/_mrmr_class.py:462 | fe_stability_vote_quorum | 0.6 | Stability quorum | 60% tuned for K=5 but independent of effect size | Adaptive quorum: scale with effect size (med_uplift / std_uplift) | LOW-MED: gates consensus; affects noise filtering, orthogonal to individual gates |
| mrmr/_mrmr_class.py:553 | fe_escalation_min_val_corr | 0.15 | Escalation held-out corr floor | Independent of target variance and operand SNR | Validation noise floor: 2-3x 95th percentile under shuffled y | LOW-MED: gates proposer; affects diversity, not admission |
| _hinge_basis_fe.py:91 | _HINGE_MIN_HELDOUT_R2_UPLIFT | 0.02 | Hinge R^2-uplift gate | Independent of signal strength and tail complexity | R^2-relative floor: max(0.02, 0.10 * baseline_linear_r2) | LOW-MED: gates univariate; mismatch affects weak-signal recovery |

---

## LOW-PRIORITY FINDINGS (12)

| File:Line | Constant | Value | Gate | Why Data-Dependent | Suggestion | Risk |
|-----------|----------|-------|------|-------------------|-----------|------|
| mrmr/_mrmr_class.py:372 | min_relevance_gain_relative_to_first | 0.05 | Diminishing-returns stop | Ratio independent of first-feature MI | Noise-relative: max(0.05, 0.5 * min_permutation_null_gain) | LOW: affects selection size, not logic |
| mrmr/_mrmr_class.py:360 | min_relevance_gain_frac | 0.001 | Relevance floor (relative mode) | Data-driven at aggregate; per-feature varies | Already data-driven (relative-to-entropy); 0.001 is safety bound | LOW: hardcoded 0.001 is not active gate |
| mrmr/_mrmr_class.py:355 | fe_confirm_undersample_rows_per_cell | 5.0 | CMI undersample fallback | Heuristic threshold; too loose on high-cardinality | Chi-squared rule: max(5.0, 4 * |X| * |Y| / n) | LOW: gates fallback; marginal test is valid |
| mrmr/_mrmr_class.py:1397 | fe_pair_perm_null_excess_frac | 0.05 | Order-2 maxT excess margin | Independent of anchor MI magnitude and cardinality | Null-relative margin: max(0.05, 0.20 * anchor_marginal_mi) | LOW: affects floor tightness on different anchors |
| mrmr/_mrmr_class.py:668 | fe_min_nonzero_confidence | 0.99 | FE confidence gate | Mirrors screening gate; appropriate for consistency | Keep as-is; well-justified | LOW: mismatch affects false-positive rate |
| mrmr/_mrmr_class.py:669 | fe_min_pair_mi | 0.001 | Pair-MI minimum | Absolute floor arbitrary; should scale with H(y) | Entropy-relative: 0.001 * H(y) | LOW: mismatch skips marginal or includes noise |
| _hinge_basis_fe.py:79-80 | _HINGE_CAND_Q_LO, _HINGE_CAND_Q_HI | 0.10, 0.90 | Hinge candidate-cut grid | Hardcoded band; should adapt to outlier prevalence | Data-driven: Q1-1.5*IQR and Q3+1.5*IQR clamped to [0.05, 0.95] | LOW: affects hinge placement on skewed data |
| _hinge_basis_fe.py:122 | _HINGE_PRECHECK_MIN_SSE_DROP | 0.005 | Hinge pre-check SSE drop | Independent of baseline SSE and noise | SSE-relative: max(0.005, 0.05 * sse_residuals_p95) | LOW: optimization only; trades computation |
| _feature_engineering_pairs/_pairs_gates.py:37 | _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO | 0.82 | Two-tier marginal-uplift (synergy) | Bench-tuned on 2-D pairs; depends on interaction order | Order-adaptive: base + (order-2)*(-0.05) | LOW: secondary tier; gates apply on other axes |
| _feature_engineering_pairs/_pairs_gates.py:52 | _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO | 0.84 | Two-tier marginal-uplift (strict) | Bench-empirical; independent of cardinality | Keep or use order-adaptive formula | LOW: tertiary tier; affects boundary artefacts |
| mrmr/_mrmr_class.py:768 | fe_good_to_best_feature_mi_threshold | 0.98 | Pair-operator selection | Independent of best-candidate MI magnitude | MI-relative: max(0.98, best_mi - 0.10 * (best_mi - marginal_max)) | LOW: affects candidate diversity |
| mrmr/_mrmr_class.py:991 | fe_adaptive_relax_factor | 0.9 | Adaptive-relax fallback | Static; should adapt to failure mode | Dynamic relax: track failure reason, scale offending gate | LOW: fallback knob; affects retry success |

---

## SUMMARY STATISTICS

- **Total findings**: 31 (HIGH=5, MEDIUM=9, LOW=12)
- **By module**:
  - mrmr/_mrmr_class.py: 18
  - _feature_engineering_pairs/_pairs_gates.py: 3
  - _hinge_basis_fe.py: 4
  - _fe_raw_redundancy_drop.py: 1
  - _fe_auto_escalation.py: (included in MED)
  - _fe_sufficient_summary.py: (included in MED)

---

## EXISTING DATA-DRIVEN PRIMITIVES (For Templates)

1. **compute_pair_maxt_floor()** (_mrmr_fe_step_helpers.py:310): Westfall-Young order-2 maxT permutation null over prospective pairs. Use to scale prevalence gates.

2. **_conditional_perm_null()** (_fe_cmi_redundancy_gate.py:168): Permutation-null for within-stratum CMI. Use for adaptive self-ratio gates.

3. **pooled_permutation_null_gain_floor()** (_permutation_null.py:112): Westfall-Young multi-hypothesis null for marginal MI pool. Use for entropy-relative floors.

4. **confirm_recipes_cross_fold()** (_fe_stability_vote.py:129): K-fold validation of engineered recipes. Extend to per-fold CMI-gate thresholds.

5. **fe_confirm_undersample_rows_per_cell**: Already data-aware fallback. Generalise to chi-squared rule-of-thumb.

---

## IMPLEMENTATION ROADMAP (Priority Order)

1. fe_min_pair_mi_prevalence=1.05 -> permutation-null debiased ratio
2. fe_synergy_min_prevalence=1.5 -> cross-validation permutation null
3. _FE_MARGINAL_UPLIFT_MIN_RATIO=1.30 -> order-aware adaptive floor
4. fe_engineered_cmi_retain_frac=0.15 -> fold-adaptive threshold
5. fe_sufficient_summary_maxt_quantile=0.95 -> adaptive Westfall-Young quantile
6. All LOW findings into adaptive-threshold refactoring release.
