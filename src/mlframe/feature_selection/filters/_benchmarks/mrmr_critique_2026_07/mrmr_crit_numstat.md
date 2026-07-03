# MRMR numerical/statistical critique (READ-ONLY)

Scope: MI/CMI/entropy estimation, permutation-null construction, bias corrections, discretization used BY MRMR.
Generic info_theory log(0)/div0/overflow issues excluded (prior wave). Focus: MRMR-specific statistical correctness.

Paths are relative to `src/mlframe/feature_selection/filters/`.

---

## Ranked findings

### F1 (P1) — Miller-Madow correction is applied to the OBSERVED relevance MI but NOT to the permutation null → double-debias / over-rejection of weak-but-real signal
- **Where:** `permutation.py:625-628` (observed) vs `permutation.py:341,229,165,403,455` (all permutation kernels). The observed relevance uses `use_mm=(use_mi_miller_madow() and not _use_su)`; every `compute_relevance_score(use_su, classes_x, freqs_x, local, freqs_y, ...)` call inside `parallel_mi_prange_with_null` / `parallel_mi_prange` / `parallel_mi(_with_null)` / `parallel_mi_besag_clifford*` omits `use_mm`, so it defaults to `use_mm=False` (plug-in). Confirmed the default in `info_theory/_class_mi_kernels.py:144`.
- **Defect:** With MM enabled, `original_mi` is MM-corrected (≈ observed − (Bx−1)(By−1)/2N) while the null distribution and `null_mean = sum_perm_mi/n_checked` are PLUG-IN (centered at ≈ +(Bx−1)(By−1)/2N). Two compounding errors:
  1. The exceedance test `mi_perm(plugin) >= original_mi(MM)` compares a plug-in null against a bias-lowered threshold → nfailed inflated → confidence deflated → genuine weak signal wrongly flagged non-significant.
  2. On the non-significant branch (`evaluation.py:550,576`) `direct_gain = max(0, observed_mm − null_mean_plugin)` subtracts the bias twice (once by MM in `observed_mm`, again via the plug-in `null_mean` ≈ bias).
- **Scenario:** n=8000, a real but weak low-cardinality feature, Bx=By=20, `mi_normalization` set so `use_mi_miller_madow()` is True. True MI ≈ 0.03 nats, MM bias ≈ 361/16000 ≈ 0.0226. observed_mm ≈ 0.007; plug-in perms center ≈ 0.0226 so ~all perms exceed → p≈1 → non-significant → direct_gain = max(0, 0.007 − 0.0226) = 0. Feature dropped though it carries real signal.
- **Severity:** P1 (only bites when MM is the active estimator, but then it silently kills weak signal).
- **Fix:** Thread `use_mm=(use_mi_miller_madow() and not use_su)` into every `compute_relevance_score` call inside the permutation kernels (both observed and permuted MUST use the identical estimator), OR prefer the analytic null path (which already matches the MM bias sign-for-sign). Add a regression test asserting perm-null-mean ≈ 0 when the observed uses MM.

### F2 (P1) — Relevance is MM/null-debiased but redundancy (conditional MI) is raw plug-in → the mRMR relevance−redundancy trade-off is cardinality-biased
- **Where:** relevance debiasing at `evaluation.py:548-576` (`max(0, direct_gain − null_mean)` + significance gate); redundancy term `conditional_mi` at `evaluation.py:371` → `info_theory/_entropy_kernels.py:342,356,...` uses plain `entropy(freqs=...)` (no MM, no null). JMIM branch `mi(...)` (`:356`) likewise plug-in.
- **Defect:** The MRMR score is `relevance − redundancy` (or Fleuret min-CMI). Relevance is pushed DOWN (debiased) while redundancy `I(X;Y|Z)` / `I({X,Z};Y)` keeps its full upward plug-in bias, which grows with the cardinality of the conditioning set Z (Bxz·Byz cells). Net effect: high-cardinality candidates and candidates conditioned on high-cardinality already-selected sets are systematically over-penalized (redundancy inflated) on top of having their relevance deflated — the trade-off is biased, not merely both-sides-conservative, because the two bias magnitudes differ per candidate.
- **Scenario:** Two equally-relevant candidates; A is low-cardinality, B is high-cardinality and conditioned on a 3-way selected combo (Bz large). B's plug-in CMI redundancy carries a large +bias while A's is small; B loses the tie purely from estimator bias, not real redundancy.
- **Severity:** P1 (systematic ranking bias, always on for the default Fleuret path).
- **Fix:** Apply a consistent bias correction (MM telescoped CMI bias `(k_xyz+k_z−k_xz−k_yz)/2N`, already derived in `_fe_cmi_redundancy_null.py:200`) to the redundancy CMI so both legs live on the same debiased scale, OR debias neither. The redundancy-gate module already computes exactly this df; reuse it.

### F3 (P2) — `_perm_pvalue(full_budget=...)` extrapolates a truncated exceedance count to the full budget → confidence over-stated on early-broken NON-significant candidates
- **Where:** `permutation.py:62-74`, called at `:824,660`.
- **Defect:** When an early-break path (`parallel_mi` outer worker at `:408-409`, or Besag-Clifford) stops because `nfailed` piled up, `nchecked < full_budget`. `_perm_pvalue` then uses `denom = full_budget` with the *truncated* `nfailed`: `p = (1+nfailed)/(full_budget+1)`. But the run stopped precisely because failures were accumulating FAST — extrapolating would give MORE failures, not the same count. Dividing the small truncated `nfailed` by the large full budget UNDER-states p → confidence OVER-stated for a feature that is actually non-significant. The docstring's justification ("plain rate overstates") is backwards for the pile-up early-break.
- **Scenario:** BC exits at nchecked=30 with nfailed=6 (rate 0.20, clearly null); full_budget=200 → reported p=(7)/(201)=0.035, confidence=0.965 — the opposite of the true ~0.20 failure rate.
- **Mitigation present:** the actual reject decision uses the RATE (`nfailed/n_checked`) at `:705-708` and `:663`, and `original_mi` is zeroed, so selection is usually protected. But the surfaced `confidence` feeds downstream `min_relevance_gain` / relative-gain / FDR floors and `evaluate_candidate` reporting.
- **Severity:** P2.
- **Fix:** For a pile-up early break, either keep `denom = nchecked` (rate-consistent) or estimate p from the sequential-test stopping boundary; never rescale a truncated numerator by a larger denominator. Reserve the full-budget denominator for the CLEARLY-SIGNIFICANT early break (nfailed≈0), where extrapolation is safe.

### F4 (P2) — Per-candidate significance gate at fixed alpha=0.05 with no multiple-testing control across the engineered candidate pool
- **Where:** `evaluation.py:549,575` (`if p_value >= _MRMR_NULL_SIGNIF_ALPHA`), alpha default 0.05.
- **Defect:** The null-debias gate decides per candidate whether to keep full observed MI (significant) or subtract the null mean. Across a wide composite/engineered pool (thousands of candidates) with a fixed 0.05 threshold and no BH/Bonferroni, ≈5% of pure-noise candidates are wrongly declared "significant" and KEEP their inflated observed MI — exactly the high-cardinality/heavy-tailed decoys the debiasing exists to demote. FDR machinery exists elsewhere (`_screen_predictors*.py`, `_permutation_null.py`) but is not applied to this gate's alpha.
- **Scenario:** 4000 engineered candidates, all noise; ~200 pass the per-candidate 0.05 gate, retain full MI, and can out-rank a genuine weak feature that the gate (per F1) demoted.
- **Severity:** P2.
- **Fix:** Apply Benjamini-Hochberg (or Benjamini-Yekutieli, since candidates are dependent) across the batch of per-candidate p-values before the keep/subtract decision, or make alpha shrink with pool size. At minimum document the uncorrected per-candidate alpha as a known FP source.

### F5 (P2) — Empirical null-mean estimator variance (default 32 perms) can flip selection between near-tied candidates at n < analytic threshold
- **Where:** `permutation.py:46` (`_NULL_MEAN_MIN_PERMS=32`), used at `:642,656`; the subtraction `max(0, observed − null_mean)` at `evaluation.py:550,576`.
- **Defect:** Below `analytic_null_min_n()` (default 25_000) the null mean is a 32-sample Monte-Carlo average with relative SE ≈ 1/sqrt(32) ≈ 18% of the bias magnitude. That noise is subtracted directly from the observed MI, so two candidates whose true debiased relevance differs by less than the null-mean SE can swap order run-to-run (seed-dependent), and the swap propagates into the greedy MRMR path (irreversible once selected).
- **Scenario:** n=6000, two candidates with debiased relevance 0.041 vs 0.039; null-mean SE ≈ 0.004 dwarfs the 0.002 gap → selection order is effectively random across `base_seed`.
- **Severity:** P2 (instability, not incorrectness).
- **Fix:** Use the deterministic analytic bias `(Bx−1)(By−1)/2N` as the null-mean at ALL n (it matched the permutation mean to 3+ digits even at n=5000 per `_analytic_mi_null.py:17-19`), reserving permutations only for the sparse-cell case; or raise the null-mean budget and/or shrink the subtraction toward the analytic value (James-Stein style).

### F6 (P3) — Chao-Shen MI and Miller-Madow are not guaranteed to be used consistently across relevance vs redundancy vs the null
- **Where:** `_chao_shen.py` (`chao_shen_mi`, `_joint_chao_shen_mi_njit`) and `_cat_mm_correction.py` provide alternative estimators; the permutation null (`permutation.py`) and redundancy CMI (`_entropy_kernels.py`) only ever use plug-in or MM.
- **Defect:** If `mi_correction='chao_shen'` is selected for the pointwise relevance/redundancy estimate, the permutation null and the analytic null (which assume plug-in/MM bias `(Bx−1)(By−1)/2N`) no longer describe the estimator being gated — the null and the statistic are mismatched estimators, so the p-value and null-mean are not the correct reference distribution for a Chao-Shen MI. Generalizes F1 beyond MM.
- **Severity:** P3 (only when Chao-Shen is active; verify wiring).
- **Fix:** Route the null through the SAME estimator as the observed value (compute the permutation null with Chao-Shen when Chao-Shen is the statistic), or document that Chao-Shen bypasses/reshapes the significance gate.

### F7 (P3) — Analytic chi-square/G-test null assumes a FIXED contingency table, but equi-frequency binning makes occupied Bx data-dependent under permutation
- **Where:** `_analytic_mi_null.py:118-143` and `_fe_cmi_redundancy_null.py:200-209`; edge binning `_fe_edge_mi.py:26-78`.
- **Defect:** The `2N·MI ~ chi2(df)` identity and the Miller-Madow bias `(Bx−1)(By−1)/2N` are derived for a FIXED number of categories. MRMR bins continuous x by equi-frequency percentile edges; the number of OCCUPIED bins can shrink on tied/low-cardinality columns and the true permutation null uses whatever occupancy the shuffle induces. Using the observed occupied-bin df as a constant slightly mis-specifies df on tied columns. The sparse-cell guard (`_min_expected_cell`, expected≥5) is a partial safeguard but does not address the fixed-vs-random-occupancy assumption.
- **Severity:** P3 (small effect once the expected≥5 gate holds; matters on tied/discretized columns near the threshold).
- **Fix:** For flagged tied/low-cardinality columns fall through to the permutation null (which is occupancy-correct by construction); the existing sparsity gate mostly does this — extend it to a "ties present" predicate.

### F8 (P3) — `analytic_batch_noise_gate` per-column applicability uses OCCUPIED bx, so it cannot fall back to permutation once the batch is gated ON; ungated columns keep raw MI
- **Where:** `_analytic_mi_null.py:253-263`. `_reject = (fe_mi>0) & _applicable & (_p>=alpha)`. When `_applicable` is False for a column (sparse cells), that column is NOT rejected and NOT routed to a permutation test — it silently keeps its ungated observed MI.
- **Mitigation:** the caller `_pairs_dispatch.py:114-117` gates the whole batch ON only when the CONSERVATIVE declared-nbins applicability passes, and occupied bx ≤ declared ⇒ denser cells ⇒ per-column `_applicable` stays True, so in practice the sparse-per-column case is largely precluded for that caller. The `_gpu_resident_basis.py:936` caller passes `disc_2d=None` with precomputed bx — verify the same invariant holds there.
- **Severity:** P3 (latent; depends on caller invariant holding).
- **Fix:** Make columns that are analytic-inapplicable route to the permutation gate instead of passing through ungated, so the safe-condition failure never silently admits noise.

---

## Statistical improvement ideas

1. **Single-estimator invariant (fixes F1/F6):** introduce one `mi_estimator` selection that is threaded identically into (a) the observed score, (b) the permutation/analytic null, and (c) the redundancy CMI. Add a meta-test asserting perm-null-mean ≈ analytic bias for the active estimator, so any future divergence trips.

2. **Symmetric debiasing of relevance AND redundancy (fixes F2):** reuse the telescoped MM CMI bias already implemented in `_fe_cmi_redundancy_null.py:200` for the Fleuret/JMIM redundancy term so both legs of `relevance − redundancy` are on the same debiased scale. Ship a biz_value test on a high-cardinality-decoy synthetic where the current asymmetry mis-ranks.

3. **BH/BY multiplicity control on the null-debias gate (fixes F4):** collect per-candidate p-values per greedy round and apply Benjamini-Yekutieli (candidates are dependent) before the keep/subtract decision, replacing the fixed per-candidate alpha.

4. **Deterministic/shrunk null mean (fixes F5):** default the null-mean to the analytic `(Bx−1)(By−1)/2N` at all n and only permute for sparse cells; where permutation is used, shrink the empirical mean toward the analytic value to cut selection-flipping variance.

5. **Occupancy-aware analytic null (fixes F7):** gate the analytic path additionally on "no ties / all-distinct column" (a cheap predicate the codebase already uses elsewhere, e.g. `_all_columns_distinct`), routing tied columns to the occupancy-correct permutation null.
