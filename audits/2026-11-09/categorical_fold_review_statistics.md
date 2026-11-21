# Review of `categorical_fold_strategy`: statistics, overfitting, target types

Angle: statistics, leakage, target types. Code was read for context only: `cap_categorical_cardinality`, `_factorize_categorical_columns`, and the mentions of `_fe_prewarp_y_continuous_`.
Notation: n rows, L levels of a column, Ly classes of the discretised target (`classes_y`), n_l level count, p_l = distribution of y inside level l (vector of length Ly), p_g = global distribution, cap = max_categorical_cardinality.

## 0. Verdict

The direction is right (the target must matter, and frequency alone is a weak criterion). Five of the design decisions are statistically wrong or underspecified, in priority order:

1. P1. The "OOF" protection does not protect what the proposal says it does (leakage of the grouping into the MI estimate).
2. P1. "delta_MI = MI(old) - MI(merged)" is not a usable criterion as written: plug-in delta MI is always >= 0 and biased, so it needs a penalty. A principled form exists (G-test / MDL) and it removes the fixed thresholds.
3. P1. The fold sits inside `_factorize_categorical_columns`, which has neither y nor a recipe. This is an architecture prerequisite for every non-frequency strategy.
4. P2. Several "no fixed thresholds" requirements are violated by the proposal itself (5, 20, 50, "NaN share >= threshold"). All of them can be derived.
5. P2. The target table (section 5) is too wide. With a discretised `classes_y` a single count-based criterion covers all targets; special statistics are needed only for diagnostics and for continuous-target shrinkage.

## 1. Merge-cost: what is correct

### 1.1 Plug-in delta MI is wrong as a stand-alone criterion
Merging levels (a coarser partition) can never increase plug-in MI (data processing inequality), so delta MI >= 0 always. Taken literally, "merge when delta MI is small" has no scale. Under H0 (the merged levels have the same p_l) the expected delta MI is about (dL)(Ly-1)/(2n) nats (Miller-Madow), where dL is the number of levels removed by the merge. Both the sign and the scale depend on n, Ly and dL, so no single fixed threshold exists.

### 1.2 The principled form is the likelihood ratio (G-statistic)
For merging groups A and B into one group:

    ΔG(A,B) = 2 · n · ΔMI = 2 · [ n_A·KL(p_A || p_AB) + n_B·KL(p_B || p_AB) ]  =  2·(n_A + n_B)·JS_π(p_A, p_B)

where JS_π is the Jensen-Shannon divergence with weights π = (n_A, n_B)/(n_A + n_B). Under H0 this is chi-squared with (Ly-1) degrees of freedom (one pair merge). This is the Ward-style cost for categorical distributions and it is exactly what the proposal calls merge-cost, but it carries its own null.

Merge rule without a fixed threshold (MDL/BIC):

    merge A and B  <=>  ΔG(A,B) < (Ly-1) · log n          (BIC; the AIC variant uses 2(Ly-1))

Since the cost is additive across targets and classes, one criterion serves all of them (see section 3). The chi-squared quantile alternative needs a significance level and a multiple-testing correction over about L²/2 pairs, so BIC is preferred (no alpha). Report the criterion in the log.

### 1.3 Algorithms
- Binary y (Ly=2): sort levels by rate p_l1. For any convex impurity, the optimal partition into G groups is contiguous in this order (Fisher/Breiman). Use DP O(L²·G) or greedy adjacent merging with ΔG. Level counts L ≤ about 10^5, so this is vectorised and cheap.
- Ly > 2 (multiclass or a regression discretised to many bins): there is NO optimal 1-D order (the "first PC projection" in the proposal gives a heuristic, not a guarantee). Use agglomerative clustering with the ΔG cost and a nearest-neighbour chain, O(L²·Ly) worst case. For L > few thousand pre-pool the target-free tail (levels with n_l below the derived support, section 4) first.
- Cost is computed from the count matrix C[L, Ly]: one `np.add.at`/bincount over (code, class_y), O(n), then everything works on L×Ly. That is the cheap path; no re-scan of the n rows per merge.

### 1.4 Important: the marginal criterion is blind to conditional signals
MRMR needs MI conditional on already selected features. A level that differs from others only given another feature (interaction or XOR-like structure) has ΔG ≈ 0 marginally and gets merged away. Failure mode of every supervised fold. Mitigation: keep the target-free `min_support`-style tail rule as the floor (never merge a level that is both frequent enough and above the support); supervised merging acts only on levels that would be lost anyway (cap exceeded). So supervised fold = better ORDER of what to drop, not a mandate to merge everything.

## 2. Overfitting and leakage

### 2.1 The OOF scheme in section 4 is incorrectly aimed
The proposal: statistics OOF, "final mapping built on all rows", strategy chosen by OOF MI. Problems:
- The output is a DISCRETE code, not an encoded float. OOF is a cure for target-encoding leakage (a row's own y inside its own feature value). Here the leakage is different: the grouping (a function of y over ALL rows) then goes into an in-sample MI estimate on the same rows. OOF statistics with a final all-rows mapping gives the same in-sample optimism as no OOF. The MI of a column whose grouping was chosen to maximise MI is inflated by the selection (the maximum over groupings), so the folded column beats unfolded competitors in MRMR and its analytic null (which assumes a FIXED partition with known df) is too lenient.
- A genuinely OOF code (different mapping per fold) cannot be a single recipe: fit rows get fold-mappings, transform rows get the global mapping, so train/test skew appears and the selected feature is not the one deployed.

Correct split of responsibilities:
1. The mapping is one global recipe (deterministic, reusable in transform).
2. K-fold cross-fitting is used ONLY to choose between strategies (and to detect a noise column): learn the mapping on K-1 folds, score held-out log-loss / conditional entropy of y given the group (section 5). This is honest because the scored rows never saw the mapping.
3. The in-sample optimism of the final column is corrected through the degrees of freedom: tell the MI estimator / null that the column has G groups chosen from L levels, i.e. effective df = (G-1)(Ly-1) plus the selection penalty. The cheapest correct version: the BIC term in 1.2 already charges (Ly-1)·log n per merge not performed; propagate that penalty as a downward correction of the column's MI (n·MI - (Ly-1)·(L-G)... use the MDL code length of the mapping, about L·log2(G) bits).
   Alternative, exact but costly: refit the fold inside the permutation null (permute y, redo fold, record MI). Only for the final candidate columns, not for every column; use it as a gate, not the default path.

### 2.2 "Permutation control" in section 6
"Shuffle y, check MI not significantly above noise; otherwise roll back to frequency" is not a decision rule (no cutoff, cost = fold repeated B times per column, 20 columns × B shuffles). The same information comes analytically from ΔG against the BIC bound and from the empirical-Bayes variance estimate (2.3): if between-level variance tau² <= 0, the whole column is noise. Keep permutation only as a test-suite check, not in the production path.

### 2.3 Shrinkage: where it is needed and how to choose k
With count-based ΔG, shrinkage is NOT needed for the merge decision (the G-statistic already weights by n_l). Shrinkage is needed only (a) to ORDER levels for 1-D contiguity (binary/regression) and (b) for continuous targets, where the criterion is not count-based. Parameter k must be estimated, not set:
- Binary: method of moments. tau² = Var_l(p̂_l) - mean_l( p̄(1-p̄)/n_l ); k = p̄(1-p̄)/tau². If tau² <= 0 then k = infinity: every level is pooled and the column is flagged as noise (a free noise detector, replaces the permutation control).
- Multiclass: Dirichlet-multinomial, precision from the pooled overdispersion (Minka fixed point), shrink p_l toward p_g.
- Continuous (on normal scores, see 3): k = sigma²_within / tau²_between from one-way random-effects ANOVA; this is the standard James-Stein/empirical-Bayes weight, no free parameter.
The proposal's `categorical_fold_smoothing` as a user parameter should default to "estimate" (None) and only exist as an override.

### 2.4 Other leakage / consistency risks
- The fold must not see validation/test rows of any outer split used later by the selector (fit-only data). Note MRMR already subsamples; the fold must use the same fit rows as the MI estimator, and `classes_y` of those rows (not full-data y).
- Interaction with grouped FE (`grouped_quantile`, `composite_group_agg`): those use the categorical column as a GROUP KEY and compute y-based aggregates per group. If the key is itself a supervised fold, the aggregates become a second-order leak (the group was built to separate y). The FE families must receive the target-free fold (frequency / min_support), not the supervised one. Add a `target_free` flag/second code matrix; the proposal's open question 5 should be closed with this rule.
- Unseen levels at transform: map to the "other" bucket, always. With the design in section 4 (non-significant levels are pooled into "other", so other ≈ global distribution) this is coherent and honest. The proposal's "group with statistic closest to the global value" is equivalent in spirit but needs one more table entry in the recipe and picks an arbitrary non-neutral group when two are near; prefer "other".
- Fit/transform consistency: the current `cap_categorical_cardinality` recomputes counts from whatever matrix it receives. Verify (not checked here) that transform uses a stored remap and not a re-ranking of the transform batch; the proposal's open question 1 depends on it. A stored remap array (code -> group) per column is the recipe.
- NaN: the proposal puts NaN into its own level only if its share >= threshold. This throws away missingness signal (MNAR is often the strongest feature) and uses a fixed threshold. Treat NaN as an ordinary level in the same significance test (it competes for a slot like any other level).

## 3. Target types: is one criterion enough?

Claim: YES for the selection criterion, provided it is evaluated on the discrete `classes_y` that MRMR already uses for its own MI. MRMR's relevance is MI against `classes_y`, so a fold that preserves MI(level; classes_y) preserves exactly what the selector sees. The type-specific machinery in section 5 of the proposal is then needed only where `classes_y` loses information that matters for the fold.

| target | what the count-based ΔG on `classes_y` does | failure mode / extension |
|---|---|---|
| binary | exact; Ly=2, contiguous in rate | none beyond conditional blindness (1.4) |
| multiclass | exact; agglomerative on ΔG, no ordering needed | L×Ly cells sparse for large Ly: pooled df penalty (Ly-1)·log n grows, so folding gets MORE aggressive with Ly, which is correct |
| regression, continuous | works if `classes_y` bins are quantile-based (rank-robust, so heavy tails do not distort). The bins coarsen: Ly-1 df vs a 1-df mean shift, so a pure location effect is detected with lower power than ANOVA | add a second criterion ONLY for ordered/continuous y: Kruskal-Wallis/ANOVA on normal scores (rank -> probit) of `_fe_prewarp_y_continuous_`; NOT raw means. Raw mean and trimmed mean are unsafe under heavy tails (one 1e6 outlier in a 30-row level moves its mean). For a location-effect target use cost = (n_A n_B/(n_A+n_B))·(m_A - m_B)²/σ̂² (1 df) with the BIC bound log n |
| count / zero-inflated | `classes_y` of a zero-inflated target is degenerate (a big tie mass at 0; quantile bins collapse). ΔG still valid on whatever bins exist; the zero bin is its own class, so P(y=0) is captured | zero-inflation: two channels (zero indicator via binary ΔG, positive part via normal scores of log y) and cost = sum of the two ΔG with summed df. Gamma-Poisson shrinkage is not needed once the count-based machinery is used |
| survival / censored | no censoring information exists in `classes_y` (unless the pipeline passes an event+time pair; unverified). A time-only discretisation treats censored times as events: biased | do not build it: declare unsupported, fall back to `min_support` (target-free) and log it. If an event indicator is available, binary ΔG on the event is the honest minimal version; log-rank only if time and event are both supplied |
| multi-target | MI sums across targets in MRMR relevance | cost = Σ_t ΔG_t, penalty = Σ_t (Ly_t-1)·log n (additive log-likelihood: joint-independence assumption). A weighted max (most demanding target) is the conservative alternative; do NOT intersect per-target partitions (the cell count multiplies, L·ΠG_t, sparse by construction) |

Edit to the table in the proposal: drop the per-type shrinkage/test columns as separate designs; keep one column "criterion" = ΔG on `classes_y` (+ the continuous add-on only for regression/count). `LevelStat(target_kind)` collapses to: `C[L, Ly]` counts always, plus `(m_l, v_l)` of normal scores when `_fe_prewarp_y_continuous_` exists.

Note on `_fe_prewarp_y_continuous_`: it may be the prewarped (transformed) y rather than the raw y. If it is a monotone prewarp, rank/normal scores are invariant to it, which is another argument for ranks. If it is not monotone, the fold would be fit on a different target than the one MI sees. Verify before wiring; the fold must be consistent with `classes_y`.

## 4. Derive every constant (no fixed thresholds)

Existing code already states the principle: "the analytic null already guards on >= 5 expected/cell". Reuse that same constant:

- Minimum level support: a level is estimable when its expected count in the rarest class is >= 5 (the null's own guard): n_l · min_c p_g(c) >= 5, i.e. `min_support = 5 / min_c p_g(c)` (balanced Ly classes: 5·Ly). This replaces `max(5, n/(cap*20))`.
- Upper bound on groups: G · Ly · 5 <= n, i.e. G <= n / (5·Ly). This replaces `n/50` (n/50 equals this bound only for Ly=10 by coincidence).
- Keep-separate rule for a level (resolves weak point 1, the 30-row fraud merchant): keep level l as its own value iff its own evidence n_l·KL(p_l || p_g) (times 2 for G) exceeds the BIC cost (Ly-1)·log n. Non-significant levels have p_l ≈ p_g, so pooling them into "other" costs nothing and "other" stays near the global distribution (this is also what makes the unseen-level policy coherent). Keep significant levels in descending order of n_l·KL, up to min(cap-1, n/(5·Ly)). If the significant set exceeds the cap, cluster them by ΔG (section 1.3).
  This is a one-pass O(L·Ly) vectorised rule. It is also close to what `target_stat` should be in practice: "frequency OR significant", not 1-D clustering of all levels. Cluster merging remains for the over-cap case only.
- No fold at all when L <= cap AND all levels pass the support: identical to today (bit-identical codes), as the proposal already says.
- Code order: assign group ids by n_l descending (as `cap_categorical_cardinality` does), so downstream code that assumes the "other" bucket is the last id (cap-1) keeps working, and the int8 range (cap <= 127) holds.

## 5. How `auto` chooses without thresholds

The proposal's `auto` ("frequency at large n/levels, target_stat at small n and heavy tail") is two fixed thresholds in disguise. Replace with an honest held-out comparison:

1. Candidates per column: `frequency`, `min_support`, `target_stat` (section 4 rule).
2. K-fold cross-fitting on `classes_y` (K from n: stratified, K = 5, or leave-one-out is unnecessary; fixing K affects precision only, not correctness). For each fold build the mapping on K-1 folds, estimate p(y | group) with the Dirichlet-smoothed counts (alpha = estimated, 2.3), and score the held-out mean log-loss
   `ℓ_s = -(1/n) Σ_i log p̂_s(y_i | g_s(x_i))`.
   Everything runs on count matrices L×Ly per fold: K·O(L·Ly), cheap.
3. Choose the strategy with minimal ℓ_s; apply the one-standard-error rule toward the simpler/target-free strategy (`frequency` < `min_support` < `target_stat`): switch to the more complex one only if its ℓ is lower by more than one standard error across folds. That removes "random wins" from a noise column and needs no constant (the SE comes from the folds).
4. Noise column: compute ℓ_0 of the global prior. If no strategy beats ℓ_0 beyond one SE, the column is noise: use `frequency` (cheapest, target-free) and skip the supervised path.
5. Optionally the final in-sample MI correction in 2.1(3).

The held-out log-loss is a proper scoring rule defined for any discrete target, so `auto` is target-type-agnostic. For regression with the continuous add-on use the held-out Gaussian log-likelihood on normal scores as a second scoring rule and add the two.

Failure modes of `auto`: selection among 3 candidates has a small optimism (negligible, mention it); cross-fitted rare levels (n_l < K) are absent from the training folds, so they are scored as "other" in held-out folds. This is the correct, conservative behaviour for rare levels, but it penalises `target_stat` for the very case it was designed for (the single rare fraud level). Mitigation: stratify the fold assignment by level for levels with n_l >= 2K and report the held-out loss separately for levels with n_l < K (do not use them in the decision).

## 6. Failure modes by strategy

| strategy | failure mode |
|---|---|
| `frequency` (current) | drops a rare informative level (the proposal's weak point 1); mixes opposite-effect levels into "other", which cancel. No dependence on n, Ly or the number of columns. Not wrong for the common case, hence the default |
| `min_support` | target-free so no leakage; folds levels that are rare but significant (30-row fraud level when support is 5·Ly = 10 passes, but at Ly=20 and 100 it does not); still blind to y |
| `target_stat` | in-sample selection bias (2.1); marginal-only blindness to conditional effects (1.4); noisy ranking when L >> n; label-specific: depends on `classes_y` discretisation (binning of a regression y changes the grouping); makes the fold dependent on the MRMR fit subsample (different subsample -> different recipe: instability across `fit` calls and across bootstrap/stability-selection resamples; a recipe built on a subsample must be reused, not recomputed, by later FE stages) |
| `auto` | per-column strategy flip changes code semantics between fit calls; log which was chosen per column and keep it in the recipe |

Also stability: supervised folds make the selected feature set less stable under resampling. Add a test: fit twice on two halves of the same data and compare the groupings (adjusted Rand index above the null) for a column with a real effect.

## 7. Concrete list of changes to the proposal

1. Section 4/5: replace "delta_MI = MI(old) - MI(merged)" by ΔG = 2n·ΔMI = 2(n_A+n_B)·JS_π(p_A,p_B) with the BIC acceptance bound (Ly-1)·log n. State explicitly that plain delta MI >= 0 always.
2. Section 4 step 2: split roles of cross-fitting. K-fold is for strategy choice and noise detection; the mapping is ONE global recipe. Delete the sentence "statistics OOF, final mapping on all rows" as a leakage protection. Add the MDL/df correction of the final column (2.1.3) and an optional permutation-with-refit gate for finalists.
3. Section 6: remove "permutation control" from the production path; use tau² <= 0 (empirical Bayes) and the BIC bound as the noise detector. Keep permutation in tests.
4. Replace constants 5, 20, n/50, and the NaN-share threshold by the derived forms in section 4 above (support = 5/min p_g, G <= n/(5·Ly), keep-iff-significant). NaN is a level like any other.
5. Make `target_stat` = "frequency OR significant" (keep significant levels, pool the rest into "other" ≈ global); 1-D contiguity clustering only for the over-cap case. Binary: sort by rate; Ly > 2: agglomerative on ΔG (drop "PC1 projection" as the ordering).
6. Section 5 table: collapse to one criterion (ΔG on `classes_y`). Add continuous add-on only for regression/count (normal scores of the continuous y, never raw mean/median/trimmed mean). Zero-inflated: two channels summed. Survival: unsupported unless event/time are available, fall back to `min_support` and log. Multi-target: sum ΔG and penalties, no partition intersection.
7. Shrinkage parameter: estimated by default (method of moments / Dirichlet-multinomial / random-effects ANOVA), user value only as an override; k = infinity is the noise-column signal.
8. `auto` per section 5 (held-out log-loss, one-SE rule toward the simpler strategy, noise-column fallback). Remove "n/levels large vs small n and strong tail".
9. Unseen levels -> "other" always. NaN is an ordinary level. Group ids ordered by count descending, "other" last (cap-1), int8-safe for cap <= 127.
10. Architecture prerequisite: `_factorize_categorical_columns` and `cap_categorical_cardinality` have no y. Introduce a fit-time step that receives the count matrix C[L, Ly] (from `classes_y` of the fit rows) and returns a remap array per column; store the remap in the recipe; transform applies the stored remap (verify the current cap does not re-rank on the transform batch). Keep the old function untouched as the `frequency` implementation (new functions, no deletions).
11. Grouped FE (`grouped_quantile`, `composite_group_agg`): feed them the target-free code (frequency or min_support) only; supervised fold feeds the MI estimator only. Close open question 5 with this rule.
12. Tests to add (beyond section 7): (a) pure-noise column with 500 levels: `auto` picks `frequency` and the fold's MI is not above the analytic null (calibration over many seeds, expected rejection rate vs nominal); (b) heavy-tailed regression (Cauchy-like noise) where the informative level is a location shift: mean-based criterion fails, normal-score criterion succeeds; (c) XOR-with-another-feature column: document that the marginal fold merges the levels (known limitation) and that the target-free floor keeps frequent ones; (d) zero-inflated count target; (e) multi-target sum; (f) two-half stability (ARI); (g) fit/transform: unseen level -> "other"; transform batch re-ranking does not change codes; (h) the in-sample inflation check: fit the fold on half A, compare MI of the folded column on A vs on held-out B; the gap must shrink to about the BIC correction (this tests 2.1 directly).
13. Pickle/recipe: store only numpy remap arrays and plain scalars (no runtime caches) so pickling and the existing recipe serialization are not broken.

## 8. Answers to the proposal's open questions (statistics-relevant)

- Q3 (one criterion enough?): yes on `classes_y` (ΔG with BIC), plus a rank-based continuous add-on for regression/count; survival unsupported; multi-target additive.
- Q4 (`auto` without thresholds): held-out log-loss comparison with the one-SE rule, section 5.
- Q5 (fe_prewarp and conditional targets, grouped FE): use ranks, so the prewarp is irrelevant provided it is monotone (verify); grouped FE get the target-free code.
- Q1/Q2 (recipes, GPU): the recipe is one int remap array per column, GPU path unaffected as long as group ids stay within int8 and "other" = cap-1.
