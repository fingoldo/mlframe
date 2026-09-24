# Benchmark pre-registration

**Status: OPEN.** This document is binding from the moment it is committed. Its git SHA is recorded in
every benchmark run's `MANIFEST.json`. Anything decided after looking at results is a `POST-HOC` finding
and is reported in a separate, labelled section — never merged into the headline.

This benchmark is designed and run by the author of one of the arms it judges (`MRMR`). The scenario
distribution is not a sample from any real problem population. Every report generated from it must carry
that sentence in its first paragraph.

## 1. Why this exists

Every degree of freedom in a benchmark — which scenarios exist, what "informative" means, what the headline
metric is, how many features each arm may keep, the compute budget — can move the result without anyone
lying. The mitigations below are mechanical, not aspirational: each is enforced by a meta-test or by a
committed lock file.

## 2. Kill criterion (evaluated at the end of Phase 0)

Phase 0 runs the existing `_benchmarks/fs_hybrid` arms against the cached OpenML beds.

**The eligible denominator is 7, not 8.** Two beds were audited before any arm ran:

- **`isolet` is excluded from the kill-criterion count.** It has 26 roughly balanced classes, so the bench's
  majority-class binarisation yields a 3.8% positive rate, and after the bench's own 3000-row budget that is
  about 115 positives — far below the roughly 5000-row floor this repository requires before any
  minority-class metric is stable. Its row is reported but carries no weight in the stop decision. Promoting
  it back requires re-binarising it (one-vs-rest on a chosen letter, or a class-pair restriction), which
  changes the bench's semantics and must be declared here first.
- **`gisette` version 1 is flagged inactive by OpenML** and is sparse-ARFF only; it is cached through a
  densifying path. It counts, but if it ever decides the stop verdict on its own, pin a newer active version
  and re-run before acting.

So the criterion below reads: 4 or more of the 7 eligible beds.

> **If no arm beats `all-features` beyond the noise band on 4 or more of the 7 eligible real beds, the full benchmark
> is cancelled.** Effort redirects to `wrappers/_noise_floor.py`, the one feature-selection mechanism in this
> repository with a measured real-data win over doing nothing.

This is not a pessimistic hedge. The repository's own recorded numbers make it the modal outcome:

| source | measurement |
|---|---|
| `wrappers/_noise_floor.py` docstring | madelon: all-features lgbm **0.872**; RFECV keeps 251/500 → **0.868** |
| `filters/_mrmr_tree_rescue.py` docstring | madelon: MRMR collapses to under 4 features → **0.6885**; with tree rescue **0.7999** |
| `wrappers/_noise_floor.py` docstring | madelon: permuted-y plateau cut N*=8 → **0.9135**, N*=12 → **0.940** |

## 2a. RESULT of the confirmatory run (recorded 2026-09-04)

2240 cells: 7 eligible beds x 16 arms x 20 reserved seeds (1000-1019), 2220 ok after the failed cells were re-run against a fixed adapter. The 20 that remain are shap-proxied on hill-valley, failing deterministically on every seed rather than flaking. Paired t on the 20 per-seed differences, m=20, df=19.

**On the primary outcome (matched K), the stop condition is MET.** Beds where no arm beats `all-features`, out of 7:

| model | k5 | k10 | k20 | k50 | k100 | k200 | self |
|---|---|---|---|---|---|---|---|
| lightgbm | 7 | 6 | 5 | 5 | 5 | 4 | 3 |
| logistic | 4 | 4 | 4 | 3 | 2 | 0 | 2 |

With a strong model at any declared cardinality, 5 to 7 of the 7 beds show no arm clearing the null. The criterion's threshold is 4. With a linear model it clears once enough features are allowed (k50+).

Two ambiguities in this document, exposed by its own first use and recorded rather than resolved after the fact: the criterion names neither the model nor the K at which it is evaluated, and it says "beyond the noise band" while the harness decides with a paired t at p<0.05. Read in the spirit it was written -- primary outcome, realistic model -- it triggers. A future pre-registration must name both.

**Hypothesis H3 is CONFIRMED on the primary outcome.** An earlier reading of the same run reported it falsified; that reading used the self-chosen-K row, the SECONDARY outcome, because a hardcoded label list in the report renderer silently emitted no matched-K section at all. The renderer now derives its labels from the records.

Residue worth keeping regardless of the stop decision:

- Only madelon and hill-valley ever produce a lightgbm win at matched K. On arcene at k20 every arm loses to `all-features`, MRMR by -0.0395 (p=0.0002).
- On madelon at k50, k100 and k200 the outright winner is `variance-sort` -- ranking by marginal variance, with no target involved -- ahead of every information-theoretic and wrapper arm. At k20 it is third (+0.0562), still ahead of MRMR. On a synthetic bed this document treats that as a broken bed; on a real one it is a statement about madelon and about the arms, and it stands.
- `random-<k>` at matched cardinality beats `all-features` on madelon under logistic (+0.0171, p=0.001), which is why every skill number here is read against that control and not against the null alone.

## 2b. POST-HOC: the synthetic control point (measured 2026-09-06)

Section 2a recorded that the kill criterion fired. It could not say whether that verdict was about real
data or about the downstream model, because a real bed has no ground truth and therefore no way to show
that selection COULD have helped. The control run answers exactly that and nothing more: the same sixteen
arms, the same protocol, the same panel, on nine adversarial beds whose informative columns are
constructed. 3 029 cells, 20 seeds per (arm, bed). Full write-up:
[`BENCHMARK_SYNTH_CONTROL.md`](BENCHMARK_SYNTH_CONTROL.md).

**Result.** With lightgbm, some arm beats the null on 5 of 9 beds at 1k and 2k, and on 7 of 9 at 5k; with
logistic, on 8 of 9 at 1k. So a gradient-boosted tree does benefit from selection when the truth is
sparse and known. The Phase 0 verdict is about the DATA, not about the model, and the kill criterion
stands as a statement about real beds rather than about feature selection in general.

**The qualification, which points the other way.** The gains measured here are +0.0032 to +0.0171 AUC.
The real leg's minimum detectable effect at R = 20 is 0.013 on the median contrast and 0.031 on the p90
one (section 6a), so all but the largest of these sit inside the real leg's blind spot. The two legs are
consistent with "real gains of this size exist and were invisible" as well as with "no real gains". The
control rules out one explanation -- that a strong model cannot use selection -- and establishes nothing
about the other.

**The beds where nothing pays were built so nothing would**: the three parity beds and the private-delta
cluster, each of which declared the arms it expects to defeat before the run. A predicted failure is a
result; the report distinguishes it from an unexplained one.

## 2c. POST-HOC: the SCM leg, and a withdrawn result (measured 2026-09-09)

A third leg ran the same roster on eight beds whose answer key is derived from the graph that generated
them: 2 560 cells, 20 seeds per (arm, bed), no failures. It sharpens 2b rather than contradicting it.
With logistic downstream some arm beats the null on 7 of 8 beds at the tightest cut; with lightgbm, on 1
of 8 and then none. The verdict about the DATA stands, and the model-dependence is stronger than the
adversarial leg suggested.

**A result is withdrawn.** An earlier reading of this leg reported `rfecv` recovering a three-way parity
answer key perfectly on all twenty seeds. It was an artefact of the harness: the arm tied every column,
the matched-K cut resolved ties with a stable sort, and a stable sort preserves the input order, which on
a generated bed whose informative columns are declared first IS the answer key. Ties are now randomised
per cell and the beds shuffle their columns; the recovery fell to 0.050, chance. Under the corrected
harness **no arm in the roster solves parity**, and `rfecv`'s pooled advantage on these beds is -0.098
with P(> 0) = 0.003 -- the opposite sign from what was claimed.

Two mis-specifications were found and fixed in the same pass and both are recorded here rather than in a
commit message alone: the adapter overrode each bed's declared row count with a uniform 4 000, and the
registry lock could not catch it because the lock hashes the SPEC while the adapter resized the bed
afterwards. The realised size now travels with the bed.

## 3. Primary outcome and null hypothesis

- **Null hypothesis: `all-features`.** Not a baseline line on a chart — the thing every arm must beat.
  Every scenario where no arm clears it is reported as "FS does not pay here", as its own leaderboard row.
- **Primary outcome: downstream quality at matched feature count `K`**, evaluated at one, two and five
  times the target-set size on the honest holdout. Unlimited-`K` comparison is not a question with an
  answer — a sufficiently strong gradient-boosted model is close to invariant to feature selection.
- **Secondary:** the quality-versus-cardinality Pareto frontier; support recovery against the primary
  target set; selection stability; `n_model_fits`.
- **Diagnostic only, never headline:** regret to the Bayes ceiling, and only within scenario families where
  the runtime check `|ceiling_analytic − ceiling_mc| <= mc_ci` passes.
- **Cost axis is `n_model_fits`,** which is deterministic. Wall-clock is advisory and every figure using it
  carries a caption stating the host was contended.

## 3a. Matched K on a real bed, where no target set exists

Section 3 evaluates at one, two and five times the target-set size. A real dataset has no declared target set, so on the real leg that multiplier has no denominator and the runner must not invent one.

On real beds, matched-K is therefore evaluated over a **declared absolute K grid**, fixed here before any run: `K in {5, 10, 20, 50, 100, 200}`, truncated to the bed's feature count. Every arm is asked for exactly K features and scored against `all-features` at each K. The self-chosen-K row is reported separately, as on the synthetic leg.

Two consequences, both accepted deliberately:

- The grid is a free parameter and therefore a rigging surface. It is pinned here, and changing it after seeing results is a POST-HOC deviation that ships labelled as one.
- An arm that cannot be asked for a specific K (it selects its own set and exposes no ranking -- every `score_kind = "none"` arm) gets no matched-K row on any bed. It is scored on the self-chosen-K row and on set metrics only. This is a property of the arm, not a gap in the protocol, and it is reported as such rather than papered over with a synthesised score.

**Partial ground truth where a published probe design exists.** Some NIPS 2003 beds were built by injecting a known number of artificial "probe" features drawn to match the real features' marginals. Where that count can be verified against the dataset's published description -- verified, not recalled -- the false-discovery rate against the probe block is reported alongside the K grid. That is the only ground truth available on real data, and it is what makes these beds worth more than an ordinary tabular dataset. A bed whose probe design cannot be verified is scored on the K grid alone.

## 4. Target set

Scored against **`markov_blanket`** as primary. `minimal_sufficient` is reported as a secondary efficiency
metric, scored by cluster coverage over the stored equivalence partition, never by exact set match.
`causal_parents` is reported only for causal-family scenarios and never in the headline.

This must be declared because it decides winners. On `mb_spouse_collider` (`Y -> C <- S`, `S` independent of
`Y`) MRMR takes `C` first by marginal MI and is **correct** under `markov_blanket`; under `causal_parents`
both `C` and `S` are wrong answers and a descendant-refusing arm wins. Same data, same arms, opposite winner.

## 5. ROPE

Defined on normalized skill, not on sampling noise:

```
skill = (Brier_baserate − Brier_method) / (Brier_baserate − Brier_Bayes)
ROPE  = 1% of attainable skill
```

Sampling noise is a property of the estimator; practical significance is a property of the decision. Using
the first as the second produces a region that shrinks as replications grow, is incomparable between cells,
and inherits the roughly 35% coefficient of variation of a 4-degrees-of-freedom variance estimate.

Reports show the full sensitivity curve `P(|delta| < r)` over `r`, with the pre-registered ROPE marked. The
single `P(rope)` number is a point on that curve, not a substitute for it.

## 6. Replication and seeds

- Inference is on the `m` independent per-`dataset_seed` paired differences within a scenario. `cv_seed` is
  a **nuisance axis**: averaged away before any test, reported separately as selection instability. A
  meta-test asserts no statistical function receives more than one row per `(arm, scenario, dataset_seed)`.
- **Floor: `R >= 20` `dataset_seed` per `(arm, scenario)`.** The exact `R` is set from a 40-cell pilot
  (2 scenarios times 2 arm pairs times 10 seeds) measuring `tau`, the seed-to-seed standard deviation of the
  paired difference, via `R ~= (z_{1−alpha/2} + z_{0.80})^2 * (tau/Delta)^2`. **`R` is recorded here before
  the confirmatory run.**
  **`R = 20` is hereby recorded** as the confirmatory replication count, taken as the declared floor rather
  than from a measured `tau`. The pilot is still worth running afterwards, but only to answer whether 20 is
  ENOUGH -- it cannot lower the floor, so the confirmatory run does not wait on it. Seeds 1000-1019.
- Replication budget goes to `dataset_seed`. Never to holdout rows (2–4% of the variance; 10k to 100k rows
  shrinks the standard error by about 2%) and never to `cv_seed`, which cannot reduce the dominant term at
  all.
- **Reserved seed ranges.** `[0..99]` — development and threshold calibration, tune freely. `[1000..1099]` —
  report-only. Tuning any threshold or scenario parameter against a reserved seed is a violation, checkable
  by grepping the test tree for literals in that range.
- `MANIFEST.json` records `n_seeds_declared`. A run containing cells beyond it is flagged in the report as
  optional stopping.

## 6a. POST-HOC: was R = 20 enough? (measured 2026-09-05)

Section 6 recorded `R = 20` as a declared floor and stated that the pilot was still worth running
afterwards to answer whether 20 is ENOUGH. It was run, off the confirmatory run's own 2 240 cells rather
than off a fresh 40-cell pilot, which is strictly more information. Full tables:
[`BENCHMARK_POWER.md`](BENCHMARK_POWER.md), regenerable from
`mlframe.feature_selection._benchmarks.fs_hybrid._power`.

Measured `tau` (sd of the per-`dataset_seed` paired difference against `all-features`, lightgbm, k10):
median 0.0205, p90 0.0477. At `R = 20` the minimum detectable effect is therefore **0.013 AUC** on the
median contrast and **0.031 AUC** on the p90 contrast.

This qualifies section 2a rather than changing it. The kill criterion fired, and it stays fired; what the
power analysis fixes is the size of the claim it licenses. The confirmatory run establishes that no arm
delivered a gain LARGER THAN roughly one to three AUC points over `all-features`. It does not establish
that no arm delivered a gain: detecting a true 0.005 AUC improvement would need 134 seeds at the median
contrast and 715 at the p90 one, between seven and thirty-six times the executed design.

Recorded here because it cuts against the conclusion the run reached, and section 2a would otherwise read
as a stronger negative than the design can support.

## 7. Control arms — permanent members of every leaderboard

| arm | purpose |
|---|---|
| `all-features` | the null hypothesis |
| `RandomSelectionArm` at matched cardinality | without it no recall number is interpretable — much apparent skill is just picking the right *number* of features |
| `VarianceSortArm` | varsortability tripwire (Reisach, Seiler & Drton, NeurIPS 2021): in an additively generated SCM, marginal variance grows with topological depth, so sorting by it recovers the causal order. A meta-test asserts this arm performs at chance on every scenario. **If it ever beats chance, the scenario is broken, not the method.** |
| oracle-informative, true-prob, all-except-informative, shuffled-prob, base-rate | reference lines |
| top-`k` by permutation importance, tail dropped | tests the "low-importance features still contribute" hypothesis |

## 8. Anti-rigging mechanics

1. `scenarios/REGISTRY.lock.json` is committed before any arm runs: per scenario, family, structural spec
   hash, ceiling grid, seed list, `expected_to_break`, declared primary target set. A meta-test asserts the
   lock hash matches the code. Adding or reparametrising a scenario after seeing results bumps the lock
   version and is visible in git.
2. `expected_to_break` is mandatory; a meta-test requires every arm to appear in at least two scenarios'
   `expected_to_break`.
3. `provenance.harvested_from` is recorded per scenario. The leaderboard is published both overall and
   stratified by which method's test suite the scenario came from.
4. Aggregation is **blind**: arms carry opaque IDs, the mapping lives in a separate file the aggregation and
   plotting code never reads, and the reveal happens once, after the aggregate is committed.
5. `rfecv_bare` and `rfecv_registry_default` are **separate arms** (the registry wraps RFECV and BorutaShap
   in a cluster-medoid `GroupAwareMRMR` with `expand=True`, which drags a whole cluster back in when its
   medoid is selected). Same for BorutaShap.
6. `test_negative_results_nonempty.py` fails if the generated report contains no scenario where MRMR ranks
   below median.
7. The **oracle is cross-checked by an outside implementation**. `assert_estimators_disjoint` keeps the
   oracle's estimator families apart from the arms' MI backends, which addresses the oracle agreeing with
   the arms; it does nothing about the oracle's own closed form being wrong, since that form and every test
   of it were written by the same hand against the same understanding. `dit` -- a third-party
   information-theory package, never an arm -- is handed the same exact joint law and must reproduce the
   oracle's exact `I(X;Y)` to within 1e-9 nats. Measured on the fixed probe every run records: 8.9e-16 nats
   at 2, 5, 17 and 64 levels. The verdict is a manifest field, and when `dit` is unavailable that field says
   the check did not run and why, rather than being omitted.
8. Free knobs are declared as **priors, not points** (`corr ~ U(0.5, 0.99)`, `prevalence ~ LogUniform(0.01,
   0.5)`, and so on) and aggregated over. A fixed value is a rigging surface; a declared prior is auditable.

## 9. Hypotheses

Directional, with the number that falsifies each. A falsified hypothesis ships as a result.

| # | Hypothesis | Prediction | Falsified if |
|---|---|---|---|
| H1 | MRMR at defaults loses to `all-features` on madelon | MRMR downstream lgbm AUC at most 0.75, versus about 0.87 | MRMR at least all-features minus 0.01 |
| H2 | The permuted-y plateau cut beats `all-features` on madelon | N* in [8, 16], AUC at least 0.91 | AUC below all-features plus the noise band |
| H3 | On most real beds, no arm beats `all-features` | at least 4 of the 7 eligible beds show no arm clearing the noise band | 3 or fewer beds |
| H4 | A univariate t-test filter beats binned-MI selection at small `n` | filter wins on `linear_gaussian_lowdim_n200` | MRMR at least matches the filter |
| H5 | MRMR recovers almost nothing on pure XOR without its synergy prefilter | recovery near 0 on `xor2` and `xor3` with the gate off | recovery above 0.3 |
| H6 | Rank aggregation beats the best single arm | Dowdall or Borda at least matches the best single arm on half or more of scenarios | fewer than half |
| H7 | RFECV's shipped default rule (`auto` resolves to `one_se_max`) over-selects on plateau-prone curves | positive bias in the selected feature count over 30 or more cells; on madelon it keeps about half the columns | the interval covers zero or is negative |
| H8 | Every search-based arm shows a winner's curse | `selection_score` exceeds the honest holdout score for RFECV, zero-importance pruning and the bandit | the interval covers zero |
| H9 | `VarianceSortArm` is at chance everywhere | recovery indistinguishable from `RandomSelectionArm` | it beats chance anywhere — **this invalidates the scenario, not the hypothesis** |
| H10 | MRMR's advantage is larger on MRMR-derived scenarios | provenance-stratified gap above zero | the gap covers zero |

## 10. Analysis plan

Per scenario: `m` paired per-`dataset_seed` differences, standard error `sd(delta)/sqrt(m)`, `m−1` degrees
of freedom. Exact, and needs no equivalence band. Across scenarios: a normal-normal random-effects model
with `mu` marginalised conjugately and a 200-point quadrature over `tau` — giving the pooled effect,
`P(mu in ROPE)`, per-scenario shrunken estimates, and the posterior for `tau` itself, the between-scenario
heterogeneity. `p(tau)` is half-Cauchy, never an inverse-gamma with a small shared parameter; sensitivity is
reported at half-Cauchy(s), half-Normal(s) and half-Cauchy(3s) on the figure.

Contrasts against the control (`all-features`) are primary; simultaneous max-t bands within a figure's
family. Benjamini-Hochberg is retained only for the support-recovery analysis, which is genuinely a
discovery problem. Scenario is a **fixed** factor, so any global number is a weighted average over an
arbitrary list and is captioned as such. The arm-by-scenario interaction is the scientific product.

Crashed cells are not missing at random: a `reliability` column (fraction of cells completed) is reported
per arm and scenario, alongside an intention-to-treat aggregate scoring a crashed cell at the base rate.

## 11. Declared limitations

- The author of one arm designed the benchmark. Adversarial scenario generation is that author's
  imagination one step removed. Not neutralizable; disclosed.
- Acquisition cost, inference latency, maintenance burden and interpretability — the reasons feature
  selection actually pays in production — are unmeasurable on synthetic data. This benchmark does not answer
  "should I use feature selection".
- External validity rests on 5 designed-probe datasets and a curated OpenML sample; neither is
  representative of production tabular pipelines.
- The Bayes ceiling exists only on the synthetic leg. The metric that makes this benchmark distinctive is
  unavailable on the only externally valid leg.

## 2d. Beds added after the first legs ran, and why each was added

The lock file makes a post-hoc addition visible; this section says what it was for. Eleven beds were added
after the three legs reported, none of them in response to an arm's result:

| bed | added because |
|---|---|
| `friedman1`, `friedman2`, `friedman3`, `weston_guyon_k4_p100` | every other bed was written by the author of one of the arms; these predate the arms and give an outside check on the harness |
| `heavy_tail_t4`, `outliers_020permille`, `zero_inflated_40pct`, `quantized_6levels` | the suite had no bed where the marginal was the difficulty, so "robust to outliers" had no measurable meaning in it |
| `graded_cardinality`, `id_trap`, `zipf_levels` | every bed was all-numeric, which excluded impurity-based importance's cardinality bias entirely |
| `missingness_trio_30pct`, `rare_class_010permille`, `shift_covariate`, `shift_concept` | the observation process was unrepresented, and it calls for different fixes than a structural failure does |
| `tail_isolation_clayton_vs_gaussian` | the registered copula pair was measured to be almost the same bed, so the tail claim had nothing supporting it |

None of these beds may be edited after their first reported run without bumping the lock.


## 2e. The roster doubled, and every new method was predicted to break BEFORE it ran

The roster went from 21 arms to 52: the information-theoretic family one criterion at a time, the MRMR
class's own variants, every wrapper, stability and bandit method in the repository, the registry-built
RFECV and BorutaShap that section 8 always required, the univariate filters, and an oracle reference pair.

Section 8 says a meta-test requires every arm to be named in at least two beds' `expected_to_break`. That
meta-test did not exist. When it was written, four methods already in the roster -- the three CatBoost
criteria and `shap-proxied` -- were named fewer than twice, so the rule had been violated for as long as
those arms had existed. It is now `test_fs_hybrid_full_roster.py`, and it fails on the predictions as they
stood before this section: 33 methods flagged.

Controls are exempt and say so in code (`_roster.is_control_arm`): `all-features`, `variance-sort`,
`random-<k>` and the two oracle arms are references, not methods. They are expected to hold still so the
others can be read against them, not to break.

Every prediction below was written from how the method works, before any of these arms ran on any bed. The
registry lock was regenerated with them, so `lock_differences()` shows sixteen beds whose expectations
changed and none whose structure did -- the git history is the record that these came after the first legs
and before the results for these arms.

Three O(d^2) wrappers are built only on beds of at most fifty columns; a prediction for one of them on a
wider bed could never be scored, so none was made, and a test keeps it that way.

| arm | predicted to break on | mechanism |
|---|---|---|
| `bandit` | `null_p100` | a fixed subset size selects k columns even when nothing is relevant |
| `bandit` | `null_p1000` | a fixed subset size selects k columns even when nothing is relevant |
| `bandit-ensemble` | `null_p100` | every seed must fill a fixed-size subset from pure noise |
| `bandit-ensemble` | `null_p1000` | every seed must fill a fixed-size subset from pure noise |
| `boruta-shap-registry` | `latent_replicates_private_delta` | the medoid pre-reduction collapses jointly necessary members |
| `boruta-shap-registry` | `id_trap` | SHAP of a tree credits the ID column |
| `boruta-shap-registry` | `graded_cardinality` | tree importance tracks cardinality |
| `cascade` | `xor3` | the forward stage adds one column at a time and no single operand improves the score |
| `cascade` | `id_trap` | the Boruta stage uses tree importance, which rewards the ID column |
| `cascade-stable` | `xor3` | the forward stage cannot start an interaction in any bootstrap |
| `cascade-stable` | `id_trap` | the Boruta stage keeps the ID column in every bootstrap |
| `catboost-loss` | `id_trap` | an ID column lowers training loss while generalising to nothing |
| `catboost-loss` | `redundant_exact_k5` | eliminating one copy costs no loss, so copies go in arbitrary order |
| `catboost-predictions` | `id_trap` | predictions shift most when the memorising ID column is removed |
| `catboost-predictions` | `graded_cardinality` | prediction change tracks split opportunities |
| `catboost-shap` | `id_trap` | tree importance rewards a unique-per-row column |
| `catboost-shap` | `graded_cardinality` | importance tracks split opportunities, i.e. cardinality |
| `forward-select` | `xor3` | no single operand improves the score, so the interaction never starts |
| `forward-select` | `xor3_plus_marginal_decoy` | the decoy is taken first and the operands never catch up |
| `greedy-backward` | `linear_gaussian_lowdim_n200` | at n=200 each removal is decided by CV noise |
| `greedy-backward` | `label_flip_uniform_15pct` | label noise makes the CV score too noisy to rank removals |
| `hetero-vote` | `id_trap` | tree members of the vote rank the ID column above its shadows |
| `hetero-vote` | `graded_cardinality` | tree members vote by split opportunities |
| `it-cmim` | `xor3` | the first pick is by marginal MI, which is zero for every operand |
| `it-cmim` | `latent_replicates_private_delta` | conditioning on one replicate zeroes the others' conditional MI though they are jointly needed |
| `it-jmim` | `xor3` | the first pick is by marginal MI, which is zero for every operand |
| `it-jmim` | `quantized_6levels` | six levels leave the joint histogram tied and sparse |
| `it-mim` | `redundant_exact_k5` | relevance alone takes every copy, spending K on one direction |
| `it-mim` | `xor3` | each operand has zero marginal MI |
| `it-mim` | `simpson_sign_reversal` | the reversing column has zero marginal MI |
| `it-relax` | `xor3` | the first pick is by marginal MI, which is zero for every operand |
| `it-relax` | `linear_gaussian_lowdim_n200` | three-way tables starve at n=200 and the interaction term drops out |
| `ksg-mi` | `xor3` | univariate: an operand's marginal MI is zero |
| `ksg-mi` | `simpson_sign_reversal` | univariate: the reversing column's marginal MI is zero |
| `mrmr-grouped` | `latent_replicates_private_delta` | collapsing a cluster to its medoid destroys jointly necessary members |
| `mrmr-grouped` | `redundancy_graded` | one medoid per cluster above the threshold loses the members that differ |
| `mrmr-grouped-expand` | `redundant_exact_k5` | expansion drags every copy back in, spending K on one direction |
| `mrmr-grouped-expand` | `redundancy_graded` | expansion re-admits whole clusters, widening the set without new information |
| `mrmr-pld` | `xor3` | marginal-first greedy, as mrmr |
| `mrmr-pld` | `linear_gaussian_lowdim_n200` | binned MI discards rows a t-statistic uses |
| `mrmr-relax` | `xor3` | marginal-first greedy, as mrmr |
| `mrmr-relax` | `linear_gaussian_lowdim_n200` | three-way cells starve at n=200 |
| `mrmr-stability` | `xor3` | the inner MRMR cannot find operands with zero marginal MI in any bootstrap |
| `mrmr-stability` | `linear_gaussian_lowdim_n200` | half-sample bootstraps leave about a hundred rows for a binned estimator |
| `mrmr-tree-rescued` | `linear_gaussian_lowdim_n200` | binned MI discards rows at small n; the rescue does not help a linear signal |
| `mrmr-tree-rescued` | `id_trap` | the tree rescue uses tree importance, which rewards the ID column |
| `near-noise-auc` | `xor3` | an operand's univariate AUC is 0.5, so it is dropped |
| `near-noise-auc` | `simpson_sign_reversal` | the reversing column's marginal AUC is 0.5, so it is dropped |
| `noise-floor` | `null_p1000` | the top of a noise ranking clears a 95th-percentile floor by chance often enough to admit columns |
| `noise-floor` | `weston_guyon_k4_p100` | equal-weight signal and same-marginal probes interleave in the importance ranking |
| `null-importance` | `redundant_exact_k5` | importance splits across copies, pulling each toward its label-shuffled null |
| `null-importance` | `graded_cardinality` | low-cardinality columns earn less real importance against the same null |
| `permutation-topk` | `redundant_exact_k5` | permuting one copy is compensated by the others, so each copy scores near zero |
| `permutation-topk` | `latent_replicates_private_delta` | each replicate is compensated by its siblings under permutation |
| `relevance-table` | `xor3` | univariate tests see no marginal association |
| `relevance-table` | `simpson_sign_reversal` | univariate tests see no marginal association |
| `rfecv-registry` | `latent_replicates_private_delta` | the medoid pre-reduction collapses jointly necessary members |
| `rfecv-registry` | `id_trap` | the inner LightGBM ranks the ID column highly, as bare rfecv does |
| `ridge-prefilter` | `xor3` | a linear coefficient cannot see a parity |
| `ridge-prefilter` | `friedman1` | a sine interaction and a centred square are invisible to a linear fit |
| `shap-proxied` | `id_trap` | SHAP of a tree proxy credits the ID column |
| `shap-proxied` | `redundant_exact_k5` | SHAP value splits across identical copies |
| `unanimous-permutation` | `redundant_exact_k5` | permuting one copy never hurts while the others remain, so every copy is unanimously dropped |
| `unanimous-permutation` | `latent_replicates_private_delta` | each replicate looks dispensable alone, so the jointly needed set is pruned away |
| `unsupervised-prescreen` | `null_p100` | it never looks at the target, so every noise column survives |
| `unsupervised-prescreen` | `linear_k5_p50` | it keeps all forty-five noise columns |
| `zero-importance` | `linear_k5_p50` | a forty-tree model gives every noise column some split, so nothing is ever exactly zero |
| `zero-importance` | `id_trap` | the ID column is never unused by a tree |

## 6b. What a non-rejection means, stated before the numbers are read

A paired contrast that fails to reject is reported in one of two states, and the report may not merge them:

* **`indistinguishable`** -- the observed difference is at or above the smallest this contrast could have
  detected at the pre-registered power, and the test did not reject. This is evidence about the arm.
* **`underpowered`** -- the observed difference is smaller than that threshold. The design could not have
  detected a difference of this size, so the non-rejection is a statement about the design.

The threshold is computed per contrast from its own paired spread and seed count, never pooled: the spread
varies by an order of magnitude across arms, and one pooled threshold would flatter the noisy ones and
slander the stable ones.

## 8b. What makes a cell's timing quotable

Wall-clock is advisory and stays advisory. Three conditions must hold before it is reported at all:

1. **The memo is drained and the drain is VERIFIED.** MRMR memoizes whole fits by content hash in two
   independent caches, the roster runs several MRMR-family arms against one training frame, and a
   memoized fit returns the correct selection instantly. A cell that cannot verify both caches are empty
   records `memo_drained=False` and its timing is not used.
2. **A calibration anchor is recorded beside it.** A fixed, deterministic, single-threaded workload runs
   next to the arm, so timings are comparable as ratios across machines and months where they are not
   comparable as seconds.
3. **The environment tuple matches.** Quality metrics are hardware-invariant and always comparable;
   timings are comparable only within one environment, and an aggregate must refuse to compare across
   environments rather than average them.

`n_model_fits` remains the primary cost axis, and every arm in the roster reports a counted figure -- an
arm whose fits nobody counted would appear in the cost table as free.

## 7a. Hypothesis 7, answered: RFECV's aggregation choice does not move its result here

Run: the full 24-point grid (8 `VotesAggregation` rules x 3 `fi_missing_policy` values), three seeds, on
four beds -- `linear_k5_p50` and `mb_spouse_collider` at 2 500 rows, `xor3` and `joint_tail_t4` at 6 000.

**Three of the four beds answer nothing, and the fourth answers the question.**

| bed | kept | precision | recall spread over 24 configs | reading |
|---|---|---|---|---|
| `linear_k5_p50` | 15 | 0.508 | 0.000, all at 1.000 | at ceiling: no knob has room to act |
| `mb_spouse_collider` | 7 | 0.450 | 0.000, all at 1.000 | at ceiling |
| `xor3` | 33 of 33 | 0.091 | 0.000, all at 1.000 | selecting EVERYTHING: recall 1.000 by not selecting |
| `joint_tail_t4` | 2.7 | 0.800 | 0.000, against a per-seed noise of 0.289 | **the configuration does not matter** |

On the one bed where RFECV is genuinely choosing -- keeping under three columns of the bed's width at a
precision of 0.800, and scoring 0.833 rather than 1.000 -- all twenty-four configurations produce the
identical result, while the same configuration varies by 0.289 across seeds. The aggregation knob is
forty times smaller than the data draw. H7's alternative is rejected on this evidence: "RFECV" names a
method here, not a family, and every RFECV row in the atlas stands as written.

The `xor3` row is the reason this section exists in this shape. Its recall of 1.000 is the artefact this
suite has already withdrawn a headline over: the arm kept all thirty-three columns, so it recovered the
answer key by not selecting. The ablation's summary now refuses to read that as a result and names it,
because the first version of this run reported `xor3` at 1.000 alongside the real numbers.

## 7b. What a null from an at-ceiling cell can and cannot say

The general form of the above, because it will recur. A configuration sweep over cells where the arm
already scores perfectly measures the ceiling, not the configuration -- and a cell where the arm selects
everything is worse, since its perfect recall is the absence of selection rather than success at it.
Before a sweep's null is reported as "this knob does not matter", the cells it ran on must have had room
for the knob to matter in: the arm choosing a proper subset, and at least one configuration scoring below
the maximum. Otherwise the sweep answers a different question than the one it was run to answer.
