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

2240 cells: 7 eligible beds x 16 arms x 20 reserved seeds (1000-1019), 2209 ok. Paired t on the 20 per-seed differences, m=20, df=19.

**On the primary outcome (matched K), the stop condition is MET.** Beds where no arm beats `all-features`, out of 7:

| model | k5 | k10 | k20 | k50 | k100 | k200 | self |
|---|---|---|---|---|---|---|---|
| lightgbm | 7 | 6 | 5 | 5 | 5 | 5 | 3 |
| logistic | 4 | 4 | 4 | 3 | 2 | 0 | 2 |

With a strong model at any declared cardinality, 5 to 7 of the 7 beds show no arm clearing the null. The criterion's threshold is 4. With a linear model it clears once enough features are allowed (k50+).

Two ambiguities in this document, exposed by its own first use and recorded rather than resolved after the fact: the criterion names neither the model nor the K at which it is evaluated, and it says "beyond the noise band" while the harness decides with a paired t at p<0.05. Read in the spirit it was written -- primary outcome, realistic model -- it triggers. A future pre-registration must name both.

**Hypothesis H3 is CONFIRMED on the primary outcome.** An earlier reading of the same run reported it falsified; that reading used the self-chosen-K row, the SECONDARY outcome, because a hardcoded label list in the report renderer silently emitted no matched-K section at all. The renderer now derives its labels from the records.

Residue worth keeping regardless of the stop decision:

- Only madelon and hill-valley ever produce a lightgbm win at matched K. On arcene at k20 every arm loses to `all-features`, MRMR by -0.0395 (p=0.0002).
- On madelon at k50, k100 and k200 the outright winner is `variance-sort` -- ranking by marginal variance, with no target involved -- ahead of every information-theoretic and wrapper arm. At k20 it is third (+0.0562), still ahead of MRMR. On a synthetic bed this document treats that as a broken bed; on a real one it is a statement about madelon and about the arms, and it stands.
- `random-<k>` at matched cardinality beats `all-features` on madelon under logistic (+0.0171, p=0.001), which is why every skill number here is read against that control and not against the null alone.

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
7. Free knobs are declared as **priors, not points** (`corr ~ U(0.5, 0.99)`, `prevalence ~ LogUniform(0.01,
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
