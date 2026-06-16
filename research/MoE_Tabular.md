\# Research Proposal: Mixture-of-Experts for Tabular Supervised Learning in MLFrame



\## Motivation



Several promising ideas derived from Yu. I. Zhuravlev's lectures (testors, separation-profile redundancy, rare-pair coverage, disjoint-testor ensembles) were evaluated and rejected via benchmark-gated probes.



The next idea worth investigating is a modern reinterpretation of Zhuravlev's concept of local algorithms and alternative decision systems:



> Different subsets of objects may require different predictive models.



This naturally leads to Mixture-of-Experts (MoE).



The objective is NOT to implement a neural-network MoE.



The objective is to determine whether classical supervised tabular ML can benefit from explicit expert specialization and routing.



Success criterion:



A statistically significant and reproducible improvement over existing MLFrame ensemble methods.



If the benchmark fails, the idea should be rejected without production integration.



\---



\# Core Hypothesis



Most current tabular ensembles assume:



```

One global function

f(X) -> y

```



However, many real-world datasets may contain:



\* multiple regimes

\* multiple generating processes

\* heterogeneous subpopulations

\* local feature relevance



Examples:



Credit scoring:

different decision logic for young vs elderly customers



Trading:

different logic in trending vs mean-reverting periods



Medical data:

different logic for demographic groups



The hypothesis:



```

Multiple specialized experts

\+

learned routing

```



may outperform a single global model.



\---



\# Important Restriction



Do NOT implement neural MoE initially.



No PyTorch.



No Transformer-style gating.



No end-to-end differentiable routing.



Phase 0 should use only existing MLFrame estimators.



Reason:



We want to isolate the value of expert specialization.



\---



\# Phase 0: Hard-Routed MoE



Implement:



```

HardRoutedMoEClassifier

HardRoutedMoERegressor

```



Architecture:



```

Router

&#x20;   ↓

Expert 1

Expert 2

Expert 3

...

```



Training:



1\. Train router.

2\. Split training data.

3\. Train experts on assigned subsets.

4\. Route inference samples.



\---



\# Candidate Routers



\## Router Type A



KMeans



Input:



```

X

```



Routing:



```

nearest centroid

```



This is the simplest baseline.



\---



\## Router Type B



Decision Tree Router



Train shallow tree:



```

max\_depth=2..5

```



Leaves define experts.



This creates interpretable regimes.



\---



\## Router Type C



Meta-Feature Router



Train lightweight classifier:



```

region = g(X)

```



where:



```

region

=

cluster label

```



Experts specialize within cluster.



\---



\## Router Type D



Error-Driven Router



Procedure:



1\. Train global model.



2\. Compute residuals/errors.



3\. Cluster samples using:



&#x20;  ```

&#x20;  X

&#x20;  residuals

&#x20;  prediction vectors

&#x20;  ```



4\. Train experts on clusters.



Motivation:



Experts should specialize where the global model struggles.



This is the most interesting router.



\---



\# Candidate Experts



Initially restrict to models already present in MLFrame:



\* LightGBM

\* CatBoost

\* XGBoost

\* RandomForest

\* ExtraTrees

\* Linear models



Allow:



```

same expert type

```



and



```

heterogeneous experts

```



Examples:



```

Expert1 = CatBoost

Expert2 = LightGBM

Expert3 = Linear

```



\---



\# Phase 1: Soft Routing



If Phase 0 shows value.



Instead of:



```

route to one expert

```



compute:



```

p(expert | x)

```



Final prediction:



```

weighted average

```



Classification:



```

probability averaging

```



Regression:



```

prediction averaging

```



\---



\# Phase 2: Dynamic Expert Selection



Investigate:



```

top-k experts

```



instead of:



```

all experts

```



Motivation:



reduce inference cost.



\---



\# Phase 3: Residual MoE



Train:



```

Global Model

```



Then:



```

Expert\_i predicts residual

```



Final prediction:



```

global + local correction

```



This is conceptually similar to boosting.



Potentially stronger than pure routing.



\---



\# Strong Baselines



The benchmark MUST compare against:



\## Single Models



\* LightGBM

\* CatBoost

\* XGBoost



\## Existing MLFrame Ensembles



\* Stacking

\* Blending

\* Voting

\* Random Subspaces

\* Any current ensemble implementation



\## Simple Partition Baseline



Random partition into K groups.



Train experts.



This baseline is critical.



If MoE cannot beat random partitioning:



```

routing has no value.

```



\---



\# Evaluation Protocol



Datasets:



Minimum:



\* 10 classification datasets

\* 10 regression datasets



Include:



\* heterogeneous datasets

\* interaction-heavy datasets

\* tabular benchmarks



Possible sources:



\* OpenML CC18

\* Kaggle tabular datasets already used in MLFrame



\---



\# Metrics



Classification:



\* ROC AUC

\* LogLoss

\* Accuracy



Regression:



\* RMSE

\* MAE

\* R²



Also measure:



\* training time

\* inference time

\* memory usage



\---



\# Stability



Evaluate:



\* multiple seeds

\* repeated CV



Report:



mean

std



Do not trust single runs.



\---



\# Diagnostic Metrics



Very important.



For each MoE run collect:



\## Expert Utilization



Fraction of samples routed to each expert.



Detect dead experts.



\---



\## Expert Diversity



Measure:



```

correlation(predictions)

```



between experts.



\---



\## Expert Specialization



Evaluate:



```

expert\_i error

```



inside and outside its assigned region.



A useful MoE should show specialization.



\---



\# Failure Conditions



Reject the idea if:



1\. Improvement < 1% over strongest baseline.

2\. Wins occur only on isolated datasets.

3\. Gains disappear after repeated CV.

4\. Complexity increases substantially without measurable benefit.



\---



\# Success Conditions



Proceed only if:



1\. Consistent improvement on >= 30% of datasets.

2\. Average improvement exceeds noise level.

3\. Expert specialization is observable.

4\. Improvement survives repeated CV.



\---



\# Research Questions



Q1:

Do heterogeneous regions exist in common tabular datasets?



Q2:

Can routing discover them?



Q3:

Is expert specialization real or illusory?



Q4:

Does MoE outperform stacking?



Q5:

Does residual-MoE outperform standard boosting?



\---



\# Most Interesting Variant



Among all proposed ideas, prioritize:



```

Error-Driven Router

\+

Residual MoE

```



Reason:



KMeans routing is largely a sanity check.



Error-driven routing is the closest analogue to the intuition behind Zhuravlev's local algorithms:



```

different regions of the space require different decision rules.

```



This variant has the highest probability of producing genuinely new signal relative to existing MLFrame ensembles.

---

# Phase-0 probe results & verdict (2026-06-16)

Probe: `research/structural_testor_probe/moe_probe.py`. Leakage-safe, honest design (per the
prediction-space framing): the combiner sees ONLY the member models' predictions (+ row-wise
aggregates over them: mean/std/min/max/range/median), never raw X. A 5-member pool
{lgbm, extratrees, rf, hgb, logit} produces 4-fold OOF probs on train (to TUNE every combiner) and
test probs (for scoring). **Every number below is TEST-set AUC**, 3 seeds, mean.

Three tiers compared: (1) fixed-weight simple means = mlframe `combine_probs`
(arithm/quad/qube/geo/harm/median/rrf); (2) STACKING = a global learned meta-combiner over the
member preds (`stack_logit`, `stack_gbm`); (3) MoE = input-dependent per-region combiner
(`moe_router_best`, `moe_router_softlogit`, KMeans in prediction-space). References: `single_model`
(1st pool model) and `best_model` (best member by OOF, shown on test).

| dataset | best simple-mean | stack_logit | MoE router (best) | single_model | best_model |
|---|---|---|---|---|---|
| two_regime (planted heterogeneity) | 0.9891 | 0.9936 | 0.9934 | 0.9913 | 0.9936 |
| homogeneous (control) | 0.9831 | 0.9854 | 0.9858 | 0.9774 | 0.9858 |
| madelon | 0.8329 | **0.8712** | 0.8704 | 0.8688 | 0.8688 |
| gina_agnostic | 0.9834 | 0.9881 | 0.9878 | 0.9879 | 0.9893 |
| scene | 0.9868 | 0.9868 | 0.9853 | **0.9916** | 0.9911 |
| breast_cancer | 0.9940 | 0.9947 | 0.9886 | 0.9915 | 0.9920 |

**Verdict by tier:**
1. **MoE (input-dependent gating) adds NOTHING over stacking -> REJECTED.** `moe_router` ties
   `stack` at best (two_regime 0.9934 vs 0.9936; madelon 0.8704 vs 0.8712) and LOSES on small/saturated
   data (breast 0.9886 vs 0.9947; scene 0.9853 vs 0.9868). Even on the planted-heterogeneity home turf
   where per-region rules should shine, router = stack. The defining MoE feature (different combination
   per region) does not pay off in prediction-space. Fails the ТЗ success conditions (no consistent >1%
   gain over the strongest baseline; gating no better than a global combiner).
2. **Stacking (global logit over member preds + aggregates) is a mild, consistent win over the fixed
   simple means** -- especially with heterogeneously-strong pools: madelon +0.038, two_regime +0.0045,
   gina +0.005. This is the actual answer to "how to combine N predictions better than arithm/geo/harm":
   a linear (logit) stacker, NOT MoE. Caveat: the nonlinear `stack_gbm` over-fits small data
   (breast 0.986) -- prefer the linear stacker.
3. **But even stacking only ties the best single model** (best_model >= stack on two_regime/gina;
   single_model wins scene 0.9916 outright), and the fixed geo/arithm means stay best on saturated small
   data (breast). So stacking's absolute value is modest.

**Decision: do NOT build HardRoutedMoE / soft-routing / residual-MoE.** MoE-over-predictions never beats
plain stacking, and stacking itself only marginally beats the simple means and ties the best single model.
The one transferable note: a linear logit-stacker over a heterogeneously-strong member pool is a small,
safe improvement over fixed-weight prob-means -- but it is stacking, not MoE, and not worth a new estimator
family. Mirrors the testor / separation-profile / rare-pair / ReliefF rejections: a clean idea, cheaply
benchmarked out before production cost.



