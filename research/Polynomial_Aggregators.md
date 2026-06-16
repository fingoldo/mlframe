\# Research Proposal: Learnable Polynomial Aggregators for Ensemble Learning



\## Motivation



One of Zhuravlev's most intriguing ideas:



```

Recognition algorithms produce evaluations.

```



The final decision is only a function applied to these evaluations.



Therefore:



```

Better aggregation

may be as important as better base models.

```



Modern analogues:



\* Stacking

\* Meta-learning

\* Learned Aggregators

\* Deep Ensembles



The question:



Can polynomial combinations of model outputs outperform standard stacking?



\---



\# Core Hypothesis



Current stacking usually uses:



\* linear models

\* shallow GBM

\* logistic regression



as meta-learners.



These may fail to capture higher-order interactions between model predictions.



Hypothesis:



```

Polynomial aggregators

```



can exploit interactions between predictors.



\---



\# Phase 0



Literature review.



Study:



\* Stacking

\* Polynomial Networks

\* Functional Link Networks

\* Product-of-Experts

\* Deep Sets

\* Learned Aggregation



Determine whether the idea is already dominated by GBM meta-learners.



\---



\# Inputs



For every sample collect:



Base predictions:



```

p1

p2

...

pk

```



Optional:



\* confidence

\* uncertainty

\* SHAP summaries

\* leaf embeddings



\---



\# Candidate Aggregators



\## A



Linear Stacking



Baseline.



\---



\## B



Polynomial Expansion



Generate:



```

p\_i

p\_i²

p\_i³

p\_i\*p\_j

```



Train:



\* Ridge

\* Lasso

\* ElasticNet



\---



\## C



Sparse Polynomial Selection



Generate many polynomial terms.



Use:



\* L1 regularization

\* feature selection



to keep only useful interactions.



\---



\## D



Gradient Boosting Aggregator



Use prediction vector only.



Compare against polynomial model.



This is the strongest baseline.



\---



\## E



Symbolic Aggregator



Optional.



Search simple formulas:



```

f(p1,p2,...)

```



using symbolic regression.



Research only.



\---



\# Baselines



Must include:



\* Mean averaging

\* Weighted averaging

\* Logistic stacking

\* Ridge stacking

\* GBM stacking



If GBM stacking wins everywhere,

reject idea.



\---



\# Datasets



Use existing MLFrame benchmark suite.



Minimum:



\* 10 classification datasets

\* 10 regression datasets



\---



\# Metrics



Classification:



\* ROC AUC

\* LogLoss

\* Calibration



Regression:



\* RMSE

\* MAE



\---



\# Diagnostic Analysis



Measure:



\* meta-feature importance

\* selected polynomial terms

\* interaction frequency



Questions:



Which interactions matter?



Do useful polynomial terms repeat across datasets?



\---



\# Failure Conditions



Reject if:



\* GBM stacking dominates.

\* Gains disappear after repeated CV.

\* Complexity greatly exceeds benefit.



\---



\# Success Criteria



Proceed only if:



\* Consistent wins over linear stacking.

\* Occasional wins over GBM stacking.

\* Improvements survive repeated CV.



\---



\# Most Important Benchmark



The key benchmark is NOT:



```

Polynomial vs Mean

```



The key benchmark is:



```

Polynomial Aggregator

vs

GBM Meta-Learner

```



If GBM already captures the same interactions,

the idea is effectively solved by existing stacking methods.

---

# Phase-0 probe results & verdict (2026-06-16)

Probe: `research/structural_testor_probe/meta_aggregators_probe.py` (shares the leakage-safe
member-prediction harness: 5-member pool -> 4-fold OOF train preds train every meta-learner,
test preds score). **TEST-set AUC, 3 seeds.** Poly aggregators = `PolynomialFeatures(deg 2/3)` over
the member preds + L2/L1 logistic, vs the ТЗ baselines (mean / linear logit / ridge / GBM stacking).

| dataset | ens_mean | stack_logit | stack_gbm | poly2_l2 | poly2_l1 | poly3_l2 |
|---|---|---|---|---|---|---|
| two_regime | 0.9869 | 0.9937 | 0.9934 | 0.9928 | 0.9928 | 0.9926 |
| homogeneous | 0.9820 | 0.9853 | 0.9831 | 0.9853 | 0.9854 | 0.9852 |
| madelon | 0.8329 | 0.8714 | 0.8585 | 0.8689 | 0.8689 | 0.8710 |
| gina_agnostic | 0.9834 | 0.9882 | 0.9871 | 0.9888 | 0.9887 | 0.9886 |
| scene | 0.9860 | 0.9870 | 0.9861 | 0.9894 | 0.9892 | 0.9890 |
| breast_cancer | 0.9939 | 0.9948 | 0.9861 | 0.9952 | 0.9956 | 0.9934 |

**Verdict: REJECT.**
- The ТЗ's headline "poly vs GBM stacking" superficially passes (poly >= stack_gbm on 5/6), but that
  is misleading: poly aggregators are essentially EQUAL to plain LINEAR stacking (logit/ridge), and
  BOTH beat GBM stacking. With only ~5 base predictions a GBM meta-learner OVER-FITS and underperforms
  a linear meta -- so the "strong baseline" the ТЗ feared is actually weaker here than linear stacking.
- Against the right baseline (linear stacking) poly shows NO consistent win: it ties or loses on
  two_regime / madelon / breast and edges only on scene (+0.0024) / gina (+0.0006). Fails the success
  criterion "consistent wins over linear stacking."
- L1 keeps only ~4-11 poly terms yet matches L2 -> the higher-order pred-interactions carry no extra
  transferable signal beyond the linear blend.

Takeaway: for combining a handful of strong model outputs, a LINEAR meta-learner (logistic/ridge) is
the robust ceiling; polynomial expansion adds nothing and nonlinear GBM-over-preds over-fits. Do NOT
build a polynomial-aggregator estimator. Consistent with the MoE-over-predictions rejection.



