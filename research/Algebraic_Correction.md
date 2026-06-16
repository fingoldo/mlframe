\# Research Proposal: Algebraic Correction Framework for MLFrame



\## Motivation



One of the central ideas of Yu. I. Zhuravlev's school is:



```

A weak or imperfect recognition algorithm

can be systematically corrected

using algebraic or logical transformations.

```



This idea predates:



\* Boosting

\* Residual Learning

\* Meta-Classifiers

\* Error-Correcting Output Codes



The question:



Can modern tabular ML benefit from explicit error-correction layers built on top of already trained models?



Unlike boosting, the objective is NOT to iteratively fit residuals.



The objective is to learn:



```

where

and why

base models fail

```



and build specialized correction models.



\---



\# Core Hypothesis



Most ensemble methods optimize:



```

prediction quality

```



but do not explicitly model:



```

prediction mistakes

```



Hypothesis:



```

Modeling error structure directly

may provide additional signal.

```



\---



\# Phase 0



Literature review.



Study:



\* Boosting

\* Gradient Boosting

\* ECOC

\* Stacked Generalization

\* Residual Learning

\* Error Prediction Models



Identify:



\* what is already solved;

\* where possible gaps remain.



\---



\# Candidate Correction Architectures



\## Type A



Binary Error Detector



Train base model.



Create target:



```

error = prediction\_wrong

```



Train:



```

ErrorClassifier(X)

```



Inference:



```

detect risky samples.

```



\---



\## Type B



Residual Correction



Train:



```

BaseModel

```



Then:



```

ResidualModel

```



Classification:



predict correction probability.



Regression:



predict residual.



Final:



```

base + correction

```



\---



\## Type C



Local Error Experts



Train base model.



Find:



\* high-error regions

\* systematic failure clusters



Train specialized correctors.



Apply only in detected regions.



\---



\## Type D



Meta-Correction Layer



Inputs:



\* original features

\* model prediction

\* model confidence

\* leaf indices

\* SHAP summaries



Target:



\* corrected prediction



This is closest to algebraic correction.



\---



\# Baselines



Must compare against:



\* LightGBM

\* CatBoost

\* XGBoost

\* Existing stacking

\* Existing blending

\* Existing ensembles



Critical:



If correction layer cannot beat stacking,

idea is rejected.



\---



\# Metrics



Classification:



\* ROC AUC

\* LogLoss

\* Brier

\* Calibration



Regression:



\* RMSE

\* MAE

\* R²



Also:



\* train time

\* inference time



\---



\# Diagnostic Questions



Q1



Are model errors predictable?



Q2



Do error regions exist?



Q3



Can correction models improve calibration?



Q4



Can correction models improve OOS performance?



\---



\# Success Criteria



Proceed only if:



\* Improvement >= 0.5% on multiple datasets.

\* Beats strongest stacking baseline.

\* Survives repeated CV.



Otherwise reject.

---

# Phase-0 probe results & verdict (2026-06-16)

Probe: `research/structural_testor_probe/meta_aggregators_probe.py` (leakage-safe: base preds are
4-fold OOF on train; corrector/meta fit on those; test preds score). **TEST-set AUC, 3 seeds.**
Two correction architectures from the ТЗ, both X-aware (the feature that distinguishes them from
pure stacking): `corr_residual` = Type B (X-aware GBM corrects the base's OOF residual, final =
clip(base + corrector(X))); `corr_meta` = Type D (GBM over [original X + member preds + aggregates]).
Baselines: linear stacking (logit/ridge) and GBM stacking.

| dataset | stack_logit | stack_ridge | stack_gbm | corr_residual | corr_meta |
|---|---|---|---|---|---|
| two_regime | 0.9937 | 0.9929 | 0.9934 | 0.9929 | **0.9947** |
| homogeneous | 0.9853 | 0.9853 | 0.9831 | 0.9584 | 0.9828 |
| madelon | 0.8714 | 0.8711 | 0.8585 | 0.8558 | 0.8603 |
| gina_agnostic | 0.9882 | 0.9894 | 0.9871 | 0.9857 | **0.9907** |
| scene | 0.9870 | 0.9868 | 0.9861 | 0.9691 | 0.9856 |
| breast_cancer | 0.9948 | 0.9955 | 0.9861 | 0.9785 | 0.9895 |

**Verdict: REJECT.**
- `corr_residual` (X-aware residual correction) is **consistently worse** than stacking and sometimes
  catastrophic (homogeneous -0.025, scene -0.017, breast -0.008 vs stack_gbm): re-fitting the base's
  residual on X double-counts X and over-fits / decalibrates.
- `corr_meta` (= "stacking + original features") beats the strongest stacking baseline on only 2/6
  (two_regime, gina -- where X still holds signal the base preds missed) and loses on 4/6, badly on
  madelon (0.8603 vs logit 0.8714) and breast (0.9895 vs ridge 0.9955; over-fits small data). No
  consistent >=0.5% win over the strongest stacking baseline -> fails the success criterion.

Diagnostic answers: Q1/Q2 errors are partially predictable and error regions exist (corr_meta helps on
gina/two_regime), but Q4 -- correction does NOT reliably improve OOS over plain stacking. Modeling error
structure explicitly adds nothing beyond a linear stacker except where original X carries residual
signal, and there a plain GBM-on-X already captures it. Do NOT build a correction-layer estimator family.
Consistent with the MoE and Polynomial-Aggregator rejections: linear stacking over member preds is the
robust ceiling.



