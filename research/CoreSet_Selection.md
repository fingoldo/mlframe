\# Research Proposal: Dataset Pruning and Coreset Selection for MLFrame



\## Motivation



MLFrame already contains substantial work on:



\* feature engineering

\* feature selection

\* model selection

\* hyperparameter optimization

\* ensemble construction



However, almost no effort has been invested into:



```

sample selection

dataset pruning

coreset construction

```



The question is:



> Can we remove a large fraction of training samples while preserving nearly the same predictive quality?



Potential benefits:



\* faster training

\* faster hyperparameter optimization

\* lower RAM usage

\* cheaper AutoML searches

\* cheaper ensembling



Unlike previous research probes (testors, separation-profile redundancy, rare-pair coverage), success is NOT defined primarily by higher predictive quality.



Success is defined by:



```

Similar quality

\+

Significant reduction in training cost

```



\---



\# Objective



Benchmark the following sample-selection methods:



1\. Forgetting Events

2\. GraNd

3\. EL2N

4\. Prototype Selection

5\. Condensed Nearest Neighbor (CNN)

6\. Core-set Selection



against simple baselines.



The goal is to determine:



\* how many samples can be removed;

\* how much training time is reduced;

\* how much predictive quality is lost;

\* whether any method is practically useful inside MLFrame.



\---



\# Important Rule



Do NOT write production code initially.



All work belongs under:



```

research/

experiments/

```



No integration into:



```

src/

```



until benchmark gates are passed.



\---



\# Phase 0: Literature Survey



For each method collect:



\* original paper

\* implementation availability

\* computational complexity

\* known strengths

\* known weaknesses

\* known benchmark results



Create:



```

research/pruning\_survey.md

```



before writing code.



\---



\# Candidate Methods



\## 1. Forgetting Events



Reference:



Toneva et al.



Idea:



Track how often samples switch from:



```

correctly classified

\->

incorrectly classified

```



during training.



Frequently forgotten samples may contain valuable information.



Never-forgotten samples may be redundant.



\---



\## 2. GraNd



Reference:



Paul et al.



Idea:



Gradient norm per sample.



Samples with small gradient contribution may be removable.



\---



\## 3. EL2N



Reference:



Paul et al.



Idea:



Per-sample prediction error magnitude.



Large errors indicate informative samples.



\---



\## 4. Prototype Selection



Representative sample selection.



Possible variants:



\* k-medoids

\* facility location

\* clustering-based prototypes



\---



\## 5. Condensed Nearest Neighbor



Classical Hart algorithm.



Keep only samples necessary for preserving decision boundaries.



\---



\## 6. Core-set Selection



Modern coreset methods.



Examples:



\* k-center greedy

\* facility location

\* diversity-based selection



Use available implementations where possible.



\---



\# Baselines



Very important.



Without proper baselines the experiment is invalid.



\## Baseline A



Random sampling.



Keep:



\* 10%

\* 20%

\* 30%

\* 50%

\* 70%



of training data.



This baseline must always be present.



Many sophisticated pruning methods fail to beat random selection.



\---



\## Baseline B



Stratified random sampling.



Classification:



preserve class proportions.



\---



\## Baseline C



No pruning.



Reference point.



\---



\# Datasets



Minimum:



10 classification datasets.



Preferred sources:



\* OpenML CC18

\* existing MLFrame benchmark datasets



Dataset sizes should vary:



Small:

<10k samples



Medium:

10k–100k



Large:

>100k



Include:



\* tabular classification

\* tabular regression



if practical.



\---



\# Models



Evaluate pruning independently of model choice.



Minimum:



\## Tree Models



\* LightGBM

\* CatBoost



\## Linear Models



\* Logistic Regression

\* Ridge/Lasso



Optional:



\* Random Forest



\---



\# Pruning Levels



For every method evaluate:



Keep:



\* 10%

\* 20%

\* 30%

\* 50%

\* 70%



of training samples.



Measure quality degradation curve.



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



Also collect:



\* train time

\* inference time

\* peak RAM



\---



\# Critical Metric: End-to-End Cost



Do NOT report only model fit time.



Report:



```

pruning\_time

\+

model\_fit\_time

```



Many pruning methods are useless if:



```

pruning takes 2 hours

training saves 5 minutes

```



Define:



```

total\_wall\_time

```



and compare against no-pruning baseline.



\---



\# Stability



Run:



\* multiple random seeds

\* repeated CV



Report:



mean

std



Do not trust single-seed wins.



\---



\# Benchmark Outputs



For each method produce:



\## Quality Retention



quality\_after\_pruning / quality\_full\_dataset



Example:



```

99%

98%

95%

```



\---



\## Speedup



\## full\_training\_time



pruned\_total\_time



Example:



```

2x

5x

10x

```



\---



\## Compression Ratio



fraction of removed samples.



Example:



```

80% removed

```



\---



\# Diagnostic Analysis



For selected datasets inspect:



\* which samples survive

\* class balance after pruning

\* distance to decision boundary

\* overlap between methods



Questions:



Do different methods preserve similar samples?



Or fundamentally different samples?



\---



\# Success Criteria



A method passes if:



1\. Average quality retention >= 98%.

2\. Average speedup >= 2x.

3\. Wins are reproducible.

4\. End-to-end wall-clock time improves.



\---



\# Strong Success



A method is highly valuable if:



1\. Quality retention >= 99%.

2\. Speedup >= 5x.

3\. Works on multiple datasets.

4\. Requires no deep-learning infrastructure.



\---



\# Failure Criteria



Reject a method if:



1\. Quality loss > 2%.

2\. Speedup disappears after including pruning time.

3\. Wins occur only on isolated datasets.

4\. Random sampling performs similarly.



\---



\# Research Questions



Q1



Can tabular datasets be compressed substantially without losing predictive power?



Q2



Which pruning methods beat random sampling?



Q3



Which methods provide the best speedup-quality tradeoff?



Q4



Does pruning improve hyperparameter optimization throughput?



Q5



Can pruning improve ensemble construction efficiency?



\---



\# Follow-up Experiments



Only if a method passes the benchmark gate.



Evaluate:



\* Hyperparameter optimization speedup

\* Ensemble training speedup

\* AutoML pipeline acceleration



These are likely more important than single-model training speed.



\---



\# Final Deliverable



Produce:



```

research/pruning\_benchmark\_report.md

```



including:



\* benchmark tables

\* plots

\* speedup curves

\* quality-retention curves

\* pass/fail verdict



For every method.



No production integration unless benchmark gates are passed.



