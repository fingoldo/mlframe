# Structural Testor Selector for MLFrame

## Motivation

This document describes a modern reinterpretation of the classical Testor Theory
developed in the school of Yu. I. Zhuravlev.

The goal is **not** to implement exact testor search.

Classical testor algorithms:

- operate mostly on binary features;
- search for all minimal separating subsets;
- are NP-hard;
- do not scale to modern ML datasets.

Instead, we propose a scalable feature selector based on the same underlying idea:

> A useful feature is a feature that separates objects belonging to different classes.

---

# Historical Background

Classical testor theory defines a testor as a subset of features capable of
distinguishing objects from different classes.

A typical (irreducible) testor is a testor for which removal of any feature
destroys this property.

Zhuravlev later proposed estimating feature importance via:

    weight(feature)
    =
    number_of_testors_containing_feature
    /
    total_number_of_testors

This can be interpreted as a structural importance measure.

---

# Modern Reformulation

Instead of searching for all testors, build a pairwise inter-class
difference matrix.

Given:

```python
X.shape = (n_samples, n_features)
y.shape = (n_samples,)
```

Construct all pairs:

```python
(i, j)
```

such that:

```python
y[i] != y[j]
```

Each pair becomes one row of matrix:

```python
D.shape = (n_pairs, n_features)
```

where

```python
D[pair, feature]
```

measures how strongly the feature separates those two objects.

---

# Difference Matrix Construction

## Numerical Features

Default:

```python
d = abs(x_i - x_j) / std(feature)
```

Alternative:

```python
d = abs(x_i - x_j) / (max(feature)-min(feature))
```

Normalization strategy should be configurable.

---

## Categorical Features

Default:

```python
d = 1 if x_i != x_j else 0
```

Future extension:

- category embeddings
- similarity matrices

---

## Boolean Features

```python
d = int(x_i != x_j)
```

---

# Scalability

Full pairwise matrix:

```python
O(N²)
```

can become too large.

Implement:

```python
max_pairs
```

If exceeded:

- randomly sample pairs
- preserve class balance

Target scale:

- 100k+ rows
- 1000+ features

---

# Structural Coverage

For every feature:

```python
coverage[f] = mean(D[:, f])
```

Interpretation:

Average inter-class separation provided by the feature.

This is analogous to a structural feature importance.

---

# Exclusive Coverage

Motivation:

Two features may separate exactly the same object pairs.

Example:

- RSI(14)
- RSI(21)

Both can have high coverage while providing redundant information.

We need a measure of unique separation.

For each pair:

```python
best_other =
max(D[pair, g])
for g != f
```

Compute:

```python
exclusive_contribution
=
max(
    D[pair, f] - best_other,
    0
)
```

Then:

```python
exclusive_coverage[f]
=
mean(exclusive_contribution)
```

Interpretation:

How much separation is uniquely provided by feature f.

---

# Rare Pair Coverage

Potentially the most interesting metric.

Some inter-class pairs are easy.

Some are difficult.

Define:

```python
pair_weight
=
1 / n_features_separating_pair
```

Then:

```python
rare_pair_coverage[f]
=
weighted_mean(
    D[:, f],
    pair_weight
)
```

Interpretation:

Features that separate difficult and rare object pairs receive higher scores.

Hypothesis:

This may discover useful features missed by:

- SHAP
- permutation importance
- mutual information

---

# Feature Similarity

Represent each feature by:

```python
v_f = D[:, f]
```

Compute similarity:

```python
corr(v_f, v_g)
```

or

```python
cosine_similarity(v_f, v_g)
```

This produces:

```python
feature_separation_similarity
```

Important:

This is NOT ordinary feature correlation.

Two features are considered similar if they separate the same object pairs.

---

# Structural Clustering

Cluster features using vectors:

```python
D[:, f]
```

Methods:

- hierarchical clustering
- HDBSCAN
- spectral clustering

Goal:

Identify Zhuravlev-style "feature orbits":

- strong features
- medium features
- weak features

Potentially useful for diversity-aware feature selection.

---

# Feature Selection Modes

## Mode 1

Top-K by:

```python
coverage
```

---

## Mode 2

Top-K by:

```python
exclusive_coverage
```

---

## Mode 3

Top-K by:

```python
rare_pair_coverage
```

---

## Mode 4

Diversity-Aware Greedy Selection

Objective:

```python
score
=
importance
+
diversity_bonus
```

Algorithm:

1. Select best feature.
2. Penalize highly similar features.
3. Continue until K selected.

This is the most promising mode.

---

# Multi-Class Support

Use all pairs:

```python
y[i] != y[j]
```

No OVR decomposition initially.

---

# sklearn-style API

```python
selector = StructuralTestorSelector(
    max_pairs=100000,
    scoring="exclusive_coverage",
    top_k=50,
)
```

Methods:

```python
fit(X, y)

transform(X)

fit_transform(X, y)
```

Attributes:

```python
feature_importances_
exclusive_importances_
rare_pair_importances_
feature_similarity_matrix_
selected_features_
```

---

# Experiments

Compare against:

- Mutual Information
- SHAP
- Permutation Importance
- CatBoost Feature Importance
- LightGBM Gain Importance

Evaluate:

- CV score
- stability across folds
- selected feature diversity
- ensemble diversity

---

# Research Hypotheses

## Hypothesis 1

Coverage behaves as a purely structural feature importance.

---

## Hypothesis 2

Exclusive coverage identifies features that are not replaceable by others.

---

## Hypothesis 3

Rare pair coverage identifies niche but valuable features.

---

## Hypothesis 4

Diversity-aware selection produces stronger ensembles than ordinary top-K importance.

---

# Probe results & verdict (2026-06-16)

A cheap go/no-go probe was run before any production code: `research/structural_testor_probe/`
(`scorers.py` pure-numpy scorers, `probe.py` driver). Two independent questions, real data
(Madelon + breast_cancer + ionosphere controls), ReliefF via `skrebate`. Verbatim numbers:

**Experiment A — is `rare_pair_coverage` just ReliefF? NO, but it has NO robust advantage either.**
- Spearman(rare_pair_soft, ReliefF) = **+0.081** (XOR), **+0.087** (niche): NOT a numeric ReliefF reskin.
- A single-seed niche run *looked* promising (rare_pair rank 5 vs ReliefF 11 vs MI 38). **A 10-seed
  stability re-check overturned it** — median niche-feature rank: ReliefF **7.0**, MI 15.5,
  rare_pair **18.0**, coverage 17.5. So robustly ReliefF is the *best* niche recoverer, and rare_pair
  is statistically indistinguishable from trivial `coverage` and no better than MI's median. The
  single-seed "win" was seed noise. REJECTED.

**Experiment B (the lead bet) — `separation_profile` redundancy for mRMR: REJECTED (7 datasets).**
Generic mRMR (MI relevance, pluggable redundancy), GB/LR downstream ROC-AUC. sep_profile **never wins**;
it almost always equals `corr`, and where a gap exists **SU wins**:

| dataset                  | corr | SU | sep_profile | note |
|--------------------------|------|----|-----|------|
| madelon (interaction)    | 0.617 | **0.748** | 0.619 | SU dominates; sep≈corr |
| two_latent_groups (synth)| 0.852 | **0.897** | 0.852 | SU wins; sep=corr |
| correlated_redundant     | 0.980 | 0.980 | 0.980 | tie |
| latent_reflections       | 0.938 | 0.911 | 0.938 | sep=corr (SU worse) |
| breast_cancer / ionosphere | ~0.99 / ~0.97 | ~ | ~ | within noise |

The home-turf synthetic built to favour sep_profile (low-corr redundant twin) **mis-constructed**:
the twin's |corr|=0.52 was not low and sep_sim=0.52 matched it, so no discrimination arose. Across the
other 6 sets sep_profile ≈ corr and is dominated by SU. "Different similarity metric ≠ better." Gate not met.

**Experiment C — add ReliefF (skrebate) to the production cross-selector bench? NO.**
`round4_broad_realdata_bench.py` + `fs_selectors.ReliefFSel`, 4 real datasets, mean of lgbm/logit/knn
held-out AUC + fit seconds. ReliefF never wins and is slow/bloated:

| dataset       | ReliefF AUC (n, s) | best shipped |
|---------------|--------------------|--------------|
| madelon       | 0.700 (252f, 27s)  | hybrid 0.841 / shap 0.840 |
| gina_agnostic | 0.960 (454f, 85s)  | hybrid 0.969 |
| scene         | 0.975 (231f, 13s)  | shap 0.991 / rfecv 0.984 |
| breast_cancer | 0.992 (30f, 0.4s)  | boruta 0.993 (ReliefF kept ALL 30 → no selection) |

On madelon (ReliefF's own interaction home turf) it is nearly the worst, and its positive-weight cut
bloats the set (252–454 feats) without an AUC gain, at no speed advantage over shap/boruta. Caveat: a
tuned top-k cut would shrink the sets, but ReliefF's *ranking* on madelon (lgbm 0.879 with 252f vs shap
0.937 with 11f) is still below the stack, so k-tuning would not flip the verdict.

**Experiment D — Zhuravlev-spirit STRUCTURE test (disjoint minimal-sufficient feature systems +
ensemble vote): most interesting, still NO-GO.** Reframe: not feature ranking, but "find multiple
near-disjoint systems that each solve the task, ensemble them."
- Disjoint-testor ensemble **beats Random Subspaces** of matched sizes by +0.034–0.069 AUC (both
  synthetic two-solution data and breast_cancer, both GB and LR). Structure > randomness — the one
  place the testor idea does something a random baseline doesn't.
- BUT it **loses to one model on the full X** everywhere (synth 0.965 vs 0.971; breast 0.975 vs 0.988).
  No predictive ROI: just fit on everything.
- "Recover the alternative explanations" is ill-posed: "two near-independent groups each solving y" is
  mathematically A⊥B | y (two interchangeable noisy detector pools of the label), so minimal subsets mix
  A/B (observed) and need not separate. Genuinely *distinct* systems differ by functional form
  (linear vs interaction) → that is interaction detection, which mlframe already does
  (`structure_discovery.py`, SU-seeded interactions).

**Overall decision: do NOT build `StructuralTestorSelector`, the numba/GPU kernels, the
`separation_profile` MRMR mode, or add ReliefF.** Every branch benchmarks out against the mature shipped
stack (MRMR/hybrid/boruta/shap/SU). The narrow true findings: (i) ReliefF is the best *niche-synthetic*
recoverer but that does not transfer to real-data downstream AUC; (ii) disjoint-system ensembling beats
random subspaces but not full-X. Mirrors the poly-vs-fourier rejection: clean ideas, cheaply falsified
(~one evening) before any production cost. Re-open only with a concrete dataset where these gaps invert.

---

# Future Extensions

1. Approximate minimal testor sampling.
2. Fuzzy testors.
3. Regression adaptation.
4. Pairwise routing features for Mixture-of-Experts.
5. Meta-features derived from structural separation profiles.
6. Automatic discovery of feature orbits.
