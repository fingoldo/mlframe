# Selection-bias / prior-shift handling in mlframe

Two complementary tools shipped in Session 7 (2026-04-26) for the
classifier-meets-shifted-prior failure mode that the production Upwork
jobs-hired drift incident exposed.

## When you need this

A binary classifier is trained on data where the marginal `P(y=1)` does
NOT match the deployment-time marginal. The classic failure mode:

- VAL ROC AUC ~0.99 (because val drifts in the same direction as train)
- TEST ROC AUC ~0.65 (test drifts the opposite direction or harder)

The drop reads as "the model overfits" but it's almost always
**prior shift** — calibration is broken because the model learned a wrong
marginal. AUC is rank-preserving, so it tracks the misclassification
across the threshold; the actual probability output is systematically
biased toward the train-time prior.

Common causes:
- Selection bias on the label: data source only surfaces positives
  (e.g. a marketplace that lists only-hired jobs)
- Temporal drift: the population's `P(y=1)` itself shifts over time
- Sampling bias: train was filtered / oversampled differently than
  deployment data

Both shipped tools target this failure class.

## Tool 1: drift_report — catch the problem in seconds

`compute_label_distribution_drift` runs ~1ms per target and is wired
into `train_mlframe_models_suite` automatically. Right after the
train/val/test split materialises and BEFORE training starts, the
suite logs:

```
label_distribution_drift report (target_type=binary_classification target=cl_act_total_hired_above_1):
  train n=  7_304_969 n_positive=  5_405_680 P(y=1)=0.7400
  val   n=    811_663 n_positive=    698_344 P(y=1)=0.8604
  test  n=    901_847 n_positive=    746_133 P(y=1)=0.8273
  WARN: VAL P(y=1)=0.860 vs train 0.740 (Δ=+12.0pp); selection-bias / prior-shift suspected — model will be miscalibrated on val.
  WARN: TEST P(y=1)=0.827 vs train 0.740 (Δ=+8.7pp); selection-bias / prior-shift suspected — model will be miscalibrated on test.
```

If you see the WARN lines: stop the run. You're about to spend hours
training a model that will be systematically miscalibrated. Decide:

1. Adjust your dataset (add unbiased data, remove biased data, change
   the splitter).
2. Use `PULearningWrapper` (next section).
3. Accept and apply post-hoc calibration on the held-out test set
   (works only if you have a calibration sample from the target
   distribution).

The structured dict is also stored on
`metadata["label_distribution_drift"][target_type][target_name]`.
Programmatic access:

```python
from mlframe.training import compute_label_distribution_drift, format_drift_report

report = compute_label_distribution_drift(y_train, y_val, y_test, "binary_classification")
print(format_drift_report(report, target_name="my_target"))
report["drifts"]["test_minus_train_pp"]   # +8.7
report["drifts"]["max_abs_drift_pp"]      # 12.0
```

Type-aware: works for `binary_classification`,
`multiclass_classification`, `multilabel_classification`, `regression`.
Threshold defaults: 5pp for binary/multi, 0.5σ for regression.

## Tool 2: PULearningWrapper — actually fix the model

For the specific case where your training data is "positive-mostly
because the data source doesn't surface negatives" (the user's
production scenario), `PULearningWrapper` recovers calibration via
three strategies. Pick via `strategy=` or let `auto` choose.

### Data layout assumed

You provide:
- `X` — feature matrix (numpy / pandas / polars)
- `y` — observed labels (0/1). In biased periods all observed rows are
  positive; in unbiased periods y is the true label.
- `is_unbiased` — boolean array, True for rows from periods where you
  observed BOTH classes (so y is reliable).

### Three strategies

**`unbiased_only`** — train on the unbiased subset only.
- Best calibration in the small-unbiased / large-biased regime.
- Discards biased data entirely (some discrimination loss).
- Use when the unbiased subset is large enough (>1k each class) AND
  you don't trust the biased data's feature distributions.
- AUC drops slightly vs naive (synthetic: 0.864 → 0.833).

**`prior_shift_correction`** — train on full data; correct at inference.
- Uses Saerens-Latinne-Decaestecker (2002):
  `f(x) ∝ P_train(y=1|x) · P_target(y=1) / P_train(y=1)`.
- Preserves discrimination (AUC unchanged from naive — Saerens is a
  monotone rescale).
- Recovers calibration to within a few pp of the target prior.
- Best of both worlds when `P(x|y)` is the same in biased and unbiased
  periods (true when bias is on the LABEL, not on the FEATURES).
- Requires `true_prior` (target population P(y=1)). Can be supplied by
  the operator OR estimated from the unbiased subset's positive rate.
- Synthetic: Brier 0.376 → 0.149 (-60%), AUC 0.864 unchanged.

**`elkan_noto`** — Elkan & Noto (KDD 2008) classical PU.
- Trains a proxy `g(x) = P(s=1|x)` where s = "observed positive".
- Recovers `f(x) = clip(g(x)/c, 0, 1)` with `c = P(s=1|y=1)`
  estimated as the mean of g(x) on truly-positive unbiased rows.
- Requires `balance_proxy=True` (default) when s is severely skewed
  (the typical case here): without proxy balancing, g(x) ≈ marginal
  P(s=1) for everything and the recovery is degenerate.
- In the small-unbiased regime usually beaten by the simpler
  strategies. Ship for completeness; pick only if you know your
  proxy classifier separates well.
- Synthetic: Brier 0.376 → 0.196 (-48%), AUC 0.856 (slight drop).

**`auto`** (default) — picks `unbiased_only` if the unbiased subset has
≥1000 rows of each class, else `prior_shift_correction`.

### Synthetic benchmark

Real numbers from the test fixture (true TEST P(y=1)=0.46, train
observed P(y=1)=0.96 — a near-copy of the user's drift pattern):

| strategy                | mean_pred | AUC   | Brier   | vs naive Brier |
|-------------------------|-----------|-------|---------|----------------|
| NAIVE                   | 0.882     | 0.864 | 0.3765  | —              |
| UNBIASED_ONLY           | 0.463     | 0.833 | 0.1812  | -51.9%         |
| **PRIOR_SHIFT_CORRECTION** | **0.448** | **0.864** | **0.1493** | **-60.3%** |
| ELKAN_NOTO              | 0.653     | 0.856 | 0.1959  | -48.0%         |

`prior_shift_correction` is the headline win: zero discrimination loss
+ near-perfect calibration recovery + 60% Brier reduction.

### Usage

```python
from mlframe.training import PULearningWrapper
from sklearn.ensemble import HistGradientBoostingClassifier

# is_unbiased: True for rows where y was reliably labeled (you scraped
# both classes during that period). False for rows where the data
# source only surfaced positives.
is_unbiased = (
    df["job_posted_at"].between("2021-10-01", "2022-02-01")
    | (df["job_posted_at"] >= "2026-04-01")  # recent month with full uid scrape
)

pu = PULearningWrapper(
    base_estimator=HistGradientBoostingClassifier(max_iter=500),
    strategy="prior_shift_correction",
    true_prior=0.40,   # known marketplace true hire rate; estimate from unbiased subset if None
)
pu.fit(X=df[features], y=df["hired"].to_numpy(), is_unbiased=is_unbiased.to_numpy())

p_test = pu.predict_proba(X_test)[:, 1]
```

### Decision matrix

| your situation                                                                | start with                  |
|-------------------------------------------------------------------------------|-----------------------------|
| ≥1k each-class unbiased samples, want simplicity                              | `unbiased_only`             |
| Want to use ALL data, know target prior, expect P(x\|y) is the same in biased | `prior_shift_correction`    |
| Want to use ALL data, don't know target prior, unbiased subset representative | `auto` (estimates the prior) |
| Need to compare to literature / reproduce a published baseline                | `elkan_noto`                 |
| Class proportions in biased vs unbiased look very different on inspection     | start with `unbiased_only`; bigger problem suspected |

### Empirical workflow for production datasets

1. Run baseline naive classifier → record TEST Brier and AUC.
2. Set `is_unbiased` based on data-collection metadata.
3. Run `PULearningWrapper(strategy="auto")` — observe `pu.strategy_`
   to see which strategy was picked.
4. Compare TEST Brier and AUC vs baseline.
5. If `prior_shift_correction` was picked and you actually know the
   target prior, re-run with `true_prior=<your value>` instead of
   letting it estimate from the unbiased subset.
6. If results don't improve, the bias may be on the FEATURES (not just
   the LABEL) — Saerens won't fix that. Consider domain adaptation
   or extra feature engineering.

## What this does NOT fix

- **Feature-distribution drift** — when `P(x|y)` itself changes between
  biased and unbiased periods (e.g. the marketplace started accepting
  a new job category mid-stream). Saerens correction assumes only the
  marginal `P(y)` shifts, not the conditionals. If feature-distribution
  drift is severe, switch to `unbiased_only` or do explicit
  domain-adaptation work.
- **Label noise** — both tools assume that observed positives are
  TRUE positives. If your biased period has false positives mixed in,
  fix the data first.
- **Multiclass / multilabel selection bias** — `PULearningWrapper` is
  binary-only by design. For multi-output drift, the `drift_report`
  alone gives you the diagnostic; remediation is per-class importance
  reweighting (not currently shipped).

## References

- Elkan, C. and Noto, K. (2008). "Learning Classifiers from Only
  Positive and Unlabeled Data." KDD 2008.
- Saerens, M., Latinne, P., Decaestecker, C. (2002). "Adjusting the
  outputs of a classifier to new a priori probabilities: a simple
  procedure." Neural Computation 14(1).
- Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). "Detecting and
  Correcting for Label Shift with Black Box Predictors." ICML 2018.
