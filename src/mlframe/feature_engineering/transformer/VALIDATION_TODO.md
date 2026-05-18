# Validation TODO — honest gaps in iter 60-86 measurement methodology

The 86-iteration session measured all records on **`_cap_rows(X, y, 4000)`** for iteration speed.
This compresses statistical confidence and is invalid for marginal records. The work below restores
honest measurement before any record is publicly cited.

## TODO 1: Full-N validation runs

Current cap effects:
- **mammography**: 11183 → 4000 (truncated to ~36% of data, 52 positives instead of 145)
- **kin8nm**: 8192 → 4000 (~half)
- **abalone**: 4177 → 4000 (effectively no truncation)
- **diabetes**: 768 → 768 (already small, no cap effect)

Action: re-run the 7 standing records (iter 61, 66, 68, 69, 72, 77, plus iter 77 ALL-5 sweep) on
**full N** for kin8nm and mammography. Diabetes stays at 768 (physical limit). Document records
that survive vs collapse.

Expected (a priori):
- mammography records likely STRENGTHEN at full N (3× more positives stabilises the rare-class signal).
- kin8nm records likely STABLE (smooth manifold, capped N still well above kNN K=32).
- diabetes records likely FLAGGED (768 rows × 5-fold = 153/fold, fold noise ~0.5pp, marginal records
  collapse into noise band).

## TODO 2: Bootstrap confidence intervals

Each record currently is a point estimate from a single 5-fold CV split with seed=42. Honest
reporting needs 95% CIs.

Action: bootstrap the test fold 50 times per record (5-fold CV × 50 resamples = 250 fold-runs).
Compute 95% CI on lift via percentile method. Records with CI overlapping zero are not records.

Specifically suspect:
- iter 68 kin8nm LGB R² +11.91% marginal (claimed +0.57pp over iter 5/6 historical +11.34%) — likely
  CI overlaps with prior record.
- iter 77 diabetes CB PR_AUC +6.75% marginal (claimed +0.26pp over iter 60 +6.49%) — almost certainly
  fold noise at 768 rows.

## TODO 3: Seed robustness

All measurements used `KFold(n_splits=5, shuffle=True, random_state=42)`. Different seeds give
different splits → different CV variances.

Action: re-run each of 7 records on seeds {0, 7, 17, 42, 99}. Report median lift + IQR per record.
Records with high IQR are fold-luck artefacts.

## TODO 4: OpenML / production-scale validation

Current test datasets are toy-scale (max 11k rows). Production tabular workloads are 50k-500k+.
Whether records translate to that scale is unknown.

Action: add to test suite:
- Adult (49k rows binary classification — class imbalance ~24% positive)
- California Housing (20k regression)
- Forest Cover Type (580k multi-class — pick binary class-1-vs-rest)
- Higgs subset (250k binary)
- MoA Mechanisms of Action (24k multi-output → pick 1 target)

Run top-3 records from each category (regression / rare binary / balanced binary) and report
records that scale. Mechanisms that win at 4000 but lose at 100k are model-class-specific noise,
not signal.

## TODO 5: Compute scaling

While at it, profile compute scaling of winning mechanisms at N=10k, 100k, 1M:
- iter 72 ldgrad: kNN O(N log N), gradient O(N·K·d) — should be linear in N.
- iter 77 curv: kNN + quadratic-fit O(N·d²·K) — d² coefficient matters at d=20+ workloads.
- iter 66 cbhrattn+rff: O(N·K) anchor routing + O(N·d_rff) RFF — linear, fast.
- iter 86 persistence diagram (gudhi RipsComplex): O(K³) per query × N queries = O(N·K³) — could
  become bottleneck at N=1M, K=30.

## Status: pending — to be addressed before any record is cited in publications / blog posts.

Records marked "marginal" in CHANGELOG.md and RESULTS.md should not be cited without first
clearing TODO 2 + TODO 3.
