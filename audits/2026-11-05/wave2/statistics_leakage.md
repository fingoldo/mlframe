# Wave 2: statistics and data leakage audit (2026-11-05)

Scope: `src/mlframe`, read-only. Candidates came from greps for shuffled splitters, `fit_transform`, quantile, percentile and searchsorted calls, and p-value construction. Each candidate below was read in context.

## Result

No new High or Medium leak was found. The obvious leak sites have already been fixed in code, and their docstrings record the fixes:
- the threshold stability folds now fit on TRAIN and score on held-out (`calibration/threshold_optimizer.py:84-124`)
- the override backtest has a K-fold held-out MAE (`calibration/smoothed_override_backtest.py:103-130`)
- the OOF target encoding takes its prior and fallback from the fold (`preprocessing/category_support.py:80-95`)
- `auto_transform_select` fits each fold's transform and NaN fill on TRAIN only (`preprocessing/auto_transform_select.py:51-89, 338-351`)
- the KNN-Shapley binarization threshold comes from `y_train` only (`data_valuation/_knn_shapley_regression_binarize.py:58`)
- the early-stopping split is shuffled and stratified (`estimators/early_stopping.py:153-180`)
- `forward_select` applies a Bonferroni correction (`feature_selection/forward_select.py:95-105`)
- the group-MI permutation p-value uses the (1+k)/(1+n) form (`feature_selection/filters/info_theory/_group_mi.py:268`)

The four findings below are residual, Low or Info.

| id | sev | file:line | statement | failure scenario | disposition |
|---|---|---|---|---|---|
| STAT-W2-1 | Low | src/mlframe/feature_selection/forward_select.py:101 | The `patience` noise test runs a paired t-test on k-fold CV scores. These scores are not independent because the training sets overlap, so the variance is underestimated (Nadeau and Bengio 2003). | With the default `cv=3` the t-test has 2 df and an optimistic variance. A noise column can still pass as "significant" after Bonferroni, which resets the patience streak and lets the greedy loop run longer. | RESOLVED-needed: use the Nadeau-Bengio corrected resampled t (variance times (1/k + n_test/n_train)), or document the bias in the `patience` docstring. |
| STAT-W2-2 | Low | src/mlframe/training/composite/compare.py:154-163 | The p-value counts how many bootstrap means of the paired difference fall on the far side of zero. That distribution is centred at the observed mean, not built under H0. | This is a percentile-interval inversion, so it agrees with the reported CI. With small n and a skewed loss difference it is anti-conservative compared with a null-centred bootstrap or a permutation (sign-flip) test, so `compare_models` can report a challenger as significant too easily. | DOC: state in the docstring that it is a CI-inversion p-value. Optionally offer a sign-flip permutation test for small n. |
| STAT-W2-3 | Low | src/mlframe/preprocessing/outlier_policy.py:128-143 | `apply_outlier_policy` computes cap and outlier-score quantiles from whatever frame it is given. There is no fit/apply split. | A caller that applies it to train and test separately gets test-frame bounds, so test statistics leak into the test transform. The bounds also differ from train, so the model sees shifted features. The docstring already carries a LEAKAGE CAVEAT, but the API invites this misuse. | RESOLVED-needed (API): accept precomputed `bounds=` or return them so that test reuses the train bounds. The leak is already documented, so this is not new. |
| STAT-W2-4 | Info | src/mlframe/preprocessing/cluster.py:33-34 | The dimensionality reducer and the StandardScaler are fitted on the full `X` passed in. | This only matters if the cluster labels are used as a supervised feature and the function is called on train and test together. As written it is an unsupervised utility. | REJECTED: no held-out data is involved in its contract. |

Sites checked and judged clean (no finding):
- the `KFold(shuffle=True)` uses in `data_valuation/_adversarial_*`, `count_weighted_blend.py` (it uses GroupKFold when entities are present), `align_feature_direction.py` and `tuning_rules.py:510`
- `conformal.py`, where the bin edges come from the calibration predictions and the finite-sample quantile helpers are used
- `_dummy_bootstrap.py`, which builds percentile CIs
- `votenrank/rank_percentile_stacking.py:105-110`, where the mid-rank uses left and right searchsorted and the off-by-one is handled correctly
