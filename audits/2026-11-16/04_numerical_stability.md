# 04 Numerical stability audit (src/mlframe, MRMR/benchmarks excluded)

| ID | severity | file:line | finding | evidence | disposition |
|----|----------|-----------|---------|----------|-------------|
| NS-01 | med | data_valuation/_knn_shapley.py:104 | Standardisation guards only `sigma == 0.0`; a constant column whose float mean is off by 1 ulp has sigma ~1e-17, so (x-mu)/sigma yields +-1 noise columns that dominate KNN distances | `np.full((1000,1),0.1).std(0)` -> 1.39e-17; `np.full((49,1),0.7).std(0)` -> 1.1e-16 | TODO: regression test `test_knn_shapley_standardize_constant_column_float_noise_not_amplified` |
| NS-02 | med | data_valuation/_training_weight_adapter.py:67 | Same exact-zero sigma guard, same amplification in the propagation step | same repro as NS-01 | TODO: `test_training_weight_adapter_constant_column_sigma_epsilon_guard` |
| NS-03 | low | calibration/_independence_check.py:62 | `E[x^2]-E[x]^2` pattern | columns centred first, annotated `moment-ok`; no cancellation | REJECTED: inputs centred, verified in code comment lines 40-62 |
| NS-04 | low | evaluation/bootstrap.py:798 | `diff/np.sqrt(var_diff)` | zero/non-finite var_diff handled at 780-789 (returns p=nan or p=1) | REJECTED: guarded |
| NS-05 | low | feature_engineering/entity_inter_event.py:138,196; recency_aggregation.py:203; hurst.py:59 | sqrt of variance | two-pass / Welford with `var<0 -> 0` clamp, or sum of squares of deviations (non-negative) | REJECTED: cannot go negative |
| NS-06 | low | feature_engineering/transformer/counterfactual_substitution.py:128, cross_feature_reconstruction.py:85 | log/division by norm | +1e-9 epsilon added | REJECTED: guarded |
| NS-07 | low | feature_engineering/spatial.py:921-938 | batched `np.linalg.solve` | ridge-regularised, comment documents LinAlgError handling | RESOLVED: already has fallback |

Scope note: grep-driven sweep of variance/sqrt/log/linalg/division patterns; not an exhaustive read of every module (softmax/logsumexp in models and float32 paths not exhaustively traced).
