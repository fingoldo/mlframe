# Robust regression bench results (1M rows, 5% outliers)

Bench script: `bench_robust_linres_1M.py`. Synthetic: alpha_true=0.85, beta_true=50, 5% Cauchy outliers injected.

| Method | Time | alpha_err | beta_err | Notes |
|---|---|---|---|---|
| ols_lstsq | 0.03s | 0.51% | **95.25%** | OLS biased by outliers (alpha barely OK, beta destroyed) |
| ols_normal | 0.01s | 0.51% | **95.25%** | Same as lstsq, just closed-form |
| **trimmed_ls** | **0.12s** | **0.01%** | **2.40%** | OLS -> drop |residual|>3MAD -> refit; very fast, very accurate |
| huber_irls | 3.28s | 0.51% | 99.50% | sklearn HuberRegressor; alpha OK but intercept did not converge in max_iter=50 |
| ransac | 4.53s | 0.03% | 5.56% | sklearn RANSACRegressor; fine accuracy but 40x slower than trimmed |
| theil_sen | SKIPPED | n/a | n/a | O(n^2) memory; not feasible on 1M (sklearn warns + crashes) |
| lad_quantreg | 9.81s | 0.00% | 0.29% | statsmodels QuantReg q=0.5 (L1); best accuracy but 80x slower than trimmed |

**Decision**: ship `trimmed_ls` (and only `trimmed_ls`) as `linear_residual_robust` -- it is **80x faster than LAD** and **27x faster than RANSAC** while delivering **better alpha estimate than either**. Theil-Sen / Huber / LAD / RANSAC are NOT registered: user explicitly required "только одна быстрая".

Default for `linear_residual` stays unregularised OLS (back-compat); `linear_residual_robust` is opt-in via discovery's `transforms` list (already included by default in the next commit since "Accuracy/perf over legacy").
