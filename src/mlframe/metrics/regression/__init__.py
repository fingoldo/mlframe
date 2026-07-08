"""Regression metrics: error metrics, association metrics, and deviance families.

Submodules (internal):
    _regression_metrics  - MAE / MSE / RMSE / R2 / max-error kernels + fused block.
    _regression_extras   - RMSLE / MAPE / SMAPE / correlation / Huber / Tier-2 deviance wrappers.
    _regression_deviance - Poisson / Gamma / Tweedie deviance kernels.

The public surface below mirrors exactly the names ``mlframe.metrics.core`` re-exports from these modules, so cross-package consumers can import from the public ``mlframe.metrics.regression`` path instead of reaching into the underscore-prefixed implementation modules.
"""

from __future__ import annotations

from ._regression_metrics import (
    _fast_mae_seq, _fast_mae_par, _fast_mse_seq, _fast_mse_par,
    _fast_max_error_seq, _fast_r2_score_seq, _fast_r2_score_par,
    _fast_r2_variance_seq,
    _fast_mae_weighted_seq, _fast_mae_weighted_par,
    _fast_mse_weighted_seq, _fast_mse_weighted_par,
    _fast_r2_score_weighted_seq, _fast_r2_score_weighted_par,
    _aggregate_multioutput, _to_2d,
    fast_mean_absolute_error, fast_mean_squared_error,
    fast_root_mean_squared_error, fast_max_error, fast_r2_score,
    _fused_regression_pass1_seq, _fused_regression_pass1_par,
    _fused_regression_pass2_seq, _fused_regression_pass2_par,
    fast_regression_metrics_block,
)

from ._regression_extras import (
    fast_rmsle,
    fast_mape_mean,
    fast_smape,
    fast_mdape,
    fast_wmape,
    fast_mase,
    fast_mean_bias_error,
    fast_cv_rmse,
    fast_nash_sutcliffe,
    fast_explained_variance,
    fast_adjusted_r2_score,
    fast_huber_loss,
    fast_pearson_corr,
    fast_spearman_corr,
    fast_kendall_tau,
    fast_concordance_index,
    fast_regression_metrics_block_extended,
    fast_poisson_deviance,
    fast_gamma_deviance,
    fast_tweedie_deviance,
)

from ._regression_deviance import (
    _maybe_warn_tweedie,
    _tweedie_deviance_gamma_kernel,
    _tweedie_deviance_general_kernel,
    _tweedie_deviance_poisson_kernel,
)

from ._regression_benchmark import (
    fast_epsilon_band_accuracy,
    fast_rel_mae,
    fast_mrae,
    fast_percent_better,
    fast_logcosh_loss,
    fast_rmspe,
)
