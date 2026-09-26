"""Cross-package API of ``mlframe.metrics.regression``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.metrics.regression`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _fast_mae_par as fast_mae_par,
    _fast_mae_seq as fast_mae_seq,
    _fast_mae_weighted_par as fast_mae_weighted_par,
    _fast_mae_weighted_seq as fast_mae_weighted_seq,
    _fast_max_error_seq as fast_max_error_seq,
    _fast_mse_par as fast_mse_par,
    _fast_mse_seq as fast_mse_seq,
    _fast_mse_weighted_par as fast_mse_weighted_par,
    _fast_mse_weighted_seq as fast_mse_weighted_seq,
    _fast_r2_score_par as fast_r2_score_par,
    _fast_r2_score_seq as fast_r2_score_seq,
    _fast_r2_score_weighted_par as fast_r2_score_weighted_par,
    _fast_r2_score_weighted_seq as fast_r2_score_weighted_seq,
    _fast_r2_variance_seq as fast_r2_variance_seq,
    _fused_regression_pass1_par as fused_regression_pass1_par,
    _fused_regression_pass1_seq as fused_regression_pass1_seq,
    _fused_regression_pass2_par as fused_regression_pass2_par,
    _fused_regression_pass2_seq as fused_regression_pass2_seq,
)
from ._regression_extras import (
    _naive_mae_kernel as naive_mae_kernel,
)
from ._regression_metrics import (
    _aggregate_multioutput as aggregate_multioutput,
    _to_2d as to_2d,
)

__all__ = [
    "aggregate_multioutput",
    "fast_mae_par",
    "fast_mae_seq",
    "fast_mae_weighted_par",
    "fast_mae_weighted_seq",
    "fast_max_error_seq",
    "fast_mse_par",
    "fast_mse_seq",
    "fast_mse_weighted_par",
    "fast_mse_weighted_seq",
    "fast_r2_score_par",
    "fast_r2_score_seq",
    "fast_r2_score_weighted_par",
    "fast_r2_score_weighted_seq",
    "fast_r2_variance_seq",
    "fused_regression_pass1_par",
    "fused_regression_pass1_seq",
    "fused_regression_pass2_par",
    "fused_regression_pass2_seq",
    "naive_mae_kernel",
    "to_2d",
]
