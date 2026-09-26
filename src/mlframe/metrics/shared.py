"""Cross-package API of ``mlframe.metrics``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.metrics`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._core_auc_brier import (
    _fast_brier_score_loss_seq as fast_brier_score_loss_seq,
)
from ._log_loss_and_separation import (
    _fast_log_loss_binary_seq as fast_log_loss_binary_seq,
)
from ._numba_params import (
    NUMBA_NJIT_PARAMS,
)
from .calibration.shared import close_unless_interactive
from .ranking import (
    _iter_group_slices as iter_group_slices,
    _lift_curve_kernel as lift_curve_kernel,
    _per_query_mrr_kernel as per_query_mrr_kernel,
    _per_query_ndcg_kernel as per_query_ndcg_kernel,
    _summary_batched_kernel as summary_batched_kernel,
)

__all__ = [
    "NUMBA_NJIT_PARAMS",
    "close_unless_interactive",
    "fast_brier_score_loss_seq",
    "fast_log_loss_binary_seq",
    "iter_group_slices",
    "lift_curve_kernel",
    "per_query_mrr_kernel",
    "per_query_ndcg_kernel",
    "summary_batched_kernel",
]
