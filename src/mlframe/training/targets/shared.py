"""Cross-package API of ``mlframe.training.targets``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.targets`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _NAN_FRACTION_THRESHOLD as NAN_FRACTION_THRESHOLD,
)
from ._target_distribution_analyzer_stats import (
    _lag1_autocorr_grouped as lag1_autocorr_grouped,
    _max_abs_lag_autocorr as max_abs_lag_autocorr,
)
from ._ttr_eval_set_scaling import (
    _TTRWithEvalSetScaling as TTRWithEvalSetScaling,
)

__all__ = [
    "NAN_FRACTION_THRESHOLD",
    "TTRWithEvalSetScaling",
    "lag1_autocorr_grouped",
    "max_abs_lag_autocorr",
]
