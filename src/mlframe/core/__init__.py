"""Core numerical primitives shared across mlframe.

Submodules:
    arrays   - numba-accelerated array operations (arrayMinMax, ...).
    stats    - statistical helpers (rolling, correlation, summary).
    ewma     - exponentially-weighted moving averages.
    recency_weights - parametric recency/importance weight vectors (poly/exp/power).
    robust_location - robust location estimators (M-estimator mean, geometric median).
    proportion_stats - proportion confidence intervals + required sample size.
    matrix_seriation - spectral (Fiedler/SVD) reordering of a similarity/correlation matrix.
    helpers  - general-purpose helper utilities used across the package.
"""

from __future__ import annotations


from mlframe.core.arrays import *  # noqa: F401,F403
from mlframe.core.stats import *  # noqa: F401,F403
from mlframe.core.ewma import *  # noqa: F401,F403
from mlframe.core.recency_weights import *  # noqa: F401,F403
from mlframe.core.robust_location import *  # noqa: F401,F403
from mlframe.core.proportion_stats import *  # noqa: F401,F403
from mlframe.core.matrix_seriation import *  # noqa: F401,F403
from mlframe.core.helpers import *  # noqa: F401,F403
