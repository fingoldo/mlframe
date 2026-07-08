"""Core numerical primitives shared across mlframe.

Submodules:
    arrays   - numba-accelerated array operations (arrayMinMax, ...).
    stats    - statistical helpers (rolling, correlation, summary).
    ewma     - exponentially-weighted moving averages.
    recency_weights - parametric recency/importance weight vectors (poly/exp/power).
    robust_location - robust location estimators (M-estimator mean, geometric median).
    proportion_stats - proportion confidence intervals + required sample size.
    matrix_seriation - spectral (Fiedler/SVD) reordering of a similarity/correlation matrix.
    set_similarity - set-similarity coefficients (Jaccard/Dice/overlap/Ochiai/Kulczynski/Tversky).
    binning  - binning-smoothing (replace each value by its bin mean/median/boundary representative).
    composite_similarity - LENKOR: coordinate-descent-tuned deformed combination of per-block similarities.
    helpers  - general-purpose helper utilities used across the package.
"""

from __future__ import annotations


from mlframe.core.arrays import *
from mlframe.core.stats import *
from mlframe.core.ewma import *
from mlframe.core.recency_weights import *
from mlframe.core.robust_location import *
from mlframe.core.proportion_stats import *
from mlframe.core.matrix_seriation import *
from mlframe.core.set_similarity import *
from mlframe.core.binning import *
from mlframe.core.composite_similarity import *
from mlframe.core.helpers import *
