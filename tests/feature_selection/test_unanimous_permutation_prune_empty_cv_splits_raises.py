"""FS_BORUTA_ROOT-6 (2026-08-05 audit): an empty ``cv_splits`` must raise a clear, actionable ValueError
naming ``cv_splits`` -- not an opaque ``np.stack([], axis=0)`` numpy ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.unanimous_permutation_prune import unanimous_permutation_prune


def test_empty_cv_splits_raises_clear_error():
    """cv_splits=[] must raise ValueError naming cv_splits, not numpy's opaque np.stack error."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20)})
    y = rng.normal(size=20)

    with pytest.raises(ValueError, match="cv_splits"):
        unanimous_permutation_prune(X, y, estimator_factory=lambda: None, cv_splits=[])
