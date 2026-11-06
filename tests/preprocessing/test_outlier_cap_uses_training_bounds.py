"""A test frame is capped by bounds fitted on the training frame, not by its own quantiles.

Computing the bounds on whatever frame is passed let the test statistics set the cap they are then scored with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.outlier_policy import apply_outlier_policy, fit_outlier_bounds


class _Linear:
    """A model that is not tree-based, so the cap branch runs."""


def test_test_frame_is_capped_by_the_training_bounds():
    """A test frame with a huge outlier is capped at the training upper bound, not at its own 99th percentile."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=1000)})
    test = pd.DataFrame({"a": np.concatenate([rng.normal(size=99), [1000.0]])})
    bounds = fit_outlier_bounds(train, ["a"])
    capped = apply_outlier_policy(test, _Linear(), columns=["a"], bounds=bounds)
    assert capped["a"].max() == bounds[1]["a"]
    assert capped["a"].max() < 10.0
