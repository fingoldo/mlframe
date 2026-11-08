"""``MRMR.transform`` replays engineered columns on a selection of the input and never changes the caller's frame."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fit():
    """A small fit with engineered features and the frame it was fitted on."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n).astype(np.float32) for i in range(6)})
    y = (X["x0"] * X["x1"] + 0.3 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    sel = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=1).fit(X, y)
    return sel, X


def test_transform_leaves_the_input_frame_unchanged():
    """Columns, dtypes and values of the input are identical before and after transform, and the output is the same twice."""
    sel, X = _fit()
    before = X.copy(deep=True)
    out1 = sel.transform(X)
    out2 = sel.transform(X)
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_frame_equal(out1, out2)
    assert list(X.columns) == list(before.columns)
