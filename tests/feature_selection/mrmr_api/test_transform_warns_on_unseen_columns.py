"""transform() on a named frame with columns the fit never saw warns, and the output is unchanged."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted():
    """A light fit on columns a, b, c."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = (X["a"] + 0.5 * X["b"] > 0).astype(np.int32).to_numpy()
    m = MRMR(full_npermutations=3, baseline_npermutations=2, n_jobs=1, verbose=0, fe_max_steps=0, random_seed=0).fit(X, y)
    return m, X


def test_transform_warns_on_unseen_columns(caplog):
    """An extra column d is named in a warning; the selected output equals the transform of the original frame."""
    m, X = _fitted()
    expected = m.transform(X)
    X_extra = X.assign(d_unseen_col=1.0)
    with caplog.at_level(logging.WARNING):
        out = m.transform(X_extra)
    assert any("d_unseen_col" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_transform_on_the_fit_columns_does_not_warn(caplog):
    """Control: the same column set raises no unseen-column warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X)
    assert not any("not seen at fit" in r.getMessage() for r in caplog.records)
