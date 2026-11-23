"""The depth cap that ``max_categorical_cardinality`` puts on the MDLP recursion is announced (AUD23-62).

With the default ``max_categorical_cardinality=127`` an ``mdlp`` fit limits every numeric column to ``2**6 = 64`` intervals, a limit the user never set and the log never mentioned.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fit(caplog, **kw):
    """One small mdlp fit with the log captured at INFO."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((600, 4)), columns=list("abcd"))
    y = (X["a"] + 0.3 * rng.standard_normal(600) > 0).astype(int)
    with caplog.at_level(logging.INFO):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MRMR(verbose=0, random_seed=0, fe_max_steps=0, nbins_strategy="mdlp", **kw).fit(X, y)
    return [r.getMessage() for r in caplog.records if "max_depth" in r.getMessage() and "caps the MDLP" in r.getMessage()]


def test_the_default_cardinality_cap_is_logged_for_mdlp(caplog):
    """The message names the depth and the interval limit."""
    msgs = _fit(caplog)
    assert msgs and "max_depth=6" in msgs[0] and "64 intervals" in msgs[0]


def test_an_explicit_max_depth_is_silent(caplog):
    """Contrast: a user-pinned depth is the user's choice, nothing to announce."""
    assert not _fit(caplog, nbins_strategy_kwargs={"max_depth": 5})
