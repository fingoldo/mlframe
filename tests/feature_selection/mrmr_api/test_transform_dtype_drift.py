"""transform() warns when a column changes dtype kind since fit or arrives entirely missing.

No fit-time dtypes were stored, so an int column arriving as float, a categorical arriving as string, or an all-NaN selected column passed
transform silently: wrong values, not a crash.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted():
    """A light fit selecting an int column ``k`` and a float column ``x``."""
    rng = np.random.default_rng(0)
    n = 800
    k = rng.integers(0, 5, size=n).astype(np.int64)
    x = rng.normal(size=n)
    X = pd.DataFrame({"k": k, "x": x, "noise": rng.normal(size=n)})
    y = ((x + 0.8 * (k - 2)) > 0).astype(np.int64)
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=0, n_jobs=1, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    names = set(map(str, m.get_feature_names_out()))
    assert {"k", "x"} <= names, f"fixture precondition: k and x must be selected, got {names}"
    return m, X


def _warnings(caplog, needle):
    """WARNING records whose message contains ``needle``."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def test_int_column_arriving_as_float_warns(caplog):
    """The int column k cast to float at transform is named in a dtype-drift warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X.assign(k=X["k"].astype(np.float64)))
    msgs = _warnings(caplog, "changed dtype kind")
    assert msgs and "k" in msgs[0], f"no dtype-drift warning naming k: {[r.getMessage() for r in caplog.records]}"


def test_all_missing_selected_column_warns(caplog):
    """A selected column that is entirely NaN at transform is named in a warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X.assign(x=np.nan))
    msgs = _warnings(caplog, "entirely missing")
    assert msgs and "x" in msgs[0]


def test_unchanged_input_does_not_warn(caplog):
    """Control: transforming the fit frame raises neither warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X)
    assert not _warnings(caplog, "changed dtype kind") and not _warnings(caplog, "entirely missing")
