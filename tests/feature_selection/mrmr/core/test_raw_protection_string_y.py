"""A string class target keeps the raw-protection re-add probe on: the labels are encoded to class codes, not cast to float."""

import logging

import numpy as np
import pandas as pd


def test_string_target_does_not_disable_raw_protection(caplog):
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(6)})
    y = np.where(X["f0"] + 0.5 * rng.standard_normal(n) > 0, "yes", "no")
    with caplog.at_level(logging.WARNING):
        MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0).fit(X, y)
    assert not [r for r in caplog.records if "raw protection is DISABLED" in r.getMessage()]
