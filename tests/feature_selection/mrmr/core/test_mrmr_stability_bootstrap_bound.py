"""MRMR holds the flat ``stability_n_bootstrap`` constructor argument to the same bound as its config dataclass (tracker 2026-11-02, VAL6-7)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def test_zero_bootstraps_raise_a_message_that_names_the_parameter():
    """stability_n_bootstrap=0 used to fail as 'all 0 bootstraps failed ... input too small'; it now names the parameter."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": rng.normal(size=200)})
    y = pd.Series((df["a"] > 0).astype(int), name="y")
    with pytest.raises(ValueError, match="stability_n_bootstrap"):
        MRMR(verbose=0, random_state=0, stability_selection_method="cluster", stability_n_bootstrap=0).fit(df, y)
