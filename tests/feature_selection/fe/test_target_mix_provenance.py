"""The target-mix second pass records which bases it mixed, their slopes and the candidates each produced."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_mix_fe as TM
from mlframe.feature_selection.filters.mrmr import MRMR


def test_provenance_names_the_base_and_slope_of_a_strong_column():
    """A design column that tracks y is chosen as the base, its OLS slope is recorded, and the provenance has one entry for it."""
    rng = np.random.default_rng(0)
    n = 2000
    u = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * u + 0.3 * x2 + 0.05 * rng.standard_normal(n)
    X = pd.DataFrame({"u": u, "x2": x2})
    found, provenance = TM.second_pass_recipes(MRMR(fe_max_steps=1), X, y, {"u": u}, max_bases=1, scan_rows=n, min_relative_gain=0.001)
    assert len(provenance) == 1
    assert provenance[0]["base"] == "u"
    assert abs(provenance[0]["beta"] - 2.0) < 0.1
    assert set(provenance[0]["features"]) <= set(found)


def test_no_base_gives_empty_provenance():
    """Pure noise against y has no base that clears the held-out bar, so no recipes and no provenance."""
    rng = np.random.default_rng(1)
    n = 2000
    u = rng.standard_normal(n)
    y = rng.standard_normal(n)
    found, provenance = TM.second_pass_recipes(MRMR(fe_max_steps=1), pd.DataFrame({"u": u}), y, {"u": u}, max_bases=1, scan_rows=n, min_relative_gain=0.001)
    assert found == {}
    assert provenance == []
