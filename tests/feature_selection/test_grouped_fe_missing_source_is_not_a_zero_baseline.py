"""A candidate whose named source is missing from raw_X must not clear the uplift gate by default."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_agg_fe import score_grouped_agg_by_cmi_uplift
from mlframe.feature_selection.filters._grouped_quantile_fe import score_grouped_quantile_by_mi_uplift


def _data():
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame({"g": rng.integers(0, 5, n), "present": rng.normal(size=n)})
    y = (raw["present"] > 0).astype(int).to_numpy()
    eng = pd.DataFrame({"eng_from_missing": raw["present"].to_numpy() + rng.normal(scale=0.1, size=n)})
    return raw, eng, y


def test_agg_uplift_is_unknown_when_the_source_is_missing():
    """A 0.0 baseline made uplift = cmi - 0, which clears any min_uplift for a candidate that carries signal."""
    raw, eng, y = _data()
    out = score_grouped_agg_by_cmi_uplift(raw, eng, y, ["g"], eng_to_source={"eng_from_missing": "not_in_raw_X"})
    assert np.isnan(out.loc[0, "uplift"])
    assert not (out["uplift"] >= 0.0).any(), "an unknown baseline must fail the gate, not pass it"


def test_quantile_uplift_is_unknown_when_the_source_is_missing():
    raw, eng, y = _data()
    out = score_grouped_quantile_by_mi_uplift(raw, eng, y, eng_to_source={"eng_from_missing": "not_in_raw_X"})
    assert np.isnan(out.loc[0, "uplift"])
