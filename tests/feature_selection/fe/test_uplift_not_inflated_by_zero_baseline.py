"""A missing or zero MI baseline must not manufacture a top-ranked engineered column.

``uplift = engineered_mi / (baseline + 1e-12)`` returns ~``engineered_mi * 1e12`` when the source column carries no information about the
target, which clears every ``min_uplift`` gate in the codebase and sorts first. Two ways to reach it: a source that genuinely is noise, and a
source name that failed to stem back to a raw column, whose baseline then defaulted to 0.0. Both promoted the column rather than rejecting it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._relative_uplift import relative_uplift


@pytest.mark.parametrize("baseline", [0.0, -0.0, None, float("nan"), float("inf"), -1e-9], ids=["zero", "negzero", "missing", "nan", "inf", "negative"])
def test_an_unusable_baseline_leaves_the_uplift_undefined(baseline):
    """None of these is a baseline the ratio is defined against, so none of them may produce a finite number."""
    assert math.isnan(relative_uplift(0.5, baseline)), f"baseline={baseline!r} produced {relative_uplift(0.5, baseline)}"


@pytest.mark.parametrize("baseline,expected", [(0.30, 2.0), (1e-4, 6000.0), (0.6, 1.0)], ids=["ordinary", "small", "no_gain"])
def test_a_usable_baseline_gives_the_plain_ratio(baseline, expected):
    """A legitimately small baseline is still a baseline: the ratio is computed from it, undamped."""
    assert relative_uplift(0.6, baseline) == pytest.approx(expected, rel=1e-12)


def test_the_zero_baseline_column_does_not_outrank_a_genuinely_uplifted_one():
    """Two candidates with identical engineered MI: the one whose source was noise must not sort above the one that improved on a real signal."""
    rows = [
        {"engineered_col": "noise_source__sq", "baseline_mi": 0.0, "engineered_mi": 0.05},
        {"engineered_col": "real_source__sq", "baseline_mi": 0.30, "engineered_mi": 0.05},
    ]
    for r in rows:
        r["uplift"] = relative_uplift(r["engineered_mi"], r["baseline_mi"])
    df = pd.DataFrame(rows).sort_values("uplift", ascending=False).reset_index(drop=True)
    assert df.loc[0, "engineered_col"] == "real_source__sq", f"ranking put the noise-source column first:\n{df}"


def test_an_undefined_uplift_fails_every_min_uplift_gate():
    """The gates are plain `>=` comparisons, so an undefined uplift must reject rather than pass."""
    undefined = relative_uplift(0.5, 0.0)
    for min_uplift in (0.95, 1.05, 1.10):
        assert not (undefined >= min_uplift), f"undefined uplift passed the {min_uplift} gate"


def test_an_engineered_column_whose_source_does_not_resolve_does_not_rank_first():
    """Through the univariate family's own scorer: a name that stems to no raw column gets no baseline, so it cannot head the ranking.

    ``_source_from_engineered_name`` returns the stem whether or not it names a real column, and the lookup then defaulted to 0.0 - so a
    name-resolution failure (the mis-stemming class the function's own comment records) read as "the source knows nothing" and promoted the
    column to the top with an uplift around ``engineered_mi * 1e12``.
    """
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import score_features_by_mi_uplift

    rng = np.random.default_rng(0)
    n = 4000
    real = rng.normal(size=n)
    y = (np.abs(real) > 0.7).astype(np.int64)  # |real| carries the signal, so real__He2 genuinely improves on its source
    raw_X = pd.DataFrame({"real": real})
    engineered_X = pd.DataFrame({"real__He2": real**2 - 1.0, "orphan__He2": rng.normal(size=n)})
    df = score_features_by_mi_uplift(raw_X, engineered_X, y, nbins=8)
    assert set(df["source_col"]) == {"real", "orphan"}
    orphan = df[df["source_col"] == "orphan"].iloc[0]
    assert orphan["baseline_mi"] == 0.0, "fixture precondition: the orphan name must resolve to no raw baseline"
    assert not (orphan["uplift"] > 1.0), f"an unresolved source produced uplift {orphan['uplift']}"
    assert df.loc[0, "source_col"] == "real", f"the unresolved column headed the ranking: {df.to_dict('records')}"
