"""`transform` rewrote categories in the caller's own DataFrame, on one branch only.

`out = df.copy(deep=False)` shares blocks with the input, and `out.loc[mask, col] = replacement` writes through
them. So `imp.transform(X_test)` mutated `X_test`. A caller keeping that frame for a second model, a baseline
comparison, or an error-analysis join silently got the substituted categories -- and a second `transform` on the
same frame reported a fallback rate of 0, because the substitution had already been applied.

Only `similarity_mode="nearest"` was affected. The `mode` branch two lines below used a whole-column rebind
(`out[col] = df[col].where(...)`), which is safe, so the two branches of the same loop body disagreed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer


def _frames():
    """Train categories {a, b, c}; the test frame carries an unseen `zzz`."""
    train = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c"], "val": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1]})
    test = pd.DataFrame({"cat": ["a", "zzz", "c"], "val": [1.0, 2.05, 3.0]})
    return train, test


@pytest.mark.parametrize("mode", ["nearest", "mode"])
class TestTheInputFrameIsNotTouched:
    """Parametrised across both branches so the two can never diverge again."""

    def _fit(self, mode, train):
        """An imputer fitted on the train frame in the requested similarity mode."""
        kw = {"value_column": "val"} if mode == "nearest" else {}
        return UnseenCategoryImputer(columns=["cat"], similarity_mode=mode, track_fallback_stats=True, **kw).fit(train)

    def test_the_caller_frame_keeps_its_values(self, mode):
        """The direct statement of the defect."""
        train, test = _frames()
        before = test["cat"].tolist()
        self._fit(mode, train).transform(test)
        assert test["cat"].tolist() == before, "transform rewrote the caller's frame"

    def test_a_second_transform_sees_the_same_work(self, mode):
        """The diagnostic consequence: the fallback rate read 0 the second time."""
        train, test = _frames()
        imp = self._fit(mode, train)
        imp.transform(test)
        first = dict(imp.fallback_stats_["cat"])
        imp.transform(test)
        assert imp.fallback_stats_["cat"] == first

    def test_the_substitution_still_happens_in_the_output(self, mode):
        """Not mutating the input must not mean not doing the work."""
        train, test = _frames()
        out = self._fit(mode, train).transform(test)
        assert "zzz" not in out["cat"].tolist()

    def test_the_known_rows_are_left_alone(self, mode):
        """Only unreliable rows may change."""
        train, test = _frames()
        out = self._fit(mode, train).transform(test)
        assert out["cat"].iloc[0] == "a" and out["cat"].iloc[2] == "c"

    def test_other_columns_are_untouched(self, mode):
        """A whole-column rebind must not disturb the rest of the frame."""
        train, test = _frames()
        out = self._fit(mode, train).transform(test)
        assert out["val"].tolist() == test["val"].tolist()


def test_nearest_picks_the_closest_known_category_by_value():
    """The branch's actual job, pinned so the rebind did not change which replacement is chosen.

    `zzz` sits at val=2.05; the fitted per-category means are a~1.05, b~2.05, c~3.05, so `b` is nearest.
    """
    train, test = _frames()
    imp = UnseenCategoryImputer(columns=["cat"], similarity_mode="nearest", value_column="val").fit(train)
    assert imp.transform(test)["cat"].iloc[1] == "b"


def test_a_non_default_index_survives_the_rebind():
    """The replacement Series is built on the masked index and has to reindex back onto the full frame."""
    train, _ = _frames()
    test = pd.DataFrame({"cat": ["a", "zzz", "c"], "val": [1.0, 2.05, 3.0]}, index=["r5", "r9", "r2"])
    imp = UnseenCategoryImputer(columns=["cat"], similarity_mode="nearest", value_column="val").fit(train)
    out = imp.transform(test)
    assert out.index.tolist() == ["r5", "r9", "r2"] and out["cat"].tolist() == ["a", "b", "c"]
