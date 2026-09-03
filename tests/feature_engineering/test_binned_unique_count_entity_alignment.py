"""The entity labels and the entity codes came from two functions that disagree about missing values.

`pd.unique` KEEPS a NaN entity as an element of its result; `pd.factorize` maps it to the sentinel -1 and omits
it from its uniques. The labels were taken from the first and the codes from the second, so one row with a
missing entity shifted every entity appearing after it in first-seen order by one position -- each entity's
distinct-bin count was written into its neighbour's row, and the NaN row received a count belonging to a real
entity.

Silent by construction: the frame has the right shape, the counts are all plausible integers, and nothing
raises. These tests compare against a groupby computed independently, which is what makes them discriminate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.binned_unique_count import binned_unique_count


def _truth(df: pd.DataFrame, n_bins: int) -> dict:
    """Distinct bins per entity, computed independently of the function under test."""
    kept = df.dropna(subset=["e"]).copy()
    kept["b"] = pd.cut(kept["v"], n_bins, labels=False)
    return kept.groupby("e")["b"].nunique().to_dict()


class TestAMissingEntityDoesNotShiftTheOthers:
    """The shift only appears once a NaN entity precedes another entity in first-seen order."""

    def test_counts_match_a_plain_groupby(self):
        """The direct statement of the contract, on the frame that triggers the bug."""
        df = pd.DataFrame({"e": ["A", "A", None, "B", "B", "B"], "v": [0.1, 0.9, 0.5, 0.1, 0.5, 0.9]})
        out = binned_unique_count(df, entity_col="e", value_col="v", n_bins=3)
        assert dict(zip(out["e"], out["binned_unique_v"])) == _truth(df, 3)

    def test_a_missing_entity_gets_no_row(self):
        """It used to get one, carrying a real entity's count -- worse than being absent."""
        df = pd.DataFrame({"e": ["A", None, "B"], "v": [0.1, 0.5, 0.9]})
        out = binned_unique_count(df, entity_col="e", value_col="v", n_bins=3)
        assert out["e"].isna().sum() == 0
        assert set(out["e"]) == {"A", "B"}

    def test_a_leading_missing_entity_shifts_everything(self):
        """The worst case: the NaN comes first, so every entity was off by one."""
        df = pd.DataFrame({"e": [None, "A", "A", "B", "B", "C"], "v": [0.5, 0.1, 0.9, 0.2, 0.8, 0.5]})
        out = binned_unique_count(df, entity_col="e", value_col="v", n_bins=4)
        assert dict(zip(out["e"], out["binned_unique_v"])) == _truth(df, 4)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_agrees_with_a_groupby_on_random_frames(self, seed):
        """Fuzzed over entity order and missing-value placement, since the bug is order-dependent."""
        rng = np.random.default_rng(seed)
        n = 300
        labels = rng.choice(list("ABCDE"), n).astype(object)
        labels[rng.random(n) < 0.12] = None
        df = pd.DataFrame({"e": labels, "v": rng.random(n)})
        out = binned_unique_count(df, entity_col="e", value_col="v", n_bins=5)
        assert dict(zip(out["e"], out["binned_unique_v"])) == _truth(df, 5)


class TestTheNoMissingCaseIsUnchanged:
    """The fix must not move the common path, where the two functions agreed all along."""

    def test_counts_and_order_are_preserved(self):
        """First-seen order is part of the documented contract."""
        df = pd.DataFrame({"e": ["B", "B", "A", "A", "A"], "v": [0.1, 0.9, 0.2, 0.5, 0.8]})
        out = binned_unique_count(df, entity_col="e", value_col="v", n_bins=3)
        assert list(out["e"]) == ["B", "A"], "entities must stay in first-seen order"
        assert dict(zip(out["e"], out["binned_unique_v"])) == _truth(df, 3)
