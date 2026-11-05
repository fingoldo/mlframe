"""The slice-finder pair cap builds only the pairs it keeps.

The full pair list used to be materialised before the budget was applied, so a wide frame paid for every one of its p*(p-1)/2 pairs and kept
only a few thousand of them.
"""

from __future__ import annotations

import itertools

import numpy as np

from mlframe.reporting.charts import slice_finder as sf


def test_pair_enumeration_stops_at_the_kept_budget(monkeypatch):
    """With 2000 features and a small budget, the generator is advanced only as far as the kept prefix, not over all 2M pairs."""
    p = 2000
    max_combos = p + 25
    produced = {"n": 0}
    real_combinations = itertools.combinations

    def counting_combinations(iterable, r):
        """Wrap ``itertools.combinations`` so the number of pairs drawn is observable."""
        for item in real_combinations(iterable, r):
            produced["n"] += 1
            yield item

    monkeypatch.setattr(sf.itertools, "combinations", counting_combinations)
    mat = np.zeros((4, p))
    err = np.zeros(4)
    capped: list = []
    combos = sf._build_combos(mat, err, [f"f{i}" for i in range(p)], p, 2, max_combos, 0, 0, capped)
    assert len(combos) == max_combos
    assert produced["n"] <= max_combos
    assert any("pair enumeration truncated" in note for note in capped)
