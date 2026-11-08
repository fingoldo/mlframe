"""A caller-owned support cache gives the CMI-uplift scores of every candidate block exactly the scores computed without it, and builds the support joint once."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as cmi_mod
from mlframe.feature_selection.filters._grouped_agg_fe import score_grouped_agg_by_cmi_uplift


def _frames():
    """A raw frame, a binary target and two blocks of engineered columns with their source map."""
    rng = np.random.default_rng(0)
    n = 3000
    raw = pd.DataFrame({f"r{i}": rng.standard_normal(n) for i in range(6)})
    y = (raw["r0"] + 0.5 * raw["r1"] + rng.standard_normal(n) > 0).astype(int).to_numpy()
    blocks, e2s = [], {}
    for b in range(2):
        cols = {}
        for j in (2 * b, 2 * b + 1):
            cols[f"agg{j}"] = raw[f"r{j}"].to_numpy() * rng.uniform(0.5, 1.5) + rng.standard_normal(n) * 0.1
            e2s[f"agg{j}"] = f"r{j}"
        blocks.append(pd.DataFrame(cols))
    return raw, y, blocks, e2s


def test_scores_equal_with_and_without_the_cache_and_the_joint_is_built_once(monkeypatch):
    """Two blocks scored with a shared cache equal two uncached calls; the conditioning joint is built once instead of twice."""
    raw, y, blocks, e2s = _frames()
    base = ["r4", "r5"]
    plain = [score_grouped_agg_by_cmi_uplift(raw, b, y, base, n_bins=8, eng_to_source=e2s) for b in blocks]
    calls = []
    real = cmi_mod._support_joint
    monkeypatch.setattr(cmi_mod, "_support_joint", lambda s, nb: (calls.append(1), real(s, nb))[1])
    cache: dict = {}
    cached = [score_grouped_agg_by_cmi_uplift(raw, b, y, base, n_bins=8, eng_to_source=e2s, support_cache=cache) for b in blocks]
    assert len(calls) == 1
    for a, b in zip(plain, cached):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
