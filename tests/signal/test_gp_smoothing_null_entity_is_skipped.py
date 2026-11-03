"""A null entity id no longer aborts the whole GP feature build with a KeyError; its rows are skipped like groupby skips null keys."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.signal.gp_smoothing import compute_gp_smoothed_features


def test_nan_entity_rows_are_left_out_and_the_build_succeeds():
    """Two real entities plus rows with a NaN entity: two output rows, no KeyError."""
    rng = np.random.default_rng(0)
    rows = []
    for ent in ("a", "b", None):
        t = np.sort(rng.random(12) * 10)
        rows.append(pd.DataFrame({"e": ent, "t": t, "v": np.sin(t)}))
    df = pd.concat(rows, ignore_index=True)
    out = compute_gp_smoothed_features(df, "e", "t", "v", np.array([2.0, 5.0, 8.0]))
    assert out.shape[0] == 2
