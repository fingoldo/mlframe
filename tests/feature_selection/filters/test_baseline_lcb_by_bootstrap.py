"""The raw-baseline LCB of the bootstrap scorer can be had without scoring any engineered column."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_bootstrap_mi_fe import baseline_lcb_by_bootstrap_mi, score_features_by_bootstrap_mi


def test_baseline_lcb_equals_the_one_the_full_scorer_reports_per_source():
    """Same seeded resamples, same arithmetic: the standalone baseline equals ``baseline_mi_lcb`` of the full scorer, source by source."""
    rng = np.random.default_rng(0)
    n = 1500
    raw = pd.DataFrame({f"c{j}": rng.normal(size=n) for j in range(6)})
    raw["bin"] = rng.integers(0, 2, n).astype(float)
    y = (raw["c0"] + 0.5 * rng.normal(size=n) > 0).astype(np.int64).to_numpy()
    eng = pd.DataFrame({f"{c}__mod_7": raw[c] % 7 for c in raw.columns})
    full = score_features_by_bootstrap_mi(raw, eng, y, n_boot=8, seed=3)
    base = baseline_lcb_by_bootstrap_mi(raw, y, n_boot=8, seed=3)
    by_source = dict(zip(full["source_col"], full["baseline_mi_lcb"]))
    assert np.allclose([by_source[c] for c in raw.columns], base, rtol=0, atol=1e-15)
