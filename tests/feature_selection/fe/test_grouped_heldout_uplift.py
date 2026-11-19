"""The held-out marginal redundancy test of the grouped aggregates (opt-in ``heldout_uplift``).

A noisy copy of a group key carries information its source lacks, so its aggregate must pass; a clean source already identifies the groups, so its aggregate must not. The default scorer is unchanged.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_agg_fe import generate_grouped_agg_features, score_grouped_agg_by_cmi_uplift

_BAR = 2.326  # one-sided 1 % per candidate, as in the composite scorer


def _frame(seed: int, noise: float, n: int = 6000):
    """Eight groups with distinct means; ``x`` is the group mean plus ``noise``; y is a function of the group."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 8, n)
    gm = np.array([1.0, 5.0, 9.0, 2.0, 7.0, 3.0, 6.0, 4.0])
    x = gm[region] + rng.normal(0.0, noise, n)
    return pd.DataFrame({"region": region, "x": x}), (gm[region] > 5.0).astype(int)


def _mean_row(df):
    """The group-mean row of a scored table."""
    return df[df.engineered_col.str.startswith("grpagg_mean")].iloc[0]


def _score(X, y, heldout):
    """Score the group-mean aggregate of ``X`` with the flag set as given."""
    enc, raw = generate_grouped_agg_features(X, ["region"], ["x"], stats=("mean",))
    return score_grouped_agg_by_cmi_uplift(X, enc, y, ["x"], n_bins=10, eng_to_source={c: raw[c]["num_col"] for c in enc.columns}, heldout_uplift=heldout)


def test_the_default_scorer_keeps_its_columns():
    """Without the flag the table has the documented five columns and no held-out column."""
    X, y = _frame(1, 0.8)
    df = _score(X, y, heldout=False)
    assert list(df.columns) == ["engineered_col", "source_col", "cmi", "source_mi", "uplift"]


def test_a_noisy_copy_of_the_key_passes_the_held_out_test():
    """The group mean of a noisy copy of the key is significantly better than its source (z above the bar) on several seeds."""
    for seed in (1, 7, 13):
        X, y = _frame(seed, 0.8)
        z = float(_mean_row(_score(X, y, heldout=True)).heldout_z)
        assert z >= _BAR, f"seed {seed}: z {z:.2f} below {_BAR}"


def test_a_clean_source_does_not_pass_the_held_out_test():
    """A clean source identifies the groups, so the aggregate adds nothing: z stays below the bar."""
    for seed in (1, 7, 13):
        X, y = _frame(seed, 0.01)
        z = float(_mean_row(_score(X, y, heldout=True)).heldout_z)
        assert z < _BAR, f"seed {seed}: z {z:.2f} not below {_BAR}"


def test_an_independent_source_and_target_stay_below_the_bar():
    """Pure null: the source and the target are both independent of the groups, so no aggregate carries information about the target. The median held-out z stays below the bar for several data seeds and group counts (MR11-95)."""
    from mlframe.feature_selection.filters import _grouped_agg_heldout_uplift as heldout

    n = 6000
    for groups in (4, 12):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            region = rng.integers(0, groups, n)
            x = rng.normal(size=n)
            y = rng.integers(0, 2, n).astype(np.int64)
            X = pd.DataFrame({"region": region, "x": x})
            enc, _raw = generate_grouped_agg_features(X, ["region"], ["x"], stats=("mean",))
            agg = enc[next(c for c in enc.columns if c.startswith("grpagg_mean"))].to_numpy()
            fine = heldout.source_fine_bins(x)
            zs = [heldout.heldout_marginal_gain(agg, x, y, (i := np.random.default_rng(s).permutation(n))[: n // 2], i[n // 2 :], fine=fine)[1] for s in range(25)]
            z = float(np.median(zs))
            assert z < _BAR, f"groups {groups} seed {seed}: median z {z:.2f} on an independent source"
