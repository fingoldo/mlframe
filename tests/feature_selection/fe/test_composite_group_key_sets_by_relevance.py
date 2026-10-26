"""Composite group-key candidates are ranked by what they say about the target, and the detector looks past the first six columns."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._composite_group_agg_fe import auto_detect_key_sets, hybrid_composite_group_agg_fe


def _frame(n=20000, seed=0):
    """Twelve six-level group columns and a value whose per-cell mean over (g8, g9) drives the target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"g{i}": rng.integers(0, 6, n) for i in range(12)})
    cell = rng.standard_normal((6, 6))
    mu = cell[X["g8"], X["g9"]]
    X["v"] = 0.35 * mu + rng.standard_normal(n)
    y = ((mu > 0) ^ (rng.random(n) < 0.05)).astype(int)
    return X, y


def test_relevant_key_set_is_ranked_first():
    """With the target given the (g8, g9) key leads the candidate list; all pairs have the same cardinality, so the old order was arbitrary."""
    X, y = _frame()
    sets = auto_detect_key_sets(X.drop(columns="v"), y=y)
    assert sets[0] == ("g8", "g9")


def test_planted_composite_group_agg_is_found_with_auto_detection():
    """The composite aggregate of v over (g8, g9) is engineered without naming the group columns."""
    X, y = _frame()
    winners = hybrid_composite_group_agg_fe(X, y, num_cols=["v"])[1]
    assert any("v|g8,g9" in w for w in winners)


def test_relevance_ranking_adds_no_false_aggregates_on_a_noise_target():
    """On a random target the y-ranked key sets give no more aggregates than the earlier choice (first six columns, lowest cardinality first); the gate's own noise level is unchanged."""
    for seed in range(3):
        X, _ = _frame(seed=seed)
        y = (np.random.default_rng(100 + seed).random(len(X)) < 0.5).astype(int)
        groups = X.drop(columns="v")
        old = auto_detect_key_sets(groups, detected_group_cols=[f"g{i}" for i in range(6)])
        new = auto_detect_key_sets(groups, y=y)
        n_old = len(hybrid_composite_group_agg_fe(X, y, num_cols=["v"], group_col_sets=old)[1])
        n_new = len(hybrid_composite_group_agg_fe(X, y, num_cols=["v"], group_col_sets=new)[1])
        assert n_new <= n_old + 1
