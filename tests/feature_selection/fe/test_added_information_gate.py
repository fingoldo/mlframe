"""The group-statistic gate asks whether a column adds information about y beyond its own source, with a significance test, instead of the sign of an MI difference and a fixed floor.

Before the test, ``hybrid_grouped_quantile_fe`` returned its ``top_k`` columns on a frame without any signal in every seed (the plug-in MI of a noise column, ``(Bx-1)(By-1)/(2n)``, is above the 1e-4 floor
at 30000 rows and the sign of its uplift over a noise source is a coin flip), and a column that only re-encoded its source passed or failed with the sign of the estimation noise.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._added_information_gate import added_information_p, passes_added_information
from mlframe.feature_selection.filters._group_distance_fe import hybrid_group_distance_fe
from mlframe.feature_selection.filters._grouped_quantile_fe import hybrid_grouped_quantile_fe
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import quantile_bin_columns

N = 20000
SEEDS = (0, 1, 2)


def _codes(a: np.ndarray) -> np.ndarray:
    """Ten equal-frequency bins of ``a``."""
    return np.asarray(quantile_bin_columns(np.asarray(a, dtype=float).reshape(-1, 1), 10)[0])


def _frame(seed: int, kind: str):
    """Eight numeric columns and two group columns; the target is noise, depends on x0 only (``copy``) or on x0's rank inside its group while the groups move x0 apart (``shifted``)."""
    r = np.random.default_rng(seed)
    g1 = r.integers(0, 8, N)
    X = pd.DataFrame({f"x{i}": r.standard_normal(N) for i in range(8)})
    X["g1"], X["g2"] = g1.astype(float), r.integers(0, 5, N).astype(float)
    if kind == "noise":
        y = (r.random(N) < 0.5).astype(int)
    elif kind == "copy":
        y = (X["x0"].to_numpy() + 0.7 * r.standard_normal(N) > 0).astype(int)
    else:
        X["x0"] = X["x0"] + 6.0 * g1
        y = (X.groupby("g1")["x0"].rank(pct=True).to_numpy() + 0.15 * r.standard_normal(N) > 0.5).astype(int)
    return X, y


def _winners(fn, kind: str, seed: int) -> list:
    """The names the family returns on a frame of the given kind."""
    X, y = _frame(seed, kind)
    return list(fn(X, y, group_cols=["g1", "g2"], num_cols=[c for c in X.columns if c.startswith("x")])[1])


def test_p_value_of_a_copy_of_the_source_is_not_significant_whatever_the_sign_of_the_mi_difference():
    """A column that re-encodes its source adds nothing: p is large."""
    r = np.random.default_rng(0)
    x = r.standard_normal(N)
    y = (x + 0.7 * r.standard_normal(N) > 0).astype(int)
    rank_in_group = pd.Series(x).groupby(r.integers(0, 8, N)).rank(pct=True).to_numpy()
    assert added_information_p(_codes(rank_in_group), _codes(x), y) > 0.05


def test_p_values_of_noise_columns_are_not_small():
    """Noise candidates against a noise source and a noise target: no p-value under 1e-3 in 20 draws."""
    ps = []
    for s in range(20):
        r = np.random.default_rng(s)
        ps.append(added_information_p(_codes(r.standard_normal(N)), _codes(r.standard_normal(N)), (r.random(N) < 0.5).astype(int)))
    assert min(ps) > 1e-3


def test_p_value_of_a_real_addition_is_tiny():
    """The rank of x inside its group carries the target while the groups move x apart: p is practically zero."""
    r = np.random.default_rng(1)
    g = r.integers(0, 8, N)
    x = r.standard_normal(N) + 6.0 * g
    rank = pd.Series(x).groupby(g).rank(pct=True).to_numpy()
    y = (rank + 0.15 * r.standard_normal(N) > 0.5).astype(int)
    assert added_information_p(_codes(rank), _codes(x), y) < 1e-12


def test_p_value_edge_cases():
    """A constant candidate has no degrees of freedom (p = 1); a table with fewer than five expected rows per cell is not tested (NaN)."""
    y = np.arange(100) % 2
    assert added_information_p(np.zeros(100, dtype=int), None, y) == 1.0
    assert np.isnan(added_information_p(np.arange(100) % 10, np.arange(100) % 10, y))


def test_the_mask_applies_bonferroni_and_keeps_untested_rows():
    """``alpha / number of candidates`` is the bar, ``alpha=None`` keeps all, a NaN p-value (untested) is kept for the other gates."""
    scores = pd.DataFrame({"p_added": [1e-9, 0.04, 0.2, np.nan]})
    assert list(passes_added_information(scores, 0.05)) == [True, False, False, True]
    assert list(passes_added_information(scores, None)) == [True] * 4
    assert list(passes_added_information(scores.drop(columns="p_added"), 0.05)) == [True] * 4


@pytest.mark.parametrize("fn", [hybrid_grouped_quantile_fe, hybrid_group_distance_fe], ids=["grouped_quantile", "group_distance"])
def test_a_frame_without_signal_yields_no_columns(fn):
    """Pure noise: no candidate is returned in any of three seeds (it returned the maximum number in every seed)."""
    assert [_winners(fn, "noise", s) for s in SEEDS] == [[], [], []]


def test_a_copy_of_the_only_informative_column_is_not_returned():
    """The target depends on x0 only and the groups are irrelevant: the percentile rank of x0 inside a group is a copy of x0 and is never returned."""
    assert SEEDS, "no seeds: the loop would pass vacuously"
    assert SEEDS, "no seeds: the loop would pass vacuously"
    for s in SEEDS:
        assert not any(n.startswith("grppct(x0|") for n in _winners(hybrid_grouped_quantile_fe, "copy", s))


def test_the_rank_inside_a_group_is_found_when_it_carries_the_target_and_nothing_else_is_returned():
    """The target depends on the rank of x0 inside its group: grppct(x0|g1) is returned in every seed, and it is the only column."""
    assert SEEDS, "no seeds: the loop would pass vacuously"
    for s in SEEDS:
        assert _winners(hybrid_grouped_quantile_fe, "shifted", s) == ["grppct(x0|g1)"]


def test_group_distance_keeps_a_group_level_signal():
    """The target depends on the group level of x0: group_distance still returns columns for it (the test must not cost power)."""
    r = np.random.default_rng(0)
    g1 = r.integers(0, 8, N)
    mu = r.standard_normal(8) * 2.0
    X = pd.DataFrame({f"x{i}": r.standard_normal(N) for i in range(8)})
    X["g1"], X["g2"] = g1.astype(float), r.integers(0, 5, N).astype(float)
    X["x0"] = X["x0"] * 0.3 + mu[g1]
    y = ((mu[g1] + 0.3 * r.standard_normal(N)) > 0).astype(int)
    won = hybrid_group_distance_fe(X, y, group_cols=["g1", "g2"], num_cols=[c for c in X.columns if c.startswith("x")])[1]
    assert any("x0" in n for n in won)
