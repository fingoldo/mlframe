"""The parsimony compaction keeps an engineered feature that is redundant in the binned information but lowers the held-out linear loss of the selection."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._compaction_linear_value import make_linear_value_check
from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import choose_compaction, prune_redundant


class _Est:
    """A bare estimator holding the continuous target."""

    def __init__(self, y):
        self._fe_prewarp_y_continuous_ = y
        self.random_seed = 0


def _quadratic_frame(n=2500, seed=0):
    """``y = x^2 + 0.2 x + noise`` with the columns x, sq = x^2 (linearly useful) and junk (pure noise)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = x**2 + 0.2 * x + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "sq": x**2, "junk": rng.standard_normal(n)}), y


def test_square_of_a_selected_source_has_linear_value_and_noise_has_none():
    """Removing ``sq`` from ``[x, sq]`` makes the linear fit of a quadratic much worse; removing noise does not."""
    X, y = _quadratic_frame()
    check = make_linear_value_check(_Est(y), X, {})
    assert check is not None
    assert check("sq", ["x"]) is True
    assert check("junk", ["x", "sq"]) is False


def test_no_continuous_target_means_no_check():
    """Without a continuous target the compaction keeps its information-only rule."""
    X, _ = _quadratic_frame()
    assert make_linear_value_check(_Est(None), X, {}) is None


def test_prune_redundant_keeps_what_the_linear_check_vouches_for():
    """A feature whose codes equal a selected one's is information-redundant and dropped, unless the linear check says it carries linear value."""
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 6, 3000)
    y_codes = (codes % 2).astype(np.int64)
    columns = {"x": codes, "e": codes.copy()}
    assert prune_redundant(columns, ["x", "e"], ["e"], y_codes) == ["e"]
    assert prune_redundant(columns, ["x", "e"], ["e"], y_codes, None, lambda feature, rest: True) == []
    assert prune_redundant(columns, ["x", "e"], ["e"], y_codes, None, lambda feature, rest: False) == ["e"]


def test_choose_compaction_skips_a_replacement_that_loses_linear_value():
    """Two engineered features covered by a pool candidate with the same information are replaced, unless the replacement loses linear value."""
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4, 4000)
    y_codes = (a == 1).astype(np.int64)
    columns = {"e1": a, "e2": a.copy(), "cand": a.copy()}
    sources = {"e1": frozenset({"s"}), "e2": frozenset({"s"}), "cand": frozenset({"s"})}
    args = (columns, ["e1", "e2"], ["e1", "e2"], sources, ["cand"], y_codes)
    assert choose_compaction(*args) == ("cand", ["e1", "e2"])
    assert choose_compaction(*args, None, lambda cand, replaced, rest: True) == ("cand", ["e1", "e2"])
    assert choose_compaction(*args, None, lambda cand, replaced, rest: False) is None
