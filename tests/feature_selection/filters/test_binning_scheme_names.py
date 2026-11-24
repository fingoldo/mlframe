"""The names of the binning schemes and count rules: one name each, no aliases, and the removed names (``nbins_strategy`` and its values, ``mah_sci`` / ``sci`` / ``marx``) are rejected.

``binning_scheme`` says where the cut points go and ``nbins_rule`` says how many bins there are (``_binning_spec``); the former single ``nbins_strategy`` mixed the two. The tests pin that ``mah`` is a scheme
of its own, that a removed or unknown name is rejected by the edge dispatcher and by the estimator's parameter check, and that a rule given to a scheme that picks its own count is an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges
from mlframe.feature_selection.filters._binning_spec import BINNING_SCHEMES, NBINS_RULES, make_binning_spec


def _xy(n=3000, seed=0):
    """Two informative columns with different shapes, a noise column, and a 3-class target."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.exponential(size=n), rng.uniform(size=n)])
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n), [0.0, 1.0])
    return X, y


def test_the_accepted_names_are_the_documented_ones():
    """Schemes and rules as listed in ``_binning_spec``; none of the removed names is among them."""
    assert set(BINNING_SCHEMES) == {"quantile", "uniform", "mdlp", "mdlp_validated", "optimal_joint", "mah", "bayesian_blocks"}
    assert set(NBINS_RULES) == {"rice_capped", "sturges", "freedman_diaconis", "knuth", "fixed"}
    for removed in ("mah_sci", "sci", "marx", "quantile_spacing", "fd", "qs", "fayyad_irani", "cv", "auto"):
        assert removed not in BINNING_SCHEMES and removed not in NBINS_RULES


@pytest.mark.parametrize("removed", ["mah_sci", "sci", "marx", "quantile_spacing", "sturges"])
def test_a_removed_name_is_rejected_by_the_edge_dispatcher(removed):
    """``per_feature_edges`` raises for a name that is not an edge method (``partition`` or a scheme that picks its own count)."""
    X, y = _xy()
    with pytest.raises(ValueError, match="unknown edge method"):
        per_feature_edges(X, y, method=removed, n_jobs=1)


@pytest.mark.parametrize("removed", ["mah_sci", "sci", "marx", "quantile_spacing", "sturges"])
def test_a_removed_name_is_rejected_as_a_scheme(removed):
    """``binning_scheme`` accepts none of the old names (the count rules ``sturges`` and the old ``quantile_spacing`` are not schemes)."""
    with pytest.raises(ValueError, match="unknown binning_scheme"):
        make_binning_spec(removed)


def test_the_old_constructor_parameter_is_gone():
    """``nbins_strategy`` and ``nbins_strategy_kwargs`` are no constructor parameters: no alias, no compatibility."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    for old in ("nbins_strategy", "nbins_strategy_kwargs"):
        with pytest.raises(TypeError):
            MRMR(**{old: "mdlp" if old == "nbins_strategy" else {}})


@pytest.mark.parametrize("scheme", ["mdlp", "mdlp_validated", "optimal_joint", "mah", "bayesian_blocks"])
def test_a_rule_given_to_a_scheme_that_picks_its_own_count_is_an_error(scheme):
    """The count of these schemes comes from the scheme itself: an explicit ``nbins_rule`` is a mistake the caller must see."""
    with pytest.raises(ValueError, match="does not apply"):
        make_binning_spec(scheme, "sturges", warn=False)


def test_the_scheme_without_a_rule_takes_its_own_default():
    """``None`` resolves to ``rice_capped`` for ``quantile`` and ``freedman_diaconis`` for ``uniform``."""
    assert make_binning_spec("quantile").rule == "rice_capped"
    assert make_binning_spec("uniform").rule == "freedman_diaconis"
    assert make_binning_spec("mdlp").rule is None


def test_mah_is_not_indistinguishable_from_an_unrelated_method():
    """Control: 'mah' must differ from a plain Sturges quantile partition somewhere."""
    X, y = _xy()
    mah = per_feature_edges(X, y, method="mah", n_jobs=1)
    sturges = per_feature_edges(X, y, method="partition", partition="quantile", count_rule="sturges", nbins_cap=48, n_jobs=1)
    assert any(np.asarray(a).shape != np.asarray(b).shape or not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(mah, sturges))
