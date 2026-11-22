"""``mah`` is the one name of the multi-dimensional adaptive histogram binning; the former aliases ``mah_sci`` / ``sci`` / ``marx`` are gone.

The three aliases mapped to the same estimator (mrmr_audit_2026-09-14 TESTGAP-1) and only made the option list ambiguous, so the owner asked for one name (2026-11-22). The test pins that the
one name works, that the removed names are rejected by the edge dispatcher and by the estimator's parameter check, and that ``mah`` differs from an unrelated method.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import _METHOD_ALIASES, per_feature_edges


def _xy(n=3000, seed=0):
    """Two informative columns with different shapes, a noise column, and a 3-class target."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.exponential(size=n), rng.uniform(size=n)])
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n), [0.0, 1.0])
    return X, y


def test_mah_resolves_to_itself_and_the_old_aliases_are_gone():
    """The alias table keeps ``mah`` and carries none of the three removed names."""
    assert _METHOD_ALIASES["mah"] == "mah"
    for removed in ("mah_sci", "sci", "marx"):
        assert removed not in _METHOD_ALIASES


@pytest.mark.parametrize("removed", ["mah_sci", "sci", "marx"])
def test_a_removed_name_is_rejected_by_the_edge_dispatcher(removed):
    """``per_feature_edges`` raises for a name that is no longer a method."""
    X, y = _xy()
    with pytest.raises(ValueError, match="unknown method"):
        per_feature_edges(X, y, method=removed, n_jobs=1)


@pytest.mark.parametrize("removed", ["mah_sci", "sci", "marx"])
def test_a_removed_name_is_rejected_by_the_estimator(removed):
    """The estimator's own validation does not list the removed names."""
    from mlframe.feature_selection.filters.mrmr._mrmr_param_constants import _VALID_NBINS_STRATEGIES

    assert removed not in _VALID_NBINS_STRATEGIES
    assert "mah" in _VALID_NBINS_STRATEGIES


def test_mah_is_not_indistinguishable_from_an_unrelated_method():
    """Control: 'mah' must differ from plain Sturges somewhere."""
    X, y = _xy()
    mah = per_feature_edges(X, y, method="mah", n_jobs=1)
    sturges = per_feature_edges(X, y, method="sturges", n_jobs=1)
    assert any(np.asarray(a).shape != np.asarray(b).shape or not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(mah, sturges))
