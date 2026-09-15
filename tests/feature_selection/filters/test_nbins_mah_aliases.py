"""The ``mah`` binning aliases must stay exact aliases (mrmr_audit_2026-09-14 TESTGAP-1).

``nbins_method`` accepts ``"mah"``, ``"mah_sci"``, ``"sci"`` and ``"marx"``, and ``_adaptive_nbins._METHOD_ALIASES`` maps all four to the
same estimator: MAH / SCI is one method (Marx 2021), so the aliasing is intentional. None of the four names had a single test, so a typo in
the alias table, or a change that made one name silently mean something else, would pass unnoticed.
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


@pytest.mark.parametrize("alias", ["mah_sci", "sci", "marx"])
def test_alias_resolves_to_mah(alias):
    """The alias table itself: every name points at the one MAH estimator."""
    assert _METHOD_ALIASES[alias] == _METHOD_ALIASES["mah"] == "mah"


@pytest.mark.parametrize("alias", ["mah_sci", "sci", "marx"])
def test_alias_produces_the_same_edges_as_mah(alias):
    """End to end: the per-feature edges an alias produces are identical to ``mah``'s."""
    X, y = _xy()
    ref = per_feature_edges(X, y, method="mah", n_jobs=1)
    got = per_feature_edges(X, y, method=alias, n_jobs=1)
    assert len(got) == len(ref)
    for j, (g, r) in enumerate(zip(got, ref)):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(r), err_msg=f"column {j}: {alias!r} edges differ from 'mah'")


def test_mah_is_not_indistinguishable_from_an_unrelated_method():
    """Control: 'mah' must differ from plain Sturges somewhere, so the equality above is not vacuously true of every method."""
    X, y = _xy()
    mah = per_feature_edges(X, y, method="mah", n_jobs=1)
    sturges = per_feature_edges(X, y, method="sturges", n_jobs=1)
    assert any(np.asarray(a).shape != np.asarray(b).shape or not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(mah, sturges))
