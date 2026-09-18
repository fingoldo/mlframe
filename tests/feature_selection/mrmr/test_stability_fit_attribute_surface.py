"""The stability-selection fit path must leave the same fitted-attribute surface as the other terminal paths (mrmr_audit_2026-09-14 CORE-2).

``_stability_outer_fit`` set six attributes where the identity shortcut sets the full diagnostic roster, and on ndarray input it stored
``feature_names_in_`` as the INTEGER column labels pandas assigns. Two consequences: ``_feature_names_in_synthesized_`` was never set, so
``get_feature_names_out(input_features=...)`` fell back to the retired name-pattern heuristic, which does not recognise ``"0"`` as a
placeholder and so rejected the caller's names; and sklearn requires ``feature_names_in_`` to be strings.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _data(seed=0, n=300, p=6):
    """Two informative columns out of six."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = ((X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _stability_fit(X, y):
    """A small, deterministic complementary-pairs stability fit."""
    m = MRMR(
        verbose=0, random_seed=0, fe_max_steps=0, fe_hinge_enable=False, dcd_enable=False, build_friend_graph=False,
        stability_selection_method="complementary_pairs", stability_n_bootstrap=4,
    )
    return m.fit(X, y)


@pytest.fixture(scope="module")
def ndarray_stability_fit():
    """One stability fit on an ndarray, shared by the assertions below."""
    X, y = _data()
    m = _stability_fit(X, y)
    # A stability fit whose every replicate fails falls back to a classic fit with only a warning, and a classic fit would pass the name
    # assertions below on its own. Integer column labels made every replicate fail, so without this the tests observed nothing.
    assert hasattr(m, "stability_freq_"), "the stability outer loop did not complete; the fit fell back to classic"
    return m, X


def test_ndarray_stability_fit_stores_string_feature_names(ndarray_stability_fit):
    """sklearn's contract: feature_names_in_ is all strings, and the placeholders match the classic path's."""
    m, X = ndarray_stability_fit
    names = list(m.feature_names_in_)
    assert names and all(isinstance(n, str) for n in names), f"feature_names_in_ is empty or holds non-strings: {names}"
    assert names == [f"feature_{i}" for i in range(X.shape[1])]
    assert m._feature_names_in_synthesized_ is True


def test_ndarray_stability_fit_accepts_caller_input_features(ndarray_stability_fit):
    """After an ndarray fit the caller's names take precedence, as they do after a classic fit."""
    m, X = ndarray_stability_fit
    names = [f"col_{i}" for i in range(X.shape[1])]
    out = m.get_feature_names_out(input_features=names)
    assert list(out) == [names[int(i)] for i in m.support_]


def test_stability_fit_has_the_identity_shortcut_diagnostic_roster(ndarray_stability_fit):
    """Every diagnostic attribute a consumer may introspect on the other terminal paths must exist here too."""
    m, _ = ndarray_stability_fit
    roster = (
        "_engineered_features_", "_engineered_recipes_", "fallback_used_", "dcd_", "cluster_members_", "cluster_hierarchy_",
        "mrmr_gains_", "friend_graph_", "cluster_aggregate_", "ran_out_of_time_", "provenance_", "_feature_names_in_synthesized_",
    )
    missing = [a for a in roster if not hasattr(m, a)]
    assert not missing, f"stability fit is missing fitted attributes: {missing}"
    assert list(m._engineered_recipes_) == [] and m.stability_freq_ is not None
