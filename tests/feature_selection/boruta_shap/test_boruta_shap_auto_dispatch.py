"""Unit tests for BorutaShap importance_measure='auto' dispatch.

'auto' runs one cheap noise/overfit probe on (X, y) at fit start and routes to permutation on
noisy / small-n-per-feature beds and gini on clean / large-n beds. These tests assert: (a) 'auto'
is a real constructor option stored verbatim, (b) the router picks permutation on a noisy small-n/p
bed and gini on a clean large-n bed, (c) a full auto fit pins the resolution + diagnostics and runs
the routed driver, (d) the default stays 'gini' (auto is opt-in), (e) auto forces the held-out split
only on the permutation branch and only when the user left train_or_test='train'.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def _clean(n=1500, p=10, inf=6, seed=0):
    X, y = make_classification(n_samples=n, n_features=p, n_informative=inf, n_redundant=0,
                               shuffle=False, random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), pd.Series(y)


def _noisy(n=250, p=50, inf=4, seed=0):
    X, y = make_classification(n_samples=n, n_features=p, n_informative=inf, n_redundant=0,
                               shuffle=False, random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), pd.Series(y)


def test_auto_is_a_real_constructor_option_stored_verbatim():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    b = BorutaShap(importance_measure="auto")
    assert b.importance_measure == "auto"  # sklearn contract: stored verbatim, no mutation in __init__.


def test_default_driver_is_still_gini_auto_is_opt_in():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    assert inspect.signature(BorutaShap.__init__).parameters["importance_measure"].default == "gini"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_router_picks_permutation_on_noisy_small_np(seed):
    from mlframe.feature_selection.boruta_shap._auto_dispatch import resolve_auto_importance_measure

    X, y = _noisy(seed=seed)
    measure, diag = resolve_auto_importance_measure(X, y, classification=True, random_state=seed)
    assert measure == "permutation", f"noisy small-n/p must route to permutation, got {measure} ({diag['reasons']})"
    assert diag["reasons"], "permutation route must carry a tripped-threshold reason"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_router_picks_gini_on_clean_large_n(seed):
    from mlframe.feature_selection.boruta_shap._auto_dispatch import resolve_auto_importance_measure

    X, y = _clean(seed=seed)
    measure, diag = resolve_auto_importance_measure(X, y, classification=True, random_state=seed)
    assert measure == "gini", f"clean large-n must route to gini (no perm cost), got {measure} ({diag['reasons']})"
    assert not diag["reasons"], f"gini route must trip no thresholds, got {diag['reasons']}"


def test_auto_fit_pins_resolution_and_diagnostics_noisy():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _noisy()
    b = BorutaShap(model=RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0),
                   importance_measure="auto", permutation_n_repeats=2, classification=True,
                   n_trials=6, percentile=95, verbose=False, random_state=0)
    b.fit(X, y)
    assert b._resolved_importance_measure_ == "permutation"
    assert b.auto_dispatch_diagnostics_["resolved_measure"] == "permutation"
    # permutation branch carved the held-out split it reads.
    assert getattr(b, "X_boruta_test", None) is not None


def test_auto_fit_pins_gini_on_clean_and_keeps_train_split():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _clean()
    b = BorutaShap(model=RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0),
                   importance_measure="auto", classification=True, n_trials=6, percentile=95,
                   verbose=False, random_state=0)
    b.fit(X, y)
    assert b._resolved_importance_measure_ == "gini"
    # gini route must NOT force the held-out split (no permutation cost path).
    assert b.train_or_test == "train"


def test_auto_only_forces_test_split_when_user_left_default():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _noisy()
    # User explicitly chose train_or_test='test' already -> auto-permutation respects it (no change needed).
    b = BorutaShap(model=RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=0),
                   importance_measure="auto", permutation_n_repeats=2, classification=True,
                   n_trials=4, percentile=95, train_or_test="test", verbose=False, random_state=0)
    b.fit(X, y)
    assert b._resolved_importance_measure_ == "permutation" and b.train_or_test == "test"


def test_explicit_measures_are_not_rerouted_by_auto_path():
    """_resolve_auto_importance_measure is a no-op for explicit measures: resolution == the explicit value."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _noisy()
    b = BorutaShap(importance_measure="gini", classification=True, random_state=0)
    b._resolve_auto_importance_measure(X, y)
    assert b._resolved_importance_measure_ == "gini" and b.auto_dispatch_diagnostics_ is None
