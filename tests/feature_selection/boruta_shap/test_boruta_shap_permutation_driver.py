"""Unit tests for the BorutaShap permutation importance driver (importance_measure='permutation').

Permutation is the third driver alongside 'Shap' and 'gini'. In its honest mode (train_or_test='test') it uses
the 30% holdout that BorutaShap already carves so the shadow comparison is on held-out degradation rather than
in-sample fit. These tests assert the driver (a) is a real constructor option exposed verbatim, (b) recovers
informative features and rejects pure noise on a clean synthetic, and (c) the unknown-measure error now lists it.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def _make(n=1200, seed=0):
    """Returns ``(pd.DataFrame(X, columns=cols), pd.Series(y), [f'inf_{i}' for i in range(4)], [f'noise_...`` (after 2 setup steps)."""
    X, y = make_classification(n_samples=n, n_features=8, n_informative=4, n_redundant=0, n_repeated=0, shuffle=False, random_state=seed)
    cols = [f"inf_{i}" for i in range(4)] + [f"noise_{i}" for i in range(4)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), [f"inf_{i}" for i in range(4)], [f"noise_{i}" for i in range(4)]


def test_permutation_is_a_real_constructor_option():
    """Permutation is a real constructor option."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    p = inspect.signature(BorutaShap.__init__).parameters
    assert "permutation_n_repeats" in p, "permutation_n_repeats must be a real constructor param"
    b = BorutaShap(importance_measure="permutation", permutation_n_repeats=4)
    # sklearn contract: params stored verbatim, no mutation.
    assert b.importance_measure == "permutation" and b.permutation_n_repeats == 4


def test_default_driver_is_gini_not_dominated_shap():
    """Measured default: SHAP is dominated on the fs_hybrid bed (worst downstream AUC AND ~137x slower than gini),
    so the default importance_measure is 'gini', not 'Shap'. Guards against a silent revert to the dominated driver."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    assert inspect.signature(BorutaShap.__init__).parameters["importance_measure"].default == "gini"
    assert BorutaShap().importance_measure == "gini"


def test_permutation_held_out_recovers_signal_and_is_clean():
    """The recommended honest mode (train_or_test='test'): recovers the informative features AND keeps the
    accepted-noise count near zero, the precision property that distinguishes held-out permutation (it drove
    accepted-noise to ~0 across the fs_hybrid bed)."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y, informative, noise = _make()
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0),
        importance_measure="permutation",
        permutation_n_repeats=4,
        classification=True,
        n_trials=25,
        percentile=95,
        train_or_test="test",
        verbose=False,
        random_state=0,
    )
    b.fit(X, y)
    selected = set(c for c in b.selected_features_ if c in X.columns)
    assert len(selected & set(informative)) >= 3, f"missed informative features: {selected}"
    assert len(selected & set(noise)) <= 1, f"held-out permutation should be clean, accepted: {selected & set(noise)}"


def test_permutation_in_bag_recovers_signal_but_leaks_more():
    """In-bag permutation (the train_or_test='train' fallback) still recovers the signal but inherits in-sample
    optimism, so it leaks more noise than held-out. This documents WHY held-out is the recommended mode rather
    than asserting an equal precision the in-bag path does not have."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y, informative, noise = _make()
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0),
        importance_measure="permutation",
        permutation_n_repeats=4,
        classification=True,
        n_trials=25,
        percentile=95,
        train_or_test="train",
        verbose=False,
        random_state=0,
    )
    b.fit(X, y)
    selected = set(c for c in b.selected_features_ if c in X.columns)
    assert len(selected & set(informative)) >= 3, f"missed informative features: {selected}"
    # rejects at least one noise column (the gate does real work) but is not held to held-out precision.
    assert len(selected & set(noise)) < len(noise), f"in-bag accepted ALL noise: {selected & set(noise)}"


def test_held_out_mode_populates_the_split_used_by_permutation():
    """train_or_test='test' must carve the 30% holdout the permutation branch reads (X_boruta_test/y_test)."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y, _, _ = _make(n=800)
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0),
        importance_measure="permutation",
        permutation_n_repeats=2,
        classification=True,
        n_trials=8,
        percentile=95,
        train_or_test="test",
        verbose=False,
        random_state=0,
    )
    b.fit(X, y)
    assert getattr(b, "X_boruta_test", None) is not None and getattr(b, "y_test", None) is not None
    # holdout is ~30% of the rows it was fit on.
    assert 0.2 * len(X) <= len(b.X_boruta_test) <= 0.4 * len(X)


def test_unknown_importance_measure_lists_permutation():
    """Unknown importance measure lists permutation."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y, _, _ = _make(n=400)
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=20, random_state=0),
        importance_measure="bogus",
        classification=True,
        n_trials=4,
        verbose=False,
        random_state=0,
    )
    with pytest.raises(ValueError, match="permutation"):
        b.fit(X, y)
