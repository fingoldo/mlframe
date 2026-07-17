"""BorutaShap cross-subsample stability gate (opt-in).

A single-sample shadow comparison leaks the top finite-sample-spurious real-noise column past the gate
(see BorutaShap class docstring). Majority vote across distinct row-subsamples WITHOUT replacement drops
it while keeping genuinely-relevant features. These tests pin: the new params round-trip, the orchestration
runs n sub-fits and votes, and stability never accepts MORE noise than a single fit (and keeps the signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _signal_noise(n=900, p_signal=3, p_noise=14, seed=0):
    """Signal noise."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_signal))
    logit = z @ np.array([1.5, -1.2, 1.0])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {f"sig_{i}": z[:, i] for i in range(p_signal)}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y), [f"sig_{i}" for i in range(p_signal)], [f"noise_{j}" for j in range(p_noise)]


def test_stability_params_roundtrip_shallow_get_params():
    """Stability params roundtrip shallow get params."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    b = BorutaShap(stability_subsamples=4, stability_subsample_fraction=0.8, stability_threshold=0.7)
    p = b.get_params(deep=False)
    assert p["stability_subsamples"] == 4
    assert p["stability_subsample_fraction"] == 0.8
    assert p["stability_threshold"] == 0.7
    # Shallow round-trip must reconstruct (the orchestration builds sub-fits via get_params(deep=False)+__class__(**p)).
    assert BorutaShap(**p).stability_subsamples == 4
    # Default threshold is intersection (1.0): the only setting that reliably drops a draw-level-spurious column.
    assert BorutaShap(stability_subsamples=2).stability_threshold == 1.0


def test_borutashap_early_terminates_when_no_tentatives():
    """On a clean separable problem every feature is confirmed/rejected well before n_trials; the loop must stop
    early (n_trials_run_ < n_trials) with no tentatives. Pure speedup: the final partition is unchanged because
    remaining trials would only re-test already-confirmed features."""
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = make_classification(n_samples=1500, n_features=12, n_informative=6, n_redundant=0, n_repeated=0, class_sep=2.0, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    sel = BorutaShap(
        model=RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=150,
        percentile=100,
        verbose=False,
        random_state=0,
    )
    sel.fit(X, pd.Series(y))
    assert sel.n_trials_run_ < 150, f"expected early stop, ran all {sel.n_trials_run_} trials"
    assert len(sel.tentative) == 0  # early stop only triggers when nothing is tentative


def test_borutashap_is_sklearn_cloneable():
    """__init__ must store params verbatim so the estimator is clone-able (GridSearchCV / Pipeline). Regression for
    the prior bug where __init__ lowercased importance_measure, defaulted fit_params None->{}, and ran check_model
    (None->RandomForest), all of which made sklearn.clone raise 'constructor does not set or modifies parameter'."""
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    for kw in (
        dict(),
        dict(importance_measure="Shap"),
        dict(importance_measure="Gini"),
        dict(model=None, fit_params=None),
        dict(model=RandomForestClassifier(), stability_subsamples=4),
    ):
        c = clone(BorutaShap(**kw))  # must not raise
        assert isinstance(c, BorutaShap)
    # Params survive verbatim (no __init__ mutation).
    b = BorutaShap(importance_measure="Shap")
    assert b.get_params(deep=False)["importance_measure"] == "Shap"
    assert b.model is None and b.fit_params is None


def test_stability_runs_subfits_votes_and_keeps_signal():
    """Stability runs subfits votes and keeps signal."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y, signal, _noise = _signal_noise(seed=0)
    sel = BorutaShap(
        model=RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=25,
        percentile=100,
        verbose=False,
        random_state=0,
        stability_subsamples=8,
        stability_subsample_fraction=0.75,
        stability_threshold=1.0,
    )
    sel.fit(X, y)
    # The vote diagnostic is populated and bounded by the number of subsamples.
    assert hasattr(sel, "stability_accept_counts_")
    assert set(sel.stability_accept_counts_) == set(X.columns)
    assert max(sel.stability_accept_counts_.values()) <= 8
    # Strong signal features must survive the vote.
    for s in signal:
        assert s in sel.accepted, f"signal feature {s} not accepted by stability vote"
    # sklearn-style outputs stay consistent.
    assert len(sel.support_) == X.shape[1]
    # Intersection mode (threshold==1.0) keeps ONLY all-accept features; optimistic does NOT re-add the
    # sub-majority 'tentative' bucket (that is the draw-level-spurious bucket intersection exists to drop).
    assert set(sel.selected_features_) == set(sel.accepted)


def test_stability_accepts_no_more_noise_than_single_fit():
    """Stability is a tightening gate: it must never accept MORE noise columns than a single fit."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y, _signal, noise = _signal_noise(seed=0)
    noise_set = set(noise)
    mk = lambda **kw: BorutaShap(
        model=RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=25,
        percentile=100,
        verbose=False,
        random_state=0,
        **kw,
    )
    single = mk()
    single.fit(X, y)
    stable = mk(stability_subsamples=8, stability_subsample_fraction=0.75, stability_threshold=1.0)
    stable.fit(X, y)
    n_noise_single = len(set(single.accepted) & noise_set)
    n_noise_stable = len(set(stable.accepted) & noise_set)
    assert n_noise_stable <= n_noise_single, f"stability accepted {n_noise_stable} noise vs single-fit {n_noise_single}; gate must not loosen"
