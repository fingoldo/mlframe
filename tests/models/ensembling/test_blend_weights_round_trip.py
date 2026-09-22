"""Blend weights fitted at train time must reach the predict-time blend.

Training fitted NNLS/Caruana weights, scored a weighted blend and stamped that flavour as the winner, but persisted no
weights: deployment loaded the same models and averaged them 0.5/0.5. The deployed ensemble was a different estimator
from the one whose metrics were reported, with no warning.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.core.predict import _combine_probs, _resolve_chosen_ensemble_params


def _probs(seed: int, n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    return np.stack([1.0 - p, p], axis=1)


def test_persisted_weights_are_applied():
    members = [_probs(1), _probs(2)]
    weighted = _combine_probs(members, "arithm", precomputed_weights=[0.8, 0.2])
    np.testing.assert_allclose(weighted, np.average(np.stack(members), axis=0, weights=[0.8, 0.2]), rtol=1e-12)


def test_without_weights_the_blend_stays_the_unweighted_mean():
    members = [_probs(1), _probs(2)]
    np.testing.assert_allclose(_combine_probs(members, "arithm"), np.mean(np.stack(members), axis=0), rtol=1e-12)


@pytest.mark.parametrize("weights", [[0.8, 0.1, 0.1], [np.nan, 1.0], [0.0, 0.0]])
def test_weights_that_cannot_belong_to_this_member_set_are_refused_loudly(weights, caplog):
    """A length mismatch maps each weight onto the wrong model; NaN or all-zero weights cannot normalise."""
    members = [_probs(1), _probs(2)]
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        got = _combine_probs(members, "arithm", precomputed_weights=weights, target_label="bin/y")
    np.testing.assert_allclose(got, np.mean(np.stack(members), axis=0), rtol=1e-12)
    assert "blend weights" in caplog.text


def test_the_resolver_returns_the_weights_beside_rrf_k():
    metadata = {"ensembles_chosen_params": {"binary": {"y": {"rrf_k": 42, "blend_weights": [0.7, 0.3]}}}}
    params = _resolve_chosen_ensemble_params(metadata, "binary", "y")
    assert params["rrf_k"] == 42
    assert params["blend_weights"] == [0.7, 0.3]


def test_a_legacy_artefact_without_the_stamp_still_resolves():
    metadata = {"ensembles_chosen_params": {"binary": {"y": {"rrf_k": 60}}}}
    assert _resolve_chosen_ensemble_params(metadata, "binary", "y").get("blend_weights") is None
