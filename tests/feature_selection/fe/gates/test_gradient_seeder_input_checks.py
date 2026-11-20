"""The gradient-interaction seeder refuses X/y of different lengths and reports an unparsable threshold override and an uncoercible X at WARNING."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gradient_interaction_seeder as gis


def test_rank_pairs_rejects_y_of_a_different_length():
    """A y longer than X used to be sampled with X's row indices and silently misaligned."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    with pytest.raises(ValueError, match="rows"):
        gis.rank_gradient_interaction_pairs(X, rng.standard_normal(120), [0, 1, 2, 3, 4])


def test_bad_threshold_env_value_warns_and_uses_the_default(monkeypatch, caplog):
    """A mistyped override is no longer swallowed without a trace."""
    monkeypatch.setenv("MLFRAME_GRAD_INTERACT_ROW_CAP", "lots")
    with caplog.at_level(logging.WARNING):
        assert gis._resolve_grad_threshold("row_cap", 2000) == 2000
    assert any(r.levelno == logging.WARNING and "ROW_CAP" in r.getMessage() for r in caplog.records)


def test_uncoercible_x_warns(caplog):
    """The proposer stays best-effort but the skip is visible at the default verbosity."""
    with caplog.at_level(logging.WARNING):
        out = gis._rank_gradient_pool(object(), np.zeros((4, 2)), [0, 1], 0, 100, "analytic")
    assert out is None
    assert any(r.levelno == logging.WARNING and "not array-coercible" in r.getMessage() for r in caplog.records)
