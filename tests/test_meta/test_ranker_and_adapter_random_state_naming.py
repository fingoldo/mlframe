"""X_ARCHITECTURE_API_CONSISTENCY-3 regression test: MLPRanker / ShortlistTransformerAdapter expose the
sklearn-standard ``random_state`` param, with ``seed`` kept only as a deprecated back-compat alias.

Pre-fix, both BaseEstimator subclasses named their RNG param ``seed`` instead of the ``random_state``
convention ~40 sibling BaseEstimator classes in this codebase already use.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter
from mlframe.training.neural.ranker import MLPRanker


def test_mlp_ranker_exposes_random_state_param():
    """random_state is a real constructor param, honored, and sklearn's get_params exposes it."""
    model = MLPRanker(n_estimators=1, random_state=7)
    assert model.random_state == 7
    assert "random_state" in model.get_params()


def test_mlp_ranker_deprecated_seed_still_works_and_warns(caplog):
    """The deprecated seed= alias still sets random_state (back-compat) and logs a deprecation warning."""
    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.ranker"):
        model = MLPRanker(n_estimators=1, seed=13)
    assert model.random_state == 13
    assert any("deprecated" in r.message and "random_state" in r.message for r in caplog.records)


def test_mlp_ranker_random_state_default_matches_prior_seed_default():
    """Sanity: the new random_state default (42) matches the old seed default, so unseeded callers see no behavior change."""
    model = MLPRanker(n_estimators=1)
    assert model.random_state == 42


def test_shortlist_transformer_adapter_exposes_random_state_param():
    """random_state is a real constructor param, honored, and sklearn's get_params exposes it."""

    def _dummy_compute_fn(X, X_query, seed=0):
        """Placeholder compute_fn; never actually invoked in this test."""
        raise NotImplementedError

    adapter = ShortlistTransformerAdapter(_dummy_compute_fn, random_state=5, needs_y=False)
    assert adapter.random_state == 5
    assert "random_state" in adapter.get_params()


def test_shortlist_transformer_adapter_deprecated_seed_still_works_and_warns(caplog):
    """The deprecated seed= alias still sets random_state (back-compat) and logs a deprecation warning."""

    def _dummy_compute_fn(X, X_query, seed=0):
        """Placeholder compute_fn; never actually invoked in this test."""
        raise NotImplementedError

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.transformer._suite_adapter"):
        adapter = ShortlistTransformerAdapter(_dummy_compute_fn, seed=9, needs_y=False)
    assert adapter.random_state == 9
    assert any("deprecated" in r.message and "random_state" in r.message for r in caplog.records)


def test_shortlist_transformer_adapter_forwards_random_state_as_seed_kwarg_to_compute_fn():
    """The adapter still calls the wrapped compute_fn with its own seed= keyword (unchanged external contract)."""
    seen_kwargs = {}

    def _spying_compute_fn(X, X_query, seed=0):
        """Records the seed kwarg it was called with and returns a trivial polars frame."""
        import polars as pl

        seen_kwargs["seed"] = seed
        return pl.DataFrame({"f0": np.zeros(X_query.shape[0])})

    adapter = ShortlistTransformerAdapter(_spying_compute_fn, random_state=21, needs_y=False, passthrough=False)
    adapter.fit(np.zeros((5, 2)))
    adapter.transform(np.zeros((3, 2)))
    assert seen_kwargs["seed"] == 21
