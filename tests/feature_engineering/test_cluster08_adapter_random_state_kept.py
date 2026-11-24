"""Cluster 08 regression: the suite adapter keeps random_state as given (audit 2026-11-20, finding 08-F14).

``__init__`` used to overwrite random_state with the deprecated seed, so get_params() no longer returned the caller's value. random_state is now
stored as given; the effective RNG seed is resolved by _rng_seed(), with seed taking precedence when given.
"""

import numpy as np
import polars as pl

from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter


def _fn(X_train, y_train, X_query=None, *, seed=0, **kwargs):
    """A stand-in shortlist transformer."""
    return pl.DataFrame({"f": np.zeros(len(X_query) if X_query is not None else len(X_train))})


def test_random_state_is_kept_as_given_when_seed_is_passed():
    """random_state is not overwritten by the deprecated seed (08-F14)."""
    a = ShortlistTransformerAdapter(_fn, random_state=7, seed=3)
    assert a.random_state == 7
    assert a.seed == 3


def test_effective_seed_prefers_seed_then_random_state():
    """The RNG seed is seed when given, otherwise random_state (08-F14)."""
    assert ShortlistTransformerAdapter(_fn, random_state=7, seed=3)._rng_seed() == 3
    assert ShortlistTransformerAdapter(_fn, random_state=7)._rng_seed() == 7
