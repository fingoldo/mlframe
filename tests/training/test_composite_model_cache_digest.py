"""A cached composite inner model is reused only for the spec it was trained on.

The model cache keyed the dump by path alone, so a rerun that rediscovered ``y-linres-lag1`` with different fitted params
(a different T) loaded the inner trained on the old T and wrapped it with the new alpha/beta. The dump now records the spec
digest, and a mismatch (or a dump with no digest) invalidates the cache.
"""

from __future__ import annotations

import types

from mlframe.training.core._composite_wrap_helpers import composite_spec_digest
from mlframe.training.train_eval import _composite_cache_mismatch


def _spec(alpha: float) -> dict:
    """A linear-residual spec differing only in its fitted slope."""
    return {"name": "y-linres-lag1", "transform_name": "linear_residual", "base_column": "lag1", "fitted_params": {"alpha": alpha, "beta": 0.5}}


def test_the_digest_follows_the_fitted_params():
    """Same name and transform, different fitted params: a different digest; identical specs: the same digest."""
    assert composite_spec_digest(_spec(0.9)) == composite_spec_digest(_spec(0.9))
    assert composite_spec_digest(_spec(0.9)) != composite_spec_digest(_spec(1.1))


def test_a_dump_trained_on_another_spec_is_invalidated():
    """A matching digest reuses the dump; a changed one, or a dump without a digest, is stale; raw targets are unaffected."""
    want = composite_spec_digest(_spec(1.1))
    same = types.SimpleNamespace(model=types.SimpleNamespace(spec_digest_=want))
    old = types.SimpleNamespace(model=types.SimpleNamespace(spec_digest_=composite_spec_digest(_spec(0.9))))
    bare = types.SimpleNamespace(model=object())
    assert _composite_cache_mismatch(same, want) is None
    assert "spec changed" in _composite_cache_mismatch(old, want)
    assert "no spec digest" in _composite_cache_mismatch(bare, want)
    assert _composite_cache_mismatch(bare, None) is None
