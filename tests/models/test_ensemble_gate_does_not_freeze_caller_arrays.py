"""The outlier gate made its caller's prediction arrays permanently read-only.

Its memo key was `id()`-based, which is only meaningful while the objects are alive and unmutated. To defend
that assumption it marked every member array `writeable = False` and held a reference to each -- so
`ensemble_probabilistic_predictions`, a public entry point whose docstring says nothing about it, silently
froze its inputs for the rest of the process and pinned up to 16 complete member sets in memory. The caller's
next in-place clip or calibration then raised `ValueError: assignment destination is read-only`, with a
traceback pointing nowhere near the call that caused it.

A content fingerprint needs neither: a mutated member simply keys differently and is recomputed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions
from mlframe.models.ensembling import predict as predict_mod


def _members(n: int = 500, seed: int = 0, m: int = 3):
    """`m` independent (N, 2) probability blocks."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(m):
        p = rng.random(n)
        out.append(np.column_stack([1.0 - p, p]))
    return out


@pytest.fixture(autouse=True)
def _clean_cache():
    """The gate memo is module-level; keep tests independent of each other's entries."""
    predict_mod._clear_gate_cache()
    yield
    predict_mod._clear_gate_cache()


class TestTheCallersArraysAreLeftAlone:
    """A function that mutates its inputs' flags has to say so; this one did not."""

    def test_the_inputs_stay_writable(self):
        """The direct statement of the defect."""
        members = _members()
        ensemble_probabilistic_predictions(*members)
        assert all(p.flags.writeable for p in members), "the call froze its caller's arrays"

    def test_the_caller_can_still_mutate_afterwards(self):
        """The failure as it actually reached a user: an in-place operation raising far from the cause."""
        members = _members()
        ensemble_probabilistic_predictions(*members)
        members[0][0, 0] = 0.5  # must not raise
        np.clip(members[1], 0.01, 0.99, out=members[1])

    def test_repeated_calls_do_not_accumulate_frozen_arrays(self):
        """The freeze was applied per insert, so a loop over splits froze every set it ever saw."""
        for seed in range(5):
            members = _members(seed=seed)
            ensemble_probabilistic_predictions(*members)
            assert all(p.flags.writeable for p in members)


class TestTheMemoStillMemoises:
    """Removing the freeze must not remove the caching it was protecting."""

    def test_the_same_inputs_hit_the_cache(self):
        """Same content, same key: the gate is computed once."""
        members = _members()
        ensemble_probabilistic_predictions(*members)
        entries = len(predict_mod._gate_cache)
        ensemble_probabilistic_predictions(*members)
        assert len(predict_mod._gate_cache) == entries, "an identical call added a second cache entry"

    def test_an_in_place_mutation_is_noticed(self):
        """The whole point of the fingerprint: a changed member must not reuse the old decision."""
        members = _members()
        ensemble_probabilistic_predictions(*members)
        entries = len(predict_mod._gate_cache)
        members[0][:] = np.column_stack([np.zeros(len(members[0])), np.ones(len(members[0]))])
        ensemble_probabilistic_predictions(*members)
        assert len(predict_mod._gate_cache) > entries, "a mutated member reused the cached gate decision"

    def test_different_members_key_differently(self):
        """A fingerprint that collided across distinct members would return another set's verdict."""
        ensemble_probabilistic_predictions(*_members(seed=0))
        ensemble_probabilistic_predictions(*_members(seed=1))
        assert len(predict_mod._gate_cache) == 2

    def test_the_cache_does_not_retain_the_arrays(self):
        """It used to hold a strong reference to every member set -- gigabytes at ensemble scale."""
        import gc
        import weakref

        members = _members()
        ref = weakref.ref(members[0])
        ensemble_probabilistic_predictions(*members)
        del members
        gc.collect()
        assert ref() is None, "the gate cache is still holding the caller's prediction arrays alive"
