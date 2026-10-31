"""A refused shared-buffer check drops the registered stage caches and asks again: a cache that holds the memory must not cost a later stage its batched path."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _cache_release as CR
from mlframe.feature_selection.filters import _family_maxt_gate as gate
from mlframe.feature_selection.filters import _feature_engineering_mem_budget as MB


def test_a_refusal_releases_the_caches_and_the_second_answer_counts(monkeypatch):
    """The first decision says no, the releaser reports bytes freed, the second decision says yes and is the result; with nothing to release the first answer stands."""
    answers = iter([(False, 100, 1), (True, 100, 5)])
    monkeypatch.setattr(MB, "_decide_hoist_shared_buffer", lambda *a, **k: next(answers))
    freed = []
    monkeypatch.setattr(CR, "_RELEASERS", [lambda: (freed.append(1), 4096)[1]])
    assert MB._can_hoist_shared_buffer(100) == (True, 100, 5) and freed == [1]
    monkeypatch.setattr(MB, "_decide_hoist_shared_buffer", lambda *a, **k: (False, 100, 1))
    monkeypatch.setattr(CR, "_RELEASERS", [lambda: 0])
    assert MB._can_hoist_shared_buffer(100) == (False, 100, 1)


def test_an_accepted_check_leaves_the_caches_alone(monkeypatch):
    """No release happens when the buffer fits."""
    monkeypatch.setattr(MB, "_decide_hoist_shared_buffer", lambda *a, **k: (True, 1, 2))
    called = []
    monkeypatch.setattr(CR, "_RELEASERS", [lambda: (called.append(1), 1)[1]])
    assert MB._can_hoist_shared_buffer(1)[0] and not called


def test_the_permutation_block_cache_is_registered_and_releases_its_bytes():
    """The family gate's block cache is in the registry; releasing it empties the cache and reports what it held."""
    assert gate._release_perm_block_cache in CR._RELEASERS
    rng = np.random.default_rng(0)
    gate._permutation_block(rng, np.arange(500) % 2, 16)
    assert gate._perm_block_cache and gate._perm_block_cache_bytes > 0
    held = gate._perm_block_cache_bytes
    assert CR.release_registered_caches() >= held
    assert not gate._perm_block_cache and gate._perm_block_cache_bytes == 0
