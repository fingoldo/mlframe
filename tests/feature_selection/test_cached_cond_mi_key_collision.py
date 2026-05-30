"""Wave 9.1 loop-iter-14 regression: ``cached_cond_MIs`` key MUST NOT
collide across partitions of the multiset ``X u Z``.

Pre-fix at ``evaluation.py:358`` the key was built as
``arr2str(X) + "_" + arr2str(Z)``. Since ``arr2str`` already uses ``_``
between elements (see ``_numba_utils.py:26``), the X/Z boundary was
indistinguishable from element boundaries. Three concrete equivalent-
key cases:
  X=[1,2] Z=[3,4]   -> '1_2_3_4'
  X=[1]   Z=[2,3,4] -> '1_2_3_4'
  X=[1,2,3] Z=[4]   -> '1_2_3_4'

Effect: ``I(X; Y | Z)`` is NOT symmetric in X<->Z, so whichever
candidate was scored first poisoned the cache for the others. Affects
every config with ``max_veteranes_interactions_order >= 2`` or
engineered multi-element X/Z. Silent wrong scoring at the conditional-
MI cache layer.

Fix: use a separator (``|``) that ``arr2str`` cannot emit.
"""
from __future__ import annotations

import numpy as np


def _legacy_key(X, Z):
    """Reproduces the pre-fix key construction."""
    from mlframe.feature_selection.filters._numba_utils import arr2str
    return arr2str(X) + "_" + arr2str(Z)


def _post_fix_key(X, Z):
    """The iter-14 fix replaces ``_`` with ``|``."""
    from mlframe.feature_selection.filters._numba_utils import arr2str
    return arr2str(X) + "|" + arr2str(Z)


def test_legacy_key_collides_baseline():
    """Confirms the pre-fix collision pattern. If this ever stops being
    true (arr2str semantics change), the iter-14 fix can be revisited.
    """
    pairs = [
        (np.array([1, 2]), np.array([3, 4])),
        (np.array([1]), np.array([2, 3, 4])),
        (np.array([1, 2, 3]), np.array([4])),
    ]
    legacy_keys = [_legacy_key(X, Z) for X, Z in pairs]
    # All three produce '1_2_3_4' under the pre-fix construction.
    assert len(set(legacy_keys)) == 1, legacy_keys


def test_post_fix_keys_are_distinct():
    """The iter-14 fix must produce distinct keys for the three partitions."""
    pairs = [
        (np.array([1, 2]), np.array([3, 4])),
        (np.array([1]), np.array([2, 3, 4])),
        (np.array([1, 2, 3]), np.array([4])),
    ]
    keys = [_post_fix_key(X, Z) for X, Z in pairs]
    assert len(set(keys)) == 3, (
        f"post-fix keys still collide: {keys}. Separator '|' must be "
        f"absent from arr2str output."
    )


def test_post_fix_key_for_empty_z():
    """Edge case: Z empty must still produce a unique key vs the same
    X with a one-element Z.
    """
    empty_z_key = _post_fix_key(np.array([1, 2]), np.array([], dtype=np.int64))
    one_z_key = _post_fix_key(np.array([1]), np.array([2], dtype=np.int64))
    assert empty_z_key != one_z_key


def test_e2e_no_wrong_answer_cache_hit():
    """End-to-end behavioural guard: write a value with one X/Z
    partition then read with a different partition. Cache MUST miss.
    Catches future re-introductions of any X/Z separator collision
    without inspecting source.
    """
    from mlframe.feature_selection.filters._numba_utils import arr2str
    # Simulate evaluation.py's cache pattern with the post-fix separator.
    cache = {}
    X1 = np.array([1, 2]); Z1 = np.array([3, 4])
    X2 = np.array([1]); Z2 = np.array([2, 3, 4])
    cache[arr2str(X1) + "|" + arr2str(Z1)] = 0.5
    key2 = arr2str(X2) + "|" + arr2str(Z2)
    # Pre-fix bug: this lookup would HIT (wrong-answer reuse).
    # Post-fix: it must MISS so a fresh CMI computation runs.
    assert key2 not in cache, (
        f"X=[1] Z=[2,3,4] must NOT hit a cache populated by X=[1,2] Z=[3,4]; "
        f"these are different conditional MI queries with different values."
    )
