"""The FE accuracy gate's baseline cache key must separate dtypes and must not copy the array to hash it.

``_baseline_cv_key`` guards a memo of the ``X_base``-only CV baseline that every engineered sibling of the
same raw source hits, so it is rebuilt once per candidate against the whole ``X_base``. It fed the hash via
``tobytes()``, materialising two full copies (``X_base`` and ``y``) each time.

Two correctness problems came with the rewrite and are pinned below. The key carried ``shape`` but not
``dtype``, so a float32 array and an int32 view of the same buffer -- same shape, same bytes -- shared one
memo entry and the second caller got the first one's baseline. And ``hash(bytes)`` truncates to 64 bits,
where a collision silently returns the wrong cached baseline rather than raising; blake2b removes that.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_accuracy_gate import _baseline_cv_key


def _key(x, y):
    """The key with every non-array argument held fixed."""
    return _baseline_cv_key(x, y, classification=True, n_splits=10, seed=0)


def test_identical_bytes_under_a_different_dtype_do_not_share_an_entry():
    """The float32 array and its int32 view have the same shape and the same bytes but different values."""
    x32 = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
    x_int = x32.view(np.int32)
    y = np.arange(64) % 2
    assert x32.tobytes() == x_int.tobytes(), "the fixture no longer exercises the collision it was built for"
    assert _key(x32, y) != _key(x_int, y), "a float32 array and an int32 view of it shared one baseline entry"


def test_the_key_still_separates_different_content_and_different_targets():
    """Guards the hash itself: dtype-awareness must not have coarsened the key onto metadata alone."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 4))
    y = np.arange(64) % 2
    other_x = x.copy()
    other_x[3, 1] += 1.0
    other_y = y.copy()
    other_y[0] ^= 1
    assert _key(x, y) != _key(other_x, y)
    assert _key(x, y) != _key(x, other_y)


def test_the_same_inputs_still_hit_the_same_key():
    """The whole point of the memo: identical content from a separate array must key identically."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 4))
    y = np.arange(64) % 2
    assert _key(x, y) == _key(x.copy(), y.copy())


def test_a_fortran_layout_keys_like_its_c_order_content():
    """`ascontiguousarray` normalises the layout, so a Fortran-ordered view is not a separate memo entry."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 4))
    y = np.arange(64) % 2
    assert _key(x, y) == _key(np.asfortranarray(x), y)


def test_a_contiguous_array_is_hashed_without_a_second_allocation():
    """The copy-free half of the fix: `ascontiguousarray` returns the caller's own object here."""
    x = np.random.default_rng(4).normal(size=(256, 4))
    assert x.flags["C_CONTIGUOUS"]
    assert np.ascontiguousarray(x) is x, "the hashed buffer is a copy, not the caller's own array"
