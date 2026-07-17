"""Wave 9.1 loop-iter-16 regression: ``mi_direct(parallelism='outer')``
must fall back to ``classes_y`` when ``classes_y_safe`` is None.

Pre-fix asymmetry at permutation.py:408: bc and inner branches both
guarded ``classes_y_safe if classes_y_safe is not None else classes_y``,
but the outer branch passed ``classes_y_safe`` directly into the
``@njit parallel_mi`` worker. When the caller didn't pre-stage a safe
buffer (a perfectly valid usage per the docstring listing
``classes_y_safe`` as optional), numba choked with::

  TypingError: No implementation of function asarray(none) found

Severity: medium. All internal callers happen to pre-stage the buffer,
but the bug:
- makes mi_direct not safely public-callable with the documented signature
- creates a latent crash any future caller / test sweep will trip
- breaks the documented parallelism-mode symmetry contract.

Fix: mirror the bc / inner pattern - fall back to classes_y on None.
``parallel_mi`` already ``.copy()``s the array internally so workers
don't race on the shared buffer.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("parallelism", ["outer", "inner", "bc"])
def test_mi_direct_works_without_classes_y_safe(parallelism):
    """All three parallel modes must accept ``classes_y_safe=None``."""
    from mlframe.feature_selection.filters.permutation import mi_direct

    rng = np.random.default_rng(0)
    n = 500
    x = rng.integers(0, 4, n).astype(np.int32)
    y = rng.integers(0, 2, n).astype(np.int32)
    factors = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([4, 2], dtype=np.int32)
    # No classes_y_safe argument supplied -> defaults to None.
    mi_val, conf = mi_direct(
        factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=500,
        n_workers=2 if parallelism != "bc" else 1,
        parallelism=parallelism,
        prefer_gpu=False,
    )
    # Validity of the numeric result is not the focus; the test is
    # passing if mi_direct does NOT crash with a numba TypingError.
    assert isinstance(mi_val, float)
    assert 0.0 <= conf <= 1.0


def test_outer_path_specifically_no_crash():
    """Pin-point the iter-16 fix: explicit ``parallelism='outer'`` +
    ``n_workers > 1`` + large enough npermutations to enter the outer
    branch. Pre-fix this was a guaranteed crash.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct

    rng = np.random.default_rng(1)
    n = 500
    x = rng.integers(0, 4, n).astype(np.int32)
    y = rng.integers(0, 2, n).astype(np.int32)
    factors = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([4, 2], dtype=np.int32)
    # This MUST complete without raising.
    mi_val, conf = mi_direct(
        factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=1000,
        n_workers=4,
        parallelism="outer",
        prefer_gpu=False,
    )
    assert isinstance(mi_val, float)
