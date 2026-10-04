"""Wave 9.1 loop-iter-25 regression: ``screen_predictors`` preamble
must handle the documented defaults correctly.

Pre-fix three bugs in ``_screen_predictors.py:170-256``:

1. ``factors_names=None`` (the documented signature default) crashed
   at ``len(None)`` -> TypeError before the auto-name branch.

2. Auto-name fallback at line 177 used
   ``range(len(factors_data))`` (= n_rows) instead of
   ``range(factors_data.shape[1])`` (= n_cols), producing a name
   list of length n_rows that immediately triggered the
   length-mismatch raise three lines down. Net effect: the documented
   "empty/None auto-generate" code path was dead.

3. The CuPy seeding block at line 254
   (``cp.random.seed(random_seed)``) ran BEFORE the ``import cupy as
   cp`` at line 332 (only inside ``if use_gpu:``). The bare
   ``except NameError`` swallowed the failure silently. Documented
   "seed it (numpy + numba + cupy) for the screening duration" was
   actually only seeding numpy + numba; CuPy RNG remained
   non-deterministic across runs with the same ``random_seed``.

Same issue at the finally-block restore (line 688).

Fix:
- (1+2) Fold into ``if factors_names is None or len(factors_names) == 0``
  AND use ``factors_data.shape[1]`` for the range.
- (3) Move the ``import cupy as cp`` next to the seed call so it's
  defined when the call fires (entry block); mirror in the
  finally-block restore.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_factors_names_none_does_not_crash():
    """``screen_predictors`` with ``factors_names=None`` (the documented
    signature default) MUST not crash on ``len(None)``. The function
    is internal but the MRMR.fit happy-path exercises it - just verify
    the public surface still works.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] > 0).astype(np.int64), name="y")
    # Happy path - if iter-25 broke the preamble we'd see a TypeError
    # / ValueError here.
    sel = MRMR(verbose=0).fit(X, y)
    assert sel.support_ is not None


def test_screen_predictors_factors_names_empty_list_no_mismatch():
    """Direct call into ``screen_predictors`` with ``factors_names=[]``
    must trigger the auto-name fallback and produce names of length
    ``factors_data.shape[1]`` (not ``len(factors_data)`` = n_rows).
    Pre-fix this raised "shape[1]=5 must equal len(factors_names)=100".
    """
    from mlframe.feature_selection.filters._screen_predictors import (
        screen_predictors,
    )

    # Build a tiny synthetic input that the screen will reject quickly.
    rng = np.random.default_rng(0)
    n_rows, n_cols = 100, 5
    factors_data = rng.integers(0, 4, (n_rows, n_cols)).astype(np.int32)
    factors_nbins = np.array([4] * n_cols, dtype=np.int64)
    # Targets must NOT share memory with factors_data.
    targets_data = rng.integers(0, 2, (n_rows, 1)).astype(np.int32)
    targets_nbins = np.array([2], dtype=np.int64)
    # The auto-name branch must succeed and the screen must run to completion on the generated names.
    result = screen_predictors(
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=[],  # triggers auto-name branch
        targets_data=targets_data,
        targets_nbins=targets_nbins,
        y=[0],
        full_npermutations=10,
        verbose=0,
    )
    predictors_log = result[1]
    assert len(predictors_log) >= 1
    assert {entry["name"] for entry in predictors_log} <= {f"F{i}" for i in range(n_cols)}
    assert len(result[4]) == n_cols, f"expected one univariate MI per column ({n_cols}); got {len(result[4])}"
