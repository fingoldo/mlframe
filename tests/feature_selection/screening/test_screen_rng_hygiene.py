"""Regression: ``screen_predictors`` must restore global ``np.random`` state on
EVERY exit path, including exceptions mid-screen.

Pre-fix code captured a snapshot on entry and restored it just before the
single happy-path ``return`` statement (around line 905). If anything in the
600-line body raised after ``np.random.seed(random_seed)`` (line ~293) the
restore line never executed and the caller's RNG state was left silently
seeded with ``random_seed`` -- defeating the entire snapshot/restore purpose.

The fix wraps the body in a ``with _preserve_global_numpy_rng_state(...)``
block whose ``finally`` clause restores state on both return AND raise.

Companion happy-path test:
``tests/feature_selection/test_mrmr_fixes_p0_p1.py::test_fix3_screen_does_not_mutate_global_numpy_rng``
must continue to pass (see that file for the no-raise case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters import screen as screen_mod
# 2026-05-22: ``screen_predictors`` body lives in ``_screen_predictors.py``;
# the live call to ``merge_vars`` resolves from THAT module's globals.
from mlframe.feature_selection.filters import _screen_predictors as screen_predictors_mod


def _toy_dataset(n_rows: int = 200, n_cols: int = 4, seed: int = 0):
    """Build a small synthetic classification fixture with signal on columns 0 and 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def test_screen_restores_np_random_state_on_exception(monkeypatch):
    """Forced raise inside ``screen_predictors`` after the seed line must still
    restore the caller's pre-call ``np.random`` state. Pre-fix: state leaked.
    """
    X, y = _toy_dataset()

    # Seed the GLOBAL state to a known sentinel BEFORE calling MRMR so we can
    # detect bleed-through. Pick a seed that differs from MRMR.random_seed
    # below so a state mismatch is unambiguous.
    np.random.seed(99)
    before_state = np.random.get_state()

    # ``merge_vars`` is the first heavy call inside ``screen_predictors``
    # AFTER ``np.random.seed(random_seed)`` is invoked (around line 330,
    # post the seeding block at lines ~290-298). Patching it to raise
    # reliably exercises the exception path through the seeded region.
    def _boom(*args, **kwargs):
        """Stand in for merge_vars and always raise."""
        raise RuntimeError("forced failure inside screen_predictors after seeding")

    monkeypatch.setattr(screen_predictors_mod, "merge_vars", _boom)

    m = MRMR(
        random_seed=42,
        verbose=0,
        n_jobs=1,
        full_npermutations=2,
        baseline_npermutations=2,
        skip_retraining_on_same_content=False,
        fe_max_steps=0,
    )
    with pytest.raises(RuntimeError, match="forced failure"):
        m.fit(X.copy(), y)

    after_state = np.random.get_state()

    # MT19937 tuple: (name, uint32 key array, pos, has_gauss, cached_gaussian)
    assert before_state[0] == after_state[0]
    assert np.array_equal(before_state[1], after_state[1]), (
        "screen_predictors left np.random global state seeded after mid-screen "
        "exception; snapshot/restore must execute on the raise path "
        "(wrap function body in `with _preserve_global_numpy_rng_state(...)`)."
    )
    assert before_state[2] == after_state[2]
    assert before_state[3] == after_state[3]
    assert before_state[4] == after_state[4]
