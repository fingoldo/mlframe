"""Wave 9.1 loop-iter-26 regression: BUR additive bonus must publish
into ``expected_gains[cand_idx]`` unconditionally.

Pre-fix at ``evaluation.py:610-611``:
  if expected_gains[cand_idx]:
      expected_gains[cand_idx] = current_gain

This gate was wrong in two ways:
1. On the dict-path (``expected_gains`` is a dict) the stopped_early
   branch at line 575-577 leaves the key ABSENT. The ``if
   expected_gains[cand_idx]:`` lookup then raised KeyError - silently
   swallowed by the surrounding ``except Exception: pass``.
2. On the ndarray-path (``expected_gains`` is preallocated to zeros)
   the stopped_early entry stayed 0 (falsy), so the bonus publish
   was skipped.

In both branches, ``partial_gains[cand_idx]`` and the local
``current_gain`` carried the BUR bonus correctly but the dense
ranking vector did not. The confirmation loop's ``lexsort`` then
ranked the BUR-bonus winner BELOW peers because ``expected_gains``
reflected pre-bonus scores. BUR became a partial no-op.

Severity: P1. ``bur_lambda > 0`` is an explicit opt-in; users who
turned it on assumed the bonus would affect ranking. It did not on the
common stopped_early path.

Fix: publish unconditionally; raise on actual invariant breaks (KeyError
/ IndexError from a missing preallocated slot); keep numerical-failure
best-effort behaviour via ``logger.warning``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_mrmr_with_bur_runs_and_publishes():
    """End-to-end smoke: MRMR with bur_lambda > 0 must complete without
    silent BUR errors and produce a non-empty support.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
    y = pd.Series((X["a"] > 0).astype(np.int64), name="y")
    sel = MRMR(verbose=0, bur_lambda=0.5).fit(X, y)
    assert len(sel.support_) >= 1


def test_bur_disabled_still_works_unchanged():
    """Negative control: bur_lambda=0 (default) behaviour unchanged."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
    y = pd.Series((X["a"] > 0).astype(np.int64), name="y")
    sel = MRMR(verbose=0).fit(X, y)
    assert len(sel.support_) >= 1


def test_bur_bonus_publishes_into_dense_expected_gains_and_dict_path(monkeypatch):
    """The BUR bonus must reach ``expected_gains`` on both the ndarray path (preallocated zeros) and the dict path
    (key absent). Bonus = lambda * max(0, I(X;Y) - max_j I(X;X_j)) = 0.5 * (1.0 - 0.25) = 0.375."""
    from mlframe.feature_selection.filters import evaluation as ev

    monkeypatch.setattr(ev, "get_bur_lambda", lambda: 0.5)
    monkeypatch.setattr(ev, "_evaluate_candidate_selected_vars", lambda selected_vars, *a, **k: 0.25)

    dense = np.zeros(3, dtype=np.float64)
    out = ev._apply_bur_bonus(2.0, 1.0, [0], None, None, None, None, 1, {}, dense)
    assert out == pytest.approx(2.375)
    assert dense[1] == pytest.approx(2.375)
    assert dense[0] == 0.0 and dense[2] == 0.0

    as_dict: dict = {}
    ev._apply_bur_bonus(2.0, 1.0, [0], None, None, None, None, 1, {}, as_dict)
    assert as_dict == {1: pytest.approx(2.375)}


def test_bur_with_collinear_features_fits_and_selects_valid_columns():
    """With clustered features and bur_lambda > 0 the fit completes and returns in-range column indices."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2)
    n = 500
    latent = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "lat0": latent,
            "lat1": latent + 0.05 * rng.standard_normal(n),
            "lat2": latent + 0.05 * rng.standard_normal(n),
            "ind": rng.standard_normal(n),
        }
    )
    y = pd.Series((latent + rng.standard_normal(n) * 0.3 > 0).astype(np.int64), name="y")
    sel_bur = MRMR(verbose=0, bur_lambda=1.0).fit(X, y)
    selected = [int(i) for i in np.asarray(sel_bur.support_).ravel()]
    assert len(selected) >= 1
    assert all(0 <= i < X.shape[1] for i in selected)
    assert any(X.columns[i].startswith("lat") for i in selected), f"no latent-cluster column selected: {selected}"
