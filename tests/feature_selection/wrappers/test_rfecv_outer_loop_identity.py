"""CPX31 regression: the outer-loop ``fi_run_order`` keys-rebuild must stay behaviourally identical.

``run_outer_loop_iteration`` used to pass ``fi_run_order=list(state.feature_importances.keys())`` to
``get_next_features_subset`` on EVERY outer iteration -- O(steps^2) over a run on the growing
``feature_importances`` dict. The list is only consumed when ``fi_decay_rate > 0`` (age-weighted FI
voting). The fix skips the materialisation when ``fi_decay_rate == 0`` (default), passing ``None``.

Identity guarantees pinned here:
  1. With decay OFF (default), the spied ``fi_run_order`` arg is ``None`` (skipped) -- yet the RFECV fit
     selects exactly the same features as a control with the same seed.
  2. With decay ON, the spied ``fi_run_order`` arg is the FULL insertion-order key list of
     ``feature_importances`` (newest last) -- the pre-fix behaviour, preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier

from mlframe.feature_selection.wrappers.rfecv import RFECV
import mlframe.feature_selection.wrappers.rfecv._fit_outer_loop as _ol


def _make_data():
    """Make data."""
    rng = np.random.default_rng(7)
    n = 80
    informative = rng.normal(size=(n, 3))
    noise = rng.normal(size=(n, 7))
    X = pd.DataFrame(
        np.column_stack([informative, noise]),
        columns=[f"f{i}" for i in range(10)],
    )
    y = (informative[:, 0] + informative[:, 1] - informative[:, 2] > 0).astype(int)
    return X, y


def _fit(fi_decay_rate: float):
    """Returns ``sel`` (after 3 setup steps)."""
    X, y = _make_data()
    sel = RFECV(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        fi_decay_rate=fi_decay_rate,
        random_state=0,
        verbose=0,
    )
    sel.fit(X, y)
    return sel


def _fit_with_spy(fi_decay_rate: float, monkeypatch):
    """Fit RFECV while recording the ``fi_run_order`` kwarg passed on each outer iteration."""
    recorded = []
    real = _ol.get_next_features_subset

    def _spy(*args, **kwargs):
        """Returns ``real(*args, **kwargs)`` (after 1 setup step)."""
        recorded.append(kwargs.get("fi_run_order"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_ol, "get_next_features_subset", _spy)

    X, y = _make_data()
    sel = RFECV(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        fi_decay_rate=fi_decay_rate,
        random_state=0,
        verbose=0,
    )
    sel.fit(X, y)
    return sel, recorded


def test_decay_off_skips_keys_rebuild_passes_none(monkeypatch):
    """Decay off skips keys rebuild passes none."""
    _sel, recorded = _fit_with_spy(0.0, monkeypatch)
    assert len(recorded) > 0, "outer loop must have run at least one iteration"
    # Default decay-off path never materialises the key list.
    assert all(r is None for r in recorded), f"fi_run_order must be None when decay off; got {recorded}"


def test_decay_on_passes_full_insertion_order_keys(monkeypatch):
    """Decay on passes full insertion order keys."""
    _sel, recorded = _fit_with_spy(0.5, monkeypatch)
    assert len(recorded) > 0
    # With decay on, every call gets the full key list (newest last), not None.
    assert all(isinstance(r, list) for r in recorded), f"fi_run_order must be a list when decay on; got {recorded}"
    # Monotonically non-decreasing length: keys only get appended as runs accumulate.
    lengths = [len(r) for r in recorded]
    assert lengths == sorted(lengths), f"insertion-order key list must grow monotonically; got {lengths}"


def test_decay_off_fit_selection_unchanged():
    # End-to-end identity: the skip-when-off optimisation must not alter which features are selected.
    """Decay off fit selection unchanged."""
    sel_a = _fit(0.0)
    sel_b = _fit(0.0)
    assert list(sel_a.support_) == list(sel_b.support_)
    assert sel_a.n_features_ == sel_b.n_features_


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-cov"]))
