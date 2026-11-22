"""Audit 2026-11-20 finding 35-F20: the GBM stacker honours group_ids and time_ordered.

The row-random early-stopping carve put same-group rows in both fit and stopping sets, so early stopping was optimistic. With groups or a time
order requested the GBM stacker now runs without row-random early stopping, and the dispatcher forwards both arguments.
"""

import numpy as np

from mlframe.training.composite.ensemble import _stackers


def _oof(n=120, k=3, seed=0):
    """A K-column OOF matrix with a known linear signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = X @ np.array([1.0, 0.5, -0.3]) + 0.1 * rng.normal(size=n)
    return X, y


def test_group_aware_gbm_disables_row_random_early_stopping():
    """With group_ids the fitted stacker has early stopping off."""
    X, y = _oof()
    groups = np.repeat(np.arange(12), 10)
    model = _stackers.fit_gbm_meta_stacker(X, y, 3, group_ids=groups)
    assert model.early_stopping is False


def test_default_gbm_keeps_early_stopping():
    """Without groups or time order the stacker keeps its row-random early stopping."""
    X, y = _oof()
    model = _stackers.fit_gbm_meta_stacker(X, y, 3)
    assert model.early_stopping is True


def test_dispatcher_forwards_group_ids_to_the_gbm_stacker(monkeypatch):
    """build_meta_stack_ensemble passes group_ids through to fit_gbm_meta_stacker."""
    seen = {}
    real = _stackers.fit_gbm_meta_stacker

    def spy(*args, **kwargs):
        """Record the group_ids the dispatcher forwards, then delegate."""
        seen["group_ids"] = kwargs.get("group_ids")
        return real(*args, **kwargs)

    monkeypatch.setattr(_stackers, "fit_gbm_meta_stacker", spy)
    X, y = _oof()
    groups = np.repeat(np.arange(12), 10)
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    _stackers.build_meta_stack_ensemble(
        CompositeCrossTargetEnsemble, [object()] * 3, ["a", "b", "c"], X, y, stacker="gbm", group_ids=groups,
    )
    assert seen.get("group_ids") is not None
