"""Cluster 15 regression: a failed SFFS swap pass rolls the selection state back (audit 2026-11-20, finding 15-F16).

The state is snapshotted before the pass and restored in place when it raises, so a half-applied swap never leaks into the selection. The fit
then continues with the state it started from.
"""

import logging
from types import SimpleNamespace

from mlframe.feature_selection.wrappers.rfecv import _finalize as fin


def _owner(swap_pass):
    """A minimal RFECV state with the swap pass enabled and the given swap implementation."""
    return SimpleNamespace(swap_top_k=2, swap_top_k_allow_no_es=True, _sffs_swap_pass=swap_pass)


def _run(owner, selected):
    """Run the swap stage on a five-feature best subset and return its ``(best_nfeatures, best_score)`` result."""
    return fin._run_sffs_swap_stage(
        owner,
        X=None,
        y=None,
        X_estimator=None,
        col_pos=None,
        estimator=None,
        cv=None,
        scoring=None,
        best_nfeatures=5,
        best_score=1.0,
        selected_features_per_nfeatures=selected,
        feature_importances={},
        original_features=["k0", "k1", "k2", "k3", "k4", "extra"],
        evaluated_scores_mean={5: 1.0},
        evaluated_scores_std={5: 0.0},
        verbose=False,
        ndigits=4,
    )


def test_failed_swap_pass_restores_the_selection_it_started_from(caplog):
    """A swap pass that corrupts the selection and then raises leaves the selection as it was, and logs the rollback (15-F16)."""

    def _corrupting_swap(**kwargs):
        """Overwrite the best subset, then fail, as a half-applied swap would."""
        kwargs["selected_features_per_nfeatures"][5][:] = ["corrupt"]
        raise RuntimeError("swap failed mid-pass")

    selected = {5: ["k0", "k1", "k2", "k3", "k4"]}
    with caplog.at_level(logging.WARNING):
        result = _run(_owner(_corrupting_swap), selected)

    assert result == (5, 1.0)
    assert selected[5] == ["k0", "k1", "k2", "k3", "k4"]
    assert any("state rolled back, continuing" in r.getMessage() for r in caplog.records)


def test_clean_swap_pass_is_not_rolled_back(caplog):
    """Control: a swap pass that finishes without error logs no rollback and leaves the selection it produced (15-F16)."""

    def _clean_swap(**kwargs):
        """Return without changing anything."""
        return None

    selected = {5: ["k0", "k1", "k2", "k3", "k4"]}
    with caplog.at_level(logging.WARNING):
        _run(_owner(_clean_swap), selected)

    assert selected[5] == ["k0", "k1", "k2", "k3", "k4"]
    assert not any("state rolled back" in r.getMessage() for r in caplog.records)
