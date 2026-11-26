"""Cluster 13 regression: a missing base honest loss raises RuntimeError in the core refine, not an assert (audit 2026-11-20, finding 13-F24).

``python -O`` strips asserts, so a None base loss used to reach the arithmetic that follows. The core refine now raises an explicit RuntimeError
before it runs.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxied_fit_steps as steps
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_revalidate as revalidate


def _owner():
    """A minimal fit state for the within-cluster refine step, in core mode."""
    return SimpleNamespace(
        within_cluster_refine=True,
        refine_mode="core",
        classification=False,
        metric="rmse",
        parsimony_tol=0.0,
        n_jobs=1,
        refine_n_estimators=10,
        refine_ucb_enabled=False,
        refine_ucb_min_eval_size=10,
        refine_ucb_slack=0.0,
        refine_ucb_stdev_multiplier=1.0,
        inner_n_jobs_cap=1,
        cache_dir=None,
    )


def _step(owner):
    """Run the step-8 refine on a two-unit, four-column fit with the honest-cache object passed through."""
    unit_to_members = {0: [0, 1], 1: [2, 3]}
    return steps.ShapProxiedFitStepsMixin._fit_step8_expand_best_proxy(
        owner,
        unit_to_members,
        [0, 1, 2, 3],
        lambda: False,
        lambda _name: contextlib.nullcontext(),
        np.array([0, 1]),
        set(),
        None,
        np.zeros((20, 4)),
        np.zeros(20),
        np.zeros((10, 4)),
        np.zeros(10),
        {},
        np.zeros((20, 4)),
        0.0,
        np.zeros(20),
        {},
    )


def test_missing_base_honest_loss_raises_runtime_error_in_core_mode(monkeypatch):
    """With the base honest loss unavailable, the core refine raises RuntimeError naming it, before any arithmetic (13-F24)."""
    monkeypatch.setattr(revalidate, "_open_disk_cache", lambda _dir: None)
    monkeypatch.setattr(revalidate, "_honest_loss", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="base honest loss missing for core refine"):
        _step(_owner())


def test_available_base_honest_loss_passes_the_guard(monkeypatch):
    """Control: with a base loss in hand the guard passes, and the refine continues past it into the collapse step (13-F24)."""

    class _Reached(Exception):
        """Raised by the stand-in core search to show the guard was passed."""

    def _collapse(*_args, **_kwargs):
        """Stand in for the cluster collapse that follows the guard and record that it was reached."""
        raise _Reached()

    monkeypatch.setattr(revalidate, "_open_disk_cache", lambda _dir: None)
    monkeypatch.setattr(revalidate, "_honest_loss", lambda *_args, **_kwargs: 0.3)
    monkeypatch.setattr(steps, "_collapse_intra_cluster_members", _collapse)
    with pytest.raises(_Reached):
        _step(_owner())
