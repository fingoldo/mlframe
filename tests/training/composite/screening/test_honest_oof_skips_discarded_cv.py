"""When honest group-OOF sets the ranking key, the group-internal CV fits it overwrites must not run at all.

With group ids and a group-disjoint holdout, ``honest_oof_reconstruction_rmse`` replaces the CV-RMSE of every spec it
can measure, so the sweep that produced those scores was pure waste: a 16-spec grouped run spent 11.6 s of model fits
to have all 16 results overwritten by a 0.26 s measurement. The sweep is now skipped for measured specs, except when
the per-bin regime gate or the Wilcoxon gate still needs its by-products.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.training.composite.discovery._screening_tiny as screening_tiny
from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig

from tests.training.composite.discovery.test_biz_val_discovery_honest_oof_select import _frame, _split_upper_tail

_FEATS = ["base_full", "base_partial", "x1"]


def _count_cv_calls(monkeypatch) -> dict:
    """Replace the multiseed CV entry point with a counting passthrough.

    Patched where the rerank's per-spec scorer (``_tiny_rerank_process.score_spec``) looks it up at call time.
    """
    seen = {"n": 0}
    original = screening_tiny._tiny_cv_rmse_y_scale_multiseed

    def counting(*args, **kwargs):
        """Count the call, then defer to the real CV."""
        seen["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(screening_tiny, "_tiny_cv_rmse_y_scale_multiseed", counting)
    return seen


def _fit(groups, train_idx, df, **cfg_kwargs) -> CompositeTargetDiscovery:
    """Run a grouped discovery whose rerank ranks by honest group-OOF."""
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, tiny_model_n_estimators=40,
        base_candidates=["base_full", "base_partial"],
        # In-process: a worker process would make the calls where this test's counter cannot see them, and "no CV fit
        # ran" would then pass for the wrong reason.
        tiny_rerank_backend="threads", **cfg_kwargs,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = groups
    disc.fit(df=df, target_col="y", feature_cols=_FEATS, train_idx=train_idx)
    return disc


@pytest.fixture(scope="module")
def _grouped():
    """A grouped frame with an upper-tail holdout, shared by the cases below."""
    df, groups, _y, levels = _frame(n_groups=20, per=120)
    train_idx, _holdout = _split_upper_tail(groups, levels, 4)
    return df, groups, train_idx


def test_no_cv_fits_run_when_honest_oof_measures_every_spec(monkeypatch, _grouped):
    """The default grouped path has no other consumer of the CV sweep, so it must not run."""
    df, groups, train_idx = _grouped
    seen = _count_cv_calls(monkeypatch)
    disc = _fit(groups, train_idx, df)
    assert disc.specs_, "the fixture must produce specs for the count to mean anything"
    assert seen["n"] == 0, f"the discarded CV sweep still ran {seen['n']} time(s)"


def test_the_reported_scores_are_the_honest_oof_values(monkeypatch, _grouped):
    """Skipping the sweep must not leave a spec scoreless: every score is its honest measurement."""
    df, groups, train_idx = _grouped
    _count_cv_calls(monkeypatch)
    disc = _fit(groups, train_idx, df)
    honest = dict(getattr(disc, "_honest_oof_rmse", {}) or {})
    assert honest, "honest-OOF selection did not run on a grouped fixture"
    scores = disc.tiny_rerank_scores_ or {}
    survivors = [n for n in scores if n in honest]
    assert survivors, "no measured spec survived the honest-OOF floor"
    for name in survivors:
        assert scores[name] == pytest.approx(honest[name]), f"{name} kept a CV score instead of its honest one"
    assert all(np.isfinite(v) for v in scores.values()), "a skipped spec was left with a non-finite score"


def test_the_wilcoxon_gate_still_gets_its_per_seed_fits(monkeypatch, _grouped):
    """The Wilcoxon gate reads the sweep's per-seed vectors, so with it on every spec must still be fitted."""
    df, groups, train_idx = _grouped
    seen = _count_cv_calls(monkeypatch)
    _fit(groups, train_idx, df, use_wilcoxon_gate=True)
    assert seen["n"] > 0, "the sweep was skipped although the Wilcoxon gate needs its per-seed RMSEs"


def test_the_per_bin_gate_still_gets_its_first_pass_breakdown(monkeypatch, _grouped):
    """The regime-aware gate reuses the sweep's per-bin breakdown; enabling it (it needs the raw-baseline gate on,
    which is what makes the breakdown load-bearing) must keep the sweep."""
    df, groups, train_idx = _grouped
    seen = _count_cv_calls(monkeypatch)
    _fit(groups, train_idx, df, per_bin_n_bins=5, require_beats_raw_baseline=True)
    assert seen["n"] > 0, "the sweep was skipped although the per-bin gate reuses its breakdown"
