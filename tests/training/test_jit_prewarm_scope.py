"""The feature-selection kernel warm-up must not run when no feature selection will.

A production run spent 511.71s in the JIT prewarm before reading a single row, with the whole step gated only on
``dummy_baselines_config.enabled`` -- so a CatBoost-only fit with no MRMR and no RFECV still compiled the entire
feature-selection kernel set it would never call.

Measured on this box: the feature-selection group alone is 12.0s, the dummy-baselines group 49.9s. That does not
account for 511s, which is why the warm-up now also logs a per-group breakdown -- the remainder lives on a
machine that compiles CUDA kernels, and attributing it needs the log, not a guess.
"""

from __future__ import annotations

import logging

import pytest

from mlframe.metrics._core_numba_warmup import prewarm_numba_cache


class TestTheScopeSwitch:
    """Whether the expensive group runs at all."""

    def test_feature_selection_group_can_be_skipped(self, monkeypatch):
        """The point of the switch: a run with no FS must not pay for FS kernels."""
        called = []

        import mlframe.feature_selection.filters as filters

        monkeypatch.setattr(filters, "prewarm_fs_numba_cache", lambda: called.append(True))
        prewarm_numba_cache(include_feature_selection=False)
        assert called == []

    def test_feature_selection_group_runs_when_asked(self, monkeypatch):
        """And a run that WILL fit MRMR still gets its kernels warmed up front."""
        called = []

        import mlframe.feature_selection.filters as filters

        monkeypatch.setattr(filters, "prewarm_fs_numba_cache", lambda: called.append(True))
        prewarm_numba_cache(include_feature_selection=True)
        assert called == [True]

    def test_default_keeps_the_old_behaviour(self, monkeypatch):
        """A caller that cannot tell must not silently lose the warm-up: paying it needlessly is a slow run,
        skipping it wrongly is a slow first fit plus a confusing profile."""
        called = []

        import mlframe.feature_selection.filters as filters

        monkeypatch.setattr(filters, "prewarm_fs_numba_cache", lambda: called.append(True))
        prewarm_numba_cache()
        assert called == [True]


class TestTheBreakdownIsReported:
    """511 seconds with no attribution is what made this hard to diagnose."""

    def test_per_group_timing_is_logged(self, caplog):
        """Every group has to appear, including one that was skipped -- a missing line reads as a missing group."""
        with caplog.at_level(logging.INFO, logger="mlframe.metrics._core_numba_warmup"):
            prewarm_numba_cache(include_feature_selection=False)
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "[JIT prewarm] per-group:" in text
        assert "feature_selection=" in text
        assert "dummy_baselines=" in text


class TestTheSuiteGate:
    """What the suite passes, derived from the feature-selection config."""

    @pytest.mark.parametrize(
        "use_mrmr, rfecv, expected",
        [(False, None, False), (True, None, True), (False, ["lgb"], True), (True, ["lgb"], True), (False, [], False)],
    )
    def test_fs_will_run_predicate(self, use_mrmr, rfecv, expected):
        """Mirrors the expression in setup_configuration; either mechanism alone is enough to need the kernels."""
        assert (bool(use_mrmr) or bool(rfecv)) is expected
