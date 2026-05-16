"""Edge-case tests for adaptive RAM cleanup logic in mlframe.training.utils.

Covers `should_clean_ram`, `maybe_clean_ram_and_gpu`, `estimate_df_size_mb`,
`get_process_rss_mb` — and their interaction at call sites in core/trainer.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training import _ram_helpers as u


# ----------------------------- helpers ----------------------------------------

def _fake_psutil(rss_mb: float, free_mb: float):
    """Return a MagicMock mimicking psutil with given rss/available (in MB)."""
    fake = MagicMock()
    fake.Process.return_value.memory_info.return_value.rss = int(rss_mb * 1024**2)
    fake.virtual_memory.return_value.available = int(free_mb * 1024**2)
    return fake


def _install_fake_psutil(monkeypatch, fake):
    """Make `import psutil` inside utils.py return our fake."""
    import sys
    monkeypatch.setitem(sys.modules, "psutil", fake)


class TestAdaptiveCleanRam:
    # ---- should_clean_ram threshold boundaries -------------------------------

    def test_growth_exactly_500_does_not_trigger(self, monkeypatch):
        """Boundary: growth == 500.0 must NOT fire (strict `>`)."""
        fake = _fake_psutil(rss_mb=1500.0, free_mb=32_000.0)
        _install_fake_psutil(monkeypatch, fake)
        assert u.should_clean_ram(baseline_rss_mb=1000.0, df_size_mb=10.0) is False

    def test_growth_just_above_500_triggers(self, monkeypatch):
        fake = _fake_psutil(rss_mb=1500.001, free_mb=32_000.0)
        _install_fake_psutil(monkeypatch, fake)
        assert u.should_clean_ram(baseline_rss_mb=1000.0, df_size_mb=10.0) is True

    def test_negative_growth_does_not_trigger(self, monkeypatch):
        """RSS shrank (OS released) — growth < 0, no clean needed."""
        fake = _fake_psutil(rss_mb=500.0, free_mb=32_000.0)
        _install_fake_psutil(monkeypatch, fake)
        assert u.should_clean_ram(baseline_rss_mb=1500.0, df_size_mb=100.0) is False

    def test_low_free_ram_fires_even_with_zero_growth(self, monkeypatch):
        """free_mb < 2 * df_size_mb -> fire even when RSS flat.

        2026-04-30 (f117fb5): the OOM branch is short-circuited for
        df_size_mb < 256 (the 2*df_size threshold cannot plausibly OOM
        below that on any modern host). Use a 300 MB DF here to keep
        the OOM-branch contract under test.
        """
        fake = _fake_psutil(rss_mb=1000.0, free_mb=500.0)  # 500 MB free
        _install_fake_psutil(monkeypatch, fake)
        # df_size=300 -> threshold free < 600; 500 < 600 triggers; df_size >= 256 keeps the branch live.
        assert u.should_clean_ram(baseline_rss_mb=1000.0, df_size_mb=300.0) is True

    def test_df_size_zero_uses_500mb_floor(self, monkeypatch):
        """df_size_mb=0 → max(500, 0)=500 growth threshold still applies."""
        fake = _fake_psutil(rss_mb=1400.0, free_mb=32_000.0)  # +400 growth
        _install_fake_psutil(monkeypatch, fake)
        assert u.should_clean_ram(baseline_rss_mb=1000.0, df_size_mb=0.0) is False

        fake2 = _fake_psutil(rss_mb=1600.0, free_mb=32_000.0)  # +600 growth
        _install_fake_psutil(monkeypatch, fake2)
        assert u.should_clean_ram(baseline_rss_mb=1000.0, df_size_mb=0.0) is True

    def test_30pct_of_df_size_dominates_on_big_df(self, monkeypatch):
        """Large DF: threshold is 0.3 * df_size_mb, not 500."""
        # df=10_000 MB → threshold max(500, 3000)=3000; growth=2000 should NOT fire
        fake = _fake_psutil(rss_mb=12_000.0, free_mb=100_000.0)
        _install_fake_psutil(monkeypatch, fake)
        assert u.should_clean_ram(baseline_rss_mb=10_000.0, df_size_mb=10_000.0) is False

        fake2 = _fake_psutil(rss_mb=13_500.0, free_mb=100_000.0)  # +3500 > 3000
        _install_fake_psutil(monkeypatch, fake2)
        assert u.should_clean_ram(baseline_rss_mb=10_000.0, df_size_mb=10_000.0) is True

    def test_tb_scale_df_always_fires_on_typical_box(self, monkeypatch):
        """df=1TB → free_mb < 2TB will virtually always be true on real machines."""
        fake = _fake_psutil(rss_mb=10_000.0, free_mb=64_000.0)  # 64 GB free
        _install_fake_psutil(monkeypatch, fake)
        assert u.should_clean_ram(baseline_rss_mb=10_000.0, df_size_mb=1_000_000.0) is True

    # ---- psutil missing fallback --------------------------------------------

    def test_psutil_importerror_fallback_to_true(self, monkeypatch):
        """When psutil import fails, should_clean_ram returns True (safe default)."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "psutil":
                raise ImportError("simulated")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert u.should_clean_ram(100.0, 50.0) is True

    def test_get_process_rss_returns_zero_without_psutil(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "psutil":
                raise ImportError("simulated")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert u.get_process_rss_mb() == 0.0

    def test_no_psutil_combined_behavior_at_call_site(self, monkeypatch):
        """Trace: baseline=0.0 (from get_process_rss_mb fallback), then
        should_clean_ram also falls through to True → clean fires."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "psutil":
                raise ImportError("simulated")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        calls = []
        monkeypatch.setattr(u, "clean_ram_and_gpu", lambda verbose=False: calls.append(1))

        baseline = u.get_process_rss_mb()  # 0.0
        new_baseline = u.maybe_clean_ram_and_gpu(baseline, df_size_mb=10.0)
        # Fires → returns refreshed baseline (0.0 when psutil missing); clean called once
        assert new_baseline == 0.0
        assert len(calls) == 1

    # ---- maybe_clean_ram_and_gpu integration ---------------------------------

    def test_ten_calls_no_growth_zero_cleans(self, monkeypatch):
        """Simulate 10 call sites in a loop with flat RSS and plenty of free RAM."""
        fake = _fake_psutil(rss_mb=1000.0, free_mb=32_000.0)
        _install_fake_psutil(monkeypatch, fake)

        calls = []
        monkeypatch.setattr(u, "clean_ram_and_gpu", lambda verbose=False: calls.append(1))

        for _ in range(10):
            u.maybe_clean_ram_and_gpu(baseline_rss_mb=1000.0, df_size_mb=5.0)
        assert len(calls) == 0

    def test_growth_over_loop_fires_exactly_when_crossed(self, monkeypatch):
        """RSS at loop iter i = 1000 + 100*i MB; baseline=1000, threshold=500 MB.

        ``maybe_clean_ram_and_gpu`` reads RSS once via ``should_clean_ram`` and (on fire) reads
        again to refresh baseline. We back the fake with an iter-index counter (not an iterator
        consumed by reads) so refresh reads return the SAME rss as the should_clean_ram read
        inside one logical loop iteration. Pre-fix this test used a sequence iterator that
        misaligned with the post-refactor refresh call, producing brittle fire-index mismatch.
        """
        loop_state = {"iter": 0}

        def fake_rss_bytes():
            return (1000.0 + 100.0 * loop_state["iter"]) * 1024**2

        fake = MagicMock()
        fake.virtual_memory.return_value.available = 32_000 * 1024**2
        type(fake.Process.return_value.memory_info.return_value).rss = property(
            lambda self: fake_rss_bytes()
        )
        _install_fake_psutil(monkeypatch, fake)

        # Patch get_process_rss_mb to reflect the same loop_state-keyed RSS so a fire's
        # baseline-refresh doesn't return a stale or jumped value.
        monkeypatch.setattr(u, "get_process_rss_mb", lambda: 1000.0 + 100.0 * loop_state["iter"])

        calls = []
        monkeypatch.setattr(u, "clean_ram_and_gpu", lambda verbose=False: calls.append(1))

        fired_iters = []
        for i in range(10):
            loop_state["iter"] = i
            calls_before = len(calls)
            u.maybe_clean_ram_and_gpu(baseline_rss_mb=1000.0, df_size_mb=10.0)
            if len(calls) > calls_before:
                fired_iters.append(i)

        # iter i: rss = 1000 + 100*i; growth = 100*i; fires when 100*i > 500 -> i >= 6
        # (i=5 gives growth=500; threshold is strict ">" so first fire is i=6).
        assert fired_iters == [6, 7, 8, 9], (
            f"expected fires at iters [6,7,8,9] (growth crosses 500 MB), got {fired_iters}"
        )
        assert len(calls) == 4

    def test_verbose_logs_reason_when_fires(self, monkeypatch, caplog):
        fake = _fake_psutil(rss_mb=2000.0, free_mb=32_000.0)  # big growth
        _install_fake_psutil(monkeypatch, fake)
        monkeypatch.setattr(u, "clean_ram_and_gpu", lambda verbose=False: None)

        with caplog.at_level(logging.INFO, logger=u.logger.name):
            u.maybe_clean_ram_and_gpu(1000.0, 10.0, verbose=True, reason="post-feature-eng")
        assert any("post-feature-eng" in r.message for r in caplog.records)

    def test_skip_produces_no_log(self, monkeypatch, caplog):
        fake = _fake_psutil(rss_mb=1000.0, free_mb=32_000.0)
        _install_fake_psutil(monkeypatch, fake)
        monkeypatch.setattr(u, "clean_ram_and_gpu", lambda verbose=False: None)

        with caplog.at_level(logging.INFO, logger=u.logger.name):
            new_baseline = u.maybe_clean_ram_and_gpu(1000.0, 5.0, verbose=True, reason="noop")
        # No fire → baseline returned unchanged
        assert new_baseline == 1000.0
        assert not any("clean_ram fired" in r.message for r in caplog.records)

    # ---- estimate_df_size_mb -------------------------------------------------

    def test_estimate_df_size_variants(self):
        pdf = pd.DataFrame({"a": np.arange(10_000), "b": ["xyz"] * 10_000})
        assert u.estimate_df_size_mb(pdf) > 0.0

        pldf = pl.DataFrame({"a": np.arange(10_000)})
        assert u.estimate_df_size_mb(pldf) > 0.0

        # LazyFrame, numpy, None, dict → inf (OOM-safe fallback; guards
        # clean_ram heuristic from silently missing Arrow/Modin/Dask inputs)
        import math
        assert math.isinf(u.estimate_df_size_mb(pldf.lazy()))
        assert math.isinf(u.estimate_df_size_mb(np.zeros((100, 100))))
        assert math.isinf(u.estimate_df_size_mb(None))
        assert math.isinf(u.estimate_df_size_mb({"a": [1, 2]}))

    def test_empty_df_size_is_small(self):
        assert u.estimate_df_size_mb(pd.DataFrame()) < 1.0
        assert u.estimate_df_size_mb(pl.DataFrame()) < 1.0
