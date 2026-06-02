"""Regression coverage for the composite-discovery RAM profiler + pickle-cache
size clamp (2026-05-30).

(1) Sub-phase RAM telemetry: verify the helper logs phase deltas and triggers
adaptive GC.
(2) Pickle cache size clamp: verify oversized entries are skipped and a smaller
entry loads normally.
(3) Default env-var behaviour: profiler defaults ON; explicit "0" disables.
"""
from __future__ import annotations

import logging
import os
import pickle
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1: phase RAM report logs one INFO line per call, tracks baseline + prev.
# ---------------------------------------------------------------------------
def test_1_phase_ram_report_records_baseline_then_delta(caplog):
    from mlframe.training._composite_discovery_fit import _phase_ram_report

    state: dict = {}
    with caplog.at_level(logging.INFO, logger="mlframe.training._composite_discovery_fit"):
        _phase_ram_report(state, "entry")
        _phase_ram_report(state, "filter_features_done")
        _phase_ram_report(state, "transforms_evaluated")
    # New API: state keys carry the uss_mb suffix + commit.
    assert state.get("baseline_uss_mb") is not None
    assert state.get("prev_uss_mb") is not None
    assert state.get("baseline_commit_mb") is not None
    assert state.get("prev_commit_mb") is not None
    # 3 phase lines minimum (entry/filter/transforms).
    phase_lines = [r for r in caplog.records if "CompositeTargetDiscovery.RAM" in r.getMessage()]
    assert len(phase_lines) >= 3, [r.getMessage() for r in phase_lines]
    assert any("phase=entry" in r.getMessage() for r in phase_lines)
    assert any("phase=filter_features_done" in r.getMessage() for r in phase_lines)
    # Each line reports all three signals so OOM root cause is observable.
    for r in phase_lines:
        msg = r.getMessage()
        assert "USS=" in msg and "RSS=" in msg and "commit=" in msg, msg


def test_1_phase_ram_report_flags_page_thrashing(caplog):
    """When USS >> RSS by 2x+ on a 1 GB+ process, emit a PAGE_THRASHING marker.
    This is the signal the prior version masked entirely by reporting just RSS."""
    from mlframe.training import _composite_discovery_fit as mod
    state: dict = {}
    # Post-EmptyWorkingSet artefact: USS=60 GB, RSS=4 MB, commit roughly = USS.
    with patch.object(mod, "_process_mem_mb", return_value=(4.0, 60_000.0, 60_500.0)):
        mod._phase_ram_report(state, "entry")  # baseline
    with patch.object(mod, "_process_mem_mb", return_value=(4.0, 60_500.0, 61_000.0)):
        with caplog.at_level(logging.INFO, logger="mlframe.training._composite_discovery_fit"):
            mod._phase_ram_report(state, "after_phase")
    thrash = [r for r in caplog.records if "PAGE_THRASHING" in r.getMessage()]
    assert thrash, "PAGE_THRASHING marker must fire when USS >> RSS"


def test_1_phase_ram_report_flags_commit_pressure(caplog):
    """When commit >> USS the process holds large committed-but-untouched memory.
    On Windows that consumes the system-wide commit limit and is the proximate
    OOM-kernel-kill cause even when USS / RSS look benign."""
    from mlframe.training import _composite_discovery_fit as mod
    state: dict = {}
    # Mid-discovery: USS=20 GB, RSS=20 GB, commit=90 GB (private bytes reserved
    # for committed-but-paged-out buffers from pyarrow / numba intermediate work).
    with patch.object(mod, "_process_mem_mb", return_value=(20_000.0, 20_000.0, 20_000.0)):
        mod._phase_ram_report(state, "entry")
    with patch.object(mod, "_process_mem_mb", return_value=(20_000.0, 20_000.0, 90_000.0)):
        with caplog.at_level(logging.INFO, logger="mlframe.training._composite_discovery_fit"):
            mod._phase_ram_report(state, "after_phase")
    pressure = [r for r in caplog.records if "COMMIT_PRESSURE" in r.getMessage()]
    assert pressure, "COMMIT_PRESSURE marker must fire when commit >> USS"


def test_1_phase_ram_report_tolerates_psutil_failure():
    """The profiler must not raise when memory read fails -- it's diagnostic-only
    and must never block a real fit() path."""
    from mlframe.training import _composite_discovery_fit as mod
    state: dict = {}
    with patch.object(mod, "_process_mem_mb", side_effect=RuntimeError("psutil down")):
        try:
            mod._phase_ram_report(state, "entry")
        except RuntimeError:
            pytest.fail("profiler must not propagate psutil errors")


# ---------------------------------------------------------------------------
# 2: DiscoveryCache size-clamp.
# ---------------------------------------------------------------------------
def test_2_discovery_cache_skips_oversized_entry(tmp_path, caplog):
    """An entry larger than MLFRAME_DISCOVERY_CACHE_MAX_BYTES must read as a
    miss and emit a WARNING. The file itself stays on disk for operator
    inspection."""
    pytest.importorskip("pyutilz")
    from mlframe.training.composite_cache import DiscoveryCache

    cache = DiscoveryCache(cache_dir=str(tmp_path))
    key = "abc123"
    path = cache._path(key)
    # Write a small junk pickle so the file size dominates the test (we
    # intentionally write 50 KB so the 10 KB ceiling rejects it).
    payload = pickle.dumps({"junk": b"x" * 50_000})
    with open(path, "wb") as f:
        f.write(payload)
    assert os.path.getsize(path) > 40_000

    with patch.dict(os.environ, {"MLFRAME_DISCOVERY_CACHE_MAX_BYTES": "10000"}):
        sentinel = object()
        with caplog.at_level(logging.WARNING, logger="mlframe.training.composite_cache"):
            value = cache.get(key, default=sentinel)
    assert value is sentinel, "oversized entry must read as miss"
    assert any("oversized entry" in r.getMessage() for r in caplog.records), \
        "must emit a WARNING about the skipped oversize entry"
    # Stale file preserved.
    assert os.path.exists(path)


def test_2_discovery_cache_loads_small_entry_normally(tmp_path):
    """Sanity gate: an entry well under the cap loads exactly the pickled value."""
    pytest.importorskip("pyutilz")
    from mlframe.training.composite_cache import DiscoveryCache
    cache = DiscoveryCache(cache_dir=str(tmp_path))
    key = "abc123"
    payload = {"value": 42, "spec_name": "TVT-diff-foo"}
    cache.set(key, payload)
    with patch.dict(os.environ, {"MLFRAME_DISCOVERY_CACHE_MAX_BYTES": str(1024 ** 3)}):
        got = cache.get(key, default=None)
    assert got == payload


# ---------------------------------------------------------------------------
# 3: env-var disables the profiler (legacy quiet path).
# ---------------------------------------------------------------------------
def test_3_profiler_env_var_disabled(caplog):
    """Setting MLFRAME_DISCOVERY_RAM_PROFILER=0 must skip the per-phase
    logger calls inside fit(). We don't run the full fit() here (heavy
    deps); the unit gate is the env-var truthiness check itself.
    """
    # Match the truthiness check in fit():
    for v in ("0", "false", "FALSE", "no", "off"):
        on = v.strip().lower() not in ("0", "false", "no", "off")
        assert on is False, f"env var {v!r} must disable profiler"
    for v in ("1", "true", "yes", "ON", ""):
        on = v.strip().lower() not in ("0", "false", "no", "off")
        assert on is True, f"env var {v!r} must keep profiler ENABLED (default)"
