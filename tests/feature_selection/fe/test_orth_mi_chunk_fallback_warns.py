"""A failed RAM probe in the orthogonal-basis MI chunker falls back to a conservative width and says so at WARNING."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as backends


def test_mi_chunk_cols_for_warns_when_psutil_probe_fails(monkeypatch, caplog):
    """A raising psutil probe must be visible at WARNING and must keep the 2 GiB default block budget."""
    broken = types.ModuleType("psutil")

    def _no_probe():
        """Stand in for psutil's RAM probe on a host where it fails."""
        raise OSError("probe unavailable")

    broken.virtual_memory = _no_probe
    monkeypatch.setitem(sys.modules, "psutil", broken)

    with caplog.at_level(logging.WARNING, logger=backends.logger.name):
        width = backends._mi_chunk_cols_for(1_000)

    expected = int(min(1024, max(1, (2 * 1024**3 // 10) // (1_000 * 8 * 3))))
    assert width == expected
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "psutil.virtual_memory() probe failed" in r.getMessage()]
    assert len(warnings) == 1
    assert "probe unavailable" in warnings[0].getMessage()


def test_mi_chunk_cols_for_is_silent_when_psutil_probe_succeeds(caplog):
    """A healthy probe logs nothing, so the new WARNING does not fire on every call."""
    pytest.importorskip("psutil")

    with caplog.at_level(logging.WARNING, logger=backends.logger.name):
        backends._mi_chunk_cols_for(1_000)

    assert not [r for r in caplog.records if "psutil.virtual_memory() probe failed" in r.getMessage()]
