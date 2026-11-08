"""``MLFRAME_BINAGG_DEVICE_PERMS_MIN_ROWS`` sets the row count from which the binned-agg null draws its permutations on the device; a value that is not an integer must be reported."""

from __future__ import annotations

import logging

from mlframe.feature_selection.filters._binned_numeric_agg_resident import _device_perms_min_rows
from mlframe.utils.log_throttle import reset_throttle_counts

_ENV = "MLFRAME_BINAGG_DEVICE_PERMS_MIN_ROWS"


def test_an_integer_value_is_used_as_is(monkeypatch):
    """The variable is read at every call: 0 means always on the device, and a large value never."""
    monkeypatch.setenv(_ENV, "123")
    assert _device_perms_min_rows() == 123
    monkeypatch.setenv(_ENV, "0")
    assert _device_perms_min_rows() == 0


def test_unset_gives_two_million(monkeypatch):
    """Without the variable the threshold is the documented 2 million rows."""
    monkeypatch.delenv(_ENV, raising=False)
    assert _device_perms_min_rows() == 2_000_000


def test_a_value_that_is_not_an_integer_warns_and_uses_the_default(monkeypatch, caplog):
    """A typo in the variable must not silently change which path the null takes: the default is used and a warning names the bad value."""
    reset_throttle_counts("binagg_device_perms_min_rows_invalid")
    monkeypatch.setenv(_ENV, "2e6")
    with caplog.at_level(logging.WARNING):
        assert _device_perms_min_rows() == 2_000_000
    assert any("'2e6'" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
