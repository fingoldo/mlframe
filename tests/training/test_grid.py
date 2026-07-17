"""Smoke tests for the ``run_grid`` sequential variant runner."""

from __future__ import annotations

import logging

import pytest

from mlframe.training.grid import run_grid


def _record_suite(**kwargs):
    # Returns a shallow snapshot of what the suite saw.
    """Record suite."""
    return {"seen": dict(kwargs)}


def test_run_grid_dict_entries_use_auto_labels():
    """Run grid dict entries use auto labels."""
    base = {"a": 1, "b": 2}
    grid = [{"a": 10}, {"b": 20}]
    out = run_grid(base, grid, suite_fn=_record_suite)
    assert list(out.keys()) == ["variant_0", "variant_1"]
    assert out["variant_0"]["seen"] == {"a": 10, "b": 2}
    assert out["variant_1"]["seen"] == {"a": 1, "b": 20}


def test_run_grid_tuple_entries_use_explicit_labels():
    """Run grid tuple entries use explicit labels."""
    out = run_grid(
        {"a": 0},
        [("fast", {"a": 1}), ("slow", {"a": 2})],
        suite_fn=_record_suite,
    )
    assert set(out) == {"fast", "slow"}
    assert out["fast"]["seen"] == {"a": 1}
    assert out["slow"]["seen"] == {"a": 2}


def test_run_grid_duplicate_labels_raise():
    """Run grid duplicate labels raise."""
    with pytest.raises(ValueError, match="duplicate"):
        run_grid({}, [("x", {}), ("x", {})], suite_fn=_record_suite)


def test_run_grid_continues_past_errors_by_default(caplog):
    """Run grid continues past errors by default."""
    def suite(**kwargs):
        """Suite."""
        if kwargs.get("bad"):
            raise RuntimeError("boom")
        return {"ok": True}

    with caplog.at_level(logging.ERROR):
        out = run_grid(
            {},
            [("good", {}), ("bad", {"bad": True}), ("good2", {})],
            suite_fn=suite,
        )
    assert out["good"] == {"ok": True}
    assert "error" in out["bad"] and "boom" in out["bad"]["error"]
    assert out["good2"] == {"ok": True}


def test_run_grid_stop_on_error_propagates():
    """Run grid stop on error propagates."""
    def suite(**kwargs):
        """Suite."""
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        run_grid({}, [{"x": 1}], suite_fn=suite, stop_on_error=True)


def test_run_grid_rejects_invalid_entry_shape():
    """Run grid rejects invalid entry shape."""
    with pytest.raises(TypeError):
        run_grid({}, [("onlylabel",)], suite_fn=_record_suite)
    with pytest.raises(TypeError):
        run_grid({}, [42], suite_fn=_record_suite)
