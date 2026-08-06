"""REPORTING_A-11 (2026-08-05 audit): ``_flat_scalar_metrics`` had inconsistent merge semantics --
top-level scalar keys overwrote unconditionally, but nested-subdict keys used ``setdefault``, so two
different nested sub-dicts sharing a key name silently dropped the second value with no documented
precedence rule. Fixed precedence: a top-level scalar always wins over any nested value of the same
name; among nested sub-dicts, the LAST one wins on a collision.
"""

from __future__ import annotations

from mlframe.reporting._diagnostics_dispatch_extra import _flat_scalar_metrics


def test_top_level_scalar_always_wins_over_nested():
    """A top-level scalar key must win even if a nested sub-dict shares its name."""
    metrics = {"auc": 0.9, "sub": {"auc": 0.1}}
    out = _flat_scalar_metrics(metrics)
    assert out["auc"] == 0.9


def test_last_nested_subdict_wins_on_collision():
    """Two nested sub-dicts sharing a key name: the LAST one (in iteration order) must win, not be silently
    dropped in favor of the first (the pre-fix setdefault behavior)."""
    metrics = {"sub_a": {"f1": 0.1}, "sub_b": {"f1": 0.9}}
    out = _flat_scalar_metrics(metrics)
    assert out["f1"] == 0.9, f"expected the LAST nested sub-dict's value (0.9) to win, got {out['f1']}"
