"""REPORTING_A-4: render_risk_coverage_diagnostic / render_model_card_diagnostic /
render_split_comparison_from_suite / render_decile_table_diagnostic (the dispatch/wiring layer)
had zero direct test coverage -- only the underlying charts.* composer functions were exercised."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.reporting._diagnostics_dispatch_extra import (
    render_decile_table_diagnostic,
    render_model_card_diagnostic,
    render_split_comparison_from_suite,
)
from mlframe.reporting._risk_coverage_diagnostic import render_risk_coverage_diagnostic

PNG = "matplotlib[png]"


def _png_exists(base: str) -> bool:
    """Helper: png (or its backend-suffixed sibling) exists at base."""
    # Charts land in a per-format subfolder by default (``png/name.png``); these tests care that the chart was
    # PRODUCED, not which layout the run used.
    directory, name = os.path.split(base)
    return any(os.path.exists(c) for c in (
        base + ".png", base + ".matplotlib.png",
        os.path.join(directory, "png", name + ".png"), os.path.join(directory, "png", name + ".matplotlib.png"),
    ))


@pytest.fixture
def binary_scores():
    """Deterministic binary y_true/y_score pair."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.uniform(0, 1, n) < 0.4).astype(int)
    score = np.clip(y * 0.6 + rng.uniform(0, 0.4, n), 0, 1)
    return y, score


def test_render_risk_coverage_diagnostic_renders_and_records_metrics(tmp_path, binary_scores):
    """render_risk_coverage_diagnostic must save a chart and record AURC/selective-gain metrics."""
    y, score = binary_scores
    base = str(tmp_path / "m")
    md = {}
    ok = render_risk_coverage_diagnostic(y_true=y, y_score=score, task="binary", plot_outputs=PNG, base_path=base, metrics_dict=md)
    assert ok and _png_exists(base + "_risk_coverage")
    assert "risk_coverage" in md["charts"]["saved"]
    assert "risk_coverage_aurc" in md and "risk_coverage_selective_gain" in md


def test_render_risk_coverage_diagnostic_skips_without_score(tmp_path, binary_scores):
    """render_risk_coverage_diagnostic is a no-op (returns False, no crash) when y_score is absent."""
    y, _score = binary_scores
    base = str(tmp_path / "m")
    md = {}
    assert render_risk_coverage_diagnostic(y_true=y, y_score=None, plot_outputs=PNG, base_path=base, metrics_dict=md) is False
    assert md.get("charts", {}).get("saved", []) == []


def test_render_model_card_diagnostic_renders_binary(tmp_path, binary_scores):
    """render_model_card_diagnostic must save a chart for a binary task with y_score."""
    y, score = binary_scores
    base = str(tmp_path / "m")
    md = {}
    ok = render_model_card_diagnostic(task="binary", y_true=y, y_score=score, plot_outputs=PNG, base_path=base, metrics_dict=md, model_name="lgb", split="test")
    assert ok and _png_exists(base + "_model_card")
    assert "model_card" in md["charts"]["saved"]


def test_render_model_card_diagnostic_skips_without_y_true(tmp_path, binary_scores):
    """render_model_card_diagnostic is a no-op when y_true is absent."""
    base = str(tmp_path / "m")
    md = {}
    assert render_model_card_diagnostic(task="binary", y_true=None, plot_outputs=PNG, base_path=base, metrics_dict=md) is False


def test_render_decile_table_diagnostic_renders(tmp_path, binary_scores):
    """render_decile_table_diagnostic must save a decile gain/lift/KS table figure."""
    y, score = binary_scores
    base = str(tmp_path / "m")
    md = {}
    ok = render_decile_table_diagnostic(y_true=y, y_score=score, plot_outputs=PNG, base_path=base, metrics_dict=md)
    assert ok and _png_exists(base + "_decile_table")
    assert "decile_table" in md["charts"]["saved"]


def test_render_decile_table_diagnostic_skips_on_empty_input(tmp_path):
    """render_decile_table_diagnostic is a no-op on zero-length arrays."""
    base = str(tmp_path / "m")
    md = {}
    assert render_decile_table_diagnostic(y_true=np.array([]), y_score=np.array([]), plot_outputs=PNG, base_path=base, metrics_dict=md) is False


def _suite_entry(y, score):
    """Build a suite-shaped SimpleNamespace record with train/val/test splits for split-comparison wiring."""
    n = len(y)
    a, b = n // 3, 2 * n // 3
    return SimpleNamespace(
        train_target=y[:a],
        train_probs=score[:a],
        val_target=y[a:b],
        val_probs=score[a:b],
        test_target=y[b:],
        test_probs=score[b:],
    )


def test_render_split_comparison_from_suite_renders_with_multiple_splits(tmp_path, binary_scores):
    """render_split_comparison_from_suite must save a cross-split overfit panel when >=2 splits are usable."""
    y, score = binary_scores
    entry = _suite_entry(y, score)
    base = str(tmp_path / "m")
    md = {}
    ok = render_split_comparison_from_suite(entry=entry, target_type="binary_classification", plot_outputs=PNG, base_path=base, metrics_dict=md, model_name="lgb")
    assert ok and _png_exists(base + "_split_comparison")
    assert "split_comparison" in md["charts"]["saved"]


def test_render_split_comparison_from_suite_skips_with_one_split(tmp_path, binary_scores):
    """render_split_comparison_from_suite is a no-op with fewer than 2 usable splits."""
    y, score = binary_scores
    entry = SimpleNamespace(test_target=y, test_probs=score)
    base = str(tmp_path / "m")
    md = {}
    assert render_split_comparison_from_suite(entry=entry, target_type="binary_classification", plot_outputs=PNG, base_path=base, metrics_dict=md) is False
