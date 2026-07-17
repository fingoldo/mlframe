"""Regression + wiring tests for the per-class calibration plot path.

Covers:
- INV-1: per-class reliability PNGs must NOT overwrite one another (a bare
  ``_perfplot.png`` was reused inside the per-class loop, so only the last
  class survived). A 3-class report must leave 3 distinct files on disk.
- INV-2: ``report_probabilistic_model_perf(plot_outputs=...)`` must route the
  reliability diagram through the DSL (build_calibration_spec) so plotly HTML
  is produced, not just matplotlib PNG.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from mlframe.training.reporting._reporting_probabilistic import (
    report_probabilistic_model_perf,
)


def _three_class_synthetic(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 3, size=n)
    # Probs correlated with the true class so AUC/calibration are non-degenerate.
    probs = rng.random((n, 3))
    probs[np.arange(n), targets] += 1.5
    probs /= probs.sum(axis=1, keepdims=True)
    return targets, probs


def test_per_class_perfplots_do_not_overwrite(tmp_path):
    """3-class report leaves 3 distinct reliability PNGs (INV-1)."""
    targets, probs = _three_class_synthetic()
    base = str(tmp_path / "split0")
    report_probabilistic_model_perf(
        targets=targets,
        columns=["f0", "f1"],
        model_name="m",
        model=None,
        probs=probs,
        classes=[0, 1, 2],
        plot_file=base,
        show_perf_chart=False,
        print_report=False,
    )
    pngs = sorted(p for p in os.listdir(tmp_path) if p.endswith(".png"))
    # One reliability PNG per class -> 3 distinct files, none overwriting another.
    assert len(pngs) == 3, f"expected 3 distinct per-class PNGs, got {pngs}"
    assert len(set(pngs)) == 3


def test_per_class_perfplot_filenames_carry_class_id(tmp_path):
    """Per-class filenames include the class id slug so they are stable + unique."""
    targets, probs = _three_class_synthetic()
    base = str(tmp_path / "split0")
    report_probabilistic_model_perf(
        targets=targets,
        columns=["f0", "f1"],
        model_name="m",
        model=None,
        probs=probs,
        classes=[0, 1, 2],
        plot_file=base,
        show_perf_chart=False,
        print_report=False,
    )
    pngs = sorted(p for p in os.listdir(tmp_path) if p.endswith(".png"))
    for cid in (0, 1, 2):
        assert any(f"_c{cid}_" in p for p in pngs), f"no file for class {cid} in {pngs}"


def test_plot_outputs_routes_through_dsl_to_html(tmp_path):
    """plot_outputs='plotly[html]' produces a plotly HTML per class via the DSL (INV-2)."""
    targets, probs = _three_class_synthetic()
    base = str(tmp_path / "split0")
    report_probabilistic_model_perf(
        targets=targets,
        columns=["f0", "f1"],
        model_name="m",
        model=None,
        probs=probs,
        classes=[0, 1, 2],
        plot_file=base,
        plot_outputs="plotly[html]",
        show_perf_chart=False,
        print_report=False,
    )
    htmls = sorted(p for p in os.listdir(tmp_path) if p.endswith(".html"))
    assert len(htmls) == 3, f"expected 3 per-class HTML files, got {htmls}"


def test_binary_perfplot_single_file(tmp_path):
    """Binary report (only class 1 reported) -> a single _perfplot.png (no slug churn)."""
    rng = np.random.default_rng(1)
    n = 500
    targets = rng.integers(0, 2, size=n)
    p1 = rng.random(n)
    p1[targets == 1] = np.clip(p1[targets == 1] + 0.4, 0, 1)
    probs = np.column_stack([1 - p1, p1])
    base = str(tmp_path / "splitb")
    report_probabilistic_model_perf(
        targets=targets,
        columns=["f0"],
        model_name="m",
        model=None,
        probs=probs,
        classes=[0, 1],
        plot_file=base,
        show_perf_chart=False,
        print_report=False,
    )
    pngs = sorted(p for p in os.listdir(tmp_path) if p.endswith(".png"))
    assert pngs == ["splitb_perfplot.png"], pngs
