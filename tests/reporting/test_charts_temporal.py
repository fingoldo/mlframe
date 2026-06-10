"""Tests for ``mlframe.reporting.charts.temporal``.

The legacy ``build_temporal_audit_spec`` rendered only the kept-bins
line, silently dropping the Pelt change-points, the per-segment mean
steps, and the sparse/dropped-bin markers. These tests pin PARITY with
the full matplotlib ``plot_target_over_time`` artifact: change-points
must surface (as shaded spans), segment means as a step series, dropped
bins as a faded marker series, and the time axis must be flagged
(``x_is_time``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.temporal import build_temporal_audit_spec
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import LinePanelSpec
from mlframe.training.targets.target_temporal_audit import (
    TemporalAuditResult, TimeBin, audit_target_over_time,
)


def _synthetic_audit_with_changepoint():
    """A 12-month series, regime change after month 6, one sparse (dropped) bin.

    Built directly so change-points / segments / kept flags are deterministic and the test does not depend on Pelt
    re-discovering the boundary. Change-point at kept-index 6; segments [0..6) mean 0.1, [6..12) mean 0.6.
    """
    starts = pd.date_range("2022-01-01", periods=13, freq="MS")
    bins = []
    for i in range(13):
        rate = 0.1 if i < 6 else 0.6
        kept = i != 3  # month index 3 is sparse / dropped
        bins.append(TimeBin(bin_label=str(starts[i].date()), bin_start=starts[i],
                            n_obs=(5 if not kept else 500), target_rate=rate, kept=kept))
    kept_bins = [b for b in bins if b.kept]  # 12 kept
    segments = [
        {"start_idx": 0, "end_idx": 6, "start_label": "", "end_label": "",
         "n_bins": 6, "n_obs": 3000, "mean_rate": 0.1},
        {"start_idx": 6, "end_idx": len(kept_bins), "start_label": "", "end_label": "",
         "n_bins": len(kept_bins) - 6, "n_obs": 3000, "mean_rate": 0.6},
    ]
    return TemporalAuditResult(
        target_name="y", target_type="binary_classification", timestamp_col="t",
        granularity="month", bins=bins, change_point_indices=[6], segments=segments,
        warnings=[],
    )


def test_x_is_time_flag_set():
    spec = build_temporal_audit_spec(_synthetic_audit_with_changepoint())
    line = spec.panels[0][0]
    assert isinstance(line, LinePanelSpec)
    assert line.x_is_time is True
    assert np.issubdtype(np.asarray(line.x).dtype, np.datetime64)


def test_changepoints_rendered_as_spans():
    """The single change-point MUST appear as a shaded vertical span (parity with the matplotlib axvline)."""
    spec = build_temporal_audit_spec(_synthetic_audit_with_changepoint())
    line = spec.panels[0][0]
    assert line.vspans is not None
    assert len(line.vspans) == 1
    x0, x1, color, alpha = line.vspans[0]
    assert x1 > x0  # a visible (non-zero-width) span
    assert color == "red"


def test_segment_means_present_as_step_series():
    """The per-segment mean must be a distinct series carrying both regime means (0.1 and 0.6)."""
    spec = build_temporal_audit_spec(_synthetic_audit_with_changepoint())
    line = spec.panels[0][0]
    assert "segment mean" in line.series_labels
    seg = np.asarray(line.y[line.series_labels.index("segment mean")])
    distinct = np.unique(np.round(seg[~np.isnan(seg)], 2))
    assert set(distinct.tolist()) == {0.1, 0.6}


def test_sparse_bins_rendered_as_marker_series():
    """The dropped (sparse) bin must surface as its own faded marker series, not vanish silently."""
    spec = build_temporal_audit_spec(_synthetic_audit_with_changepoint())
    line = spec.panels[0][0]
    sparse_idx = [i for i, lbl in enumerate(line.series_labels) if lbl.startswith("sparse")]
    assert len(sparse_idx) == 1
    sparse_series = np.asarray(line.y[sparse_idx[0]])
    # Exactly one finite point (the single dropped bin); all others NaN.
    assert int(np.isfinite(sparse_series).sum()) == 1
    # And it draws as markers only.
    assert line.line_styles[sparse_idx[0]] == "markers"


def test_three_series_kept_segment_sparse():
    spec = build_temporal_audit_spec(_synthetic_audit_with_changepoint())
    line = spec.panels[0][0]
    assert len(line.y) == 3
    assert line.series_labels[0] == "y"


def test_degenerate_no_bins_does_not_crash():
    res = TemporalAuditResult(
        target_name="y", target_type="regression", timestamp_col="t",
        granularity="month", bins=[], change_point_indices=[], segments=[], warnings=[],
    )
    spec = build_temporal_audit_spec(res)
    assert isinstance(spec.panels[0][0], LinePanelSpec)


@pytest.mark.parametrize("backend", ["matplotlib[png]", "plotly[html]", "matplotlib[png]+plotly[html]"])
def test_render_smoke_both_backends(tmp_path, backend):
    """Datetime x + change-point spans must render on BOTH backends (plotly add_vline crashes on datetime, hence spans)."""
    spec = build_temporal_audit_spec(_synthetic_audit_with_changepoint())
    base = os.path.join(str(tmp_path), "temporal")
    render_and_save(spec, parse_plot_output_dsl(backend), base)
    assert any(os.scandir(str(tmp_path)))


def test_end_to_end_real_audit_surfaces_changepoint(tmp_path):
    """Through the real Pelt audit: a sharp regime change must produce a change-point span + 2 segment means."""
    rng = np.random.default_rng(0)
    n = 4000
    ts = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 40 * 30, n), unit="D")
    df = pd.DataFrame({"t": ts}).sort_values("t").reset_index(drop=True)
    y = np.where(df["t"] < pd.to_datetime("2022-09-01"),
                 rng.binomial(1, 0.1, n), rng.binomial(1, 0.6, n))
    df["y"] = y
    res = audit_target_over_time(df, timestamp_col="t", target_col="y",
                                 target_type="binary_classification", granularity="month")
    assert len(res.change_point_indices) >= 1
    assert len(res.segments) >= 2
    spec = build_temporal_audit_spec(res)
    line = spec.panels[0][0]
    assert line.vspans is not None and len(line.vspans) >= 1
    seg = np.asarray(line.y[1])
    assert np.unique(seg[~np.isnan(seg)]).size >= 2  # both regime means present
    base = os.path.join(str(tmp_path), "t_e2e")
    render_and_save(spec, parse_plot_output_dsl("matplotlib[png]+plotly[html]"), base)
    assert any(os.scandir(str(tmp_path)))
