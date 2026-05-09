"""Render every mlframe chart type x both backends for visual inspection.

This is NOT a pytest test -- visual inspection cannot be automated
reliably. The script generates one PNG per (chart_type, backend) pair
into the output directory, with realistic synthetic data + a multiline
suptitle that mimics the prod log header (model identity / iter / rows).

Run manually before shipping any change to:
- ``mlframe.reporting.charts.*`` (chart spec builders)
- ``mlframe.reporting.renderers.*`` (matplotlib / plotly backends)
- the ``ReportingConfig.plot_outputs`` DSL or its panel-template fields

Usage::

    python -m mlframe.tests.reporting.visual_audit.render_all_charts
    python -m mlframe.tests.reporting.visual_audit.render_all_charts --out /tmp/audit

What gets rendered (one matplotlib + one plotly PNG each):

  01_calibration_binary    - 2-panel (scatter + histogram) for binary calib
  02_regression            - 3-panel (scatter | residual hist | resid-vs-pred)
  03_multiclass            - 6-panel grid (confusion / PR_F1 / ROC / calib / ...)
  04_multilabel            - 5-panel grid (PR_F1 / calib / co-occurrence / ...)
  05_ltr                   - 5-panel grid (NDCG@k / NDCG_dist / lift / MRR / ...)
  06_quantile              - 5-panel grid (reliability / pinball / interval / ...)
  07_temporal              - single-line target rate over time

Inspect the PNGs side-by-side. Look for:
- suptitle overlapping per-panel titles (matplotlib y-position bug)
- per-panel titles overflowing horizontally into adjacent panels (plotly font-size bug)
- multiline titles getting collapsed (plotly needs ``<br>`` not ``\\n``)
- text clipping at figure edges
- unreadable / overlapping axis labels on dense grids (multiclass / multilabel)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Sequence

import numpy as np

# Production chart headers go through ``_strip_internal_model_suffixes``
# in training/trainer.py so internal mixin class names
# (XGBClassifierWithDMatrixReuse / LGBMClassifierWithDatasetReuse /
# *WithFastpath) get reduced to the canonical upstream name.
# Mirror that here so the audit reflects what real users see.
PROD_HEADER = (
    "TEST XGBClassifier-recency_07 "
    "[114F+11491.3506 trained on 4.1M rows @iter=37/125F/527.4K rows]"
)


def _render(out_dir: str, name: str, spec) -> None:
    """Render a FigureSpec to matplotlib + plotly PNG."""
    from mlframe.reporting.output import parse_plot_output_dsl
    from mlframe.reporting.renderers import render_and_save
    base = os.path.join(out_dir, name)
    render_and_save(spec, parse_plot_output_dsl("matplotlib[png] + plotly[png]"), base)
    print(f"  {name}.matplotlib.png + {name}.plotly.png")


# ---------------------------------------------------------------- 1. calibration
def audit_calibration(out_dir: str) -> None:
    print("[1/7] calibration (binary)...")
    from mlframe.reporting.charts.calibration import build_calibration_spec
    rng = np.random.default_rng(0)
    nbins = 10
    freqs_predicted = np.linspace(0.05, 0.95, nbins)
    freqs_true = freqs_predicted + rng.normal(0, 0.05, nbins)
    hits = rng.integers(50, 1500, nbins).astype(np.int64)
    spec = build_calibration_spec(
        freqs_predicted=freqs_predicted,
        freqs_true=freqs_true,
        hits=hits,
        plot_title=PROD_HEADER + "\nMAE=2.1% STD=1.8% COV=98% ICE=0.067",
    )
    _render(out_dir, "01_calibration_binary", spec)


# ---------------------------------------------------------------- 2. regression
def audit_regression(out_dir: str) -> None:
    print("[2/7] regression (3-panel)...")
    from mlframe.reporting.charts.regression import build_regression_panel_spec
    from mlframe.training.regression_residual_audit import audit_residuals
    rng = np.random.default_rng(1)
    n = 5000
    y_true = rng.standard_normal(n) * 100 + 11000
    y_pred = y_true + rng.standard_normal(n) * 15
    audit = audit_residuals(y_true, y_pred)
    spec = build_regression_panel_spec(
        y_true, y_pred, audit=audit,
        header_str=PROD_HEADER,
        metrics_str="MAE=10.6531 RMSE=14.8344 MaxError=156.1475 R2=0.9995",
    )
    _render(out_dir, "02_regression", spec)


# ---------------------------------------------------------------- 3. multiclass
def audit_multiclass(out_dir: str) -> None:
    print("[3/7] multiclass (6-panel grid)...")
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure
    rng = np.random.default_rng(2)
    n = 3000
    K = 4
    y_true = rng.integers(0, K, size=n)
    proba = rng.dirichlet([1.0] * K, size=n)
    for i in range(n):
        if rng.random() < 0.75:
            proba[i] = np.zeros(K)
            proba[i, y_true[i]] = 0.7
            other = rng.dirichlet([1.0] * (K - 1))
            proba[i, [k for k in range(K) if k != y_true[i]]] = 0.3 * other
    spec = compose_multiclass_figure(
        y_true, proba, classes=[f"cls_{i}" for i in range(K)],
        panels_template="CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
        suptitle=PROD_HEADER,
    )
    _render(out_dir, "03_multiclass", spec)


# ---------------------------------------------------------------- 4. multilabel
def audit_multilabel(out_dir: str) -> None:
    print("[4/7] multilabel (5-panel grid)...")
    from mlframe.reporting.charts.multilabel import compose_multilabel_figure
    rng = np.random.default_rng(3)
    n = 2000
    K = 4
    y_true = (rng.standard_normal((n, K)) > 0.3).astype(np.int8)
    y_proba = np.clip(y_true * 0.7 + rng.standard_normal((n, K)) * 0.2, 0, 1)
    spec = compose_multilabel_figure(
        y_true, y_proba, labels=[f"label_{i}" for i in range(K)],
        panels_template="PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST",
        suptitle=PROD_HEADER,
    )
    _render(out_dir, "04_multilabel", spec)


# ---------------------------------------------------------------- 5. LTR
def audit_ltr(out_dir: str) -> None:
    print("[5/7] LTR (5-panel grid)...")
    from mlframe.reporting.charts.ltr import compose_ltr_figure
    rng = np.random.default_rng(4)
    n_queries = 200
    docs_per = 10
    n = n_queries * docs_per
    y_true = rng.integers(0, 5, size=n).astype(np.float64)
    y_score = y_true * 0.3 + rng.standard_normal(n) * 0.5
    group_ids = np.repeat(np.arange(n_queries), docs_per)
    spec = compose_ltr_figure(
        y_true, y_score, group_ids,
        panels_template="NDCG_K NDCG_DIST LIFT MRR_DIST SCORE_BY_REL",
        suptitle=PROD_HEADER,
    )
    _render(out_dir, "05_ltr", spec)


# ---------------------------------------------------------------- 6. quantile
def audit_quantile(out_dir: str) -> None:
    print("[6/7] quantile regression (5-panel grid)...")
    from mlframe.reporting.charts.quantile import compose_quantile_figure
    from scipy import stats
    rng = np.random.default_rng(5)
    n = 5000
    alphas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    K = len(alphas)
    y_true = rng.standard_normal(n) * 50 + 1000
    base_pred = y_true + rng.standard_normal(n) * 30
    # Per-row sigma so widths vary across samples; otherwise all
    # ``q_hi - q_lo`` values collapse to a constant and the WIDTH_DIST
    # histogram has zero data range.
    per_row_sigma = 30 + rng.uniform(-5, 5, n)
    preds_NK = np.empty((n, K))
    for j, a in enumerate(alphas):
        preds_NK[:, j] = base_pred + stats.norm.ppf(a) * per_row_sigma
    spec = compose_quantile_figure(
        y_true, preds_NK, alphas=alphas,
        panels_template="RELIABILITY PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST",
        suptitle=PROD_HEADER,
    )
    _render(out_dir, "06_quantile", spec)


# ---------------------------------------------------------------- 7. temporal
def audit_temporal(out_dir: str) -> None:
    print("[7/7] temporal audit (line)...")
    from mlframe.reporting.charts.temporal import build_temporal_audit_spec

    # The spec builder reads from a TemporalAuditResult duck-typed object
    # (``bins`` list with ``bin_start`` / ``target_rate`` / ``kept``,
    # plus ``target_name`` / ``granularity`` / ``segments``). Fake the
    # shape with a SimpleNamespace -- we don't depend on the audit
    # module here.
    from types import SimpleNamespace
    rng = np.random.default_rng(6)
    n = 200
    bin_starts = np.arange(n).astype(np.int64)
    rates = 0.5 + 0.1 * np.sin(bin_starts / 10) + rng.standard_normal(n) * 0.05
    bins = [
        SimpleNamespace(bin_start=int(t), target_rate=float(r), kept=True)
        for t, r in zip(bin_starts, rates)
    ]
    audit = SimpleNamespace(
        bins=bins, target_name="y_test",
        granularity="day", target_type="binary",
        segments=[],
    )
    spec = build_temporal_audit_spec(audit)
    # spec_builder doesn't take a suptitle; figure title is built from
    # audit attributes. Replace it with the prod-style header for parity.
    spec_with_title = spec.__class__(
        suptitle=PROD_HEADER + "\ntarget rate over time",
        panels=spec.panels,
        figsize=spec.figsize,
    )
    _render(out_dir, "07_temporal", spec_with_title)


def main(argv: Sequence[str] = ()) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default="D:/Temp/chart_audit",
        help="Output directory (default: D:/Temp/chart_audit)",
    )
    args = ap.parse_args(list(argv) if argv else None)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    print(f"Outputs -> {out_dir}\n")

    failures: list[tuple[str, BaseException]] = []
    for name, fn in [
        ("calibration", audit_calibration),
        ("regression", audit_regression),
        ("multiclass", audit_multiclass),
        ("multilabel", audit_multilabel),
        ("ltr", audit_ltr),
        ("quantile", audit_quantile),
        ("temporal", audit_temporal),
    ]:
        try:
            fn(out_dir)
        except Exception as e:
            failures.append((name, e))
            import traceback
            traceback.print_exc()

    print(f"\nDONE. Inspect PNGs in {out_dir}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for name, e in failures:
            print(f"  - {name}: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
