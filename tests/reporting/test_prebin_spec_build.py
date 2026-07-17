"""PERF-4: histogram-style panels pre-bin at spec-build so the FigureSpec never carries length-n arrays.

For a 1e6-row input the built HistogramPanelSpec must carry O(bins) data (counts + bin_centers + bin_width),
not O(n) raw values -- relevant under keep_figure_handles + large notebooks where the spec is retained.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.ltr import compose_ltr_figure
from mlframe.reporting.charts.multilabel import compose_multilabel_figure
from mlframe.reporting.charts.quantile import compose_quantile_figure
from mlframe.reporting.spec import HistogramPanelSpec

_N = 1_000_000
_BIN_CAP = 64  # generous ceiling: every pre-binned panel here uses <= 30 bins


def _assert_prebinned(panel, n):
    assert isinstance(panel, HistogramPanelSpec)
    assert panel.bin_centers is not None, "spec carries no bin_centers -> still raw"
    assert panel.bin_width is not None and panel.bin_width > 0.0
    # The three pre-binned arrays are all O(bins); none may be length-n.
    assert len(panel.values) == len(panel.bin_centers)
    assert len(panel.values) <= _BIN_CAP
    assert len(panel.values) < n // 1000, "spec data is not O(bins)"


def test_multilabel_jaccard_hamming_prebinned_at_1e6():
    rng = np.random.default_rng(0)
    K = 4
    y_true = rng.integers(0, 2, (_N, K)).astype(np.int8)
    y_proba = rng.random((_N, K))
    for tok in ("JACCARD_DIST", "HAMMING_DIST"):
        spec = compose_multilabel_figure(y_true, y_proba, [f"l{k}" for k in range(K)], panels_template=tok)
        _assert_prebinned(spec.panels[0][0], _N)


def test_quantile_width_pit_prebinned_at_1e6():
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    y = rng.standard_normal(_N)
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    # Heteroscedastic preds so interval widths are non-degenerate (true pre-bin, not a single padded bin).
    scale = 0.5 + np.abs(rng.standard_normal(_N))
    preds = np.column_stack([y * 0.0 + scale * norm.ppf(a) for a in alphas])
    for tok in ("WIDTH_DIST", "PIT_HIST"):
        spec = compose_quantile_figure(y, preds, alphas, panels_template=tok)
        _assert_prebinned(spec.panels[0][0], _N)


def test_sensor_rejects_pre_fix_raw_spec():
    """Proves the regression sensor fails on the pre-fix shape: a HistogramPanelSpec carrying raw length-n values
    with no bin_centers (what the builders emitted before the spec-build pre-bin) must NOT pass _assert_prebinned."""
    n = 50_000
    raw = HistogramPanelSpec(values=np.random.default_rng(0).random(n), bins=20)
    with pytest.raises(AssertionError):
        _assert_prebinned(raw, n)


def test_ltr_mrr_prebinned_at_1e6():
    # 1e6 rows across ~100k queries (avg 10 docs/query).
    rng = np.random.default_rng(0)
    n_queries = 100_000
    sizes = rng.integers(2, 19, n_queries)
    total = int(sizes.sum())
    group_ids = np.repeat(np.arange(n_queries), sizes)
    rels = rng.integers(0, 4, total)
    scores = rels.astype(float) + rng.normal(0, 0.7, total)
    spec = compose_ltr_figure(rels, scores, group_ids, panels_template="MRR_DIST")
    _assert_prebinned(spec.panels[0][0], total)
