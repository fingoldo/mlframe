"""W3-E multilabel chart fixes: vectorized P/R/F1 parity, ROC chance diagonal, vectorized calib grid."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.multilabel import (
    _per_label_prf1, compose_multilabel_figure,
)


@pytest.fixture
def synth_3label():
    rng = np.random.default_rng(7)
    n = 400
    K = 3
    y_true = rng.integers(0, 2, (n, K)).astype(np.int8)
    base = rng.uniform(0.0, 0.5, (n, K))
    y_proba = np.clip(base + y_true * 0.4 + rng.normal(0, 0.05, (n, K)), 0.01, 0.99)
    return y_true, y_proba, ["spam", "promo", "social"]


class TestPRF1Parity:
    """PERF-13: vectorized per-label P/R/F1 must match sklearn bit-for-bit (zero_division=0)."""

    def test_parity_vs_sklearn_small_K(self):
        from sklearn.metrics import precision_recall_fscore_support

        rng = np.random.default_rng(3)
        n, K = 500, 6
        y_true = rng.integers(0, 2, (n, K)).astype(np.int8)
        y_pred = rng.integers(0, 2, (n, K)).astype(np.int8)
        p, r, f = _per_label_prf1(y_true, y_pred)
        for k in range(K):
            ps, rs, fs, _ = precision_recall_fscore_support(
                y_true[:, k], y_pred[:, k], average="binary", zero_division=0, labels=[0, 1],
            )
            assert p[k] == pytest.approx(ps, abs=1e-12), f"precision col {k}"
            assert r[k] == pytest.approx(rs, abs=1e-12), f"recall col {k}"
            assert f[k] == pytest.approx(fs, abs=1e-12), f"f1 col {k}"

    def test_parity_with_empty_label_columns(self):
        """A column with no positives (precision/recall denominators zero) must give 0, not crash."""
        from sklearn.metrics import precision_recall_fscore_support

        n, K = 200, 4
        y_true = np.zeros((n, K), dtype=np.int8)
        y_pred = np.zeros((n, K), dtype=np.int8)
        y_true[:50, 0] = 1
        y_pred[:40, 0] = 1                       # col 0 has TP; cols 1-3 all-zero
        p, r, f = _per_label_prf1(y_true, y_pred)
        for k in range(K):
            ps, rs, fs, _ = precision_recall_fscore_support(
                y_true[:, k], y_pred[:, k], average="binary", zero_division=0, labels=[0, 1],
            )
            assert p[k] == pytest.approx(ps, abs=1e-12)
            assert r[k] == pytest.approx(rs, abs=1e-12)
            assert f[k] == pytest.approx(fs, abs=1e-12)


class TestReferenceLines:
    """INV-17: multilabel ROC must carry a chance diagonal series."""

    def test_roc_has_chance_diagonal(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="ROC")
        panel = spec.panels[0][0]
        assert panel.series_labels[0] == "chance"
        # K labels + 1 chance series.
        assert len(panel.y) == len(lbl) + 1
        # Chance diagonal is y == x on the shared grid.
        np.testing.assert_allclose(panel.y[0], panel.x)
        assert panel.line_styles[0] == ":"


class TestCalibGridVectorized:
    """INV-33: vectorized calib grid must equal the reference masking loop."""

    def test_matches_reference_loop(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="CALIB_GRID")
        panel = spec.panels[0][0]
        K = len(lbl)
        n_bins = 10
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for k in range(K):
            proba_k = p[:, k]
            true_k = y[:, k].astype(np.float64)
            bin_idx = np.clip(np.digitize(proba_k, edges[1:-1]), 0, n_bins - 1)
            ref = np.full(n_bins, np.nan)
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.any():
                    ref[b] = float(true_k[mask].mean())
            got = panel.y[k + 1]                 # +1 skips the perfect diagonal
            np.testing.assert_allclose(got, ref, equal_nan=True)
