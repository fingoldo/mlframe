"""Tests for per-query NDCG bootstrap CI (charts/ltr.py::bootstrap_ndcg_ci + panels).

Covers: the percentile CI brackets the per-query mean, the vectorised (B, n_eff)
gather is deterministic + degenerate-safe (0 / 1 query), the query cap bounds the
per-resample gather, the NDCG_DIST / NDCG_BY_QSIZE panel titles carry the CI, and
biz_value -- the CI narrows as the number of QUERIES grows (resampling the right unit).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.ltr import (
    _ndcg_by_qsize_panel,
    _ndcg_dist_panel,
    bootstrap_ndcg_ci,
)

# ----------------------------------------------------------------------------
# Unit: bootstrap CI helper
# ----------------------------------------------------------------------------


def test_ci_brackets_mean():
    """Ci brackets mean."""
    rng = np.random.default_rng(0)
    vals = rng.beta(5, 2, size=500)  # per-query NDCG in (0,1)
    mean, lo, hi = bootstrap_ndcg_ci(vals)
    assert lo <= mean <= hi
    assert mean == pytest.approx(vals.mean(), abs=1e-12)


def test_nan_queries_dropped():
    """Nan queries dropped."""
    vals = np.array([0.5, np.nan, 0.7, np.nan, 0.9])
    mean, _lo, _hi = bootstrap_ndcg_ci(vals)
    assert mean == pytest.approx(np.mean([0.5, 0.7, 0.9]), abs=1e-12)


def test_empty_and_single_query():
    """Empty and single query."""
    assert all(np.isnan(v) for v in bootstrap_ndcg_ci(np.array([np.nan, np.nan])))
    m, lo, hi = bootstrap_ndcg_ci(np.array([0.42]))
    assert m == lo == hi == 0.42


def test_deterministic_seed():
    """Deterministic seed."""
    vals = np.random.default_rng(1).random(300)
    assert bootstrap_ndcg_ci(vals, seed=7) == bootstrap_ndcg_ci(vals, seed=7)


def test_query_cap_bounds_gather(monkeypatch):
    """Query cap bounds gather."""
    import mlframe.reporting.charts.ltr as ltr_mod

    monkeypatch.setattr(ltr_mod, "_BOOTSTRAP_QUERY_CAP", 1000)
    vals = np.random.default_rng(2).random(50_000)
    # Should not allocate a (B, 50000) gather; just verify it returns a sane CI quickly.
    mean, lo, hi = bootstrap_ndcg_ci(vals, n_boot=200)
    assert lo <= mean <= hi


# ----------------------------------------------------------------------------
# Unit: panel titles carry the CI
# ----------------------------------------------------------------------------


def _synth_ltr(n_queries=400, docs=6, seed=0):
    """Helper: Synth ltr."""
    rng = np.random.default_rng(seed)
    gids, rels, scores = [], [], []
    for q in range(n_queries):
        rels_q = rng.integers(0, 3, docs)
        # Score correlated with relevance so NDCG is non-trivial.
        scores_q = rels_q + rng.standard_normal(docs) * 0.5
        gids.extend([q] * docs)
        rels.extend(rels_q.tolist())
        scores.extend(scores_q.tolist())
    return np.array(rels), np.array(scores, dtype=float), np.array(gids)


def test_ndcg_dist_title_carries_ci():
    """Ndcg dist title carries ci."""
    rels, scores, gids = _synth_ltr()
    panel = _ndcg_dist_panel(rels, scores, gids, shared={})
    assert "95% CI" in panel.title


def test_ndcg_by_qsize_title_and_bins_carry_ci():
    """Ndcg by qsize title and bins carry ci."""
    rels, scores, gids = _synth_ltr()
    panel = _ndcg_by_qsize_panel(rels, scores, gids, shared={})
    assert "overall 95% CI" in panel.title
    assert any("CI[" in c for c in panel.categories)


# ----------------------------------------------------------------------------
# biz_value: CI narrows as query count grows
# ----------------------------------------------------------------------------


def test_biz_val_ci_narrows_with_query_count():
    """Resampling QUERIES, the CI half-width must shrink ~1/sqrt(n_queries). Going from 200 to
    3200 queries (16x) should cut the half-width by roughly sqrt(16)=4x. Floor: ratio >= 3.0
    (below the ~4x theoretical, above bootstrap noise). A regression that resamples rows or a
    fixed-width CI trips this."""
    rng = np.random.default_rng(11)

    def _half_width(nq):
        """Helper: Half width."""
        vals = rng.beta(5, 2, size=nq)  # per-query NDCG draws
        _, lo, hi = bootstrap_ndcg_ci(vals, seed=3)
        return (hi - lo) / 2.0

    w_small = _half_width(200)
    w_large = _half_width(3200)
    assert w_small / w_large >= 3.0, (w_small, w_large)


def test_biz_val_ci_contains_true_mean_of_known_distribution():
    """For a beta(5,2) per-query NDCG distribution (true mean = 5/7 ~ 0.714), the 95% bootstrap
    CI at 4000 queries must contain the true mean. Pins that the CI is honest (covers truth)."""
    true_mean = 5.0 / 7.0
    vals = np.random.default_rng(21).beta(5, 2, size=4000)
    _mean, lo, hi = bootstrap_ndcg_ci(vals, seed=5)
    assert lo <= true_mean <= hi, (lo, true_mean, hi)
