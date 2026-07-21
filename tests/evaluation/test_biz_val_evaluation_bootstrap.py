"""biz_value: BCa bootstrap CI achieves closer-to-nominal coverage than the percentile interval on a skewed metric.

The default ``bootstrap_metric`` method was flipped percentile -> BCa (qual-5). The quantitative win is COVERAGE:
on a skewed / bounded sampling distribution (ROC-AUC near 1.0) the plain percentile interval UNDER-COVERS the
nominal 95% level, while BCa (bias-corrected + accelerated) recovers close-to-nominal coverage. This test runs a
small Monte-Carlo coverage experiment and asserts BCa's miscoverage gap is materially smaller than percentile's,
so a regression that silently reverts the default (or breaks the z0 / acceleration term) trips by FAILING the win.

Measured (bench `mlframe/evaluation/_benchmarks/bench_bootstrap_ci_coverage.py`, 600 trials x 5 seeds, AUC~0.97 @
n=150): percentile mean coverage 0.919 (gap 0.031), BCa mean coverage 0.948 (gap 0.002) -- BCa wins 10/10 cells.
The thresholds below are set with wide margin against the much smaller MC sample used here for test-speed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mlframe.evaluation.bootstrap import bootstrap_metric

pytestmark = [pytest.mark.fast]

_SEP = 2.66  # true AUC = Phi(sep/sqrt2) ~ 0.97 (strongly left-skewed, capped at 1.0)


def _draw_auc(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(y, score)`` (after 3 setup steps)."""
    y = rng.integers(0, 2, size=n)
    score = rng.normal(loc=_SEP * y, scale=1.0)
    if y.min() == y.max():
        y[0], y[1] = 0, 1
    return y, score


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Test helper: ranks = stats.rankdata(s, method='average'); pos = y == 1; n_pos = int(pos.sum())."""
    ranks = stats.rankdata(s, method="average")
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def test_biz_val_bootstrap_bca_covers_closer_to_nominal_than_percentile_on_skewed_auc():
    """BCa's |coverage - 0.95| must beat percentile's on a skewed-AUC coverage experiment.

    Floor: BCa gap at least 0.01 smaller than percentile's (measured full-bench delta ~0.029; wide margin for
    the reduced MC sample). Also: BCa coverage must clear 0.92 while percentile under-covers below it -- catching
    a silent revert of the default to percentile.
    """
    true_auc = float(stats.norm.cdf(_SEP / np.sqrt(2.0)))
    rng = np.random.default_rng(2026)
    n_trials, n_sample, n_boot = 200, 150, 500
    hits = {"percentile": 0, "bca": 0}
    for _ in range(n_trials):
        y, score = _draw_auc(rng, n_sample)
        seed = int(rng.integers(0, 2**31))
        r_pct = bootstrap_metric(y, score, metric_fn=_auc, n_bootstrap=n_boot, random_state=seed, method="percentile")
        r_bca = bootstrap_metric(y, score, metric_fn=_auc, n_bootstrap=n_boot, random_state=seed, method="bca")
        if r_pct["lo"] <= true_auc <= r_pct["hi"]:
            hits["percentile"] += 1
        if r_bca["lo"] <= true_auc <= r_bca["hi"]:
            hits["bca"] += 1
    cov_pct = hits["percentile"] / n_trials
    cov_bca = hits["bca"] / n_trials
    gap_pct = abs(cov_pct - 0.95)
    gap_bca = abs(cov_bca - 0.95)
    assert gap_bca + 0.01 <= gap_pct, f"BCa should cover closer to nominal: cov pct={cov_pct:.3f} (gap {gap_pct:.3f}) vs bca={cov_bca:.3f} (gap {gap_bca:.3f})"
    assert cov_bca >= 0.92, f"BCa coverage {cov_bca:.3f} should be near nominal 0.95"
    assert cov_pct < cov_bca, f"percentile {cov_pct:.3f} should under-cover relative to BCa {cov_bca:.3f}"
