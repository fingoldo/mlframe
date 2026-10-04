"""Cluster and moving-block bootstrap: intervals must reflect within-group and serial dependence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.evaluation.bootstrap import bootstrap_metric, bootstrap_metric_clustered, bootstrap_metrics_clustered
from mlframe.metrics.core import fast_roc_auc
from mlframe.training.honest_diagnostics import _bootstrap_block, _bootstrap_block_clustered, run_honest_diagnostics


def _panel(rng: np.random.Generator, n_groups: int = 40, per_group: int = 50):
    """One grouped panel whose label rate and score both carry a shared group effect."""
    g = np.repeat(np.arange(n_groups), per_group)
    eff = rng.normal(0.0, 1.5, n_groups)[g]
    y = (rng.random(n_groups * per_group) < 1.0 / (1.0 + np.exp(-eff))).astype(np.int64)
    s = eff * 0.8 + rng.normal(0.0, 1.0, n_groups * per_group) + 0.8 * y
    return y, s, g


def test_cluster_bootstrap_width_matches_true_sd_while_iid_is_too_narrow():
    """Across fresh grouped panels the cluster CI implies an SD near the true AUC SD; the i.i.d. CI implies less than half of it."""
    rng = np.random.default_rng(0)
    aucs, w_iid, w_cluster = [], [], []
    for k in range(40):
        y, s, g = _panel(rng)
        aucs.append(fast_roc_auc(y.astype(np.float64), s))
        r_iid = bootstrap_metric(y, s, fast_roc_auc, n_bootstrap=200, method="percentile", random_state=k)
        r_cl = bootstrap_metric_clustered(y, s, fast_roc_auc, n_bootstrap=200, groups=g, random_state=k)
        w_iid.append(r_iid["hi"] - r_iid["lo"])
        w_cluster.append(r_cl["hi"] - r_cl["lo"])
    true_sd = float(np.std(aucs))
    sd_iid = float(np.mean(w_iid)) / 3.92
    sd_cluster = float(np.mean(w_cluster)) / 3.92
    assert sd_iid < 0.6 * true_sd
    assert 0.75 * true_sd < sd_cluster < 1.3 * true_sd


def test_block_bootstrap_widens_interval_for_autocorrelated_series():
    """A moving-block interval on an AR(1) error series is wider than the i.i.d. interval for the same metric."""
    rng = np.random.default_rng(1)
    n = 3000
    e = np.empty(n)
    e[0] = 0.0
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal()
    y = e
    p = np.zeros(n)

    def _mse(a, b):
        """Mean squared error."""
        return float(np.mean((a - b) ** 2))

    iid = bootstrap_metric(y, p, _mse, n_bootstrap=300, method="percentile", random_state=0)
    blk = bootstrap_metric_clustered(y, p, _mse, n_bootstrap=300, block_length=60, random_state=0)
    assert (blk["hi"] - blk["lo"]) > 1.5 * (iid["hi"] - iid["lo"])


def test_bootstrap_metrics_groups_returns_every_metric_and_names_scheme():
    """``bootstrap_metrics_clustered(groups=...)`` yields point/lo/hi per metric, tagged with the cluster scheme, and is seed-deterministic."""
    y, s, g = _panel(np.random.default_rng(2))
    fns = {"roc_auc": fast_roc_auc}
    a = bootstrap_metrics_clustered(y, s, fns, n_bootstrap=100, groups=g, random_state=5)
    b = bootstrap_metrics_clustered(y, s, fns, n_bootstrap=100, groups=g, random_state=5)
    assert a["roc_auc"]["resampling"].startswith("cluster")
    assert a["roc_auc"]["lo"] <= a["roc_auc"]["point"] <= a["roc_auc"]["hi"]
    assert a["roc_auc"]["lo"] == b["roc_auc"]["lo"]


def test_bootstrap_metrics_groups_length_mismatch_raises():
    """A group vector of the wrong length is a loud error, never a silent i.i.d. fallback."""
    y, s, g = _panel(np.random.default_rng(3))
    with pytest.raises(ValueError, match="groups length"):
        bootstrap_metrics_clustered(y, s, {"roc_auc": fast_roc_auc}, n_bootstrap=10, groups=g[:-1])


def test_honest_block_uses_group_resampling_and_tags_iid_otherwise():
    """``_bootstrap_block`` widens the AUC interval when given groups and labels the scheme on every metric entry."""
    y, s, g = _panel(np.random.default_rng(4))
    probs = 1.0 / (1.0 + np.exp(-s))
    iid = _bootstrap_block(y, probs, rng_seed=1)
    grouped = _bootstrap_block_clustered(y, probs, 1, g, None)
    assert iid["roc_auc"]["resampling"] == "iid"
    assert grouped["roc_auc"]["resampling"].startswith("cluster")
    w_iid = iid["roc_auc"]["ci_hi"] - iid["roc_auc"]["ci_lo"]
    w_grp = grouped["roc_auc"]["ci_hi"] - grouped["roc_auc"]["ci_lo"]
    assert w_grp > 1.5 * w_iid
    assert set(grouped) == {"roc_auc", "brier", "log_loss", "ece"}


def test_run_honest_diagnostics_plumbs_ctx_group_ids_of_the_test_split():
    """The aggregator slices ``ctx.group_ids`` by ``ctx.test_idx`` and the stamped bootstrap entry reports cluster resampling."""
    y, s, g = _panel(np.random.default_rng(5))
    n = y.shape[0]
    group_ids_full = np.concatenate([np.full(100, -1), g + 1000])
    test_idx = np.arange(100, 100 + n)
    entry = SimpleNamespace(model_name="m", model=object(), test_target=y, test_probs=1.0 / (1.0 + np.exp(-s)), oof_probs=None, oof_target=None)
    ctx = SimpleNamespace(test_idx=test_idx, group_ids=group_ids_full, timestamps=None, split_config=None, reporting_config=None)
    payload = run_honest_diagnostics(ctx, {"BINARY_CLASSIFICATION": {"t": [entry]}}, {})
    cis = next(iter(payload["bootstrap_ci"].values()))
    assert cis["roc_auc"]["resampling"].startswith("cluster(groups=40")
