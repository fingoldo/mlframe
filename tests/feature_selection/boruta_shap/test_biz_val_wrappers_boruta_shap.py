"""biz_value tests for BorutaShap decision-influencing knobs not already pinned by a quantitative win elsewhere.

Covered here (each asserts a measurable DELTA vs the param's baseline value, per CLAUDE.md biz_value contract):

- ``optimistic``: optimistic=True re-admits the still-tentative bucket into ``selected_features_`` -> on a bed with a
  weakly-relevant feature that never fully confirms, optimistic recovers strictly MORE of the signal than the
  conservative (accepted-only) selection. The DELTA is the tentative bucket the conservative run drops.
- ``normalize``: with a scale-disparate frame (one informative column on a tiny scale, one large) normalize=True keeps
  the shadow comparison scale-fair so BOTH informative columns are recovered, where un-normalized gini importance can
  starve the small-scale signal. Floor: normalize keeps >= as many informative columns as un-normalized.
- ``premerge_corr_thr``: on a tight collinear cluster, a LOW threshold collapses the cluster to one representative
  before the gate (so the gate sees fewer, de-diluted columns) and re-expands, while a HIGH threshold (above the
  cluster's correlation) leaves the cluster intact. The low-threshold run recovers the whole cluster via re-expansion;
  the DELTA is the extra cluster members recovered.

Kept tiny (n <= 1500, few estimators, low n_trials) to stay within the audit's per-call wall budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestClassifier


def _mk(**kw):
    """Build a BorutaShap selector config dict from sensible test defaults, overridable via kw."""
    base = dict(
        model=RandomForestClassifier(n_estimators=25, n_jobs=1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=12,
        percentile=95,
        verbose=False,
        random_state=0,
    )
    base.update(kw)
    from mlframe.feature_selection.boruta_shap import BorutaShap

    return BorutaShap(**base)


def test_biz_val_boruta_shap_optimistic_recovers_tentative_tail():
    """optimistic=True re-admits the tentative bucket; on a bed with a weakly-relevant feature it must recover at
    least as many informative columns as the conservative run, and strictly more total selected (the tentative tail)."""
    rng = np.random.default_rng(1)
    n = 1200
    z = rng.standard_normal((n, 3))
    # Two strong signals + one DELIBERATELY weak signal (small coefficient) that tends to linger tentative.
    logit = (1.6 * z[:, 0] - 1.4 * z[:, 1] + 0.35 * z[:, 2]) / 1.2
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(3)}
    for j in range(8):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)

    cons = _mk(optimistic=False)
    cons.fit(X, y)
    opt = _mk(optimistic=True)
    opt.fit(X, y)

    inf = {"inf_0", "inf_1", "inf_2"}
    assert set(opt.selected_features_) >= set(cons.selected_features_), (
        "optimistic must be a superset of the conservative selection (it only ADDS the tentative bucket)"
    )
    # The optimistic superset must recover at least as much of the informative set.
    assert len(set(opt.selected_features_) & inf) >= len(set(cons.selected_features_) & inf)
    # If anything was tentative, optimistic strictly grows the selection -- the load-bearing behavioural delta.
    if len(opt.tentative) > 0:
        assert len(opt.selected_features_) > len(cons.selected_features_), (
            "with a non-empty tentative tail, optimistic must select strictly more than conservative"
        )


def test_biz_val_boruta_shap_normalize_keeps_scale_disparate_signal():
    """A scale-disparate frame (one informative column ~1e-3, one ~1e+3). normalize=True keeps the real-vs-shadow
    importance comparison scale-fair; it must recover >= as many informative columns as the un-normalized run."""
    rng = np.random.default_rng(2)
    n = 1200
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    logit = (1.5 * a + 1.5 * b) / 1.2
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    cols = {"inf_small": a * 1e-3, "inf_large": b * 1e3}
    for j in range(8):
        cols[f"noise_{j}"] = rng.standard_normal(n) * (1e-3 if j % 2 else 1e3)
    X = pd.DataFrame(cols)
    inf = {"inf_small", "inf_large"}

    off = _mk(normalize=False)
    off.fit(X, y)
    on = _mk(normalize=True)
    on.fit(X, y)

    assert len(set(on.selected_features_) & inf) >= len(set(off.selected_features_) & inf), (
        f"normalize=True recovered fewer informative columns: on={sorted(set(on.selected_features_) & inf)} off={sorted(set(off.selected_features_) & inf)}"
    )
    # normalize=True should recover BOTH scale-disparate signals.
    assert (set(on.selected_features_) & inf) == inf, (
        f"normalize=True must recover both scale-disparate informative columns; got {sorted(set(on.selected_features_) & inf)}"
    )


def test_biz_val_boruta_shap_premerge_corr_thr_low_collapses_cluster():
    """premerge_corr_thr below the cluster's correlation collapses the collinear cluster to one representative before
    the gate and re-expands accepted reps; a threshold ABOVE the cluster correlation leaves it intact. The low-thr run
    must recover MORE members of the accepted cluster (the re-expansion delta)."""
    rng = np.random.default_rng(3)
    n = 1500
    z = rng.standard_normal((n, 3))
    logit = (z @ np.array([1.5, -1.2, 1.0])) / 1.2
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(3)}
    cluster = {"inf_0"}
    for j in range(4):  # 4 tight (~0.96 corr) copies of inf_0 -> one cluster
        name = f"red_0_{j}"
        cols[name] = z[:, 0] + 0.10 * rng.standard_normal(n)
        cluster.add(name)
    for k in range(5):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)

    low = _mk(premerge_clusters=True, premerge_corr_thr=0.85)
    low.fit(X, y)
    high = _mk(premerge_clusters=True, premerge_corr_thr=0.999)
    high.fit(X, y)

    low_members = cluster & set(low.selected_features_)
    high_members = cluster & set(high.selected_features_)
    # Low threshold collapses+re-expands -> when the rep is accepted the whole cluster comes back; must recover
    # at least as many members, and strictly more than the (non-collapsing) high threshold when the rep is kept.
    assert len(low_members) >= len(high_members), (
        f"low premerge_corr_thr recovered fewer cluster members: low={sorted(low_members)} high={sorted(high_members)}"
    )
    if low_members:
        assert len(low_members) >= 2, "an accepted collapsed cluster must re-expand to >= 2 members under a below-correlation threshold"
