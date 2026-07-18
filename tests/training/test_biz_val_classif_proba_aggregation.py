"""biz_value: geometric (log-average) member-probability blend beats arithmetic mean under log-loss.

Pins the qual-13 flip: ``_ChainEnsemble`` / ``aggregate_member_probas`` default to the logarithmic opinion pool
("geometric"), measured to win the majority of honest-holdout NLL/Brier/ECE cells over diverse probabilistic members
(see ``bench_proba_aggregation_arith_vs_geom``). A regression that silently reverts the default to arithmetic, or breaks
the geometric renormalisation, fails the quantitative win below.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._classif_helpers import aggregate_member_probas, _build_classifier_chain_ensemble


def _nll(y, p, eps=1e-12):
    """Nll."""
    pc = np.clip(p, eps, 1.0)
    return float(-np.log(pc[np.arange(y.shape[0]), y]).mean())


def _make_diverse_members(rng, n=4000, k=2, n_members=5, diversity=1.0):
    """Make diverse members."""
    y = rng.integers(0, k, size=n)
    base = np.zeros((n, k))
    base[np.arange(n), y] = rng.uniform(1.2, 2.2)
    members = []
    for _ in range(n_members):
        logits = base + rng.normal(0.0, diversity, size=(n, k)) + rng.normal(0.0, 0.3, size=(1, k))
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        members.append(p)
    return y, np.stack(members, axis=0)


def test_biz_val_proba_aggregation_geometric_beats_arithmetic_nll():
    """Geometric NLL must beat arithmetic by >=1.20x on diverse calibrated members (measured ~1.29x; 7% margin)."""
    ratios = []
    for seed in (11, 23, 37, 51, 67):
        rng = np.random.default_rng(seed)
        y, stacked = _make_diverse_members(rng)
        pa = aggregate_member_probas(stacked, "arithmetic", simplex=True)
        pg = aggregate_member_probas(stacked, "geometric", simplex=True)
        ratios.append(_nll(y, pa) / _nll(y, pg))
    mean_ratio = float(np.mean(ratios))
    assert mean_ratio >= 1.20, f"geometric should cut NLL >=1.20x; got {mean_ratio:.3f} over {ratios}"
    assert all(r > 1.0 for r in ratios), f"geometric should win every seed; got {ratios}"


def test_biz_val_proba_aggregation_default_is_geometric():
    """The shipped default for the chain-ensemble blend is the logarithmic opinion pool."""
    ens = _build_classifier_chain_ensemble(None, n_labels=3)
    assert ens.proba_aggregation == "geometric"


def test_biz_val_proba_aggregation_simplex_renorm_and_multilabel_independence():
    """Geometric blend renormalises across classes when simplex=True, but keeps labels independent when False."""
    stacked = np.array([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]])
    g = aggregate_member_probas(stacked, "geometric", simplex=True)
    assert np.allclose(g.sum(axis=1), 1.0)
    ml = np.array([[[0.9, 0.1]], [[0.7, 0.3]]])  # 1 row, 2 independent labels
    gm = aggregate_member_probas(ml, "geometric", simplex=False)
    # Per-label complement renorm: label0 = geo(.9,.7)/(geo(.9,.7)+geo(.1,.3)).
    geo0 = np.sqrt(0.9 * 0.7)
    comp0 = np.sqrt(0.1 * 0.3)
    assert gm[0, 0] == pytest.approx(geo0 / (geo0 + comp0))


def test_biz_val_proba_aggregation_unknown_method_raises():
    """Biz val proba aggregation unknown method raises."""
    with pytest.raises(ValueError, match="unknown method"):
        aggregate_member_probas(np.ones((2, 3, 2)) / 2, "harmonic", simplex=True)
