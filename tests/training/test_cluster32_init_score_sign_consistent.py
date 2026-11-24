"""Cluster 32 regression: a negative init_score delta counts as sufficient, as the module's note says (audit 2026-11-20, finding 32-F15).

The note says a negative delta means the init_score baseline already matches raw. The code used ``abs(delta)``, so a strongly better init_score
was treated as insufficient and reported as high potential. The test pins that a large negative delta is not high_potential.
"""

from types import SimpleNamespace

from mlframe.training.baselines._baseline_diagnostics_recommend import _build_recommendation


def _self():
    """Stand-in carrying the thresholds the classifier reads."""
    cfg = SimpleNamespace(
        high_potential_min_dominance_pct=5.0,
        marginal_threshold_pct=1.0,
        init_score_optimal_threshold_pct=1.0,
    )
    return SimpleNamespace(config=cfg)


def test_strongly_negative_init_delta_is_sufficient():
    """init_score far better than raw (delta -50%) is not reported as high_potential (32-F15)."""
    ablation = [SimpleNamespace(name="f", delta_pct=10.0)]
    verdict, _reason = _build_recommendation(_self(), ablation, SimpleNamespace(delta_vs_raw_pct=-50.0))
    assert verdict != "high_potential"
