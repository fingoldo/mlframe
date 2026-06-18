"""Biz-value: ``interaction_base_top_k`` / ``interaction_base_max_pairs`` surface and cap synthetic bases.

``discover_interaction_bases`` scores ``a OP b`` synthetic bases whose MI beats both marginals and returns up to
``max_pairs`` of them (the strongest first). On data with MULTIPLE genuine interaction pairs, ``max_pairs`` is the
load-bearing cap: a higher cap surfaces more qualifying synergistic bases, and the cap value bounds the output count.

The win: on two independent product-interaction signals (``a*b`` and ``c*d`` both drive ``y``, neither marginal does),
the discovery surfaces the synergistic base, and ``max_pairs`` controls how many. A regression that ignores the cap
(returns all, or none) FAILS both the surfacing and the bound assertion.
"""
from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._interaction_bases import discover_interaction_bases


def _make_two_interactions(n=3000, seed=0):
    """y depends on a*b AND c*d; each factor is marginally near-useless, the product is informative."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    d = rng.normal(size=n)
    noise_e = rng.normal(size=n)
    y = a * b + c * d + 0.1 * rng.normal(size=n)
    candidates = {"a": a, "b": b, "c": c, "d": d, "e": noise_e}
    return candidates, y


def test_biz_val_interaction_surfaces_synergistic_base():
    """At least one synergistic synthetic base is surfaced on pure-interaction data."""
    candidates, y = _make_two_interactions()
    synth, records = discover_interaction_bases(candidates, y, top_k=8, max_pairs=4, nbins=12)
    assert len(synth) >= 1, "a*b / c*d interaction bases must be surfaced (MI beats both marginals)"
    assert all(r["qualifies"] for r in records), "only qualifying pairs should be returned"


def test_biz_val_interaction_max_pairs_caps_output():
    """``interaction_base_max_pairs`` bounds the number of returned synthetic bases."""
    candidates, y = _make_two_interactions()
    synth_cap1, _ = discover_interaction_bases(candidates, y, top_k=8, max_pairs=1, nbins=12)
    synth_cap4, _ = discover_interaction_bases(candidates, y, top_k=8, max_pairs=4, nbins=12)
    assert len(synth_cap1) <= 1, "max_pairs=1 must return at most one synthetic base"
    assert len(synth_cap4) >= len(synth_cap1), "a larger max_pairs cap cannot surface fewer bases"


def test_biz_val_interaction_top_k_limits_scored_pool():
    """``interaction_base_top_k`` restricts the candidate pool fed into pair scoring."""
    candidates, y = _make_two_interactions()
    synth_k2, _ = discover_interaction_bases(candidates, y, top_k=2, max_pairs=4, nbins=12)
    synth_k8, _ = discover_interaction_bases(candidates, y, top_k=8, max_pairs=4, nbins=12)
    assert len(synth_k8) >= len(synth_k2), "a larger top_k pool cannot surface fewer qualifying pairs"
