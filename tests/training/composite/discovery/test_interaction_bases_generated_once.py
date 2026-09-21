"""The interaction-base step must synthesise its product/ratio columns once, not once to score and again to select.

``discover_interaction_bases`` scored every pairwise column over the top bases and then regenerated all of them to pick
the qualifying ones; both passes produced identical arrays. The scorer's columns are now reused.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery import _interaction_bases as ib


def _candidates(n: int = 4000, seed: int = 0):
    """Four base columns and a target driven by the product of two of them."""
    rng = np.random.default_rng(seed)
    cands = {f"b{i}": rng.normal(size=n) + (3.0 if i % 2 else 0.0) for i in range(4)}
    y = cands["b0"] * cands["b1"] + 0.1 * rng.normal(size=n)
    return cands, y


def test_the_columns_are_synthesised_once_per_discovery(monkeypatch):
    """One ``generate_interaction_bases`` call serves both scoring and selection."""
    calls = {"n": 0}
    real = ib.generate_interaction_bases

    def counting(*args, **kwargs):
        """Count every synthesis pass."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ib, "generate_interaction_bases", counting)
    cands, y = _candidates()
    out, records = ib.discover_interaction_bases(cands, y)
    assert out and records, "the product target must surface at least one interaction base"
    assert calls["n"] == 1, f"the interaction columns were synthesised {calls['n']} times"


def test_the_surfaced_columns_equal_a_fresh_synthesis():
    """Reuse changes nothing: every surfaced column equals what a fresh synthesis gives for the same name."""
    cands, y = _candidates()
    out, _records = ib.discover_interaction_bases(cands, y)
    fresh, _prov = ib.generate_interaction_bases(cands, top_k=ib._INTERACTION_TOP_K_DEFAULT, forbid_self_pairs=True)
    for name, col in out.items():
        np.testing.assert_array_equal(col, fresh[name])


def test_the_public_scorer_still_returns_records_only():
    """``score_interaction_pairs`` keeps its public contract: a list of scoring records."""
    cands, y = _candidates()
    scored = ib.score_interaction_pairs(cands, y)
    assert isinstance(scored, list) and scored and {"synth_name", "gain", "qualifies"} <= set(scored[0])
