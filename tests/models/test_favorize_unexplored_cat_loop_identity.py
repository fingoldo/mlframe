"""Pins favorize_unexplored's cat-features-only loop to be bit-identical to the prior candidate.items() body.

The optimization iterates ``cat_features`` directly rather than every candidate key; this test guards against a
future change silently altering the first-occurrence-wins novelty weighting or the normalize gating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.models.tuning import favorize_unexplored


def _normalize_probs(probs):
    """Helper: Normalize probs."""
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs[:] = 1.0 / len(probs)
        return
    np.divide(probs, total, out=probs)


def _old_favorize(candidates, probs, trials, cat_features, order: int = 1) -> None:
    """Verbatim pre-optimization reference body."""
    if len(cat_features) == 0 or len(trials) == 0:
        return
    already_sampled = {col: set(trials[col].unique().tolist()) for col in cat_features}
    newly_seen = {col: set() for col in cat_features}
    favorized_items = []
    for i in range(len(probs)):
        candidate = candidates[i]
        novel_factor = 1.0
        for param, value in candidate.items():
            if param in cat_features:
                if value not in already_sampled[param] and value not in newly_seen[param]:
                    novel_factor *= 2.0
                    favorized_items.append({param: value})
                    newly_seen[param].add(value)
        if novel_factor > 1.0:
            probs[i] *= novel_factor
    if len(favorized_items) > 0:
        _normalize_probs(probs)


def _make_workload(ncand: int, seed: int):
    """Helper: Make workload."""
    rng = np.random.default_rng(seed)
    cat_features = ["grow_policy", "bootstrap_type", "model_shrink_mode", "score_function"]
    choices = {c: [f"{c}_{i}" for i in range(6)] for c in cat_features}
    extra = ["learning_rate", "depth", "l2_leaf_reg"]
    cands = []
    for _ in range(ncand):
        d = {c: str(rng.choice(choices[c])) for c in cat_features}
        for e in extra:
            d[e] = float(rng.random())
        cands.append(d)
    trials = pd.DataFrame([{c: str(rng.choice(choices[c])) for c in cat_features} for _ in range(20)])
    return cands, trials, cat_features


def test_favorize_unexplored_bit_identical_to_legacy_body():
    """Favorize unexplored bit identical to legacy body."""
    for seed in (0, 1, 7, 42):
        cands, trials, cat_features = _make_workload(ncand=400, seed=seed)
        p_old = np.ones(len(cands)) / len(cands)
        _old_favorize(cands, p_old, trials, cat_features)
        p_new = np.ones(len(cands)) / len(cands)
        favorize_unexplored(cands, p_new, trials, cat_features)
        assert np.array_equal(p_old, p_new), f"divergence at seed={seed}"


def test_favorize_unexplored_handles_missing_cat_key():
    """A candidate missing a cat_features key must be skipped (no KeyError), matching the old ``param in`` guard."""
    cat_features = ["a", "b"]
    cands = [{"a": "x"}, {"a": "y", "b": "z"}]  # first candidate omits "b"
    trials = pd.DataFrame({"a": ["x"], "b": ["z"]})
    probs = np.ones(2) / 2

    p_old = probs.copy()
    _old_favorize(cands, p_old, trials, cat_features)
    p_new = probs.copy()
    favorize_unexplored(cands, p_new, trials, cat_features)
    assert np.array_equal(p_old, p_new)


def test_favorize_unexplored_noop_on_empty_trials():
    """Favorize unexplored noop on empty trials."""
    cat_features = ["a"]
    cands = [{"a": "x"}]
    probs = np.array([1.0])
    favorize_unexplored(cands, probs, pd.DataFrame({"a": []}), cat_features)
    assert probs[0] == 1.0
