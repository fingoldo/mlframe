"""A/B bench for favorize_unexplored: iterate cat_features directly vs candidate.items() + list-membership.

OLD walked every key of each candidate dict and tested ``param in cat_features`` (a list -> O(len(cat_features))
membership) for every param, including the many numeric CatBoost params the favorization ignores. NEW iterates
the (typically short) ``cat_features`` list directly and pulls ``candidate.get(param)``, doing set-membership only
on the cat values. Bit-identical (same first-occurrence-wins ordering, same multiplicative novelty factor, same
favorized_items count gating normalize_probs).

Run: CUDA_VISIBLE_DEVICES="" python bench_favorize_unexplored_cat_only_loop.py

Measured (n_candidates=5000, 6 cat params, 8 numeric params, 50 prior trials), python 3.14.3:
  OLD 10.85 ms  NEW 5.68 ms  -> 1.91x, bit-identical.
"""

from __future__ import annotations

import sys
import time
from os.path import abspath, join, dirname

import numpy as np
import pandas as pd

sys.path.insert(0, abspath(join(dirname(__file__), "..", "..", "..")))

import logging

logging.disable(logging.CRITICAL)

from mlframe.models.tuning import favorize_unexplored as NEW

_MISS = object()


def _normalize_probs(probs):
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs[:] = 1.0 / len(probs)
        return
    np.divide(probs, total, out=probs)


def OLD(candidates, probs, trials, cat_features, order: int = 1) -> None:
    """Verbatim pre-optimization body (candidate.items() + list membership)."""
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


def _make_workload(ncand: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    cat_features = ["grow_policy", "bootstrap_type", "model_shrink_mode", "sampling_unit", "leaf_estimation_method", "score_function"]
    choices = {c: [f"{c}_{i}" for i in range(8)] for c in cat_features}
    extra = ["learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature", "iterations", "border_count", "min_data_in_leaf"]
    cands = []
    for _ in range(ncand):
        d = {c: str(rng.choice(choices[c])) for c in cat_features}
        for e in extra:
            d[e] = float(rng.random())
        cands.append(d)
    trials = pd.DataFrame([{c: str(rng.choice(choices[c])) for c in cat_features} for _ in range(50)])
    return cands, trials, cat_features, ncand


def _bench(fn, cands, trials, cat_features, ncand, n_iter: int = 200, reps: int = 5):
    p = np.ones(ncand) / ncand
    fn(cands, p, trials, cat_features)  # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n_iter):
            p = np.ones(ncand) / ncand
            fn(cands, p, trials, cat_features)
        best = min(best, (time.perf_counter() - t) / n_iter)
    return best * 1000.0


def main() -> None:
    cands, trials, cat_features, ncand = _make_workload()

    p_old = np.ones(ncand) / ncand
    OLD(cands, p_old, trials, cat_features)
    p_new = np.ones(ncand) / ncand
    NEW(cands, p_new, trials, cat_features)
    identical = np.array_equal(p_old, p_new)

    old_ms = _bench(OLD, cands, trials, cat_features, ncand)
    new_ms = _bench(NEW, cands, trials, cat_features, ncand)
    print(f"OLD {old_ms:.4f} ms  NEW {new_ms:.4f} ms  ->  {old_ms / new_ms:.3f}x")
    print(f"bit-identical probs: {identical}")


if __name__ == "__main__":
    main()
