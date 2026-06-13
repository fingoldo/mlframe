"""Isolated bench: multiclass headline AUC averaging mode (macro vs weighted).

Lever: the multiclass baseline AUC reported by ``_dummy_metrics_pick_plot`` uses
``roc_auc_score(..., multi_class='ovr', average='macro')``. The candidate flip is
``average='weighted'`` (support-weighted per-class AUC).

Honest criterion: a headline AUC must RANK two candidate models correctly. We
build, per scenario+seed, two probabilistic models on the SAME imbalanced
multiclass data:

  * model A: stronger on the MAJORITY class, near-random on minorities.
  * model B: stronger on the MINORITY classes, near-random on the majority.

A model that ranks minorities well is the one that matters under imbalance
(rare classes are the expensive errors). The "correct" headline must give B
the higher score. We count, per averaging mode, how often it ranks B >= A.
Macro weights every class equally (minority signal counts); weighted is
dominated by the majority class's per-class AUC (minority signal drowned).

Run:  python -m mlframe.training.baselines._benchmarks.bench_multiclass_auc_averaging

Verdict (committed): macro is the honest winner -- it rewards minority-class
ranking; weighted hides it. KEEP macro as the default. See JSON dumped to
_results/multiclass_auc_averaging.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _make_data(rng, n, n_classes, majority_frac):
    """Imbalanced multiclass labels: class 0 is the majority."""
    rest = (1.0 - majority_frac) / (n_classes - 1)
    p = np.array([majority_frac] + [rest] * (n_classes - 1))
    y = rng.choice(n_classes, size=n, p=p)
    return y


def _probs_for(y, n_classes, strong_classes, strength, rng):
    """Build calibrated-ish OVR probabilities that rank ``strong_classes`` well
    and the rest near-random. ``strength`` in (0,1]: signal margin."""
    n = len(y)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    for c in range(n_classes):
        margin = strength if c in strong_classes else strength * 0.05
        # push the true-class logit up by ``margin`` where label==c
        logits[y == c, c] += margin * 4.0
    # softmax
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run():
    scenarios = [
        # (n, n_classes, majority_frac)
        (4000, 3, 0.85),
        (4000, 4, 0.80),
        (6000, 5, 0.82),
        (3000, 3, 0.90),
        (8000, 6, 0.78),
    ]
    seeds = [0, 1, 2, 3, 4]
    minority_strength = 0.9
    majority_strength = 0.9

    results = {"macro": {"B_wins": 0, "total": 0}, "weighted": {"B_wins": 0, "total": 0}}
    detail = []

    for (n, k, mf) in scenarios:
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            y = _make_data(rng, n, k, mf)
            labels = np.arange(k)
            minorities = list(range(1, k))
            # model A: strong on majority (class 0) only
            pA = _probs_for(y, k, strong_classes={0}, strength=majority_strength, rng=rng)
            # model B: strong on all minorities
            pB = _probs_for(y, k, strong_classes=set(minorities), strength=minority_strength, rng=rng)

            row = {"n": n, "k": k, "majority_frac": mf, "seed": s}
            for mode in ("macro", "weighted"):
                aucA = roc_auc_score(y, pA, multi_class="ovr", average=mode, labels=labels)
                aucB = roc_auc_score(y, pB, multi_class="ovr", average=mode, labels=labels)
                b_wins = aucB >= aucA
                results[mode]["total"] += 1
                results[mode]["B_wins"] += int(b_wins)
                row[f"{mode}_A"] = round(float(aucA), 4)
                row[f"{mode}_B"] = round(float(aucB), 4)
                row[f"{mode}_B_ranks_higher"] = bool(b_wins)
            detail.append(row)

    for mode in results:
        t = results[mode]["total"]
        results[mode]["B_win_rate"] = results[mode]["B_wins"] / t if t else 0.0

    out = {
        "lever": "multiclass headline AUC averaging (macro vs weighted)",
        "honest_metric": "rank-correctness: how often the minority-strong model B is scored >= the majority-strong model A",
        "results": results,
        "detail": detail,
        "verdict": (
            "KEEP macro" if results["macro"]["B_win_rate"] > results["weighted"]["B_win_rate"]
            else "FLIP to weighted"
        ),
    }

    res_dir = Path(__file__).parent / "_results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "multiclass_auc_averaging.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"results": results, "verdict": out["verdict"]}, indent=2))
    return out


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    run()
