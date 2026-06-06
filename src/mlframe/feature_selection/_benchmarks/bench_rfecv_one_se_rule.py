"""Bench: RFECV 'auto' resolved rule one_se_max vs one_se_min for tree downstreams (A1-08).

RFECV's 'auto' default resolves to 'one_se_max' (LARGEST N within 1 SE of the best CV mean) -- recall-oriented.
A1-08 asks whether 'one_se_min' (SMALLEST N within 1 SE; parsimonious) is preferable for tree downstreams,
given the cluster-medoid pre-reduction already protects recall. Compares OOS AUC + n_features across seeds.

Run:
    python -m mlframe.feature_selection._benchmarks.bench_rfecv_one_se_rule

Verdict (this machine, seeds 0-5, CatBoost downstream): default stays 'auto' -> 'one_se_max'.
  one_se_max: OOS AUC mean 0.9779 (std 0.0117), n_features mean 34.8
  one_se_min: OOS AUC mean 0.9762 (std 0.0137), n_features mean 21.2
  one_se_min beats one_se_max on 3/6 seeds (a TIE, not a majority).
one_se_min is much more parsimonious (~40% fewer features at -0.0018 AUC) but does NOT win OOS on the majority
of seeds, so it is not promoted to the default. The resolved rule is now surfaced via RFECV.resolved_n_features_rule_
so a report can show which rule actually fired. Pin n_features_selection_rule='one_se_min' when parsimony matters
more than the last 0.002 of recall-side AUC.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers import RFECV


def _make(seed, n=2500, p=40, n_informative=12):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=n_informative,
        n_redundant=10, n_repeated=0, random_state=seed, shuffle=True,
    )
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), y


def _eval(seed, rule):
    from catboost import CatBoostClassifier
    X, y = _make(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)

    def _mk():
        return CatBoostClassifier(iterations=120, depth=4, learning_rate=0.1, verbose=0, random_seed=seed)

    sel = RFECV(estimator=_mk(), n_features_selection_rule=rule, cv=3, random_state=seed,
                max_refits=12, verbose=0)
    t0 = time.perf_counter()
    sel.fit(Xtr, ytr)
    kept = list(getattr(sel, "support_names_", [])) or [c for c, m in zip(X.columns, sel.support_) if m]
    if not kept:
        kept = list(X.columns)
    model = _mk()
    model.fit(Xtr[kept], ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte[kept])[:, 1])
    return auc, len(kept), time.perf_counter() - t0


def main():
    """Run the RFECV one_se_max-vs-one_se_min rule comparison across seeds and print the table."""
    seeds = list(range(6))
    results = []
    for seed in seeds:
        for rule in ("one_se_max", "one_se_min"):
            try:
                auc, nfeat, wall = _eval(seed, rule)
            except Exception as e:  # keep bench resilient across seeds
                print(f"seed={seed} rule={rule} FAILED: {e}")
                continue
            results.append({"seed": seed, "rule": rule, "auc": auc, "n_features": nfeat, "wall_s": wall})
            print(f"seed={seed} rule={rule:11s} auc={auc:.4f} n={nfeat:2d} wall={wall:.1f}s")
    df = pd.DataFrame(results)
    summary = df.groupby("rule").agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                                     n_mean=("n_features", "mean")).reset_index()
    print(summary.to_string(index=False))
    wins = 0
    for seed in seeds:
        sub = df[df.seed == seed]
        if {"one_se_max", "one_se_min"} <= set(sub.rule):
            mx = sub[sub.rule == "one_se_max"].auc.values[0]
            mn = sub[sub.rule == "one_se_min"].auc.values[0]
            wins += int(mn > mx + 1e-6)
    print(f"one_se_min beats one_se_max on {wins}/{len(seeds)} seeds")
    out = Path(__file__).parent / "_results" / f"rfecv_one_se_rule_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
