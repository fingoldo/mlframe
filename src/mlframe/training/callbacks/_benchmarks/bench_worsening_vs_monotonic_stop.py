"""Benchmark: which early-stop RULE picks the best honest-holdout iteration?

Compares three rules applied to the SAME trained booster's per-iteration validation trajectory, so the
only thing that varies is WHERE each rule stops (training is identical -> a fair, low-variance A/B):

  * patience(P)         -- stop P iters after the best val (classic lgb/xgb early stopping).
  * old_worsening       -- UniversalCallback._update_worsening_streak: budget-scaled threshold
                           max(n//coeff, min_iters), PLATEAU (==prev) COUNTS as worsening, resets on a
                           strict improvement-over-prev.
  * new_monotonic(N)    -- MonotonicDeclineStopper: fixed N consecutive STRICT declines, plateau/bounce RESETS.

For each rule we read the HONEST TEST metric at the stopped iteration and compare to the oracle (test
metric at the val-argbest iteration). The winner is the rule whose stop gives test closest to the oracle
(accuracy first), tie-broken by EARLIER stop (fewer trees = cheaper + less overfit). Also reports whether
old_worsening ever fires before patience (settles the "effectively never fires" claim empirically).

Run: CUDA_VISIBLE_DEVICES="" python -m mlframe.training.callbacks._benchmarks.bench_worsening_vs_monotonic_stop
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


N_TREES = 500
SEEDS = list(range(6))


def _data(scenario, seed, n=2400, d=20):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = np.r_[rng.randn(5), np.zeros(d - 5)]
    logit = X @ w
    if scenario == "overfit":          # heavy label noise -> val/test peak early then degrade
        y = (logit + 2.6 * rng.randn(n) > 0).astype(int)
    elif scenario == "clean":          # low noise -> val keeps improving (stop must NOT fire early)
        y = (logit + 0.4 * rng.randn(n) > 0).astype(int)
    else:                               # "noisy_plateau" -> moderate noise, long flat tail
        y = (logit + 1.6 * rng.randn(n) > 0).astype(int)
    tr, va, te = slice(0, n // 2), slice(n // 2, 3 * n // 4), slice(3 * n // 4, n)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def _auc_curve(booster, X, y):
    """val/test AUC at every boosting iteration (higher is better)."""
    out = np.empty(N_TREES)
    for k in range(1, N_TREES + 1):
        p = booster.predict(X, num_iteration=k)
        out[k - 1] = roc_auc_score(y, p)
    return out


# ---- the three stop rules, operating on a higher-is-better val curve ----

def stop_patience(val, patience):
    best, best_i = -np.inf, 0
    for i, v in enumerate(val):
        if v > best:
            best, best_i = v, i
        elif i - best_i >= patience:
            return i
    return len(val) - 1


def stop_old_worsening(val, coeff=5, min_iters=5):
    thr = max(len(val) // coeff, min_iters)
    best, prev, streak = -np.inf, None, 0
    for i, v in enumerate(val):
        improved = v > best
        if improved:
            best = v
        if prev is None:
            prev = v
            continue
        improved_over_prev = v > prev
        prev = v
        if improved or improved_over_prev:   # OLD: plateau (==prev) does NOT reset -> falls through to +=1
            streak = 0
            continue
        streak += 1
        if streak >= thr:
            return i
    return len(val) - 1


def stop_new_monotonic(val, n=3):
    best, prev, streak = -np.inf, None, 0
    for i, v in enumerate(val):
        if v > best:                          # NEW: new best resets
            best, streak = v, 0
        elif prev is not None and v < prev:    # strict decline
            streak += 1
        else:                                  # plateau / bounce-up resets
            streak = 0
        prev = v
        if streak >= n:
            return i
    return len(val) - 1


def main():
    scenarios = ["overfit", "noisy_plateau", "clean"]
    rules = {
        "patience20": lambda v: stop_patience(v, 20),
        "patience50": lambda v: stop_patience(v, 50),
        "old_worsening": stop_old_worsening,
        "new_monotonic3": lambda v: stop_new_monotonic(v, 3),
        "new_monotonic5": lambda v: stop_new_monotonic(v, 5),
    }
    agg = {r: {"test_gap": [], "stop": []} for r in rules}      # gap = oracle_test - rule_test (lower=better)
    old_fires_before_patience = 0
    total = 0
    for sc in scenarios:
        for seed in SEEDS:
            Xtr, ytr, Xva, yva, Xte, yte = _data(sc, seed)
            booster = lgb.train(
                {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": seed},
                lgb.Dataset(Xtr, ytr), num_boost_round=N_TREES,
            )
            val_curve = _auc_curve(booster, Xva, yva)
            test_curve = _auc_curve(booster, Xte, yte)
            oracle_i = int(np.argmax(val_curve))             # best-val iteration (what ES tries to find)
            oracle_test = test_curve[oracle_i]
            stops = {r: f(val_curve) for r, f in rules.items()}
            for r, si in stops.items():
                agg[r]["test_gap"].append(oracle_test - test_curve[si])
                agg[r]["stop"].append(si)
            total += 1
            if stops["old_worsening"] < stops["patience20"]:
                old_fires_before_patience += 1
            print(f"{sc:14s} seed={seed} oracle@{oracle_i:3d}(test={oracle_test:.4f}) "
                  + " ".join(f"{r}@{stops[r]:3d}(test={test_curve[stops[r]]:.4f})" for r in rules))
    print("\n=== AGGREGATE (mean over %d fits) ===" % total)
    print(f"{'rule':16s} {'mean_test_gap':>13s} {'median_gap':>11s} {'mean_stop':>10s}  (gap=oracle_test - rule_test, LOWER better)")
    for r in rules:
        g = np.array(agg[r]["test_gap"]); s = np.array(agg[r]["stop"])
        print(f"{r:16s} {g.mean():13.5f} {np.median(g):11.5f} {s.mean():10.1f}")
    print(f"\nold_worsening fired BEFORE patience20 in {old_fires_before_patience}/{total} fits")


if __name__ == "__main__":
    main()
