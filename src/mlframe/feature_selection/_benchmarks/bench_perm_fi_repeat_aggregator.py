"""Cross-repeat aggregator for sklearn permutation importance: mean vs median vs trimmed-mean.

Production default path: ``get_feature_importances`` (feature_selection/wrappers/_helpers_importance.py)
resolves importance_getter='auto' -> 'permutation' below the cost cap, then aggregates the (p x n_repeats)
``permutation_importance(...).importances`` matrix across repeats with ``importances_mean`` (the sklearn mean).
The lever: would a robust cross-repeat aggregator (median / 20%-trimmed-mean) recover the true feature
relevance better when an unlucky permutation produces an outlier loss?

Honest metric: spearman(importance_vector, |true_weight| relevance) over 5 synthetic scenarios x 4 seeds,
at the two realistic small n_repeats (3, 5).

VERDICT (KEEP mean; median/trim REJECTED as default):
  n_repeats=3:  mean=0.4450  median=0.4416 (9 win / 11 lose)  trim20=0.4450 (all tie -- 20% of 3 trims nothing)
  n_repeats=5:  mean=0.4639  median=0.4619 (12 win /  8 lose) trim20=0.4606 (12 win / 8 lose)
Neither alternative beats the mean on the metric mean: median's cell-wins are tiny while its losses are
larger (net negative mean-of-means at both repeat counts), and at n_repeats=3 it loses the cell vote too.
With only 3-5 repeats the median throws away the very averaging that suppresses per-permutation noise, so
the arithmetic mean stays the most accurate default. Re-run on a host that affords more repeats (n_repeats
>= 15, where a trimmed mean could plausibly win) before reconsidering.

Run: python -m mlframe.feature_selection._benchmarks.bench_perm_fi_repeat_aggregator
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

import numpy as np
from scipy.stats import spearmanr, trim_mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

SCENARIOS = ("linear", "interaction", "heavy_noise", "corr", "weak")
SEEDS = (0, 1, 2, 3)


def make(scn, seed, n=600, p=30, n_inf=6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    w = np.zeros(p)
    w[:n_inf] = rng.uniform(0.5, 1.2, n_inf) * rng.choice([-1, 1], n_inf)
    truth = np.abs(w)
    if scn == "linear":
        logit = 0.7 * (X @ w)
    elif scn == "interaction":
        logit = 1.2 * X[:, 0] * X[:, 1] + 1.0 * X[:, 2] * X[:, 3] + 0.8 * X[:, 4] + 0.8 * X[:, 5]
        truth = np.zeros(p)
        truth[:n_inf] = 1.0
    elif scn == "heavy_noise":
        logit = 0.7 * (X @ w)
        X[:, n_inf:] += 1.5 * rng.standard_t(2.0, size=(n, p - n_inf))
    elif scn == "corr":
        X[:, 8] = X[:, 0] + 0.1 * rng.standard_normal(n)
        X[:, 9] = X[:, 1] + 0.1 * rng.standard_normal(n)
        logit = 0.7 * (X @ w)
    else:  # weak
        logit = 0.3 * (X @ w)
    p1 = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p1).astype(int)
    return X, y, truth


def run():
    for nrep in (3, 5):
        agg = {"mean": [], "median": [], "trim20": []}
        for scn in SCENARIOS:
            for sd in SEEDS:
                X, y, truth = make(scn, sd)
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=sd, stratify=y)
                model = RandomForestClassifier(n_estimators=120, random_state=sd, n_jobs=-1).fit(Xtr, ytr)
                mat = permutation_importance(model, Xte, yte, n_repeats=nrep, random_state=sd, n_jobs=-1).importances
                agg["mean"].append(spearmanr(mat.mean(axis=1), truth).statistic)
                agg["median"].append(spearmanr(np.median(mat, axis=1), truth).statistic)
                agg["trim20"].append(spearmanr(np.array([trim_mean(mat[i], 0.2) for i in range(mat.shape[0])]), truth).statistic)
        mn, md, tr = (np.array(agg[k]) for k in ("mean", "median", "trim20"))
        print(f"\n=== spearman(imp, true_relevance), {len(SCENARIOS) * len(SEEDS)} cells, n_repeats={nrep} ===")
        print(f"  mean={mn.mean():.4f}  median={md.mean():.4f}  trim20={tr.mean():.4f}")
        print(f"  median vs mean: win={int((md > mn + 1e-9).sum())} lose={int((md < mn - 1e-9).sum())}")
        print(f"  trim20 vs mean: win={int((tr > mn + 1e-9).sum())} lose={int((tr < mn - 1e-9).sum())}")


if __name__ == "__main__":
    run()
