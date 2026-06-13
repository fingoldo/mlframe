"""Isolated bench: WoE Laplace-smoothing alpha for ``LeakageSafeEncoder(method='woe')``.

The WoE per-category smoothing adds ``alpha`` to positive/negative cell mass
before the log-odds: ``log((pos_c+a)/(n_pos+a)) - log((neg_c+a)/(n_neg+a))``.
A large alpha pulls every category's WoE toward 0 (no-evidence), which on rare
categories is desirable shrinkage but on common, genuinely-discriminative
categories destroys real signal. The default ``smoothing=10.0`` is shared with
the mean encoders; this bench isolates whether 10 is the right WoE Laplace alpha.

Honest metric: held-out TEST ROC-AUC of the single WoE-encoded column. The
encoder is fit on train (OOF not needed for the held-out score), transformed
onto a disjoint test split, and the resulting log-odds column is scored
directly against the test target. Higher = the encoding preserves more true
per-category signal on data the encoder never saw.

Run:
  python -m mlframe.training.feature_handling._benchmarks.bench_woe_laplace_alpha
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def _make_scenario(name, rng):
    """Return (cats_train, y_train, cats_test, y_test). Each category has a
    fixed latent positive-rate; the WoE encoder must recover it."""
    n = 8000
    if name == "balanced_midcard":
        K, base = 40, 0.5
    elif name == "imbalanced_5pct":
        K, base = 40, 0.05
    elif name == "highcard_rare":
        K, base = 400, 0.2  # many categories -> each rare
    elif name == "lowcard_strong":
        K, base = 6, 0.5
    elif name == "imbalanced_1pct_highcard":
        K, base = 200, 0.01
    else:
        raise ValueError(name)

    # Latent per-category log-odds spread around base; some categories strongly
    # discriminative, drawn so signal is real but cardinality controls rarity.
    cat_logodds = rng.normal(np.log(base / (1 - base)), 1.3, size=K)
    cats = rng.integers(0, K, size=n)
    p = 1.0 / (1.0 + np.exp(-cat_logodds[cats]))
    y = (rng.random(n) < p).astype(np.float64)
    half = n // 2
    return (
        cats[:half].astype(str),
        y[:half],
        cats[half:].astype(str),
        y[half:],
    )


def _score(alpha, cats_tr, y_tr, cats_te, y_te):
    enc = LeakageSafeEncoder(method="woe", smoothing=alpha, random_state=0)
    enc.fit(cats_tr, y_tr)
    woe_te = enc.transform(cats_te)
    if len(np.unique(y_te)) < 2:
        return np.nan
    return roc_auc_score(y_te, woe_te)


def run():
    scenarios = [
        "balanced_midcard",
        "imbalanced_5pct",
        "highcard_rare",
        "lowcard_strong",
        "imbalanced_1pct_highcard",
    ]
    seeds = [0, 1, 2]
    # results[alpha] = list of test-AUCs
    agg = {a: [] for a in ALPHAS}
    wins = {a: 0 for a in ALPHAS}
    cells = 0
    print(f"{'scenario':<26} {'seed':<5} " + " ".join(f"a={a:<6}" for a in ALPHAS))
    for sc in scenarios:
        for seed in seeds:
            rng = np.random.default_rng(seed * 100 + 7)
            ctr, ytr, cte, yte = _make_scenario(sc, rng)
            row = {a: _score(a, ctr, ytr, cte, yte) for a in ALPHAS}
            for a in ALPHAS:
                agg[a].append(row[a])
            best_a = max(ALPHAS, key=lambda a: row[a])
            wins[best_a] += 1
            cells += 1
            print(f"{sc:<26} {seed:<5} " + " ".join(f"{row[a]:<8.4f}" for a in ALPHAS))
    print("\nMean test-AUC by alpha:")
    for a in ALPHAS:
        print(f"  alpha={a:<6} mean_auc={np.nanmean(agg[a]):.4f}  win_cells={wins[a]}/{cells}")
    overall_best = max(ALPHAS, key=lambda a: np.nanmean(agg[a]))
    print(f"\nBest mean-AUC alpha: {overall_best}")
    print(f"Most-wins alpha: {max(ALPHAS, key=lambda a: wins[a])}")


if __name__ == "__main__":
    run()
