"""Honest-holdout biz_value benchmark for the curated FE transformers, to decide DEFAULT-WIRING.

For a regression and a binary synthetic engineered to carry signal the curated transformers should capture
(a high-cardinality categorical whose target-mean drives y; a nonlinear interaction that model-disagreement
flags; noise columns), fit each ShortlistTransformerAdapter on the TRAIN fold only and transform train +
held-out separately (Mode-B, leak-safe), then compare a downstream LightGBM trained on [raw] vs [raw + FE]
scored on the HELD-OUT fold. Averaged over seeds. A transformer earns default-wiring only if it lifts the
honest-holdout metric on a majority of seeds without hurting the others.

Run: python -m mlframe.feature_engineering._benchmarks.bench_curated_fe_holdout_value
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score

from mlframe.feature_engineering.curated_fe import curated_fe_pipelines, CURATED_FE_NAMES


def _make(task: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    n_cat = 40  # high-cardinality categorical whose per-level target-mean carries signal
    cat = rng.integers(0, n_cat, size=n)
    cat_effect = rng.standard_normal(n_cat) * 2.0
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    noise_cols = rng.standard_normal((n, 4))
    signal = cat_effect[cat] + 1.5 * np.sin(2.0 * x1) * x2 + 0.7 * (x3 ** 2)
    X = np.column_stack([cat.astype(np.float64), x1, x2, x3, noise_cols])
    cols = ["cat", "x1", "x2", "x3", "n0", "n1", "n2", "n3"]
    if task == "regression":
        y = signal + 0.3 * rng.standard_normal(n)
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")
    p = 1.0 / (1.0 + np.exp(-(signal - np.median(signal))))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def _fe(name, task, Xtr, ytr, Xho, seed):
    pipe = curated_fe_pipelines(task=task, names=[name], seed=seed, passthrough=False)[name]
    # fit_transform -> Mode A (OOF) train features (no in-sample skew); transform -> Mode B honest holdout.
    ftr = np.asarray(pipe.fit_transform(Xtr, ytr), dtype=np.float64)
    fho = np.asarray(pipe.transform(Xho), dtype=np.float64)
    return np.nan_to_num(ftr), np.nan_to_num(fho)


def _score(task, Xtr, ytr, Xho, yho):
    import lightgbm as lgb
    if task == "regression":
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, verbose=-1, n_jobs=-1)
        m.fit(Xtr, ytr)
        return r2_score(yho, m.predict(Xho))
    m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    return roc_auc_score(yho, m.predict_proba(Xho)[:, 1])


def run(task: str, n: int = 6000, seeds=range(5)):
    print(f"\n===== {task.upper()} (n={n}, {len(list(seeds))} seeds) =====")
    variants = list(CURATED_FE_NAMES) + ["ALL"]
    deltas = {v: [] for v in variants}
    base_scores = []
    for seed in seeds:
        X, y = _make(task, n, seed)
        Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.4, random_state=seed)
        raw_tr, raw_ho = Xtr.values, Xho.values
        base = _score(task, raw_tr, ytr.values, raw_ho, yho.values)
        base_scores.append(base)
        fe_cache = {}
        for name in CURATED_FE_NAMES:
            try:
                ftr, fho = _fe(name, task, Xtr, ytr, Xho, seed)
                fe_cache[name] = (ftr, fho)
                s = _score(task, np.hstack([raw_tr, ftr]), ytr.values, np.hstack([raw_ho, fho]), yho.values)
                deltas[name].append(s - base)
            except Exception as e:
                print(f"  seed {seed} {name}: FAILED {e!r}")
        if len(fe_cache) == len(CURATED_FE_NAMES):
            allftr = np.hstack([raw_tr] + [fe_cache[n_][0] for n_ in CURATED_FE_NAMES])
            allfho = np.hstack([raw_ho] + [fe_cache[n_][1] for n_ in CURATED_FE_NAMES])
            s = _score(task, allftr, ytr.values, allfho, yho.values)
            deltas["ALL"].append(s - base)
    metric = "R2" if task == "regression" else "AUC"
    print(f"  baseline {metric}: mean={np.mean(base_scores):.4f}")
    for v in variants:
        d = np.array(deltas[v])
        if d.size:
            wins = int(np.sum(d > 0))
            print(f"  {v:24s} delta_{metric} mean={d.mean():+.4f} median={np.median(d):+.4f}  wins={wins}/{d.size}  (min={d.min():+.4f} max={d.max():+.4f})")


if __name__ == "__main__":
    run("regression")
    run("binary")
