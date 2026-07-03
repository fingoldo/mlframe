"""Honest-holdout biz_value benchmark for the curated FE transformers, to decide DEFAULT-WIRING.

For a regression and a binary synthetic engineered to carry signal the curated transformers should capture
(a high-cardinality categorical whose target-mean drives y; a nonlinear interaction that model-disagreement
flags; noise columns), the ShortlistTransformerAdapter's fit_transform (Mode-A OOF) builds honest train
features while transform (Mode-B) builds the held-out features, then a downstream LightGBM trained on [raw]
vs [raw + FE] is scored on the HELD-OUT fold. Reports the FULL metric block via mlframe's fused kernels
(regression: R2/RMSE/MAE/MAPE/wMAPE/SMAPE; binary: ROC-AUC/PR-AUC/Brier/LogLoss/Accuracy/BalAcc/MacroF1) --
a single AUC hides that a transformer can improve PROBABILITY quality (Brier/LogLoss) while leaving rank-AUC
flat. Averaged over seeds; per metric we report the mean delta + how many seeds improved (direction-aware).

Run: python -m mlframe.feature_engineering._benchmarks.bench_curated_fe_holdout_value
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
# metric scoring is via mlframe kernels (see _score)

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


# Higher-is-better metrics; everything else (errors / losses) is lower-is-better.
_HIGHER_BETTER = {"R2", "ROC_AUC", "PR_AUC", "Accuracy", "BalAcc", "MacroF1"}
_REG_METRICS = ["R2", "RMSE", "MAE", "MAPE_mean", "wMAPE", "SMAPE"]
_CLF_METRICS = ["ROC_AUC", "PR_AUC", "Brier", "LogLoss", "Accuracy", "BalAcc", "MacroF1"]


def _score(task, Xtr, ytr, Xho, yho):
    """Full metric block via mlframe's fused kernels (not a single scalar)."""
    import lightgbm as lgb
    from mlframe.metrics.regression._regression_extras import fast_regression_metrics_block_extended
    from mlframe.metrics.regression._regression_metrics import fast_r2_score
    from mlframe.metrics._core_auc_brier import fast_aucs, fast_brier_score_loss
    from mlframe.metrics._core_precision_mape import fast_classification_report

    if task == "regression":
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, verbose=-1, n_jobs=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xho)
        blk = fast_regression_metrics_block_extended(yho, pred)
        out = {"R2": float(fast_r2_score(yho, pred))}
        for k in ("RMSE", "MAE", "MAPE_mean", "wMAPE", "SMAPE"):
            out[k] = float(blk.get(k, np.nan))
        return out

    m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    proba = np.clip(m.predict_proba(Xho)[:, 1], 1e-7, 1 - 1e-7)
    roc, pr = fast_aucs(np.asarray(yho, dtype=np.int64), proba)
    brier = float(fast_brier_score_loss(np.asarray(yho, dtype=np.int64), proba))
    logloss = float(-np.mean(yho * np.log(proba) + (1 - yho) * np.log(1 - proba)))
    hard = (proba > 0.5).astype(np.int64)
    rep = fast_classification_report(np.asarray(yho, dtype=np.int64), hard, nclasses=2)
    accuracy, bal_acc, macro = float(rep[2]), float(rep[3]), rep[8]  # macro = [precision, recall, f1]
    return {"ROC_AUC": float(roc), "PR_AUC": float(pr), "Brier": brier, "LogLoss": logloss,
            "Accuracy": accuracy, "BalAcc": bal_acc, "MacroF1": float(macro[2])}


def _improved(metric, delta):
    """True iff a positive/negative delta is an improvement for this metric's direction."""
    return delta > 0 if metric in _HIGHER_BETTER else delta < 0


def run(task: str, n: int = 6000, seeds=range(5)):
    seeds = list(seeds)
    metrics = _REG_METRICS if task == "regression" else _CLF_METRICS
    print(f"\n===== {task.upper()} (n={n}, {len(seeds)} seeds) -- full metric block =====")
    variants = list(CURATED_FE_NAMES) + ["ALL"]
    # deltas[variant][metric] = list of (variant_metric - base_metric) per seed
    deltas = {v: {m: [] for m in metrics} for v in variants}
    base_acc = {m: [] for m in metrics}
    for seed in seeds:
        X, y = _make(task, n, seed)
        Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.4, random_state=seed)
        raw_tr, raw_ho = Xtr.values, Xho.values
        base = _score(task, raw_tr, ytr.values, raw_ho, yho.values)
        for m in metrics:
            base_acc[m].append(base[m])
        fe_cache = {}
        for name in CURATED_FE_NAMES:
            try:
                ftr, fho = _fe(name, task, Xtr, ytr, Xho, seed)
                fe_cache[name] = (ftr, fho)
                s = _score(task, np.hstack([raw_tr, ftr]), ytr.values, np.hstack([raw_ho, fho]), yho.values)
                for m in metrics:
                    deltas[name][m].append(s[m] - base[m])
            except Exception as e:
                print(f"  seed {seed} {name}: FAILED {e!r}")
        if len(fe_cache) == len(CURATED_FE_NAMES):
            allftr = np.hstack([raw_tr] + [fe_cache[k][0] for k in CURATED_FE_NAMES])
            allfho = np.hstack([raw_ho] + [fe_cache[k][1] for k in CURATED_FE_NAMES])
            s = _score(task, allftr, ytr.values, allfho, yho.values)
            for m in metrics:
                deltas["ALL"][m].append(s[m] - base[m])
    print("  baseline: " + "  ".join(f"{m}={np.mean(base_acc[m]):.4f}" for m in metrics))
    arrow = {m: ("^" if m in _HIGHER_BETTER else "v") for m in metrics}
    print("  (delta signs: " + " ".join(f"{m}{arrow[m]}" for m in metrics) + " ; a metric IMPROVES when delta matches its arrow)")
    for v in variants:
        parts = []
        for m in metrics:
            d = np.array(deltas[v][m])
            if not d.size:
                continue
            imp = int(sum(_improved(m, x) for x in d))
            mark = "+" if _improved(m, d.mean()) else "-"
            parts.append(f"{m}={d.mean():+.4f}[{mark}{imp}/{d.size}]")
        print(f"  {v:24s} " + "  ".join(parts))


if __name__ == "__main__":
    run("regression")
    run("binary")
