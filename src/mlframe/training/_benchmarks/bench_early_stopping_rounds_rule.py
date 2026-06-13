"""Multi-scenario bench for the AUTO early-stopping-rounds rule in ``get_training_configs``.

When the caller leaves ``early_stopping_rounds=0`` (the suite default), the patience is auto-derived as ``max(2, iterations // 3)``
(``_helpers_training_configs.py:156``). With the default ``iterations=5000`` that is a patience of 1666 non-improving rounds --
effectively almost no early stopping. The competition-canonical patience is a small fixed value (50-100). This bench compares the
current auto rule against fixed patience 50 / 100 on an HONEST independent holdout (never seen by training nor by the early-stopping
val carve) across 5 synthetic scenarios x 3 seeds x {clf, reg}, early stopping ON so patience actually drives iteration selection.

Decision rule (CLAUDE.md "Variant defaults: most ACCURATE first"): flip the default only if an alternative wins on the MAJORITY of
scenario+seed cells on the honest metric. Else keep the auto rule and record the numbers here (REJECTED != DELETED).

Run: CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.bench_early_stopping_rounds_rule
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless: avoids the interactive-backend tkinter hang + 60s render timeouts that dominate suite wall in this bench

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

from mlframe.training.core import train_mlframe_models_suite, predict_mlframe_models_suite
from mlframe.training.configs import ReportingConfig, OutputConfig, TrainingSplitConfig

ITERATIONS = 3000
AUTO_ES = max(2, ITERATIONS // 3)  # the current default rule, materialised for the bench cap used here


def _make_fte(regression):
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    return SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression)


def _scenario_linear_noisy(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 4000, 20
    X = rng.randn(n, p)
    coefs = np.concatenate([rng.randn(6), np.zeros(p - 6)])
    signal = X @ coefs
    y = (signal + rng.randn(n) * 3.0) if regression else ((signal + rng.randn(n) * 4.0) > 0).astype(int)
    return _split(X, y, p, seed)


def _scenario_nonlinear(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 4000, 16
    X = rng.randn(n, p)
    signal = np.sin(X[:, 0] * 1.5) * 3 + X[:, 1] * X[:, 2] + np.abs(X[:, 3]) * 2
    y = (signal + rng.randn(n) * 2.0) if regression else ((signal + rng.randn(n) * 3.0) > np.median(signal)).astype(int)
    return _split(X, y, p, seed)


def _scenario_sparse_signal(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 3000, 40
    X = rng.randn(n, p)
    coefs = np.zeros(p)
    coefs[:3] = [3.0, -2.5, 2.0]
    signal = X @ coefs
    y = (signal + rng.randn(n) * 2.5) if regression else ((signal + rng.randn(n) * 3.5) > 0).astype(int)
    return _split(X, y, p, seed)


def _scenario_high_noise(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 3500, 18
    X = rng.randn(n, p)
    coefs = np.concatenate([rng.randn(4), np.zeros(p - 4)])
    signal = X @ coefs
    y = (signal + rng.randn(n) * 6.0) if regression else ((signal + rng.randn(n) * 7.0) > 0).astype(int)
    return _split(X, y, p, seed)


def _scenario_interactions(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 4000, 22
    X = rng.randn(n, p)
    signal = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5] + X[:, 6] * 1.5
    y = (signal + rng.randn(n) * 2.0) if regression else ((signal + rng.randn(n) * 2.5) > 0).astype(int)
    return _split(X, y, p, seed)


def _split(X, y, p, seed):
    rng = np.random.RandomState(seed + 777)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_hold = n // 4
    hold, train = idx[:n_hold], idx[n_hold:]
    cols = [f"f_{i}" for i in range(p)]
    tr = pd.DataFrame(X[train], columns=cols)
    tr["target"] = y[train]
    ho = pd.DataFrame(X[hold], columns=cols)
    ho["target"] = y[hold]
    return tr, ho


SCENARIOS = {
    "linear_noisy": _scenario_linear_noisy,
    "nonlinear": _scenario_nonlinear,
    "sparse_signal": _scenario_sparse_signal,
    "high_noise": _scenario_high_noise,
    "interactions": _scenario_interactions,
}


def _run_one(train_df, holdout_df, regression, es_rounds, seed, workdir, sc_name="sc"):
    fte = _make_fte(regression)
    # mname MUST be unique per (scenario, regression, seed, es): the workdir is shared across all
    # cells, and scenarios have different feature counts (p=16/18/20/40), so a name that omits the
    # scenario/mode collides model dirs -> predict loads a model trained on a different feature space.
    mname = f"{sc_name}_{'reg' if regression else 'clf'}_es{es_rounds}_s{seed}"
    data_dir = os.path.join(workdir, mname)
    train_mlframe_models_suite(
        df=train_df,
        target_name="t",
        model_name=mname,
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        verbose=0,
        split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
        hyperparams_config={"iterations": ITERATIONS, "early_stopping_rounds": es_rounds, "learning_rate": 0.05},
    )
    models_path = os.path.join(data_dir, "models", "t", mname)
    res = predict_mlframe_models_suite(
        df=holdout_df,
        models_path=models_path,
        features_and_targets_extractor=fte,
        return_probabilities=not regression,
        verbose=0,
    )
    yt = holdout_df["target"].to_numpy()
    if regression:
        preds = np.asarray(next(iter(res["predictions"].values())), dtype=float)
        return float(np.sqrt(mean_squared_error(yt, preds)))  # RMSE: lower better
    if res.get("probabilities"):
        probs = np.asarray(next(iter(res["probabilities"].values())))
        score = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
    else:
        score = np.asarray(next(iter(res["predictions"].values())), dtype=float)
    return float(roc_auc_score(yt, score))  # AUROC: higher better


def _results_path():
    rdir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(rdir, exist_ok=True)
    return os.path.join(rdir, "early_stopping_rounds_rule.json")


def _flush(results, decision=None):
    import json

    payload = {"baseline_auto": AUTO_ES,
               "cells": {f"{s}|{r}|{sd}|{es}": v for (s, r, sd, es), v in results.items()}}
    if decision is not None:
        payload["decision"] = decision
        payload["summary"] = {str(k): v for k, v in summarise(results)[0].items()}
    with open(_results_path(), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def run_bench(es_variants=(AUTO_ES, 100, 50), seeds=(0, 1, 2), regression_modes=(False, True)):
    results = {}  # (scenario, regression, seed, es_rounds) -> metric
    with tempfile.TemporaryDirectory() as workdir:
        for sc_name, sc_fn in SCENARIOS.items():
            for regression in regression_modes:
                for seed in seeds:
                    train_df, holdout_df = sc_fn(seed, regression)
                    for es in es_variants:
                        results[(sc_name, regression, seed, es)] = _run_one(train_df, holdout_df, regression, es, seed, workdir, sc_name)
                        _flush(results)  # checkpoint after every cell so a native at-exit crash never loses progress
    return results


def summarise(results, baseline=AUTO_ES, alternatives=(100, 50)):
    cells = sorted({(s, r, sd) for (s, r, sd, _) in results})
    out = {}
    for alt in alternatives:
        wins = ties = losses = 0
        deltas = []
        for (s, r, sd) in cells:
            base = results[(s, r, sd, baseline)]
            cand = results[(s, r, sd, alt)]
            delta = (cand - base) if not r else (base - cand)  # signed so >0 means alt better
            deltas.append(delta)
            if abs(delta) < 1e-9:
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1
        out[alt] = {
            "wins": wins, "ties": ties, "losses": losses, "n": len(cells),
            "mean_delta": float(np.mean(deltas)), "median_delta": float(np.median(deltas)),
        }
    return out, cells


def main():
    results = run_bench()
    summary, cells = summarise(results)
    print(f"\nearly_stopping_rounds rule bench: {len(cells)} cells (5 scenarios x 3 seeds x 2 task types); baseline auto={AUTO_ES}\n")
    print(f"{'scenario':<16}{'task':<6}{'seed':<5}{'auto':>10}{'es=100':>10}{'es=50':>10}")
    for (s, r, sd) in cells:
        task = "reg" if r else "clf"
        print(f"{s:<16}{task:<6}{sd:<5}{results[(s, r, sd, AUTO_ES)]:>10.4f}{results[(s, r, sd, 100)]:>10.4f}{results[(s, r, sd, 50)]:>10.4f}")
    print(f"\n=== Win/tie/loss vs baseline auto={AUTO_ES} (honest holdout; higher AUROC / lower RMSE = better) ===")
    for alt, st in summary.items():
        print(f"es_rounds={alt}: wins={st['wins']}/{st['n']} ties={st['ties']} losses={st['losses']} "
              f"mean_delta={st['mean_delta']:+.5f} median_delta={st['median_delta']:+.5f}")
    best_alt = max(summary, key=lambda a: summary[a]["wins"])
    majority = summary[best_alt]["wins"] > summary[best_alt]["n"] / 2
    decision = ("FLIP to es_rounds=%s" % best_alt) if majority else "KEEP auto rule (iterations//3)"
    print(f"\nDECISION: {decision} (best alt {best_alt} won {summary[best_alt]['wins']}/{summary[best_alt]['n']})")
    _flush(results, decision=decision)  # final machine-readable verdict; survives a teardown segfault / lost stdout
    return results, summary


if __name__ == "__main__":
    main()
