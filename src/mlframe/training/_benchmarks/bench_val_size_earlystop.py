"""Multi-scenario bench for the early-stopping eval-set carve fraction (``TrainingSplitConfig.val_size``).

The val set is what early stopping uses to pick the best iteration. Too small => noisy stopping signal (overfit / underfit the iteration count);
too large => fewer training rows. Current default is 0.1. This bench compares 0.1 vs 0.15 vs 0.2 on an HONEST independent holdout
(never seen by training nor by the early-stopping val carve) across 5 synthetic scenarios x 3 seeds, with early stopping ON so the val
carve actually drives iteration selection.

Decision rule (CLAUDE.md "Variant defaults: most ACCURATE first"): flip the default only if an alternative wins on the MAJORITY of
scenario+seed cells on the honest metric. Else keep 0.1 and record the numbers here (REJECTED != DELETED).

Run: CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.bench_val_size_earlystop
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

from mlframe.training.core import train_mlframe_models_suite, predict_mlframe_models_suite
from mlframe.training.configs import ReportingConfig, OutputConfig
from mlframe.training.configs import TrainingSplitConfig


def _make_fte(regression):
    # Reuse the canonical test extractor so the suite contract (8-tuple transform) is exercised exactly.
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    return SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression)


# ----------------------------------------------------------------------------------------
# Synthetic scenarios. Each returns (train_df, holdout_df, regression).
# The holdout is an independent split the suite never sees (honest metric).
# ----------------------------------------------------------------------------------------

def _scenario_linear_noisy(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 4000, 20
    X = rng.randn(n, p)
    coefs = np.concatenate([rng.randn(6), np.zeros(p - 6)])
    signal = X @ coefs
    if regression:
        y = signal + rng.randn(n) * 3.0
    else:
        y = ((signal + rng.randn(n) * 4.0) > 0).astype(int)
    return _split(X, y, p, seed)


def _scenario_nonlinear(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 4000, 16
    X = rng.randn(n, p)
    signal = np.sin(X[:, 0] * 1.5) * 3 + X[:, 1] * X[:, 2] + np.abs(X[:, 3]) * 2
    if regression:
        y = signal + rng.randn(n) * 2.0
    else:
        y = ((signal + rng.randn(n) * 3.0) > np.median(signal)).astype(int)
    return _split(X, y, p, seed)


def _scenario_sparse_signal(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 3000, 40
    X = rng.randn(n, p)
    coefs = np.zeros(p)
    coefs[:3] = [3.0, -2.5, 2.0]
    signal = X @ coefs
    if regression:
        y = signal + rng.randn(n) * 2.5
    else:
        y = ((signal + rng.randn(n) * 3.5) > 0).astype(int)
    return _split(X, y, p, seed)


def _scenario_high_noise(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 3500, 18
    X = rng.randn(n, p)
    coefs = np.concatenate([rng.randn(4), np.zeros(p - 4)])
    signal = X @ coefs
    if regression:
        y = signal + rng.randn(n) * 6.0
    else:
        y = ((signal + rng.randn(n) * 7.0) > 0).astype(int)
    return _split(X, y, p, seed)


def _scenario_interactions(seed, regression):
    rng = np.random.RandomState(seed)
    n, p = 4000, 22
    X = rng.randn(n, p)
    signal = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5] + X[:, 6] * 1.5
    if regression:
        y = signal + rng.randn(n) * 2.0
    else:
        y = ((signal + rng.randn(n) * 2.5) > 0).astype(int)
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


def _run_one(train_df, holdout_df, regression, val_size, seed, workdir):
    fte = _make_fte(regression)
    mname = f"vs_{int(val_size*100)}_s{seed}"
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
        split_config=TrainingSplitConfig(test_size=0.1, val_size=val_size),
        # Large iteration cap + early stopping ON: the val carve drives iteration selection.
        hyperparams_config={"iterations": 3000, "early_stopping_rounds": 30, "learning_rate": 0.05},
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
        return float(np.sqrt(mean_squared_error(yt, preds)))  # RMSE: lower is better
    if res.get("probabilities"):
        probs = np.asarray(next(iter(res["probabilities"].values())))
        score = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
    else:
        score = np.asarray(next(iter(res["predictions"].values())), dtype=float)
    return float(roc_auc_score(yt, score))  # AUROC: higher is better


def run_bench(val_sizes=(0.1, 0.15, 0.2), seeds=(0, 1, 2), regression_modes=(False, True)):
    results = {}  # (scenario, regression, seed, val_size) -> metric
    with tempfile.TemporaryDirectory() as workdir:
        for sc_name, sc_fn in SCENARIOS.items():
            for regression in regression_modes:
                for seed in seeds:
                    train_df, holdout_df = sc_fn(seed, regression)
                    for vs in val_sizes:
                        m = _run_one(train_df, holdout_df, regression, vs, seed, workdir)
                        results[(sc_name, regression, seed, vs)] = m
    return results


def summarise(results, baseline=0.1, alternatives=(0.15, 0.2)):
    """For each alternative, count scenario+seed cells where it beats the baseline on the honest metric."""
    cells = sorted({(s, r, sd) for (s, r, sd, _) in results})
    out = {}
    for alt in alternatives:
        wins = ties = losses = 0
        deltas = []
        for (s, r, sd) in cells:
            base = results[(s, r, sd, baseline)]
            cand = results[(s, r, sd, alt)]
            # AUROC higher is better; RMSE lower is better.
            better = (cand > base) if not r else (cand < base)
            worse = (cand < base) if not r else (cand > base)
            delta = (cand - base) if not r else (base - cand)  # signed so >0 means alt is better
            deltas.append(delta)
            if abs(delta) < 1e-9:
                ties += 1
            elif better:
                wins += 1
            elif worse:
                losses += 1
        out[alt] = {
            "wins": wins, "ties": ties, "losses": losses, "n": len(cells),
            "mean_delta": float(np.mean(deltas)), "median_delta": float(np.median(deltas)),
        }
    return out, cells


def main():
    results = run_bench()
    summary, cells = summarise(results)
    print(f"\nval_size early-stopping bench: {len(cells)} scenario+seed cells (5 scenarios x 3 seeds x 2 task types)\n")
    print(f"{'scenario':<16}{'task':<6}{'seed':<5}{'vs=0.10':>10}{'vs=0.15':>10}{'vs=0.20':>10}")
    for (s, r, sd) in cells:
        task = "reg" if r else "clf"
        v10 = results[(s, r, sd, 0.1)]
        v15 = results[(s, r, sd, 0.15)]
        v20 = results[(s, r, sd, 0.2)]
        print(f"{s:<16}{task:<6}{sd:<5}{v10:>10.4f}{v15:>10.4f}{v20:>10.4f}")
    print("\n=== Win/tie/loss vs baseline val_size=0.1 (honest holdout; higher AUROC / lower RMSE = better) ===")
    for alt, st in summary.items():
        print(f"val_size={alt}: wins={st['wins']}/{st['n']} ties={st['ties']} losses={st['losses']} "
              f"mean_delta={st['mean_delta']:+.5f} median_delta={st['median_delta']:+.5f}")
    best_alt = max(summary, key=lambda a: summary[a]["wins"])
    majority = summary[best_alt]["wins"] > summary[best_alt]["n"] / 2
    print(f"\nDECISION: {'FLIP to val_size=%s' % best_alt if majority else 'KEEP val_size=0.1'} "
          f"(best alt {best_alt} won {summary[best_alt]['wins']}/{summary[best_alt]['n']})")
    return results, summary


if __name__ == "__main__":
    main()
