"""Multi-seed honest-holdout A/B: Caruana greedy blend weights vs the NNLS default, for the score_ensemble gate.

Question: should ``use_caruana_weights_in_ensemble`` flip ON by default? Per the project "accurate default, then
fastest" rule a flip needs a MAJORITY win on an HONEST holdout across scenarios + seeds, not a single-seed anecdote.

Design (per scenario, per seed):
  - Draw a latent signal z and label y ~ Bernoulli(sigmoid(1.5 z)); split rows into an OOF part (weights are fit here,
    the train-analog) and a TEST part (the honest holdout the blend is scored on).
  - Simulate M ensemble members as probabilistic classifiers of consistent QUALITY across OOF/TEST: member m emits
    p_m = sigmoid(alpha_m z + noise_m * eps) with per-sample eps, so a weight fit on OOF must GENERALISE to TEST.
  - Fit blend weights three ways on the OOF member preds -- caruana greedy (metric-direct AUC), NNLS (squared-error),
    equal-weight (reference) -- then apply each weight vector to the TEST member preds and score the blend.

Metrics on the honest TEST holdout: ROC-AUC (higher better) and log-loss (lower better). We report per-scenario means,
the Caruana-minus-NNLS delta, and the win-rate (fraction of seeds Caruana >= NNLS), plus an overall verdict.

Run: ``python -m mlframe.models.ensembling._benchmarks.bench_caruana_vs_nnls``
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from mlframe.models.ensembling.selection import caruana_greedy_selection
from mlframe.training.composite.ensemble.stacking import stacking_aware_gate


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _members(z: np.ndarray, alphas, noises, rng) -> np.ndarray:
    """(M, N) member positive-class probabilities of consistent per-member quality (alpha=signal, noise=imperfection)."""
    out = np.empty((len(alphas), z.shape[0]), dtype=np.float64)
    for m, (a, s) in enumerate(zip(alphas, noises)):
        out[m] = _sigmoid(a * z + s * rng.standard_normal(z.shape[0]))
    return np.clip(out, 1e-6, 1 - 1e-6)


SCENARIOS = {
    # (alphas, noises): member signal strengths + noise scales.
    "diverse_strengths": ([2.5, 1.8, 1.2, 0.8, 0.4], [0.4, 0.6, 0.8, 1.0, 1.4]),
    "one_strong_many_weak": ([3.0, 0.5, 0.4, 0.3, 0.3], [0.3, 1.3, 1.4, 1.5, 1.5]),
    "correlated_redundant": ([1.6, 1.6, 1.6, 1.5, 0.6], [0.7, 0.7, 0.7, 0.75, 1.3]),
    "all_similar": ([1.4, 1.4, 1.4, 1.4, 1.4], [0.8, 0.8, 0.8, 0.8, 0.8]),
    "mixed_quality": ([2.2, 1.5, 1.5, 0.9, 0.2], [0.5, 0.7, 0.9, 1.1, 1.6]),
}


def _fit_apply(stacked_oof, y_oof, stacked_test):
    """Return (auc_by_method, logloss_by_method) blending TEST preds with weights fit on OOF."""
    m = stacked_oof.shape[0]
    tags = [f"m{i}" for i in range(m)]

    car = caruana_greedy_selection(stacked_oof, y_oof)
    w_car = np.asarray(car.weights, dtype=np.float64)

    survivors, wd = stacking_aware_gate({t: stacked_oof[i] for i, t in enumerate(tags)}, y_oof, min_weight=0.05)
    w_nnls = np.array([wd.get(t, 0.0) if t in set(survivors) else 0.0 for t in tags], dtype=np.float64)
    if w_nnls.sum() <= 0:
        w_nnls = np.ones(m) / m

    w_eq = np.ones(m) / m

    aucs, lls = {}, {}
    for name, w in (("caruana", w_car), ("nnls", w_nnls), ("equal", w_eq)):
        ws = w / w.sum() if w.sum() > 0 else np.ones(m) / m
        blend = np.clip(ws @ stacked_test, 1e-6, 1 - 1e-6)
        aucs[name] = roc_auc_score(y_test_cache[0], blend)
        lls[name] = log_loss(y_test_cache[0], blend)
    return aucs, lls


y_test_cache = [None]  # filled per iteration (keeps _fit_apply signature small)


def main(n: int = 4000, seeds: int = 25) -> None:
    print(f"Caruana vs NNLS honest-holdout A/B  (n={n}/scenario, {seeds} seeds, OOF/TEST 50/50)\n")
    overall_auc_delta, overall_ll_delta, overall_win = [], [], []
    for sc, (alphas, noises) in SCENARIOS.items():
        d_auc, d_ll, wins = [], [], 0
        car_auc, nnls_auc, eq_auc = [], [], []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            z = rng.standard_normal(n)
            y = (rng.random(n) < _sigmoid(1.5 * z)).astype(np.int64)
            half = n // 2
            oof, test = slice(0, half), slice(half, n)
            memb = _members(z, alphas, noises, rng)
            y_test_cache[0] = y[test]
            aucs, lls = _fit_apply(memb[:, oof], y[oof], memb[:, test])
            d_auc.append(aucs["caruana"] - aucs["nnls"])
            d_ll.append(lls["caruana"] - lls["nnls"])
            wins += int(aucs["caruana"] >= aucs["nnls"] - 1e-12)
            car_auc.append(aucs["caruana"]); nnls_auc.append(aucs["nnls"]); eq_auc.append(aucs["equal"])
        d_auc = np.array(d_auc); d_ll = np.array(d_ll)
        overall_auc_delta.extend(d_auc.tolist()); overall_ll_delta.extend(d_ll.tolist()); overall_win.append(wins / seeds)
        print(f"[{sc:22}] AUC caruana={np.mean(car_auc):.4f} nnls={np.mean(nnls_auc):.4f} equal={np.mean(eq_auc):.4f} "
              f"| dAUC={d_auc.mean():+.4f}+-{d_auc.std():.4f} dLogLoss={d_ll.mean():+.4f} win={wins}/{seeds}")
    da = np.array(overall_auc_delta)
    dll = np.array(overall_ll_delta)
    print(f"\nOVERALL: mean dAUC(caruana-nnls)={da.mean():+.5f}  median={np.median(da):+.5f}  " f"scenario AUC win-rates={[round(w, 2) for w in overall_win]}")
    print(f"         mean dLogLoss(caruana-nnls)={dll.mean():+.5f}  (positive = caruana WORSE calibrated)")
    # Flip the default ONLY on a majority AUC win AND no calibration regression -- caruana optimizes ranking (AUC)
    # directly, so it can improve AUC while degrading log-loss; mlframe's ensemble consumes calibrated probs and many
    # downstream metrics/decisions need probability quality, so a log-loss regression blocks the flip.
    flip = da.mean() > 0.001 and np.mean(overall_win) > 0.6 and dll.mean() <= 0.0
    verdict = "FLIP to caruana" if flip else "KEEP nnls default (caruana stays opt-in for AUC-primary tasks)"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
