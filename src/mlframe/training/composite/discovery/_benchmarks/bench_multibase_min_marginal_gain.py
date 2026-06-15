"""qual-18 bench: composite multi-base discovery gate ``multi_base_min_marginal_rmse_gain``.

Lever: ``CompositeTargetDiscoveryConfig.multi_base_min_marginal_rmse_gain`` (default 0.02), the relative CV-RMSE marginal-gain threshold a candidate base must clear in
``forward_stepwise_multi_base`` to be ADDED to a ``linear_residual_multi`` composite target. Lower threshold -> admits weaker bases (richer composite, risk of overfitting
on the discovery sample); higher threshold -> conservative (may leave genuinely-helpful weak orthogonal bases on the table).

Honest method (leakage-safe): the gate runs forward-stepwise selection on TRAIN ONLY (internal TimeSeriesSplit CV). The selected base set + the joint-OLS ``alphas/beta`` are
fit on TRAIN, defining the composite target ``T = y - base@alphas - beta``. A LightGBM model is trained to predict T from features X on TRAIN, then T_hat is predicted on a
DISJOINT holdout never seen by discovery; the composite prediction ``y_hat = T_hat + base_hold@alphas + beta`` is scored against y_hold. We compare the y-scale holdout RMSE
across candidate thresholds. The discovered-target advantage is measured ENTIRELY on held-out rows -> no leakage.

Scenarios (>=2) x seeds (>=5). Run:
    python -m mlframe.training.composite.discovery._benchmarks.bench_multibase_min_marginal_gain
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from mlframe.training.composite.discovery.forward_stepwise import forward_stepwise_multi_base
from mlframe.training.composite.transforms.linear import (
    _linear_residual_multi_fit,
    _linear_residual_multi_inverse,
)


def _make_scenario(name: str, n: int, rng: np.random.Generator):
    """Return (X, y, base_pool_names, base_cols dict) for a scenario.

    All scenarios: y = f(features) + additive contributions from several BASE columns of decreasing strength, plus noise. The bases are columns the downstream LGBM does NOT
    see (they are residualised out via the composite); X carries only the predictive FEATURES. A genuinely-helpful weak base lowers honest holdout RMSE when admitted; a
    near-useless / collinear base hurts (overfit on train) when admitted.
    """
    p = 6
    X = rng.normal(size=(n, p))
    # Strong primary base + medium secondary + weak-but-real tertiary + a near-useless decoy base.
    b_strong = rng.normal(size=n) * 3.0
    b_medium = rng.normal(size=n) * 1.2
    b_weak = rng.normal(size=n) * 0.45
    b_decoy = rng.normal(size=n) * 0.15  # contributes almost nothing to y

    if name == "three_real_bases":
        # Three bases carry genuine additive signal of decreasing strength; decoy is near-zero. A LOWER gate that admits the weak (real) base should generalise better.
        feat_signal = 2.0 * X[:, 0] + 1.0 * np.tanh(X[:, 1]) + 0.7 * X[:, 2] * X[:, 3]
        y = feat_signal + b_strong + b_medium + b_weak + 0.02 * b_decoy + rng.normal(size=n) * 0.6
    elif name == "one_dominant_plus_decoys":
        # One dominant base; medium/weak/decoy bases are noise-only. A HIGHER gate that stops after the dominant base should avoid overfitting the noise bases on the disc. sample.
        feat_signal = 2.2 * X[:, 0] - 1.1 * X[:, 1] + 0.6 * np.sin(X[:, 4])
        y = feat_signal + b_strong + 0.0 * b_medium + 0.0 * b_weak + 0.0 * b_decoy + rng.normal(size=n) * 0.6
    elif name == "graded_weak_real":
        # A boundary scenario: medium + a clearly-real-but-weak base whose marginal CV gain lands NEAR the 0.01-0.02 band. The honest holdout tells us whether admitting it (lower thr) helps.
        feat_signal = 1.8 * X[:, 0] + 0.9 * X[:, 2] - 0.5 * X[:, 5]
        y = feat_signal + b_strong + b_medium + 0.8 * b_weak + 0.0 * b_decoy + rng.normal(size=n) * 0.6
    else:
        raise ValueError(name)

    base_cols = {"b_strong": b_strong, "b_medium": b_medium, "b_weak": b_weak, "b_decoy": b_decoy}
    return X, y, list(base_cols.keys()), base_cols


def _fast_rmse(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _eval_threshold(thr, X_tr, y_tr, base_tr, X_ho, base_ho, y_ho, seed):
    import lightgbm as lgb

    # Gate runs on TRAIN only. Seed the forward-stepwise with the strongest single base (the discovery single-base winner analog).
    kept, _diag = forward_stepwise_multi_base(
        y_tr, base_tr, seed_bases=["b_strong"], max_k=4,
        min_marginal_rmse_gain=thr, time_aware=False, random_state=seed,
    )
    base_mat_tr = np.column_stack([base_tr[n] for n in kept])
    params = _linear_residual_multi_fit(y_tr, base_mat_tr)
    alphas = np.asarray(params["alphas"], np.float64)
    beta = float(params["beta"])
    # Composite target T on TRAIN.
    T_tr = y_tr - (base_mat_tr @ alphas) - beta
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05, n_jobs=1, verbosity=-1, random_state=seed)
    model.fit(X_tr, T_tr)
    base_mat_ho = np.column_stack([base_ho[n] for n in kept])
    T_hat = model.predict(X_ho)
    y_hat = _linear_residual_multi_inverse(T_hat, base_mat_ho, params)
    return _fast_rmse(y_hat, y_ho), len(kept)


def main():
    scenarios = ["three_real_bases", "one_dominant_plus_decoys", "graded_weak_real"]
    thresholds = [0.005, 0.01, 0.02, 0.05]
    seeds = list(range(11, 21))  # 10 seeds
    n = 3000

    results = {s: {t: [] for t in thresholds} for s in scenarios}
    kcount = {s: {t: [] for t in thresholds} for s in scenarios}
    for scen in scenarios:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X, y, _names, base_cols = _make_scenario(scen, n, rng)
            # Disjoint holdout split (discovery never sees holdout rows).
            n_tr = int(n * 0.6)
            idx = rng.permutation(n)
            tr, ho = idx[:n_tr], idx[n_tr:]
            X_tr, X_ho = X[tr], X[ho]
            y_tr, y_ho = y[tr], y[ho]
            base_tr = {k: v[tr] for k, v in base_cols.items()}
            base_ho = {k: v[ho] for k, v in base_cols.items()}
            for thr in thresholds:
                rmse, k = _eval_threshold(thr, X_tr, y_tr, base_tr, X_ho, base_ho, y_ho, seed)
                results[scen][thr].append(rmse)
                kcount[scen][thr].append(k)

    print("=== qual-18: multi_base_min_marginal_rmse_gain honest-holdout RMSE (lower=better) ===")
    print(f"n={n}, seeds={seeds}, thresholds={thresholds}\n")
    for scen in scenarios:
        print(f"-- scenario: {scen} --")
        for thr in thresholds:
            arr = np.array(results[scen][thr])
            ks = np.array(kcount[scen][thr])
            print(f"  thr={thr:<6}  mean RMSE={arr.mean():.4f}  median={np.median(arr):.4f}  mean#bases={ks.mean():.2f}")
        print()

    # Pairwise per-seed comparison: new candidate (0.005) vs current default (0.02).
    for cand in (0.005, 0.01):
        print(f"=== per-seed pairwise: thr={cand} (candidate) vs thr=0.02 (current default) ===")
        for scen in scenarios:
            a = np.array(results[scen][cand])
            b = np.array(results[scen][0.02])
            wins = int((a < b).sum())
            ties = int(np.isclose(a, b).sum())
            print(f"  {scen}: {cand} better in {wins}/{len(a)} (ties {ties}) | mean {a.mean():.4f} vs {b.mean():.4f} (delta {a.mean()-b.mean():+.4f})")
        print()


if __name__ == "__main__":
    main()
