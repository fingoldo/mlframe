"""Cascaded pair-FE composition + nested-CV honest validator.

Both utilities sit on top of ``optimise_hermite_pair``:

* **FE composition round 2** -- a single round of pair-FE only captures pair-wise (rank-1 separable) structure. Feeding round-1 engineered features back as new inputs and running a second round captures cross-pair interactions of effective arity 4 (every round-2 feature is a pair of pairs-of-features). Useful when the true target depends on combinations of two pair signals, e.g. ``y = sign((x_a*x_b) + (x_c*x_d))``.
* **Nested-CV honest validator** -- ``optimise_hermite_pair`` fits coefficients on the same data used to score the result; that bias-up uplift estimate is suspect. A K-fold wrapper that fits coefs on K-1 folds and scores on the heldout fold gives an honest out-of-sample uplift; comparing to the in-sample number quantifies optimism (and reveals leakage).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compose_pair_fe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_rounds: int = 2,
    top_k_per_round: int = 3,
    discrete_target: bool = True,
    basis: str = "chebyshev",
    n_trials: int = 30,
    max_degree: int = 3,
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    optimizer: str = "cma",
    use_trivial_baseline: bool = True,
    baseline_uplift_threshold: float = 1.05,
    feature_names: list | None = None,
    verbose: bool = False,
) -> dict:
    """Run cascaded pair-FE for ``n_rounds`` rounds. Returns the final augmented feature matrix plus engineered-feature provenance.

    Pipeline:
        for round r in 1..n_rounds:
            1. Rank current features by single-feature MI(x_i, y). Keep top ``2 * top_k_per_round`` candidates.
            2. Form all pairs from the top features. For each, run ``optimise_hermite_pair``. Keep at most ``top_k_per_round`` engineered features per round.
            3. Append the engineered features to X. Repeat.

    Returns
    -------
    dict with keys:
        ``X_aug`` -- final augmented feature matrix (n, p_orig + n_eng)
        ``names`` -- list of column names (orig + ``round{r}_pair{i,j}``)
        ``rounds`` -- list of per-round dicts with ``selected_pairs`` and ``engineered_features`` provenance.

    Caution
    -------
    Cascaded FE is prone to overfitting on small N. ``baseline_uplift_threshold=1.05`` (5% improvement over the HONEST trivial baseline) gates each engineered feature; round 2 typically adds 0-2 features instead of ``top_k_per_round``. Use ``validate_pair_fe_cv`` for nested-CV-validated cascading.
    """
    from .hermite_fe import optimise_hermite_pair
    from .fe_baselines import _mi_1d
    from itertools import combinations

    X = np.asarray(X, dtype=np.float64)
    _n, p_orig = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p_orig)]
    else:
        feature_names = list(feature_names)
    rounds_log = []
    cur_X = X.copy()
    cur_names = feature_names[:]

    for r in range(n_rounds):
        if verbose:
            logger.debug("[compose] round %d/%d, current X shape = %s", r + 1, n_rounds, cur_X.shape)
        # Single-feature MI ranking on current X.
        single_mi = []
        for j in range(cur_X.shape[1]):
            xj = cur_X[:, j]
            if np.std(xj) < 1e-12:
                continue
            mi = _mi_1d(xj, y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
            single_mi.append((j, mi))
        # Wave 58 (2026-05-20): plugin MI quantises -> ties realistic; secondary
        # key on feature index for deterministic top-K across runs.
        single_mi.sort(key=lambda kv: (-kv[1], kv[0]))
        top_idx = [j for j, _ in single_mi[: 2 * top_k_per_round]]
        if len(top_idx) < 2:
            if verbose:
                logger.debug("[compose] not enough features with MI>0; stopping")
            break
        # Pair candidates.
        pairs = list(combinations(top_idx, 2))
        # Score each pair via joint MI on the train data, take top ``top_k_per_round`` candidates for FE.
        pair_scores = []
        for i, j in pairs:
            xi, xj = cur_X[:, i], cur_X[:, j]
            mi_pair = max(
                _mi_1d(xi * xj, y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins),
                _mi_1d(xi + xj, y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins),
            )
            pair_scores.append((i, j, mi_pair))
        # Wave 58 (2026-05-20): secondary key on (i, j) for deterministic
        # pair selection across runs when MIs tie.
        pair_scores.sort(key=lambda kv: (-kv[2], kv[0], kv[1]))
        top_pairs = pair_scores[:top_k_per_round]

        new_cols = []
        new_names = []
        prov = []
        for i, j, _ in top_pairs:
            xi, xj = cur_X[:, i], cur_X[:, j]
            try:
                res = optimise_hermite_pair(
                    xi, xj, y,
                    discrete_target=discrete_target,
                    n_trials=n_trials, max_degree=max_degree,
                    basis=basis, mi_estimator=mi_estimator,
                    plugin_n_bins=plugin_n_bins,
                    use_trivial_baseline=use_trivial_baseline,
                    baseline_uplift_threshold=baseline_uplift_threshold,
                    optimizer=optimizer,
                )
            except Exception as e:
                if verbose:
                    logger.debug("pair (%d,%d) FE failed: %s", i, j, e)
                continue
            if res is None:
                continue
            engineered = res.transform(xi, xj)
            if not np.all(np.isfinite(engineered)):
                continue
            name = f"r{r + 1}_pair_{cur_names[i]}_{cur_names[j]}_{res.bin_func_name}"
            new_cols.append(engineered)
            new_names.append(name)
            prov.append({
                "round": r + 1, "i": i, "j": j,
                "name_i": cur_names[i], "name_j": cur_names[j],
                "bf": res.bin_func_name, "mi": res.mi,
                "uplift": res.uplift,
            })
        rounds_log.append({
            "round": r + 1,
            "selected_pairs": [(i, j) for i, j, _ in top_pairs],
            "engineered_features": prov,
            "n_added": len(new_cols),
        })
        if not new_cols:
            if verbose:
                logger.debug("[compose] round %d added no features; stopping", r + 1)
            break
        cur_X = np.column_stack([cur_X] + new_cols)
        cur_names = cur_names + new_names
        if verbose:
            logger.debug("[compose] round %d added %d features: %s", r + 1, len(new_cols), new_names)

    return {
        "X_aug": cur_X, "names": cur_names,
        "rounds": rounds_log,
    }


def validate_pair_fe_cv(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
    *,
    n_splits: int = 5,
    discrete_target: bool = True,
    basis: str = "chebyshev",
    n_trials: int = 40,
    max_degree: int = 3,
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    optimizer: str = "cma",
    use_trivial_baseline: bool = True,
    seed: int = 42,
) -> dict:
    """K-fold honest-uplift validator for pair-FE.

    For each fold:
      1. Fit coefficients on K-1 folds via ``optimise_hermite_pair``.
      2. Apply learned ``HermiteResult.transform`` to the held-out fold (NO additional fitting on heldout).
      3. Score MI of (transformed_heldout, y_heldout) using the same estimator the optimizer used.

    Compare to:
    * **In-sample MI** (``HermiteResult.mi``) -- the value the optimizer reports.
    * **Train-only baseline** (best trivial pair MI on full data).
    * **Out-of-sample MI** (mean across folds, with std).

    Returns dict with ``in_sample_mi``, ``oos_mean``, ``oos_std``, ``oos_per_fold``, ``optimism_ratio = in_sample_mi / oos_mean``. Optimism > 1.5 is suspect (bias from same-data fit + score).
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    from .hermite_fe import optimise_hermite_pair
    from .fe_baselines import _mi_1d

    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if discrete_target else KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # In-sample reference (full data).
    res_full = optimise_hermite_pair(
        x_a, x_b, y, discrete_target=discrete_target,
        n_trials=n_trials, max_degree=max_degree, basis=basis,
        mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
        use_trivial_baseline=use_trivial_baseline,
        baseline_uplift_threshold=0.0, optimizer=optimizer,
        seed=seed,
    )
    in_sample_mi = res_full.mi if res_full else 0.0

    oos_per_fold = []
    folds_with_pos_uplift = 0
    for fold_idx, (tr, va) in enumerate(cv.split(x_a.reshape(-1, 1), y)):
        try:
            res = optimise_hermite_pair(
                x_a[tr], x_b[tr], y[tr], discrete_target=discrete_target,
                n_trials=n_trials, max_degree=max_degree, basis=basis,
                mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
                use_trivial_baseline=use_trivial_baseline,
                baseline_uplift_threshold=0.0, optimizer=optimizer,
                seed=seed + fold_idx,
            )
        except Exception:
            res = None
        if res is None:
            oos_per_fold.append({
                "fold": fold_idx, "oos_mi": 0.0,
                "in_fold_mi": 0.0, "ratio": 0.0,
            })
            continue
        # Apply to held-out fold.
        eng_va = res.transform(x_a[va], x_b[va])
        if not np.all(np.isfinite(eng_va)):
            oos_per_fold.append({
                "fold": fold_idx, "oos_mi": 0.0,
                "in_fold_mi": res.mi, "ratio": 0.0,
            })
            continue
        oos_mi = _mi_1d(eng_va, y[va], discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
        # Compute trivial baseline on heldout for honest uplift.
        from .fe_baselines import best_trivial_pair

        triv = best_trivial_pair(x_a[va], x_b[va], y[va], discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
        triv_mi = triv[2] if triv else 0.0
        if oos_mi > triv_mi:
            folds_with_pos_uplift += 1
        oos_per_fold.append({
            "fold": fold_idx,
            "in_fold_mi": res.mi,
            "oos_mi": oos_mi,
            "trivial_oos_mi": triv_mi,
            "uplift_vs_trivial": oos_mi / max(triv_mi, 1e-9),
            "ratio": oos_mi / max(res.mi, 1e-9),
        })

    oos_arr = np.array([f["oos_mi"] for f in oos_per_fold])
    triv_arr = np.array([f.get("trivial_oos_mi", 0.0) for f in oos_per_fold])
    return {
        "in_sample_mi": in_sample_mi,
        "oos_mean": float(np.mean(oos_arr)),
        "oos_std": float(np.std(oos_arr)),
        "oos_per_fold": oos_per_fold,
        "optimism_ratio": (in_sample_mi / max(float(np.mean(oos_arr)), 1e-9) if in_sample_mi > 0 else 0.0),
        "trivial_oos_mean": float(np.mean(triv_arr)),
        "honest_uplift_vs_trivial": (float(np.mean(oos_arr)) / max(float(np.mean(triv_arr)), 1e-9)),
        "folds_with_positive_uplift": folds_with_pos_uplift,
    }
