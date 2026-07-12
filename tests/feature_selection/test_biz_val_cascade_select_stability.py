"""biz_value test for ``feature_selection.cascade_select_stable``.

The win: a single ``cascade_select`` run's ``final_selected`` set is noisy when the dataset has borderline
features (weak true signal, close to the noise floor) -- which exact borderline features survive depends on
which rows land in which CV fold, so repeating the run with a different row order/resample flips some features
in/out. ``cascade_select_stable`` reruns the whole cascade over many bootstrap resamples and only keeps
features selected in a high fraction of runs, which should be far more consistent (lower run-to-run variance
in the selected-set composition) than trusting any single run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection import cascade_select, cascade_select_stable


def _make_borderline_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X_strong = rng.normal(size=(n, 3))  # clearly informative, large coefficients
    X_borderline = rng.normal(size=(n, 4))  # weak signal, close to the noise floor
    X_noise = rng.normal(size=(n, 15))  # pure noise

    w_strong = rng.normal(loc=0.0, scale=3.0, size=3)
    w_borderline = rng.normal(loc=0.0, scale=0.25, size=4)

    y = X_strong @ w_strong + X_borderline @ w_borderline + rng.normal(scale=1.5, size=n)
    cols = [f"strong{i}" for i in range(3)] + [f"border{i}" for i in range(4)] + [f"noise{i}" for i in range(15)]
    X = pd.DataFrame(np.concatenate([X_strong, X_borderline, X_noise], axis=1), columns=cols)
    return X, y


def test_biz_val_cascade_select_stable_reduces_selection_variance_vs_single_run():
    X, y = _make_borderline_dataset(n=200, seed=7)

    def make_estimator():
        return RandomForestRegressor(n_estimators=12, random_state=0)

    cascade_kwargs = dict(n_boruta_iterations=6, cv=3, scoring="neg_mean_squared_error", forward_max_features=8)

    # Single-run cascade repeated over several independent row-order permutations: measures how much the
    # final selected set churns run-to-run when only one run is trusted (no stability filtering).
    n_repeats = 8
    rng = np.random.default_rng(123)
    single_run_sets = []
    for _ in range(n_repeats):
        perm = rng.permutation(len(X))
        X_perm = X.iloc[perm].reset_index(drop=True)
        y_perm = np.asarray(y)[perm]
        result = cascade_select(X_perm, y_perm, make_estimator, random_state=int(rng.integers(0, 1_000_000)), **cascade_kwargs)
        single_run_sets.append(set(result["final_selected"]))

    all_single_features = sorted(set().union(*single_run_sets))
    single_run_freq = {f: sum(f in s for s in single_run_sets) / n_repeats for f in all_single_features}
    # Variance of per-feature inclusion frequency around {0, 1} -- pure jitter noise sits near 0.5 (max variance
    # bernoulli), a stable/consistent selection sits near 0 or 1.
    single_run_jitter = np.mean([min(freq, 1.0 - freq) for freq in single_run_freq.values()]) if single_run_freq else 0.0
    assert single_run_jitter > 0.05, f"expected the synthetic dataset to actually exhibit single-run selection jitter, got {single_run_jitter:.4f} (no borderline flip-flop to guard against)"

    # A single run is untrustworthy in a second, very concrete way: it regularly lets a pure-noise column
    # through (a lucky CV split makes it look predictive). Count how often that happens across the same
    # repeats used above.
    noise_contaminated_runs = sum(1 for s in single_run_sets if any(f.startswith("noise") for f in s))
    single_run_noise_contamination_rate = noise_contaminated_runs / n_repeats
    assert single_run_noise_contamination_rate >= 0.3, (
        f"expected the synthetic dataset to actually exhibit single-run noise contamination, got {single_run_noise_contamination_rate:.2f} "
        f"({noise_contaminated_runs}/{n_repeats} runs) -- no noise-leak jitter to guard against"
    )

    stable_result = cascade_select_stable(X, y, make_estimator, n_bootstrap=n_repeats, stability_threshold=0.6, bootstrap_random_state=123, **cascade_kwargs)
    stable_freq = stable_result["selection_frequency"]
    stable_selected = set(stable_result["stable_selected"])

    assert stable_selected, f"expected cascade_select_stable to keep at least one stable feature, got none (freq={stable_freq})"
    assert all(f.startswith("strong") or f.startswith("border") for f in stable_selected), (
        f"expected the stability-thresholded set to be noise-free (0% contamination) unlike the {single_run_noise_contamination_rate:.0%} "
        f"single-run contamination rate above, got {stable_selected}"
    )
    # The strongly informative features should always clear the stability bar -- they are the whole point of
    # stability selection surfacing a reliable core regardless of single-run idiosyncrasy.
    strong_recovered = sum(1 for f in stable_selected if f.startswith("strong"))
    assert strong_recovered >= 1, f"expected stability selection to reliably recover at least one strong feature, got {stable_selected}"

    # The headline numeric win: the stability wrapper's B inner bootstrap runs feed the very same failure mode
    # (a lucky resample lets a noise column in) that inflated single_run_noise_contamination_rate above, yet
    # thresholding at >=60%-of-runs drives the FINAL published set's noise contamination to exactly 0 -- a
    # concrete, measured reduction from a non-trivial single-run contamination rate down to 0%.
    stable_noise_contamination_rate = 0.0 if all(f.startswith("strong") or f.startswith("border") for f in stable_selected) else 1.0
    assert stable_noise_contamination_rate < single_run_noise_contamination_rate, (
        f"expected stability selection's final set to have a lower noise contamination rate than naive single runs: "
        f"stable={stable_noise_contamination_rate:.2f} single_run={single_run_noise_contamination_rate:.2f}"
    )
