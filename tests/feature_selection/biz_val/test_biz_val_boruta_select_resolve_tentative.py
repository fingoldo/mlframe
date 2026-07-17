"""biz_value test for ``boruta_select``'s opt-in ``resolve_tentative`` multi-round decision mode.

Gap in the original single-shot mode: it runs ONE binomial test per feature at the final round, uncorrected
across the simultaneously-tested features -- with many candidate columns this is a classic multiple-comparisons
problem (family-wise error rate inflation): at ``alpha=0.05`` and, say, 20 pure-noise columns, roughly one is
expected to cross significance by chance alone and get wrongly "confirmed" every run.

``resolve_tentative=True`` (``correction="bonferroni"``) instead tests every still-undecided feature EVERY round
against a threshold corrected for both the repeated per-round testing and the simultaneously-tested feature
count, and never falls back to an uncorrected end-of-run test -- a feature that isn't significant under the
correction stays "tentative" rather than being handed a lucky uncorrected pass.

The win measured below: across 12 seeded synthetic datasets (2 genuinely weak-but-real signal columns + 20 pure
noise columns each), the single-shot mode falsely "confirms" 5 noise columns in total (one dataset even confirms
2), while the corrected multi-round mode falsely confirms 0 -- at a real, honestly-reported cost in recall (13 of
24 possible true confirms recovered vs. 17), the expected trade-off of controlling the family-wise error rate
rather than testing every feature independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection.filters._boruta import boruta_select

N_SEEDS = 12
N = 200
N_WEAK = 2
N_NOISE = 20
N_ITERATIONS = 15


def _importance_fn(X, y):
    """Importance fn."""
    model = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=0, n_jobs=1)
    model.fit(X, y)
    return model.feature_importances_


def _make_data(seed: int):
    """Make data."""
    rng = np.random.default_rng(seed)
    cols = {}
    y = np.zeros(N)
    for i in range(N_WEAK):
        x = rng.normal(0, 1, N)
        cols[f"weak_{i}"] = x
        y += 0.35 * x
    for i in range(N_NOISE):
        cols[f"noise_{i}"] = rng.normal(0, 1, N)
    y += rng.normal(0, 1.2, N)
    return pd.DataFrame(cols), y


def test_biz_val_boruta_select_resolve_tentative_cuts_false_confirms():
    """Biz val boruta select resolve tentative cuts false confirms."""
    total_false_confirm_single = 0
    total_false_confirm_multi = 0
    total_true_confirm_single = 0
    total_true_confirm_multi = 0

    for seed in range(N_SEEDS):
        X, y = _make_data(seed)

        res_single = boruta_select(X, y, _importance_fn, n_iterations=N_ITERATIONS, random_state=seed)
        res_multi = boruta_select(
            X,
            y,
            _importance_fn,
            n_iterations=N_ITERATIONS,
            random_state=seed,
            resolve_tentative=True,
            correction="bonferroni",
        )
        by_single = dict(zip(res_single["feature_names"], res_single["decision"]))
        by_multi = dict(zip(res_multi["feature_names"], res_multi["decision"]))

        total_false_confirm_single += sum(1 for k, v in by_single.items() if k.startswith("noise") and v == "confirmed")
        total_false_confirm_multi += sum(1 for k, v in by_multi.items() if k.startswith("noise") and v == "confirmed")
        total_true_confirm_single += sum(1 for k, v in by_single.items() if k.startswith("weak") and v == "confirmed")
        total_true_confirm_multi += sum(1 for k, v in by_multi.items() if k.startswith("weak") and v == "confirmed")

    # Measured: single=5, multi=0 false confirms across N_SEEDS*N_NOISE=240 noise-column decisions.
    assert total_false_confirm_single >= 4, total_false_confirm_single
    assert total_false_confirm_multi <= 1, total_false_confirm_multi

    # Measured: single=17, multi=13 true confirms out of N_SEEDS*N_WEAK=24 possible -- corrected mode still
    # recovers a clear majority of the real signal, not just suppressing everything to "tentative".
    assert total_true_confirm_multi >= 10, total_true_confirm_multi


def test_boruta_select_resolve_tentative_default_off_is_bit_identical():
    """``resolve_tentative`` is opt-in -- omitting it must reproduce the exact single-shot decisions/counts."""
    X, y = _make_data(seed=0)
    baseline = boruta_select(X, y, _importance_fn, n_iterations=N_ITERATIONS, random_state=0)
    explicit_off = boruta_select(X, y, _importance_fn, n_iterations=N_ITERATIONS, random_state=0, resolve_tentative=False)

    assert baseline["decision"] == explicit_off["decision"]
    assert np.array_equal(baseline["hit_counts"], explicit_off["hit_counts"])
    assert np.array_equal(baseline["win_rate"], explicit_off["win_rate"])
    assert baseline["n_rounds_run"] == explicit_off["n_rounds_run"] == N_ITERATIONS


def test_boruta_select_convergence_rounds_stops_early_same_confirmed_set():
    """Early stopping (opt-in) must reach the SAME confirmed set as running the full cap, using fewer rounds."""
    rng = np.random.default_rng(3)
    n = 400
    x_rel_1 = rng.normal(0, 1, n)
    x_rel_2 = rng.normal(0, 1, n)
    x_noise = rng.normal(0, 1, n)
    y = 3.0 * x_rel_1 - 2.0 * x_rel_2 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"rel_1": x_rel_1, "rel_2": x_rel_2, "noise": x_noise})

    full = boruta_select(
        X,
        y,
        _importance_fn,
        n_iterations=30,
        random_state=3,
        resolve_tentative=True,
        correction="bonferroni",
    )
    early = boruta_select(
        X,
        y,
        _importance_fn,
        n_iterations=30,
        random_state=3,
        resolve_tentative=True,
        correction="bonferroni",
        convergence_rounds=3,
    )

    full_confirmed = {n for n, d in zip(full["feature_names"], full["decision"]) if d == "confirmed"}
    early_confirmed = {n for n, d in zip(early["feature_names"], early["decision"]) if d == "confirmed"}
    assert early_confirmed == full_confirmed
    assert early["n_rounds_run"] <= full["n_rounds_run"]
