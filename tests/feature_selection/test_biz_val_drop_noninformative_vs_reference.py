"""biz_value test for ``feature_selection.drop_noninformative_vs_reference.drop_noninformative_vs_reference``.

The win (3rd_mechanisms-of-action-moa-prediction.md): among features measured for both a reference/control
cohort and the rest (e.g. treated samples), some carry pure batch/instrument noise with NO relationship to
the control-vs-treated distinction, while others genuinely shift between the two groups (the actual signal).
A KS-test p > alpha correctly identifies the noise features (fails to reject "same distribution"), while a
downstream model trained keeping only the KS-significant features should retain (or improve) predictive power
versus keeping everything, since the dropped columns carried no discriminative information about the very
distinction being modeled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.drop_noninformative_vs_reference import drop_noninformative_vs_reference


def _make_control_treated_dataset(n: int, n_signal: int, n_noise: int, seed: int):
    """Make control treated dataset."""
    rng = np.random.default_rng(seed)
    is_treated = rng.integers(0, 2, n).astype(bool)

    cols = {}
    for i in range(n_signal):
        # genuinely shifts between control and treated
        cols[f"signal{i}"] = rng.normal(loc=np.where(is_treated, 2.0, 0.0), scale=1.0)
    for i in range(n_noise):
        # identical distribution regardless of treated/control -- pure batch noise
        cols[f"noise{i}"] = rng.normal(loc=0.0, scale=1.0, size=n)

    df = pd.DataFrame(cols)
    reference_mask = ~is_treated  # control = reference
    y = is_treated.astype(int)
    return df, reference_mask, y


def test_biz_val_drop_noninformative_vs_reference_correctly_identifies_noise_columns():
    """Biz val drop noninformative vs reference correctly identifies noise columns."""
    df, reference_mask, _y = _make_control_treated_dataset(n=2000, n_signal=5, n_noise=15, seed=0)

    kept = drop_noninformative_vs_reference(df, reference_mask, alpha=0.1)

    n_signal_kept = sum(1 for c in kept if c.startswith("signal"))
    n_noise_kept = sum(1 for c in kept if c.startswith("noise"))
    assert (
        n_signal_kept == 0
    ), f"expected the drop-candidate list to contain NO genuinely-shifting signal columns, got {[c for c in kept if c.startswith('signal')]}"
    assert n_noise_kept >= 12, f"expected most pure-noise columns to be correctly flagged as drop candidates, got {n_noise_kept}/15"


def test_biz_val_dropping_noninformative_columns_preserves_downstream_auc():
    """Biz val dropping noninformative columns preserves downstream auc."""
    df, reference_mask, y = _make_control_treated_dataset(n=2000, n_signal=5, n_noise=15, seed=1)
    drop_candidates = drop_noninformative_vs_reference(df, reference_mask, alpha=0.1)
    kept_cols = [c for c in df.columns if c not in drop_candidates]

    auc_full = cross_val_score(LogisticRegression(max_iter=500), df, y, cv=5, scoring="roc_auc").mean()
    auc_pruned = cross_val_score(LogisticRegression(max_iter=500), df[kept_cols], y, cv=5, scoring="roc_auc").mean()

    assert len(kept_cols) < df.shape[1], "expected the pruned set to be strictly smaller than the full feature set"
    assert (
        auc_pruned >= auc_full - 0.02
    ), f"expected pruning non-informative columns to not meaningfully hurt downstream AUC, got pruned={auc_pruned:.4f} full={auc_full:.4f}"


def _make_two_cohort_dataset(seed: int):
    """One genuinely informative feature ("tricky") that, by chance, looks similar to a SMALL reference
    cohort (weak/no shift, easy for a single KS-test to miss at n=20) but clearly differs from a second,
    LARGE reference cohort (strong shift, reliably detected) -- plus a real noise column as a sanity check.
    """
    rng = np.random.default_rng(seed)
    n_treated, n_small_cohort, n_large_cohort = 800, 20, 300

    treated = rng.normal(loc=0.0, scale=1.0, size=n_treated)
    small_cohort = rng.normal(loc=0.15, scale=1.0, size=n_small_cohort)  # weak shift -- easily missed at n=20
    large_cohort = rng.normal(loc=2.0, scale=1.0, size=n_large_cohort)  # strong shift -- reliably detected

    tricky = np.concatenate([treated, small_cohort, large_cohort])
    noise = rng.normal(loc=0.0, scale=1.0, size=n_treated + n_small_cohort + n_large_cohort)

    df = pd.DataFrame({"tricky": tricky, "noise": noise})
    is_small = np.zeros(len(df), dtype=bool)
    is_small[n_treated : n_treated + n_small_cohort] = True
    is_large = np.zeros(len(df), dtype=bool)
    is_large[n_treated + n_small_cohort :] = True
    return df, is_small, is_large


def test_biz_val_drop_noninformative_vs_reference_multi_cohort_avoids_false_drop():
    """Single-cohort mode, screened only against the small cohort, spuriously flags the genuinely
    informative "tricky" column as a drop candidate whenever the KS-test happens to miss the weak shift at
    n=20 -- a real false-drop risk from relying on one reference batch. Multi-cohort mode (opt-in via
    ``require_all_cohorts=True``, passing both cohorts) only drops a feature that is noninformative against
    EVERY cohort, so the large cohort's clear shift protects "tricky" from ever being falsely dropped, while
    the genuine noise column is still correctly flagged as a drop candidate by both modes.
    """
    n_seeds = 40
    single_false_drops = 0
    multi_false_drops = 0
    single_noise_hits = 0
    multi_noise_hits = 0

    for seed in range(n_seeds):
        df, is_small, is_large = _make_two_cohort_dataset(seed)

        single_kept = drop_noninformative_vs_reference(df, is_small, alpha=0.1)
        if "tricky" in single_kept:
            single_false_drops += 1
        if "noise" in single_kept:
            single_noise_hits += 1

        multi_kept = drop_noninformative_vs_reference(df, [is_small, is_large], alpha=0.1, require_all_cohorts=True)
        if "tricky" in multi_kept:
            multi_false_drops += 1
        if "noise" in multi_kept:
            multi_noise_hits += 1

    single_rate = single_false_drops / n_seeds
    multi_rate = multi_false_drops / n_seeds
    assert single_rate >= 0.3, f"expected single-cohort mode to falsely drop the informative column often (weak shift at n=20), got rate={single_rate:.2f}"
    assert (
        multi_rate == 0.0
    ), f"expected multi-cohort mode to NEVER falsely drop the informative column (protected by the large cohort's clear shift), got rate={multi_rate:.2f}"
    assert (
        single_noise_hits >= n_seeds - 6
    ), f"expected single-cohort mode to still correctly flag genuine noise most of the time (n=20 KS test has some irreducible false-negative rate), got {single_noise_hits}/{n_seeds}"
    # multi-cohort mode requires the noise column to fail to differ against BOTH cohorts, so its hit rate is
    # necessarily <= the single-cohort rate (the small n=20 cohort alone already has a nontrivial miss rate) --
    # it should still catch a clear majority, never falling anywhere close to the false-drop-avoidance floor.
    assert (
        multi_noise_hits >= n_seeds * 0.6
    ), f"expected multi-cohort mode to still correctly flag genuine noise in a clear majority of seeds, got {multi_noise_hits}/{n_seeds}"
