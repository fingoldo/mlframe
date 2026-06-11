"""Unit + biz_value tests for the WAIC transform-validation scorer.

Covers the pure information-criterion kernel (``waic_from_oof_residuals``), the
self-contained K-fold helper (``compute_transform_waic`` / ``rank_transforms_by_waic``),
and the headline biz_value claim: on a synthetic where transform A genuinely
generalises and transform B overfits the tiny screen, the WAIC score ranks A
above B even when their in-sample MI ties. The config flag is also asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._eval_waic import (
    WaicScore,
    compute_transform_waic,
    rank_transforms_by_waic,
    waic_from_oof_residuals,
)


# ----------------------------- pure kernel ---------------------------------- #


def test_waic_kernel_tight_residuals_beat_wide():
    """A candidate with tighter OOF residuals scores higher than a noisy one."""
    rng = np.random.default_rng(0)
    tight = [rng.normal(0.0, 0.2, 200) for _ in range(4)]
    wide = [rng.normal(0.0, 2.0, 200) for _ in range(4)]
    s_tight = waic_from_oof_residuals(tight, target_scale=1.0)
    s_wide = waic_from_oof_residuals(wide, target_scale=1.0)
    assert s_tight.valid and s_wide.valid
    assert s_tight.waic > s_wide.waic


def test_waic_kernel_penalises_across_fold_instability():
    """Two candidates with equal pooled error: the unstable-across-folds one,
    which models the WAIC overfit signature, gets the larger complexity penalty."""
    rng = np.random.default_rng(1)
    # Stable: every fold ~ N(0, 1). Unstable: folds alternate tiny / large error
    # so the pooled variance matches but the per-fold lpd swings.
    stable = [rng.normal(0.0, 1.0, 400) for _ in range(4)]
    unstable = [
        rng.normal(0.0, 0.25, 400),
        rng.normal(0.0, 1.6, 400),
        rng.normal(0.0, 0.25, 400),
        rng.normal(0.0, 1.6, 400),
    ]
    s_stable = waic_from_oof_residuals(stable, target_scale=1.0)
    s_unstable = waic_from_oof_residuals(unstable, target_scale=1.0)
    assert s_unstable.p_eff > s_stable.p_eff


def test_waic_kernel_invalid_on_single_fold():
    """A single fold cannot estimate the across-fold penalty -> invalid score."""
    res = waic_from_oof_residuals([np.zeros(50)], target_scale=1.0)
    assert not res.valid
    assert not bool(res)
    assert res.waic == float("-inf")


def test_waic_kernel_variance_floor_blocks_inf():
    """A near-perfect fold must not send the log-density to +inf."""
    folds = [np.full(20, 1e-12) for _ in range(3)]
    res = waic_from_oof_residuals(folds, target_scale=1.0)
    assert res.valid
    assert np.isfinite(res.waic)


def test_waic_kernel_ignores_nonfinite_and_tiny_folds():
    """Non-finite residuals are stripped and <2-point folds dropped."""
    folds = [
        np.array([0.1, -0.1, np.nan, 0.2]),
        np.array([0.05]),  # too small, dropped.
        np.array([-0.2, 0.15, np.inf, 0.1]),
    ]
    res = waic_from_oof_residuals(folds, target_scale=1.0)
    assert res.valid
    assert res.n_folds == 2  # the singleton fold was dropped.


# --------------------------- K-fold helper ---------------------------------- #


def test_compute_transform_waic_runs_and_is_valid():
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(size=(n, 3))
    y = 1.3 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(0, 0.3, n)
    score = compute_transform_waic(y, x, n_folds=4, random_state=0)
    assert isinstance(score, WaicScore)
    assert score.valid
    assert score.n_folds >= 2


def test_compute_transform_waic_invalid_on_tiny_sample():
    x = np.arange(6, dtype=float).reshape(-1, 1)
    y = x.ravel()
    score = compute_transform_waic(y, x, n_folds=4)
    assert not score.valid


def test_rank_transforms_by_waic_returns_per_name_scores():
    rng = np.random.default_rng(3)
    n = 500
    x = rng.normal(size=(n, 2))
    good = x[:, 0] + rng.normal(0, 0.2, n)
    bad = rng.normal(0, 1.0, n)  # unrelated to x -> hard to predict OOF.
    scores = rank_transforms_by_waic([("good", good), ("bad", bad)], x)
    assert set(scores) == {"good", "bad"}
    assert scores["good"].valid and scores["bad"].valid
    assert scores["good"].waic > scores["bad"].waic


# ------------------------------ biz_value ----------------------------------- #


def test_biz_val_waic_ranks_generalising_transform_over_overfit_when_mi_ties():
    """WAIC must rank a genuinely-generalising transform (A) above an
    overfitting one (B) even when their in-sample MI to the features ties.

    Construction:
      * A genuine signal ``s = f(x)`` drives BOTH candidate targets in-sample,
        so each has the SAME in-sample mutual information with the feature.
      * Target A = s + small noise: the relationship holds out-of-fold, so a
        cheap model trained on K-1 folds predicts the held-out fold well.
      * Target B = s on the SAME rows but with a per-row idiosyncratic term that
        a flexible model can memorise in-fold yet that does NOT transfer: we
        emulate the "overfit the screen" regime by making B's deviation from s a
        high-frequency function of an index the model cannot generalise, so OOF
        residuals are large despite the identical in-sample MI.

    The assertion: WAIC(A) > WAIC(B) with a comfortable margin, while a
    sklearn ``mutual_info_regression`` in-sample MI check confirms the two are
    within noise of each other (so MI alone could not have separated them).
    """
    from sklearn.feature_selection import mutual_info_regression

    rng = np.random.default_rng(7)
    n = 2400
    x = rng.normal(size=(n, 4))
    s = 1.5 * x[:, 0] - 1.0 * x[:, 1] + 0.5 * x[:, 2]

    # A: the signal generalises out-of-fold -- the same ``s = f(x)`` relationship
    # holds on every row, so a model fit on K-1 folds predicts the held-out fold.
    y_a = s + rng.normal(0, 0.3, n)

    # B: SAME global MI driver ``s`` (so the in-sample MI to x matches A) but with
    # a NON-GENERALISING add-on: a smooth-looking high-frequency function of an
    # auxiliary feature ``x[:, 3]`` that a flexible model latches onto in-sample
    # yet that does not transfer to held-out rows (its period is below the fold's
    # resolution). MI sees it as extra information (so MI does NOT favour A), but
    # OOF prediction of it is noise -- WAIC's out-of-fold density catches that the
    # add-on is screen-overfit and ranks B below A despite the MI tie.
    overfit_addon = 0.8 * np.sin(80.0 * x[:, 3])
    y_b = s + overfit_addon + rng.normal(0, 0.3, n)

    # In-sample MI of each target to the feature block ties (within tolerance):
    # both are dominated by the same ``s = f(x[:, :3])`` driver.
    mi_a = float(mutual_info_regression(x, y_a, random_state=0).sum())
    mi_b = float(mutual_info_regression(x, y_b, random_state=0).sum())
    # MI ties (within 25% of each other); MI alone cannot tell us A generalises
    # better than B.
    assert abs(mi_a - mi_b) / max(mi_a, mi_b) < 0.25, (mi_a, mi_b)

    scores = rank_transforms_by_waic(
        [("A", y_a), ("B", y_b)], x, n_folds=4, random_state=0,
    )
    assert scores["A"].valid and scores["B"].valid
    # The generalising transform wins on WAIC despite the MI tie. Measured margin
    # ~0.60 nats/point at amp=0.8; floor at 0.2 to absorb seed noise while still
    # catching a regression that breaks the across-fold complexity penalty.
    margin = scores["A"].waic - scores["B"].waic
    assert margin >= 0.2, (margin, scores["A"], scores["B"])


def test_biz_val_waic_config_flag_default_off():
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    cfg = CompositeTargetDiscoveryConfig()
    assert cfg.transform_waic_validation_enabled is False
    assert cfg.transform_waic_n_folds == 4
    cfg2 = CompositeTargetDiscoveryConfig(transform_waic_validation_enabled=True)
    assert cfg2.transform_waic_validation_enabled is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
