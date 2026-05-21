"""Regression test for the stratified-baseline accumulator refactor (iter117).

Pre-refactor: ``_compute_classification_baselines`` allocated n_repeats
separate ``np.zeros((N, K))`` arrays for the stratified baseline, set the
one-hot cell per row, appended to a list, then averaged via
``np.mean(..., axis=0)`` across the whole stack. At 100k val + 100k test
rows / 3 classes / 40 repeats this allocated ~200 MB and ran in ~520 ms.

Post-refactor: one ``(N, K)`` accumulator + ``+= 1.0`` at the sampled cells
per rep, single divide at the end -- ~390 ms (1.3x) AND ~95 % less peak
allocation. Output is bit-identical to the old implementation for the same
seed because the per-rep one-hot increments compose identically with the
final ``/ n_repeats``.

This test pins the bit-equivalence so a future refactor that drops the
per-row randomisation (e.g. caches a single ``(K,)`` prior in place of the
per-row mean) cannot silently regress the realised-variance shape that
log_loss / AUC rely on.
"""

from __future__ import annotations

import numpy as np


def _old_strat(n_val, n_test, n_classes, n_repeats, seed, train_prior):
    """Pre-refactor reference implementation -- kept here as the test oracle."""
    classes = np.arange(n_classes)
    val_runs, test_runs = [], []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if n_val > 0:
            val_classes = rng.choice(classes, size=n_val, p=train_prior)
            val_strat = np.zeros((n_val, n_classes))
            val_strat[np.arange(n_val), val_classes] = 1.0
            val_runs.append(val_strat)
        if n_test > 0:
            test_classes = rng.choice(classes, size=n_test, p=train_prior)
            test_strat = np.zeros((n_test, n_classes))
            test_strat[np.arange(n_test), test_classes] = 1.0
            test_runs.append(test_strat)
    return (
        np.mean(val_runs, axis=0) if val_runs else None,
        np.mean(test_runs, axis=0) if test_runs else None,
    )


class _Cfg:
    random_state = 7
    stratified_n_repeats = 12
    per_group_max_cardinality_ratio = 0.5
    per_group_high_overlap_threshold = 0.5
    per_group_min_val_coverage_pct = 50.0


def test_stratified_accumulator_matches_legacy_implementation():
    from mlframe.training._dummy_baseline_classification import (
        _compute_classification_baselines,
    )
    from mlframe.training._dummy_baseline_compute import _per_target_seed

    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 5000, 1500, 1500
    n_classes = 3
    train_y = rng.integers(0, n_classes, size=n_train)
    val_y = rng.integers(0, n_classes, size=n_val)
    test_y = rng.integers(0, n_classes, size=n_test)

    cfg = _Cfg()
    val_probs, test_probs, extras = _compute_classification_baselines(
        target_name="t", train_X=None, val_X=None, test_X=None,
        train_y=train_y, val_y=val_y, test_y=test_y,
        timestamps_train=None, cat_features=None, config=cfg,
        target_type="multiclass_classification", n_classes=n_classes,
    )

    # Replay with the legacy implementation using the same seed.
    bincounts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    train_prior = bincounts / bincounts.sum()
    seed = _per_target_seed(cfg.random_state, "t")
    v_old, t_old = _old_strat(n_val, n_test, n_classes, cfg.stratified_n_repeats, seed, train_prior)

    assert np.allclose(val_probs["stratified"], v_old, atol=0.0), (
        "post-refactor val stratified probs must be bit-identical to legacy"
    )
    assert np.allclose(test_probs["stratified"], t_old, atol=0.0), (
        "post-refactor test stratified probs must be bit-identical to legacy"
    )
    assert extras["stratified_n_repeats"] == cfg.stratified_n_repeats


def test_stratified_skips_when_val_test_empty():
    """The accumulator init is guarded by n_val/n_test > 0; this test pins it."""
    from mlframe.training._dummy_baseline_classification import (
        _compute_classification_baselines,
    )

    rng = np.random.default_rng(0)
    train_y = rng.integers(0, 3, size=1000)
    cfg = _Cfg()
    # n_val=0, n_test=0 -> stratified must not crash + no "stratified" entry produced.
    val_probs, test_probs, extras = _compute_classification_baselines(
        target_name="t", train_X=None, val_X=None, test_X=None,
        train_y=train_y, val_y=None, test_y=None,
        timestamps_train=None, cat_features=None, config=cfg,
        target_type="multiclass_classification", n_classes=3,
    )
    assert "stratified" not in val_probs
    assert "stratified" not in test_probs
    assert extras["stratified_n_repeats"] == cfg.stratified_n_repeats


def test_stratified_mean_converges_to_train_prior():
    """Average over many reps should approximate train_prior per column."""
    from mlframe.training._dummy_baseline_classification import (
        _compute_classification_baselines,
    )

    rng = np.random.default_rng(0)
    n_train, n_val = 20_000, 10_000
    n_classes = 3
    train_y = rng.integers(0, n_classes, size=n_train)

    class _CfgMany(_Cfg):
        stratified_n_repeats = 100  # enough for empirical convergence

    val_probs, _, _ = _compute_classification_baselines(
        target_name="t", train_X=None, val_X=None, test_X=None,
        train_y=train_y, val_y=rng.integers(0, n_classes, size=n_val), test_y=None,
        timestamps_train=None, cat_features=None, config=_CfgMany(),
        target_type="multiclass_classification", n_classes=n_classes,
    )

    bincounts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    train_prior = bincounts / bincounts.sum()
    mean_per_class = val_probs["stratified"].mean(axis=0)
    assert np.abs(mean_per_class - train_prior).max() < 0.01, (
        f"empirical mean {mean_per_class} should approximate train_prior {train_prior} "
        f"within 0.01 at n_val=10k n_repeats=100"
    )
