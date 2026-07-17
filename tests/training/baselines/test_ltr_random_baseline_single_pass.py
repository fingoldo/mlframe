"""Regression: ``random_within_query`` LTR baseline computes ONE rng pass
per split (iter121, 2026-05-21).

Pre-iter121 the code built a list of ``n_repeats`` independent random
vectors per split, then kept only ``[0]`` -- wasting ~9 * (n_val + n_test)
PRNG draws per fit. Averaging across repeats would converge to a constant
0.5 (mean of i.i.d. uniforms), collapsing the baseline to the degenerate
``mean_relevance`` variant; storing only the first run was the correct
semantic. This test pins:

  (1) Bit-equivalence between the prior ``runs[0]`` value and the new
      single-pass output (the rng draws ``val`` then ``test`` from the
      same default_rng(seed) state, so consuming val advances the state
      before test gets its draw -- exactly matching the legacy r=0 path).
  (2) ``random_within_query_n_repeats`` extra is still stamped so any
      downstream metadata reader detects the simplification.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.baselines._dummy_baseline_compute import _per_target_seed


class _LtrCfg:
    random_state = 7
    random_within_query_n_repeats = 10
    stratified_n_repeats = 5
    per_group_max_cardinality_ratio = 0.5
    per_group_high_overlap_threshold = 0.5
    per_group_min_val_coverage_pct = 50.0


def _legacy_first_run(seed: int, n_val: int, n_test: int, n_repeats: int):
    """Pre-iter121 reference: build the list, keep only [0]."""
    val_runs, test_runs = [], []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        val_runs.append(rng.random(n_val) if n_val > 0 else np.array([]))
        test_runs.append(rng.random(n_test) if n_test > 0 else np.array([]))
    return (val_runs[0] if val_runs else np.array([]), test_runs[0] if test_runs else np.array([]))


def test_random_within_query_matches_legacy_first_run():
    from mlframe.training.baselines.dummy import _compute_ltr_baselines

    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 600, 200, 200
    train_y = rng.integers(0, 4, size=n_train).astype(np.float64)
    val_y = rng.integers(0, 4, size=n_val).astype(np.float64)
    test_y = rng.integers(0, 4, size=n_test).astype(np.float64)
    # ~10 docs per query in each split.
    g_train = np.repeat(np.arange(n_train // 10), 10)
    g_val = np.repeat(np.arange(n_val // 10), 10)
    g_test = np.repeat(np.arange(n_test // 10), 10)

    val_preds, test_preds, extras = _compute_ltr_baselines(
        target_name="t",
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        group_ids_train=g_train,
        group_ids_val=g_val,
        group_ids_test=g_test,
        ts_train=None,
        ts_val=None,
        ts_test=None,
        config=_LtrCfg(),
    )

    seed = _per_target_seed(_LtrCfg.random_state, "t")
    legacy_val, legacy_test = _legacy_first_run(
        seed,
        n_val,
        n_test,
        n_repeats=_LtrCfg.random_within_query_n_repeats,
    )

    assert "random_within_query" in val_preds
    assert "random_within_query" in test_preds
    assert np.array_equal(val_preds["random_within_query"], legacy_val), "post-iter121 val random_within_query must be bit-identical to legacy [0]"
    assert np.array_equal(test_preds["random_within_query"], legacy_test), "post-iter121 test random_within_query must be bit-identical to legacy [0]"
    assert extras["random_within_query_n_repeats"] == _LtrCfg.random_within_query_n_repeats


def test_random_within_query_handles_empty_splits():
    from mlframe.training.baselines.dummy import _compute_ltr_baselines

    rng = np.random.default_rng(0)
    n_train = 600
    train_y = rng.integers(0, 4, size=n_train).astype(np.float64)
    g_train = np.repeat(np.arange(n_train // 10), 10)

    # Empty val + test: the baseline must still produce empty arrays
    # without raising.
    val_preds, test_preds, _ = _compute_ltr_baselines(
        target_name="t",
        train_y=train_y,
        val_y=np.array([], dtype=np.float64),
        test_y=np.array([], dtype=np.float64),
        group_ids_train=g_train,
        group_ids_val=np.array([]),
        group_ids_test=np.array([]),
        ts_train=None,
        ts_val=None,
        ts_test=None,
        config=_LtrCfg(),
    )
    assert val_preds.get("random_within_query", np.array([])).shape == (0,)
    assert test_preds.get("random_within_query", np.array([])).shape == (0,)
