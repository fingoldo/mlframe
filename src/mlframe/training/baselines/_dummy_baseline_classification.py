"""Classification-side dummy baseline computation.

Wave 92 (2026-05-21): split out from `_dummy_baseline_compute.py` to keep
that file below the 1k-line threshold. Behaviour preserved bit-for-bit;
the function is re-exported from `_dummy_baseline_compute` so existing
`from ._dummy_baseline_compute import _compute_classification_baselines`
imports continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _compute_classification_baselines(
    target_name: str,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    timestamps_train: np.ndarray | None,
    cat_features: Sequence[str] | None,
    config: Any,
    target_type: str,
    n_classes: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build {baseline: probs} dicts for binary / multiclass.

    Returns ``(val_probs, test_probs, extras)`` where probs are
    ``(N, K)`` matrices.
    """
    # Lazy import to break the circular load with _dummy_baseline_compute.
    from ._dummy_baseline_compute import _per_target_seed, _pick_per_group_categorical, _per_group_predict

    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    seed = _per_target_seed(config.random_state, target_name)

    # Compute train priors
    train_y_int = train_y.astype(np.int64)
    bincounts = np.bincount(train_y_int, minlength=n_classes).astype(np.float64)
    train_prior = bincounts / bincounts.sum() if bincounts.sum() > 0 else np.full(n_classes, 1.0 / n_classes)

    # prior baseline: constant per-class prob = train prior
    prior_probs = np.tile(train_prior, (max(n_val, 1), 1)) if n_val > 0 else np.empty((0, n_classes))
    if n_val > 0:
        val_probs["prior"] = prior_probs
        test_probs["prior"] = np.tile(train_prior, (n_test, 1))

    # most_frequent: predict argmax of prior with one-hot probs
    most_freq_class = int(np.argmax(train_prior))
    mf_probs_row = np.zeros(n_classes)
    mf_probs_row[most_freq_class] = 1.0
    val_probs["most_frequent"] = np.tile(mf_probs_row, (n_val, 1))
    test_probs["most_frequent"] = np.tile(mf_probs_row, (n_test, 1))

    # uniform: 1/K per row
    uniform_probs_row = np.full(n_classes, 1.0 / n_classes)
    val_probs["uniform"] = np.tile(uniform_probs_row, (n_val, 1))
    test_probs["uniform"] = np.tile(uniform_probs_row, (n_test, 1))

    # all_zeros / all_ones (binary only)
    if target_type == "binary_classification" and n_classes == 2:
        # all-class-0: probs = [1, 0]
        z_row = np.array([1.0, 0.0])
        val_probs["all_zeros"] = np.tile(z_row, (n_val, 1))
        test_probs["all_zeros"] = np.tile(z_row, (n_test, 1))
        # all-class-1: probs = [0, 1]
        o_row = np.array([0.0, 1.0])
        val_probs["all_ones"] = np.tile(o_row, (n_val, 1))
        test_probs["all_ones"] = np.tile(o_row, (n_test, 1))

    # stratified: n_repeats over different seeds
    # Predicted class sampled from prior; probs = one-hot of sampled class.
    # The final prob per row is the MEAN over reps of one-hot[rng.choice].
    # Accumulator path: one (N, K) zeros allocation + per-rep += 1.0 at the
    # sampled cells, divided once at the end. The old per-rep
    # ``np.zeros((N, K))`` + one-hot fill + list.append + np.mean across the
    # whole stack allocated n_repeats * N * K * 8 bytes (~200 MB at N=200k /
    # K=3 / R=40) and added a final np.mean across a 4-D stack -- 9ms+ of
    # allocs and a few ms of stack-mean at the c0137 shape.
    n_repeats = config.stratified_n_repeats
    val_acc = np.zeros((n_val, n_classes)) if n_val > 0 else None
    test_acc = np.zeros((n_test, n_classes)) if n_test > 0 else None
    val_row_idx = np.arange(n_val) if n_val > 0 else None
    test_row_idx = np.arange(n_test) if n_test > 0 else None
    # Inline the inverse-CDF sampler that ``rng.choice(classes, size=N,
    # p=train_prior)`` runs internally: choice builds ``cdf = cumsum(p); cdf /=
    # cdf[-1]`` then returns ``cdf.searchsorted(rng.random(N), side="right")``
    # (replace=True). Computing the cdf ONCE (it's constant across reps) and
    # calling searchsorted directly skips choice's per-call probability
    # validation + array dispatch and the redundant per-rep cumsum; ``classes``
    # is ``arange(n_classes)`` so the searchsorted result is already the class
    # index (no gather). Bit-identical to the old rng.choice output for the
    # same seed and draw order (the legacy-equivalence regression test pins
    # this), ~1.16x per rep at n=150k.
    _cdf = np.cumsum(train_prior)
    _cdf = _cdf / _cdf[-1]
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if val_acc is not None:
            val_classes = _cdf.searchsorted(rng.random(n_val), side="right")
            val_acc[val_row_idx, val_classes] += 1.0
        if test_acc is not None:
            test_classes = _cdf.searchsorted(rng.random(n_test), side="right")
            test_acc[test_row_idx, test_classes] += 1.0
    if val_acc is not None and n_repeats > 0:
        val_probs["stratified"] = val_acc / n_repeats
    if test_acc is not None and n_repeats > 0:
        test_probs["stratified"] = test_acc / n_repeats
    extras["stratified_n_repeats"] = n_repeats

    # per_group_prior (binary only for now)
    if target_type == "binary_classification":
        cat_col = _pick_per_group_categorical(
            train_X, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
        )
        if cat_col is not None:
            try:
                _, val_pg, test_pg, pg_diag = _per_group_predict(
                    train_X, val_X, test_X, train_y.astype(np.float64), cat_col, target_type,
                )
                # Convert to (N, 2) probs: [1-p, p]
                val_pg_2d = np.column_stack([1 - val_pg, val_pg])
                test_pg_2d = np.column_stack([1 - test_pg, test_pg])
                label = "per_group_prior"
                if pg_diag["repeat_entity_rate"] >= config.per_group_high_overlap_threshold:
                    label = f"per_group_prior (high_entity_overlap={pg_diag['repeat_entity_rate']:.2f})"
                val_probs[label] = val_pg_2d
                test_probs[label] = test_pg_2d
                extras["per_group"] = {"cat_col": cat_col, **pg_diag}
                if (
                    pg_diag["val_coverage_pct"] < config.per_group_min_val_coverage_pct
                    or pg_diag["test_coverage_pct"] < config.per_group_min_val_coverage_pct
                ):
                    extras.setdefault("strongest_pick_excluded", []).append(label)
            except Exception as e:
                logger.info(
                    "[dummy-baselines] target='%s' per_group_prior failed (%s); skipping",
                    target_name, e,
                )

    return val_probs, test_probs, extras
