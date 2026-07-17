"""Wave-24 sensor: off-by-one + sentinel collision fixes (4 sites).

#1 P0 estimators/early_stopping.py:49 -- ``X[:-n_val_samples]`` with
   n_val_samples=0 silently returns EMPTY (Python's ``arr[:-0]``
   semantics). Pre-fix small-X + small-fraction silently collapsed
   training to no rows, no exception. Post-fix clamps to >=1 +
   raises when training would be empty.

#2 P0 calibration/probabilities.py:50 -- ``np.empty`` + chunked loop
   left tail rows (when n % chunk_size != 0) uninitialised. The
   final np.clip(0, 1) only fixed values outside [0,1]; garbage in
   range slipped through. Post-fix uses np.zeros init + explicit
   tail-row synthesis.

#3 P1 training/composite_screening.py:292 -- size-gate filtered on
   ``feature_binned[:, 0] >= 0`` (column 0's -1 sentinel = NaN row);
   when column 0 was NaN-heavy but other columns clean, MI returned
   0 for EVERY feature in the batch. Post-fix gates on target finite
   only; the inner per-column loop already filters its own NaN.

#4 P2 training/_reporting.py:820 -- ``np.searchsorted(classes_, ...)``
   IndexError if predict() returns an unseen value. Post-fix uses a
   dict lookup + warn-log on unseen + map-to-class-0 fallback.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


# ---- #1 early_stopping zero-val-samples ----------------------------------


def _make_es_estimator(validation_fraction):
    """Build an EarlyStoppingEstimator over a partial_fit-spy base model.

    The wrapper clones ``base_model`` (sklearn contract: never mutate the caller's
    instance), so the spy is a ``BaseEstimator`` to be cloneable; the FITTED clone is
    reachable post-fit as ``est.estimator_`` -- that is where ``train_sizes`` lands.
    """
    from sklearn.base import BaseEstimator

    from mlframe.estimators.early_stopping import EarlyStoppingWrapper

    class _SpyBase(BaseEstimator):
        def __init__(self):
            self.train_sizes = []

        def partial_fit(self, X, y, classes=None):
            self.train_sizes.append(len(X))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    return EarlyStoppingWrapper(
        base_model=_SpyBase(),
        start_iter=1,
        max_iter=1,
        validation_fraction=validation_fraction,
    )


def test_early_stopping_zero_val_samples_raises_not_silent():
    """Pre-fix: n_val_samples=int(9 * 0.05)=0 -> X[:-0] is EMPTY -> silent
    no-training. Post-fix: a fraction that leaves zero TRAIN rows raises."""
    X = np.arange(9 * 2, dtype=np.float64).reshape(9, 2)
    y = np.array([0, 1] * 4 + [0])
    # validation_fraction=1.0 -> n_val_samples=9 == len(X) -> zero train rows.
    est = _make_es_estimator(validation_fraction=1.0)
    with pytest.raises(ValueError, match="zero training rows"):
        est.fit(X, y)


def test_early_stopping_clamps_val_samples_to_at_least_one():
    """Tiny fraction that floors to 0 must clamp to >=1 val sample and STILL
    train on a non-empty slice (pre-fix X[:-0] silently gave an empty trainset)."""
    X = np.arange(9 * 2, dtype=np.float64).reshape(9, 2)
    y = np.array([0, 1] * 4 + [0])
    est = _make_es_estimator(validation_fraction=0.05)  # int(9*0.05)=0 -> clamp to 1
    est.fit(X, y)
    # One val row clamped off; the rest (8) are the training slice. The wrapper fits a
    # CLONE of base_model, so inspect the fitted clone (est.estimator_), not the original.
    assert est.estimator_.train_sizes
    assert all(sz == 8 for sz in est.estimator_.train_sizes)


# ---- #2 probabilities chunked-tail garbage --------------------------------


def test_probabilities_chunked_tail_no_garbage():
    """n=9, chunk=4 -> chunks process [0:4], [4:8], residual [8:9].
    Pre-fix the [8:9] slot was np.empty garbage. Post-fix it's a
    synthesized freq value.

    ``flip_percent=0.0`` disables the random flip that the default
    (0.6) would otherwise apply -- it would scramble 5 of 9 outcomes
    nondeterministically, making the tail-row reasoning brittle. The
    no-garbage invariant is the same either way."""
    from mlframe.calibration.probabilities import generate_probs_from_outcomes

    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1])
    probs = generate_probs_from_outcomes(
        y,
        chunk_size=4,
        nbins=2,
        scale=0.0,
        bins_std=0.0,
        flip_percent=0.0,
    )
    assert np.all(np.isfinite(probs)), f"Wave 24 P0 regression: probs contains non-finite values {probs.tolist()} - the chunked-tail garbage bug returned."
    assert np.all((probs >= 0.0) & (probs <= 1.0)), f"Probs out of range: {probs.tolist()}"
    # Tail row 8: outcomes[8]=1 -> freq=1.0 -> probs[8] should be 1.0
    # (or clipped to 1.0 from any noise). Pre-fix could return any
    # garbage value including 0.0.
    assert probs[8] >= 0.5, (
        f"Tail row 8 (outcomes=1, flip_percent=0) should fall in the "
        f"high-prob bucket; got {probs[8]}. Pre-fix could return any "
        f"garbage value (often 0.0 from np.empty's recently-released "
        f"memory blocks)."
    )


def test_probabilities_tail_row_synthesized_not_garbage():
    """The residual tail rows (n % chunk_size != 0) must be SYNTHESIZED from the
    outcome frequency, not left as uninitialised buffer memory. n=10, chunk=4 ->
    chunks [0:4],[4:8], tail [8:10]; the all-1 tail must land in the high bucket."""
    from mlframe.calibration.probabilities import generate_probs_from_outcomes

    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    probs = generate_probs_from_outcomes(
        y,
        chunk_size=4,
        nbins=2,
        scale=0.0,
        bins_std=0.0,
        flip_percent=0.0,
    )
    assert probs.shape == y.shape
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Tail rows 8,9 are outcome=1 -> synthesized high probability.
    assert probs[8] >= 0.5 and probs[9] >= 0.5


# ---- #3 composite_screening column-0 NaN gate ----------------------------


def test_composite_screening_gate_on_target_only_not_col0():
    """Behaviour: a NaN-heavy column-0 must NOT zero MI for the OTHER (clean,
    informative) columns in the batch. Pre-fix the size-gate ANDed
    ``feature_binned[:, 0] >= 0`` and silently returned 0.0 for the whole batch."""
    from mlframe.training.composite.discovery.screening import (
        _prebin_feature_columns,
        _mi_to_target_prebinned,
    )

    rng = np.random.default_rng(0)
    n, nbins = 4000, 5
    target = rng.normal(size=n)
    informative = target + rng.normal(scale=0.2, size=n)  # high MI with target
    col0 = informative.copy()
    col0[: int(n * 0.9)] = np.nan  # column-0 NaN-heavy -> -1 sentinel rows
    feature_matrix = np.column_stack([col0, informative])
    binned = _prebin_feature_columns(feature_matrix, nbins=nbins)
    mi = _mi_to_target_prebinned(binned, target, nbins=nbins, aggregation="sum")
    # Pre-fix this was 0.0 (whole-batch gate masked col-0 sentinels); post-fix
    # the clean informative column still contributes its MI.
    assert mi > 0.0, mi


# ---- #4 _reporting searchsorted IndexError -------------------------------


def test_reporting_unseen_predict_warns(caplog):
    """A model with no predict_proba whose predict() returns a value OUTSIDE
    classes_ must (a) not IndexError on the one-hot fill (pre-fix
    ``np.searchsorted`` returned index==n_classes -> IndexError) and (b) WARN
    about the unseen outputs, mapping them to class-0."""

    class _UnseenModel:
        classes_ = np.array([0, 1])

        def predict(self, X):
            # 2 is OUTSIDE classes_ -- the unseen value that pre-fix crashed.
            return np.array([0, 1, 2, 1])

        # Deliberately NO predict_proba so the fallback path is taken.

    targets = np.array([0, 1, 0, 1])
    df = pd.DataFrame({"f0": [0.1, 0.2, 0.3, 0.4]})
    from mlframe.training.reporting._reporting_probabilistic import (
        report_probabilistic_model_perf,
    )

    with caplog.at_level(logging.WARNING):
        preds, probs = report_probabilistic_model_perf(
            targets=targets,
            columns=["f0"],
            model_name="unseen",
            model=_UnseenModel(),
            df=df,
            print_report=False,
            show_perf_chart=False,
            verbose=False,
        )
    # The unseen value 2 was mapped to a valid one-hot column (no IndexError).
    assert probs.shape[0] == 4
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert any("were NOT in" in r.getMessage() for r in caplog.records), "unseen-class WARN not emitted"
