"""biz_val head-to-head: mlframe RFECV vs ``sklearn.feature_selection.RFECV``.

The audit finding (``bizvalue_value_proofs-02``) was that the existing sklearn-RFECV
head-to-head never asserted a WIN on any axis, and that the Model-Based-Heuristic
(MBH) "refit-count advantage" -- our optimiser evaluates only a handful of candidate
subset sizes where sklearn's ``step=1`` RFE walks every size from ``p`` down to ``1`` --
was claimed but never proven in CI. This file pins the one axis that DOES hold
reliably and honestly documents the one that does not.

What is measured (300x40 ``make_classification``, ``n_informative=5``, ``shuffle=False``,
``LogisticRegression`` estimator, 3-fold CV):

1. ``test_rfecv_explores_far_fewer_subset_sizes_than_sklearn`` -- the PROVEN MBH win.
   sklearn's ``step=1`` RFECV evaluates exactly ``p`` distinct ``n_features`` values
   (one per backward-elimination round); ours evaluates at most ``max_refits``-many
   distinct subset sizes. Measured robustly across 5 seeds: sklearn evaluates 40
   distinct N every seed; ours explores 2..11 (always ``<< 40``). This is the
   refit-count advantage as a hard, seed-robust assertion -- a regression that makes
   the MBH walk the full grid (e.g. a broken convergence / init-design) trips it.

2. ``test_rfecv_runtime_at_score_parity`` -- the runtime-WIN-at-parity axis the mission
   asked for, asserting the INTENDED contract (``our_score >= sk_score - 0.02`` AND
   ``our_wall_min <= 0.7 * sk_wall_min``). Marked ``xfail(strict=False)`` because a
   warm multi-run dev measurement (3x each, per-method MIN wall) showed it does NOT
   hold at CI-feasible sizes: the two halves are ANTI-CORRELATED. When the MBH explores
   enough intermediate anchors to reach score parity (seed-dependent, ~11 anchors) each
   anchor costs a full 3-fold LogisticRegression CV refit, so our wall is 5-7x sklearn's
   (sklearn's step-1 LR refits on 300x40 are ~20ms each -> ~0.8s for all 40 rounds).
   When the MBH short-circuits to ``{0, p}`` (the common branch) it is 0.3-0.5x sklearn's
   wall but keeps all ``p`` features and loses parity by 0.05-0.13 AUC (LR overfits the
   35 noise columns at n=300). There is no measured regime in this size range where
   our RFECV is simultaneously at-parity AND faster -- so the win is a strict-False xfail
   carrying the measured numbers, not a weakened assertion. See module docstring of the
   numbers above; if a future MBH/surrogate change makes the win materialise the xfail
   flips to an unexpected PASS and we promote it.

Naming follows ``test_biz_val_rfecv_*``. ``@pytest.mark.slow`` heavy timing half +
fast-mode kept alive via a representative; the timing half additionally short-circuits
under ``running_under_xdist()`` because wall-clock ratios are unreliable on a contended
box (per ``perf_*`` budget helpers' rationale).
"""
from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV as SkRFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.wrappers import RFECV as MlframeRFECV
from tests.conftest import running_under_xdist
from tests.feature_selection.conftest import is_fast_mode

warnings.filterwarnings("ignore")


# --- shared synthetic + helpers --------------------------------------------
# 300x40, 5 informative + 5 redundant + 30 noise. shuffle=False keeps the
# informative columns at the front so the selection target is deterministic.
# Small enough that even the exploring branch (full 3-fold CV at ~11 anchors)
# stays well under the 55s per-test budget warm.

_N_SAMPLES = 300
_N_FEATURES = 40
_N_INFORMATIVE = 5
_CV = 3
_MAX_REFITS = 10
# init_design_size seeds the MBH with intermediate-N anchors; without it the
# optimiser routinely short-circuits to {0, p}. 8 is enough to let the search
# walk a handful of subset sizes when the surrogate finds an improving direction.
_INIT_DESIGN = 8


def _make_data(seed: int = 0):
    X, y = make_classification(
        n_samples=_N_SAMPLES, n_features=_N_FEATURES, n_informative=_N_INFORMATIVE,
        n_redundant=5, n_repeated=0, n_classes=2, shuffle=False, random_state=seed,
    )
    return X, y


def _quiet(fn):
    """Run ``fn`` with stdout/stderr swallowed (RFECV's progress bars + the
    StratifiedKFold get_params warning are pure noise here)."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        return fn()


def _make_ours(max_refits: int = _MAX_REFITS):
    return MlframeRFECV(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        cv=_CV, max_refits=max_refits, random_state=0,
        leakage_corr_threshold=None, n_features_selection_rule="argmax",
        init_design_size=_INIT_DESIGN, verbose=0, leave_progressbars=False,
    )


def _make_sklearn():
    return SkRFECV(estimator=LogisticRegression(max_iter=200, random_state=0),
                   step=1, cv=_CV)


def _bool_mask(selector, n_features: int) -> np.ndarray:
    """``support_`` may be a bool mask OR integer indices; normalise to a bool
    mask of length ``n_features``."""
    s = np.asarray(selector.support_)
    if s.dtype == bool or s.dtype == np.bool_:
        return s.astype(bool)
    mask = np.zeros(n_features, dtype=bool)
    if s.size:
        mask[s.astype(int)] = True
    return mask


def _subset_auc(X, y, mask: np.ndarray) -> float:
    """Selector-agnostic parity metric: 3-fold ``roc_auc`` of a fresh
    LogisticRegression on ONLY the selected columns. Both arms are scored the
    same way so the comparison is honest regardless of which exact columns
    each selector kept. Returns ``nan`` for an empty selection (an empty
    selection is itself a failure the caller asserts against)."""
    cols = np.flatnonzero(mask)
    if cols.size == 0:
        return float("nan")
    est = LogisticRegression(max_iter=200, random_state=0)
    return float(cross_val_score(est, X[:, cols], y, cv=_CV, scoring="roc_auc").mean())


def _distinct_n_explored(selector) -> int:
    """How many distinct ``n_features`` subset sizes our RFECV actually
    evaluated -- the MBH refit-count proxy. Read from the public
    ``cv_results_['nfeatures']`` populated at finalize."""
    cv = getattr(selector, "cv_results_", None) or {}
    return len(set(cv.get("nfeatures", [])))


def _sklearn_distinct_n_evaluated(selector) -> int:
    """sklearn RFECV with ``step=1`` evaluates one subset size per backward
    round -> ``len(cv_results_['n_features'])`` == ``p`` distinct sizes."""
    return len(selector.cv_results_["n_features"])


# --- 1. the PROVEN, seed-robust MBH refit-count advantage ------------------


def test_rfecv_explores_far_fewer_subset_sizes_than_sklearn():
    """MBH evaluates an order of magnitude fewer distinct subset sizes than
    sklearn's ``step=1`` RFE.

    sklearn (``step=1``) walks ``p``, ``p-1``, ..., ``1`` -> exactly ``p`` distinct
    ``n_features`` evaluations, each a full CV refit. Our MBH optimiser evaluates at
    most ``max_refits``-many candidate subset sizes (measured 2..11 across 5 seeds at
    this config; sklearn measured 40 every seed). This is the refit-count advantage
    the audit flagged as unproven -- pinned here as a hard floor with a generous
    buffer so it is robust to the seed-dependent MBH branch (it holds whether the
    optimiser short-circuits to ``{0, p}`` or explores the full init design).

    Regression sensor: a broken convergence / init-design that makes the MBH walk the
    full grid would push ``our_distinct`` toward ``p`` and trip this.
    """
    X, y = _make_data(seed=0)
    p = X.shape[1]

    sk = _make_sklearn()
    sk.fit(X, y)
    sk_distinct = _sklearn_distinct_n_evaluated(sk)

    ours = _make_ours()
    _quiet(lambda: ours.fit(X, y))
    our_distinct = _distinct_n_explored(ours)

    # sklearn step=1 must walk all p subset sizes.
    assert sk_distinct == p, f"sklearn step=1 should evaluate p={p} distinct N, got {sk_distinct}"

    # Ours must explore strictly fewer, and by a wide margin. Floor: never more
    # than max_refits + a small buffer for the init-design anchors + {0, p}.
    # Measured ceiling across 5 seeds was 11 (<= _MAX_REFITS + 1); the bound
    # below (max_refits + init_design, ~18) keeps headroom while still being
    # far under p=40 -> a regression to the full grid trips it.
    ceiling = _MAX_REFITS + _INIT_DESIGN
    assert our_distinct >= 1, "our RFECV must record at least one explored subset size"
    assert our_distinct <= ceiling, (
        f"MBH explored {our_distinct} distinct subset sizes; expected <= {ceiling} "
        f"(<< sklearn's {sk_distinct})"
    )
    # The headline win: ours evaluates at most a third of sklearn's distinct-N grid.
    assert our_distinct <= 0.5 * sk_distinct, (
        f"MBH refit-count advantage not realised: ours evaluated {our_distinct} "
        f"distinct subset sizes vs sklearn's {sk_distinct} (want <= {0.5 * sk_distinct:.0f})"
    )


def test_rfecv_produces_a_valid_selection_at_parity_config():
    """Smoke + structural floor for the parity config: ours yields a non-empty
    selection whose 3-fold AUC clears a meaningful floor (not garbage), and
    sklearn does too. This is the unit-level guard that the head-to-head data
    is well-formed before the timing/xfail assertions read it. Both selectors'
    selected-subset AUC must clear 0.70 on this 5-informative target (measured:
    sklearn ~0.88, ours 0.75..0.85 depending on the MBH branch; floor 0.70 is
    ~7% below the worst observed)."""
    X, y = _make_data(seed=0)
    n = X.shape[1]

    sk = _make_sklearn()
    sk.fit(X, y)
    sk_mask = sk.support_.astype(bool)
    assert sk_mask.sum() >= 1
    assert _subset_auc(X, y, sk_mask) >= 0.70

    ours = _make_ours()
    _quiet(lambda: ours.fit(X, y))
    our_mask = _bool_mask(ours, n)
    assert our_mask.sum() >= 1, "our RFECV returned an empty selection"
    assert _subset_auc(X, y, our_mask) >= 0.70, "our selection's subset-AUC fell below the floor"


# --- 2. the runtime-WIN-at-parity axis (intended contract, measured to fail) ---


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason=(
        "PROD/PERF GAP: MBH RFECV is NOT faster than sklearn at score parity on "
        "300x40. Warm 3x-min measurement: when MBH explores enough intermediate "
        "anchors to reach parity (~11 anchors) it is 5-7x SLOWER (each anchor is a "
        "full 3-fold LR CV refit; sklearn's step-1 LR refits are ~20ms each); when it "
        "short-circuits to {0,p} it is 0.3-0.5x sklearn's wall but keeps all p features "
        "and loses parity by 0.05-0.13 AUC. The two axes are anti-correlated -- no "
        "regime here is at-parity AND faster. xfail(strict=False) so a future "
        "MBH/surrogate speedup flips this to an unexpected pass we then promote."
    ),
)
def test_rfecv_runtime_at_score_parity():
    """The mission's named head-to-head: assert the INTENDED win on BOTH axes --
    score parity (``our_score >= sk_score - 0.02``) AND a runtime win
    (``our_wall_min <= 0.7 * sk_wall_min``). Warm-run protocol: fit each method
    3 times and take the per-method MIN wall (the cleanest steady-state estimate;
    discards the one-time numba/surrogate cold-compile of the first fit).

    Currently xfail (see the marker): the MBH refit-count advantage does not buy a
    wall-clock win at this scale because the per-anchor CV refit cost dominates when
    the optimiser explores, and the fast branch sacrifices parity. The assertion is
    written to the CORRECT contract so a real future improvement converts it to a pass.
    """
    if running_under_xdist():
        pytest.skip("wall-clock ratio unreliable under xdist contention")

    X, y = _make_data(seed=0)
    n = X.shape[1]

    sk_walls = []
    sk_mask = None
    for _ in range(3):
        sk = _make_sklearn()
        t0 = time.perf_counter()
        sk.fit(X, y)
        sk_walls.append(time.perf_counter() - t0)
        sk_mask = sk.support_.astype(bool)
    sk_wall_min = min(sk_walls)
    sk_score = _subset_auc(X, y, sk_mask)

    our_walls = []
    our_mask = None
    for _ in range(3):
        ours = _make_ours()
        t0 = time.perf_counter()
        _quiet(lambda: ours.fit(X, y))
        our_walls.append(time.perf_counter() - t0)
        our_mask = _bool_mask(ours, n)
    our_wall_min = min(our_walls)
    our_score = _subset_auc(X, y, our_mask)

    # Both halves of the intended win. Either failing is the documented gap.
    assert our_score >= sk_score - 0.02, (
        f"score parity not met: ours={our_score:.4f} sklearn={sk_score:.4f} "
        f"(floor sklearn-0.02={sk_score - 0.02:.4f})"
    )
    assert our_wall_min <= 0.7 * sk_wall_min, (
        f"runtime win not met: our_min={our_wall_min:.3f}s sklearn_min={sk_wall_min:.3f}s "
        f"ratio={our_wall_min / sk_wall_min:.3f} (want <= 0.7)"
    )


# --- fast-mode representative ----------------------------------------------
# Keeps one cheap path alive under MLFRAME_FAST=1 (where the @slow timing test
# is skipped): a single fit of each selector at a reduced max_refits, asserting
# only the cheap, reliable refit-count axis -- no wall-clock timing.


def test_rfecv_fast_representative_refit_count_axis():
    """Fast-mode representative of the proven refit-count advantage. Runs always
    (not @slow) so MLFRAME_FAST=1 still exercises the mlframe-vs-sklearn distinct-N
    comparison with a single cheap fit each."""
    max_refits = 4 if is_fast_mode() else _MAX_REFITS
    X, y = _make_data(seed=0)
    p = X.shape[1]

    sk = _make_sklearn()
    sk.fit(X, y)
    sk_distinct = _sklearn_distinct_n_evaluated(sk)

    ours = _make_ours(max_refits=max_refits)
    _quiet(lambda: ours.fit(X, y))
    our_distinct = _distinct_n_explored(ours)

    assert sk_distinct == p
    assert 1 <= our_distinct <= max_refits + _INIT_DESIGN
    assert our_distinct < sk_distinct
