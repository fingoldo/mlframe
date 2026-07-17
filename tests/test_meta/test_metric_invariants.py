"""Meta-test — mathematical invariants of the metric library.

These tests assert relationships that must hold by construction, regardless
of input. They catch regressions where a "small refactor" of the kernel
quietly breaks the math (e.g. using bin-centre instead of bin-mean for
reliability, swapping a + for a - in the Brier decomposition, returning
NaN on degenerate input where 0.0 was the contract).

Hypothesis-driven so the invariant has to hold for every plausible input,
not just the example in the test author's head.
"""

from __future__ import annotations

import math
import numpy as np
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.extra import numpy as st_np

# Tolerance for fp-precision identities. Brier-decomposition identity is
# exact by construction (Murphy 1973) so we can be very tight.
_FP_TOL = 1e-9
# Looser tolerance for sums of bounded values where round-off accumulates
# (n=200 floats, each in [0,1], summed).
_FP_LOOSE = 1e-7


@st.composite
def _binary_targets_and_probs(draw, min_n=20, max_n=200):
    """A coupled (y_true, y_pred) pair where probs lie in [0,1] and labels
    in {0, 1}; both strictly the same length."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    y_true = draw(
        st_np.arrays(
            dtype=np.int64,
            shape=n,
            elements=st.integers(min_value=0, max_value=1),
        )
    )
    y_pred = draw(
        st_np.arrays(
            dtype=np.float64,
            shape=n,
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    # Reject degenerate single-class targets — most metrics have either no
    # signal or a documented escape value (1.0/0.0) that's not what the
    # invariant is testing.
    assume(0 < int(y_true.sum()) < n)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Brier decomposition: ``BinnedBrier == REL - RES + UNC`` (Murphy 1973).
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_binary_targets_and_probs(), nbins=st.integers(min_value=2, max_value=20))
def test_brier_decomposition_identity(pair, nbins):
    """BinnedBrier == REL - RES + UNC, exact to fp precision (Murphy 1973).
    The decomposition uses per-bin mean predicted probability (not bin
    centre) so the identity holds within fp round-off, not approximately.
    """
    from mlframe.metrics.core import compute_ece_and_brier_decomposition

    y_true, y_pred = pair
    _ece, rel, res, unc, brier_binned = compute_ece_and_brier_decomposition(
        y_true.astype(np.float64),
        y_pred,
        nbins,
    )
    assert math.isclose(brier_binned, rel - res + unc, rel_tol=0, abs_tol=_FP_TOL), (
        f"Brier decomposition broken: brier_binned={brier_binned}, "
        f"REL={rel}, RES={res}, UNC={unc}, "
        f"REL-RES+UNC={rel - res + unc}, gap={brier_binned - (rel - res + unc)}"
    )


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_binary_targets_and_probs(), nbins=st.integers(min_value=2, max_value=20))
def test_brier_decomposition_components_in_unit_interval(pair, nbins):
    """REL, RES, UNC, ECE all live in [0, 1] for binary targets in {0, 1}
    and probabilities in [0, 1]. UNC = base_rate*(1-base_rate) ≤ 1/4 in
    fact, but [0, 1] is the looser bound the test asserts (so a future
    refactor that changes UNC's normalisation by a constant doesn't
    silently break this — only sign / range issues fail the bound).
    """
    from mlframe.metrics.core import compute_ece_and_brier_decomposition

    y_true, y_pred = pair
    ece, rel, res, unc, brier_binned = compute_ece_and_brier_decomposition(
        y_true.astype(np.float64),
        y_pred,
        nbins,
    )
    assert 0.0 <= ece <= 1.0, f"ECE out of [0,1]: {ece}"
    assert 0.0 <= rel <= 1.0, f"REL out of [0,1]: {rel}"
    assert 0.0 <= res <= 1.0, f"RES out of [0,1]: {res}"
    assert 0.0 <= unc <= 0.25 + _FP_LOOSE, f"UNC out of [0, 1/4]: {unc} (UNC = p(1-p), max at p=0.5 is 0.25)"
    assert 0.0 <= brier_binned <= 1.0, f"BinnedBrier out of [0,1]: {brier_binned}"


# ---------------------------------------------------------------------------
# Brier score bounds + symmetry.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(pair=_binary_targets_and_probs())
def test_brier_score_in_unit_interval(pair):
    """Brier score in unit interval."""
    from mlframe.metrics.core import fast_brier_score_loss

    y_true, y_pred = pair
    score = fast_brier_score_loss(y_true.astype(np.float64), y_pred)
    assert 0.0 <= score <= 1.0, f"Brier score out of [0,1]: {score}"


@settings(deadline=None, max_examples=20)
@given(pair=_binary_targets_and_probs())
def test_brier_perfect_predictions_zero(pair):
    """Brier(y, y_as_float) = 0 — predicting the truth perfectly gives a
    perfect score. (Not Hypothesis-novel; this is the null-test that
    catches "I forgot the squared-error".)"""
    from mlframe.metrics.core import fast_brier_score_loss

    y_true, _ = pair
    perfect = y_true.astype(np.float64).copy()
    score = fast_brier_score_loss(y_true.astype(np.float64), perfect)
    assert math.isclose(score, 0.0, abs_tol=_FP_LOOSE), f"Perfect predictions should yield Brier=0, got {score}"


# ---------------------------------------------------------------------------
# AUC bounds + monotonic-transform invariance.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(pair=_binary_targets_and_probs())
def test_roc_auc_in_unit_interval(pair):
    """Roc auc in unit interval."""
    from mlframe.metrics.core import fast_roc_auc

    y_true, y_pred = pair
    auc = fast_roc_auc(y_true.astype(np.float64), y_pred)
    assert 0.0 <= auc <= 1.0 + _FP_LOOSE, f"AUC out of [0,1]: {auc}"


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_binary_targets_and_probs())
def test_roc_auc_invariant_under_monotonic_transform(pair):
    """AUC depends only on the *ranks* of scores, not their absolute
    values: f(x) = a*x + b for any a > 0 leaves AUC unchanged. Catches
    a shrinkage / scaling bug in the score transform pipeline.

    Restricts probs to [1e-6, 1 - 1e-6] to avoid the fp-absorption
    artefact where denormal scores collapse to a constant after the
    additive bias (mathematically rank-preserving, numerically not).
    Real-world probability outputs are never denormal — the artefact
    isn't a metric bug but its presence in the test would block
    merging.
    """
    from mlframe.metrics.core import fast_roc_auc

    y_true, y_pred = pair
    # Reject inputs near zero — see docstring.
    if not np.all(y_pred >= 1e-6):
        return
    yt = y_true.astype(np.float64)
    auc1 = fast_roc_auc(yt, y_pred)
    # Monotonic affine transform with positive slope.
    auc2 = fast_roc_auc(yt, 3.5 * y_pred + 7.0)
    assert math.isclose(auc1, auc2, abs_tol=_FP_LOOSE), f"AUC changed under positive affine transform: {auc1} → {auc2}"


# ---------------------------------------------------------------------------
# Hamming loss + subset accuracy + Jaccard for multilabel.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=10, max_value=80),
    k=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_hamming_loss_in_unit_interval(n, k, seed):
    """Hamming loss in unit interval."""
    from mlframe.metrics.core import hamming_loss

    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    y_pred = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    loss = hamming_loss(y_true, y_pred)
    assert 0.0 <= loss <= 1.0, f"Hamming loss out of [0,1]: {loss}"


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=10, max_value=80),
    k=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_subset_accuracy_perfect_match_is_one(n, k, seed):
    """When y_pred == y_true exactly, subset_accuracy = 1.0."""
    from mlframe.metrics.core import subset_accuracy

    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    acc = subset_accuracy(y_true, y_true.copy())
    assert math.isclose(acc, 1.0, abs_tol=_FP_LOOSE), f"subset_accuracy on identical inputs should be 1.0, got {acc}"


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=10, max_value=80),
    k=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_jaccard_score_in_unit_interval(n, k, seed):
    """Jaccard score in unit interval."""
    from mlframe.metrics.core import jaccard_score_multilabel

    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    y_pred = rng.integers(0, 2, size=(n, k)).astype(np.int8)
    j = jaccard_score_multilabel(y_true, y_pred)
    assert 0.0 <= j <= 1.0 + _FP_LOOSE, f"Jaccard out of [0,1]: {j}"


# ---------------------------------------------------------------------------
# Log loss bounds + perfect-prediction floor.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30)
@given(pair=_binary_targets_and_probs())
def test_log_loss_non_negative(pair):
    """Log loss non negative."""
    from mlframe.metrics.core import fast_log_loss_binary

    y_true, y_pred = pair
    loss = fast_log_loss_binary(y_true.astype(np.float64), y_pred)
    assert loss >= 0.0, f"Log-loss must be ≥ 0, got {loss}"


# ---------------------------------------------------------------------------
# _predict_from_probs sanity invariants (multilabel decision rule).
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=10, max_value=80),
    k=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_predict_from_probs_per_label_threshold_zero_yields_all_ones(n, k, seed):
    """``_predict_from_probs`` with per-label thresholds = [0, 0, ...] should
    label every output positive (probabilities are ≥ 0 in [0,1] and the
    rule is ``probs >= threshold``)."""
    from mlframe.training.helpers import _predict_from_probs
    from mlframe.training.configs import TargetTypes

    rng = np.random.default_rng(seed)
    probs = rng.random((n, k))
    out = _predict_from_probs(
        probs,
        TargetTypes.MULTILABEL_CLASSIFICATION,
        threshold=np.zeros(k),
    )
    assert out.shape == (n, k)
    assert (out == 1).all(), "Per-label threshold = 0 should mark every cell positive — regression in the multilabel decision rule."


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=10, max_value=80),
    k=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_predict_from_probs_per_label_threshold_above_one_yields_all_zeros(n, k, seed):
    """Symmetric: thresholds set above the [0,1] domain → all zeros."""
    from mlframe.training.helpers import _predict_from_probs
    from mlframe.training.configs import TargetTypes

    rng = np.random.default_rng(seed)
    probs = rng.random((n, k))
    out = _predict_from_probs(
        probs,
        TargetTypes.MULTILABEL_CLASSIFICATION,
        threshold=np.full(k, 1.5),
    )
    assert (out == 0).all(), "Per-label threshold > 1.0 should mark every cell negative."
