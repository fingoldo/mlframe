"""Regression tests for the extended auto-stratification (FE-L-6).

Pre-fix: ``_phase_train_val_test_split`` stratified ONLY when
``len(_classification_targets) == 1``. Multiple binary targets + multilabel
got no stratification, silently producing all-class-0 val slices on rare-
imbalance fixtures.

Post-fix:
- Multiple classification targets -> composite-key stratification (row-tuple
  encoded as an int class id; gated on combined cardinality).
- Multilabel target -> iterative-stratification when available, else first-
  label fallback.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Unit-level checks of the stratify-key construction (the contract that the
# audit row FE-L-6 actually flags). We import the phase module and exercise the
# branch on a hand-built ``target_by_type`` dict; that way we don't have to
# spin up the full suite for a one-branch test.
# ---------------------------------------------------------------------------


def _build_stratify_key(target_by_type):
    """Stratify key the split phase derives from ``target_by_type`` (production helper, bucket-stratify off, default cardinality cap)."""
    from mlframe.training.core._phase_helpers_fit_split import _stratify_labels_for_split

    return _stratify_labels_for_split(None, target_by_type, 200, False, None)


class _BinaryClass:
    """Groups tests covering binary class."""
    name = "BINARY_CLASSIFICATION"


class _MultiLabel:
    """Groups tests covering multi label."""
    name = "MULTILABEL_CLASSIFICATION"


def test_single_target_classification_stratifies():
    """Baseline contract from before the FE-L-6 fix; still must hold. Behavioural: key shape
    matches input, dtype is integer-like, and the produced key reflects the binary classes."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    tbt = {_BinaryClass(): {"y": y}}
    out = _build_stratify_key(tbt)
    assert out is not None
    assert out.shape == (200,)
    # Binary classes -> stratify key has at most 2 unique values.
    uniq = np.unique(out)
    assert 1 <= len(uniq) <= 2, f"unexpected stratify-key cardinality {len(uniq)}"


def test_two_binary_targets_compose_into_stratify_key():
    """FE-L-6: pre-fix, two binary targets -> no stratification. Post-fix,
    composite key (2x2 = 4 classes) gets stratified."""
    rng = np.random.default_rng(1)
    y0 = rng.integers(0, 2, 200)
    y1 = rng.integers(0, 2, 200)
    tbt = {_BinaryClass(): {"y0": y0, "y1": y1}}
    out = _build_stratify_key(tbt)
    assert out is not None, "composite key not produced for 2-binary-targets"
    assert out.shape == (200,)
    # Combined cardinality up to 4 (rare-event-on-both could lower it).
    assert 2 <= len(np.unique(out)) <= 4


def test_high_cardinality_composite_key_skipped():
    """Composite cardinality cap (200): too many distinct row-tuples means
    every val slice would have unique-class rows; stratification refused."""
    rng = np.random.default_rng(2)
    # 4 targets x 5 classes each -> 625 distinct tuples over 5000 rows (every tuple well populated), so only the cap can refuse.
    targets = {f"y{i}": rng.integers(0, 5, 5000) for i in range(4)}
    assert _build_stratify_key({_BinaryClass(): targets}) is None
    # Control: 3 targets x 5 classes -> 125 tuples, under the cap, so the same construction does stratify.
    allowed = {f"y{i}": rng.integers(0, 5, 5000) for i in range(3)}
    out = _build_stratify_key({_BinaryClass(): allowed})
    assert out is not None
    assert out.shape == (5000,)
    assert len(np.unique(out)) == 125


def test_multilabel_target_first_label_fallback():
    """FE-L-6: multilabel (N, K) ndarray. With iterstrat absent, falls back
    to first-label stratification. With iterstrat present, uses full ndarray."""
    rng = np.random.default_rng(3)
    Y = rng.integers(0, 2, size=(150, 3))
    # Ensure first-label has both classes present with >=2 each.
    Y[0, 0] = 0
    Y[1, 0] = 0
    Y[2, 0] = 1
    Y[3, 0] = 1
    tbt = {_MultiLabel(): {"y_multi": Y}}
    out = _build_stratify_key(tbt)
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # noqa

        # iterstrat available: full ndarray returned
        assert out is not None
        assert out.shape == (150, 3)
    except ImportError:
        # Fallback: first-label 1-D ndarray
        assert out is not None
        assert out.shape == (150,)


def test_rare_class_disables_stratification():
    """Existing contract: single class with only 1 row -> sklearn would
    raise; we must NOT pass that as stratify_y."""
    y = np.zeros(100, dtype=int)
    y[0] = 1  # single-row positive class
    tbt = {_BinaryClass(): {"y": y}}
    out = _build_stratify_key(tbt)
    assert out is None, "rare-class target should disable stratification"


def test_regression_target_no_stratify():
    """Sanity: regression targets are never stratified."""

    class _Reg:
        """Groups tests covering reg."""
        name = "REGRESSION"

    rng = np.random.default_rng(4)
    y = rng.standard_normal(100)
    tbt = {_Reg(): {"y": y}}
    out = _build_stratify_key(tbt)
    assert out is None
