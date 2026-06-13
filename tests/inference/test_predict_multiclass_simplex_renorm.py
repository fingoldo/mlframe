"""Regression: multiclass ensemble combine must renormalise rows to the probability simplex.

Pre-fix ``_combine_probs`` only clipped to [0, 1]; the harm / geo / quad / qube flavours blend valid
per-member simplices into rows that no longer sum to 1 (geo ~0.86, harm ~0.74, quad/qube > 1). For a
MULTICLASS (K > 2) target the resulting ``ensemble_probabilities`` matrix is consumed as a simplex
(argmax, log_loss, calibration, reporting), so non-normalised rows silently corrupt every downstream
probabilistic consumer. The chooser is task-agnostic and will pick a non-arithm flavour for multiclass
whenever it scores best on OOF log_loss / AUC, so this is reachable on a default suite.

The renorm is gated to MULTICLASS only: binary keeps the ``[:, 1]`` semantics and multilabel's per-label
columns are independent (not a simplex) -- neither may be renormalised.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.configs import TargetTypes
from mlframe.training.core.predict import _combine_probs

_M1 = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
_M2 = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])


@pytest.mark.parametrize("flavour", ["harm", "geo", "quad", "qube"])
def test_multiclass_combine_renormalises_rows_to_simplex(flavour):
    """K=3 multiclass blend with a simplex-breaking flavour returns rows summing to exactly 1."""
    out = _combine_probs([_M1, _M2], flavour, target_type=TargetTypes.MULTICLASS_CLASSIFICATION)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)
    assert (out >= -1e-12).all() and (out <= 1.0 + 1e-12).all()


def test_multiclass_renorm_accepts_string_target_type():
    """The predict suite path passes the target_type as a plain string key, not the enum."""
    out = _combine_probs([_M1, _M2], "geo", target_type="multiclass_classification")
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)


def test_multiclass_renorm_preserves_argmax_ordering():
    """Renorm is a positive per-row rescale, so it must not change which class wins argmax."""
    raw = _combine_probs([_M1, _M2], "geo", target_type=None)  # un-normalised reference
    norm = _combine_probs([_M1, _M2], "geo", target_type=TargetTypes.MULTICLASS_CLASSIFICATION)
    np.testing.assert_array_equal(raw.argmax(axis=1), norm.argmax(axis=1))


def test_multilabel_columns_are_not_renormalised():
    """Multilabel per-label columns are independent P(label=1); renormalising them would be wrong."""
    ml1 = np.array([[0.9, 0.8, 0.1]])
    ml2 = np.array([[0.7, 0.6, 0.3]])
    out = _combine_probs([ml1, ml2], "geo", target_type=TargetTypes.MULTILABEL_CLASSIFICATION)
    # geo blend of these columns sums to ~1.66; must be left untouched.
    assert out.sum(axis=1)[0] > 1.5


def test_binary_two_column_blend_untouched():
    """Binary (K=2) keeps the per-column blend; the predict path reads column 1 directly."""
    b1 = np.array([[0.7, 0.3]])
    b2 = np.array([[0.5, 0.5]])
    out = _combine_probs([b1, b2], "geo", target_type=TargetTypes.BINARY_CLASSIFICATION)
    # geomean of the positive column: sqrt(0.3 * 0.5) ~ 0.3873, NOT renormalised to a simplex.
    np.testing.assert_allclose(out[:, 1], np.sqrt(0.3 * 0.5), rtol=1e-9)


def test_mixed_suite_none_target_type_not_renormalised():
    """A heterogeneous suite passes target_type=None; the multiclass renorm must not fire."""
    out = _combine_probs([_M1, _M2], "geo", target_type=None)
    assert not np.allclose(out.sum(axis=1), 1.0)
