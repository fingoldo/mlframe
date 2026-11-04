"""A second full fit of a neural classifier learns its own classes instead of reusing the first fit's label encoder."""
from types import SimpleNamespace

import numpy as np

from mlframe.training.neural.base._base_fit_helpers import _FitCommonHelpersMixin


def _fit_labels(owner, y):
    """Run only the label-encoding block of a full (non-partial) fit on ``owner``."""
    return _FitCommonHelpersMixin._fit_common_inference_time(owner, True, False, None, np.asarray(y), (None, None), {})


def test_second_fit_with_other_classes_does_not_keep_the_first_encoder():
    """First fit sees 3 classes, the second only 2: classes_ must become 2 and the binary sigmoid head must be chosen."""
    owner = SimpleNamespace()
    _fit_labels(owner, [0, 1, 2, 0, 1, 2])
    assert len(owner.classes_) == 3 and owner._binary_sigmoid_head is False
    _fit_labels(owner, [0, 1, 0, 1])
    assert list(owner.classes_) == [0, 1] and owner._binary_sigmoid_head is True


def test_new_labels_in_the_second_fit_do_not_raise():
    """Labels the first encoder never saw used to raise 'y contains previously unseen labels'."""
    owner = SimpleNamespace()
    _fit_labels(owner, ["a", "b"])
    _fit_labels(owner, ["a", "b", "c"])
    assert list(owner.classes_) == ["a", "b", "c"]


def test_validation_rows_of_a_class_absent_from_training_are_dropped_not_fatal():
    """Validation holds a class (2) the training labels never had: the fit used to raise; those rows are excluded and the rest is encoded."""
    owner = SimpleNamespace()
    X_val = np.arange(8, dtype=np.float32).reshape(4, 2)
    eval_set, _y = _FitCommonHelpersMixin._fit_common_inference_time(owner, True, False, None, np.asarray([0, 1, 0, 1]), (X_val, np.asarray([0, 1, 2, 1])), {})
    assert eval_set[0].shape == (3, 2) and list(eval_set[1]) == [0, 1, 1]
