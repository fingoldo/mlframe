"""batch_univariate_auc takes the larger of two labels as positive instead of demanding a 0/1 target."""
import numpy as np

from mlframe.preprocessing.align_feature_direction import batch_univariate_auc


def test_labels_zero_and_two_and_strings_work():
    """y = [0, 2, 0, 2] raised a wrong 'single-class' error because only y == 1 was positive."""
    X = np.array([[0.1], [0.9], [0.2], [0.8]])
    assert np.allclose(batch_univariate_auc(X, np.array([0, 2, 0, 2])), batch_univariate_auc(X, np.array([0, 1, 0, 1])))
    assert np.allclose(batch_univariate_auc(X, np.array(["no", "yes", "no", "yes"])), [1.0])
