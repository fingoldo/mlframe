"""Binary log loss maps two labels outside {0, 1} with the larger as positive and refuses NaN labels."""
import numpy as np
from sklearn.metrics import log_loss

from mlframe.metrics import fast_log_loss_binary


def test_labels_one_and_two_match_sklearn():
    """Labels {1, 2} scored every 1 as positive and every 2 as a negative, i.e. the loss of the flipped problem."""
    y = np.array([1, 2, 1, 2, 2, 1])
    p = np.array([0.2, 0.9, 0.3, 0.8, 0.7, 0.1])
    assert abs(fast_log_loss_binary(y, p) - log_loss(y, p)) < 1e-9


def test_nan_label_makes_the_loss_nan():
    """A NaN label used to be charged -log(1 - p) as a negative."""
    assert np.isnan(fast_log_loss_binary(np.array([np.nan, 1.0, 0.0]), np.array([0.9, 0.9, 0.1])))
