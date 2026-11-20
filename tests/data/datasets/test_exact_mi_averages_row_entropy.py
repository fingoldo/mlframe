"""The exact MI averages each row's entropy within a bin, not the entropy of the bin's mean probability (audit 2026-11-20, finding 04-F8)."""
import numpy as np

from mlframe.data.datasets._oracle import _exact_mi_independent


def test_mixed_probabilities_inside_one_bin_are_averaged_per_row():
    """Bin 0 holds p = 0.1 and 0.9, bin 1 holds p = 0.5 twice. I = H(0.5) - (0.5*mean H(0.1,0.9) + 0.5*H(0.5)) = 0.184 nats (0.2655 bits).

    Taking the entropy at the bin mean (0.5 in bin 0) gave I = 0, which denies the probability variation inside the bin.
    """
    column = np.array([0.0, 0.0, 1.0, 1.0])
    true_prob = np.array([0.1, 0.9, 0.5, 0.5])
    got = _exact_mi_independent(column, true_prob, n_bins=2)
    h = lambda q: -(q * np.log(q) + (1 - q) * np.log(1 - q))  # nats, as binary_entropy
    expected = h(0.5) - (0.5 * h(0.1) + 0.5 * h(0.5))
    assert got is not None
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    assert got > 0.18
