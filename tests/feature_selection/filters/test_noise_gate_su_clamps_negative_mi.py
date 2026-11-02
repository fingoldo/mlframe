"""The GPU-path symmetric-uncertainty reduction floors a tiny negative MI at zero like the CPU relevance path (tracker 2026-11-02 wave 13, PAR13-4)."""

import numpy as np

from mlframe.feature_selection.filters._batch_mi_noise_gate_kernels import _mi_from_counts_cpu


def test_su_of_a_constant_column_is_not_negative():
    """Counts [3, 7] against class frequencies that are a hair too large: every log term is negative, giving MI about -1e-9 and a negative SU."""
    joint = np.array([[3, 7]], dtype=np.int64)
    freqs_y = np.array([0.3 + 1e-9, 0.7 + 1e-9], dtype=np.float64)
    assert _mi_from_counts_cpu(joint, 1, freqs_y, 10, True) >= 0.0
