"""The three MI kernels write into an ``(n_bins, n_bins)`` histogram with bounds checking off; codes and targets outside it must be refused up front."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.mi import chatgpt_compute_mutual_information, deepseek_compute_mutual_information, grok_compute_mutual_information

_KERNELS = [grok_compute_mutual_information, chatgpt_compute_mutual_information, deepseek_compute_mutual_information]


@pytest.mark.parametrize("fn", _KERNELS)
def test_code_at_or_above_n_bins_is_refused(fn):
    """An int8 code of 20 with ``n_bins=15`` would write past the histogram; it raises instead."""
    data = np.random.default_rng(0).integers(0, 15, size=(100, 4)).astype(np.int8)
    data[3, 2] = 20
    with pytest.raises(ValueError, match="n_bins"):
        fn(data, np.array([0]), n_bins=15)


@pytest.mark.parametrize("fn", _KERNELS)
def test_negative_int8_code_is_refused(fn):
    """A negative int8 code skipped the range check and indexed the histogram from the end."""
    data = np.random.default_rng(1).integers(0, 15, size=(100, 4)).astype(np.int8)
    data[5, 1] = -1
    with pytest.raises(ValueError, match=r"\[0, 127\]"):
        fn(data, np.array([0]), n_bins=15)


@pytest.mark.parametrize("fn", _KERNELS)
def test_target_index_out_of_range_is_refused(fn):
    """A target column index past the last column read outside the array."""
    data = np.random.default_rng(2).integers(0, 15, size=(100, 4)).astype(np.int8)
    with pytest.raises(ValueError, match="target_indices"):
        fn(data, np.array([4]), n_bins=15)
