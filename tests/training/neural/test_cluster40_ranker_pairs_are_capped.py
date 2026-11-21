"""Regression tests for the audit findings in this module."""
import pytest
"""Audit 2026-11-20 finding 40-F11: the precomputed ranking pairs are capped per query, like the loss, and stay valid pairs."""

import torch

pytest.importorskip("lightning")
from mlframe.training.neural.ranker import _cap_pair_indices


def test_pairs_above_the_cap_are_subsampled_to_it():
    """A pair set above the cap is cut to exactly the cap."""
    i_idx = torch.arange(100)
    j_idx = torch.arange(100) + 1
    out_i, out_j = _cap_pair_indices(i_idx, j_idx, cap=30)
    assert out_i.numel() == 30 and out_j.numel() == 30


def test_pairs_under_the_cap_are_unchanged():
    """A pair set within the cap is returned as is."""
    i_idx = torch.arange(10)
    j_idx = torch.arange(10) + 1
    out_i, out_j = _cap_pair_indices(i_idx, j_idx, cap=30)
    assert torch.equal(out_i, i_idx) and torch.equal(out_j, j_idx)


def test_kept_pairs_are_original_pairs_and_deterministic():
    """Every kept pair is one of the originals, and two calls keep the same pairs."""
    i_idx = torch.arange(100)
    j_idx = torch.arange(100) * 2
    a = _cap_pair_indices(i_idx, j_idx, cap=20)
    b = _cap_pair_indices(i_idx, j_idx, cap=20)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert set(zip(a[0].tolist(), a[1].tolist())) <= set(zip(i_idx.tolist(), j_idx.tolist()))
