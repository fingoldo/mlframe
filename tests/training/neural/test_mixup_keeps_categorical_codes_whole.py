"""Mixup never interpolates the integer category codes that feed the embedding layer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural._mixup import mixup_batch, mixup_sequence_batch


def test_category_code_columns_stay_integer_valued_codes_from_one_of_the_two_rows():
    """Column 0 holds codes 0 and 5; after mixup every row's code is 0 or 5, never a blended 1.5 that would embed a third category."""
    torch.manual_seed(0)
    x = torch.tensor([[0.0, 1.0], [5.0, 2.0]] * 8)
    y = torch.arange(16, dtype=torch.float32)
    for _ in range(20):
        mixed, *_ = mixup_batch(x, y, alpha=0.4, n_cat_prefix=1)
        assert set(mixed[:, 0].tolist()) <= {0.0, 5.0}


def test_numeric_columns_are_still_interpolated():
    """The numeric column is blended (negative control): some mixed value is neither 1.0 nor 2.0."""
    torch.manual_seed(1)
    x = torch.tensor([[0.0, 1.0], [5.0, 2.0]] * 8)
    mixed, *_ = mixup_batch(x, torch.zeros(16), alpha=0.4, n_cat_prefix=1)
    assert any(v not in (1.0, 2.0) for v in mixed[:, 1].tolist())


def test_sequence_mixup_keeps_aux_codes_whole():
    """Same rule for the aux features of the sequence path."""
    torch.manual_seed(2)
    seq = torch.randn(8, 4, 3)
    lengths = torch.full((8,), 4)
    aux = torch.tensor([[0.0, 1.0], [7.0, 2.0]] * 4)
    for _ in range(10):
        _s, _l, aux_mixed, *_ = mixup_sequence_batch(seq, lengths, torch.zeros(8), alpha=0.4, aux_features=aux, aux_n_cat_prefix=1)
        assert set(aux_mixed[:, 0].tolist()) <= {0.0, 7.0}
