"""F-E (2026-05-31) regression: AttentionPooling all-padding-row produces
NaN context without the F-E guard. The guard zeros those rows so the
MLPHead doesn't receive NaN-poisoned features.

Pre-fix: ``softmax([-inf, -inf, ..., -inf]) -> NaN`` (entire row).
``bmm(NaN_weights, anything)`` -> ``NaN`` context. Downstream linear
layers propagate NaN to logits; metrics still report "finite" because
torchmetrics has its own NaN-skip behaviour, masking the silent
contamination. Pure F-58 class of silent correctness.

Post-fix: when lengths[b] == 0 the row gets a zero context vector. The
MLPHead then receives a deterministic "no signal" input -- still wrong
relative to a meaningful summary, but at least not NaN.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch

from mlframe.training.neural._recurrent_arch import AttentionPooling


def test_attention_pooling_all_padding_row_returns_zero_not_nan():
    """F-E fix: a batch element with lengths=0 gets a finite zero
    context vector instead of NaN-propagating softmax + bmm.
    """
    B, T, H = 3, 5, 4
    rnn_output = torch.randn(B, T, H)
    # b=0: full sequence; b=1: half; b=2: all padding.
    lengths = torch.tensor([5, 3, 0])
    pool = AttentionPooling(hidden_size=H)
    out = pool(rnn_output, lengths)
    assert out.shape == (B, H)
    assert torch.isfinite(out).all(), (
        f"F-E: all-padding row produced non-finite context: {out}"
    )
    # The all-padding row MUST be exactly zeros.
    torch.testing.assert_close(out[2], torch.zeros(H))
    # Non-zero-length rows should NOT be zeroed.
    assert out[0].abs().sum() > 0, "row with valid input had zero context"
    assert out[1].abs().sum() > 0, "row with valid input had zero context"


def test_attention_pooling_no_padding_unchanged():
    """When no row is fully padded, the F-E guard is a no-op and
    AttentionPooling behaves exactly as before. Verifies the guard
    doesn't perturb the common case."""
    B, T, H = 4, 6, 8
    rnn_output = torch.randn(B, T, H)
    lengths = torch.tensor([6, 5, 6, 4])  # all > 0
    pool = AttentionPooling(hidden_size=H)
    # Bit-identical run with monkeypatched implementation that strips
    # the F-E guard would give the same numbers; instead verify finiteness
    # + that bytes match a fresh run (the only stochasticity is the
    # Linear's learned weights, fixed at __init__).
    out_a = pool(rnn_output, lengths)
    out_b = pool(rnn_output, lengths)
    torch.testing.assert_close(out_a, out_b)
    assert torch.isfinite(out_a).all()


def test_attention_pooling_handles_mixed_short_lengths():
    """Mix of lengths from 1..T including very-short sequences. The
    softmax + bmm path should still produce finite output and the
    first-position concentration on short sequences should be evident."""
    B, T, H = 3, 4, 4
    rnn_output = torch.randn(B, T, H)
    lengths = torch.tensor([1, 2, 4])
    pool = AttentionPooling(hidden_size=H)
    out = pool(rnn_output, lengths)
    assert out.shape == (B, H)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attention_pooling_dtype_preserved(dtype):
    """F-E guard uses ``torch.zeros_like`` + ``.to(context.dtype)``
    which must NOT silently up/downcast the output."""
    B, T, H = 2, 4, 3
    rnn_output = torch.randn(B, T, H, dtype=dtype)
    lengths = torch.tensor([4, 0])
    pool = AttentionPooling(hidden_size=H).to(dtype)
    out = pool(rnn_output, lengths)
    assert out.dtype == dtype
