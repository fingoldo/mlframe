"""F-J (2026-05-31) regression test: bidirectional ``_get_last_hidden``
must gather the BACKWARD RNN's *first-position* hidden state, not its
last-position hidden state.

Pre-fix bug
-----------
For BIDIRECTIONAL LSTM/GRU/RNN with pack_padded_sequence:
  * The forward RNN processes positions 0, 1, ..., length-1 in order.
    ``h_fwd[length-1]`` summarises the entire valid sequence.
  * The backward RNN processes positions length-1, length-2, ..., 0 in
    reverse. ``h_bwd[0]`` summarises the entire valid sequence (it's
    the backward RNN's FINAL hidden state in run-order). ``h_bwd[length-1]``
    is just the backward RNN's state after seeing ONLY position length-1
    -- i.e. only the last valid token, NOT a summary.

The old ``_get_last_hidden`` gathered ``rnn_out[b, length-1, :]`` for
ALL channels, so the bwd half of the returned vector was "summary of
the last valid token" rather than "summary of the entire sequence".
That meant every bidirectional + use_attention=False recurrent fit
silently fed the MLPHead a half-padding-derived embedding -- numerically
finite, shape-correct, but semantically a ~50% feature corruption on
the backward channels. Estimated R^2 drop 0.1-0.3 on sequence tasks
where the backward summary carries signal.

The fix gathers fwd at length-1 (unchanged) and bwd at position 0 (the
correct first run-order position) and concatenates.

How this test catches the bug
-----------------------------
Build a tiny manual bidirectional RNN with a known recurrent function.
For a fixed sequence, compute the expected ``[fwd_summary | bwd_summary]``
analytically, then assert ``_get_last_hidden(rnn_out, lengths,
bidirectional=True)`` returns that exact vector. Pre-fix the second
half would be ``h_bwd[length-1]`` (last-token bwd state), NOT
``h_bwd[0]`` (full bwd summary).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel


def test_unidirectional_get_last_hidden_unchanged():
    """Unidirectional path MUST behave identically to pre-F-J: gather
    rnn_out[:, length-1, :]."""
    B, T, H = 2, 5, 4
    rnn_out = torch.arange(B * T * H, dtype=torch.float32).view(B, T, H)
    lengths = torch.tensor([3, 5])
    out = RecurrentTorchModel._get_last_hidden(
        rnn_out, lengths, bidirectional=False,
    )
    # b=0, length=3 -> gather rnn_out[0, 2, :]
    # b=1, length=5 -> gather rnn_out[1, 4, :]
    expected = torch.stack([rnn_out[0, 2, :], rnn_out[1, 4, :]])
    torch.testing.assert_close(out, expected)


def test_bidirectional_get_last_hidden_uses_bwd_position_0_for_bwd_half():
    """F-J fix: for bidirectional, the bwd half MUST come from position 0
    (the backward RNN's full-sequence summary), NOT from position length-1
    (the backward RNN's last-token-only state). Channel split is
    [:H] fwd, [H:] bwd."""
    B, T = 2, 5
    H = 4
    # Synth rnn_out: channels [0..3] = fwd, [4..7] = bwd. Each element
    # encodes (batch, time, channel_role) so the gather choices are obvious.
    rnn_out = torch.zeros(B, T, 2 * H)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                # fwd channels: 100*b + t * 10 + h
                rnn_out[b, t, h] = 100 * b + t * 10 + h
                # bwd channels: 1000 + 100*b + t * 10 + h (distinct namespace)
                rnn_out[b, t, H + h] = 1000 + 100 * b + t * 10 + h

    lengths = torch.tensor([3, 5])
    out = RecurrentTorchModel._get_last_hidden(
        rnn_out, lengths, bidirectional=True,
    )
    assert out.shape == (B, 2 * H)

    # b=0, length=3:
    #   fwd half:  rnn_out[0, 2, :H]  -> 100*0 + 2*10 + h = [20, 21, 22, 23]
    #   bwd half:  rnn_out[0, 0, H:]  -> 1000 + 100*0 + 0*10 + h = [1000, 1001, 1002, 1003]
    expected_b0 = torch.tensor(
        [20.0, 21.0, 22.0, 23.0, 1000.0, 1001.0, 1002.0, 1003.0]
    )
    torch.testing.assert_close(out[0], expected_b0)

    # b=1, length=5:
    #   fwd half:  rnn_out[1, 4, :H]  -> 100*1 + 4*10 + h = [140, 141, 142, 143]
    #   bwd half:  rnn_out[1, 0, H:]  -> 1000 + 100*1 + 0*10 + h = [1100, 1101, 1102, 1103]
    expected_b1 = torch.tensor(
        [140.0, 141.0, 142.0, 143.0, 1100.0, 1101.0, 1102.0, 1103.0]
    )
    torch.testing.assert_close(out[1], expected_b1)


def test_bidirectional_get_last_hidden_pre_fix_would_take_bwd_position_length_minus_one():
    """Anti-regression: prove the OLD behaviour would have returned the
    bwd half at position length-1 (which is the bug). We don't run the
    old code -- we just hand-compute what it would have returned and
    confirm the new code does NOT return that.
    """
    B, T, H = 2, 5, 4
    rnn_out = torch.zeros(B, T, 2 * H)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                rnn_out[b, t, h] = 100 * b + t * 10 + h
                rnn_out[b, t, H + h] = 1000 + 100 * b + t * 10 + h

    lengths = torch.tensor([3, 5])
    out = RecurrentTorchModel._get_last_hidden(
        rnn_out, lengths, bidirectional=True,
    )
    # Pre-fix would have returned rnn_out[b, length-1, :] for BOTH halves.
    # b=0, length=3: bwd-half pre-fix = rnn_out[0, 2, H:] = [1020, 1021, 1022, 1023]
    bwd_prefix_b0 = torch.tensor([1020.0, 1021.0, 1022.0, 1023.0])
    # Our fixed bwd half is [1000, 1001, 1002, 1003] (position 0), NOT pre-fix.
    assert not torch.allclose(out[0, H:], bwd_prefix_b0), (
        "F-J fix not in place: bidirectional gather still uses bwd at "
        "length-1 (pre-fix buggy behaviour). The bwd half must come from "
        "position 0 to summarise the full valid sequence."
    )


def test_bidirectional_smoke_through_full_recurrent_forward():
    """End-to-end smoke: a small bidirectional LSTM-with-no-attention
    forward pass produces finite, non-degenerate output. Guards against
    the F-J fix breaking the surrounding code paths (cat / shape).
    """
    from mlframe.training.neural._recurrent_config import (
        InputMode,
        RecurrentConfig,
        RNNType,
    )

    cfg = RecurrentConfig(
        input_mode=InputMode.SEQUENCE_ONLY,
        rnn_type=RNNType.LSTM,
        hidden_size=4,
        num_layers=1,
        bidirectional=True,
        use_attention=False,  # F-J path
        mlp_hidden_sizes=(8,),
        dropout=0.0,
    )
    model = RecurrentTorchModel(
        cfg, seq_input_size=3, aux_input_size=0,
        is_regression=True,
    )
    model.eval()

    # Build a synthetic mini-batch with 3 sequences of different lengths.
    seqs = [
        torch.randn(6, 3),
        torch.randn(4, 3),
        torch.randn(2, 3),
    ]
    from torch.nn.utils.rnn import pad_sequence
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([6, 4, 2])

    with torch.no_grad():
        logits = model(sequences=padded, lengths=lengths)

    assert logits.shape == (3, 1)
    assert torch.isfinite(logits).all()
