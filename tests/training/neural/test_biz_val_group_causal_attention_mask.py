"""biz_value test for ``training.neural.group_causal_attention_mask``.

Claim: the mask correctly implements "full attention within a group, strict causal order across groups" --
proven directly at the attention-mechanism level (not just checking mask values): changing a position's input
in a STRICTLY LATER group must NOT change any earlier-group position's attention output (no future-group
leakage), while changing a position WITHIN the same group (or an earlier group, attendable per the causal
order) DOES change a later position's output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mlframe.training.neural._recurrent_arch import TransformerSequenceEncoder
from mlframe.training.neural.group_causal_attention_mask import group_causal_attention_mask


def test_group_causal_attention_mask_values():
    group_ids = torch.tensor([0, 0, 1, 2])
    mask = group_causal_attention_mask(group_ids)
    assert mask.shape == (4, 4)
    # position 0 (group 0) can attend to position 1 (same group 0) and itself, not positions 2/3 (later groups).
    assert mask[0, 0] == 0.0
    assert mask[0, 1] == 0.0
    assert mask[0, 2] == float("-inf")
    assert mask[0, 3] == float("-inf")
    # position 3 (group 2, latest) can attend to everything (all earlier-or-equal groups).
    assert torch.all(mask[3, :] == 0.0)
    # position 2 (group 1) can attend to 0, 1 (group 0, earlier) and itself (group 1), not 3 (group 2, later).
    assert mask[2, 0] == 0.0
    assert mask[2, 1] == 0.0
    assert mask[2, 2] == 0.0
    assert mask[2, 3] == float("-inf")


def test_group_causal_attention_mask_rejects_bad_ndim():
    import pytest

    with pytest.raises(ValueError):
        group_causal_attention_mask(torch.zeros(2, 3, 4))


def test_biz_val_group_causal_mask_blocks_future_group_leakage_at_attention_level():
    torch.manual_seed(0)
    d_model = 8
    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)
    mha.eval()

    group_ids = torch.tensor([0, 0, 1, 2])  # positions 0,1 simultaneous; 2 next; 3 latest
    mask = group_causal_attention_mask(group_ids)

    x1 = torch.randn(1, 4, d_model)

    with torch.no_grad():
        out1, _ = mha(x1, x1, x1, attn_mask=mask)

        # Change ONLY the latest-group position (3) -- strictly later than every other position's group.
        x_future_changed = x1.clone()
        x_future_changed[:, 3, :] = torch.randn(1, d_model)
        out_future_changed, _ = mha(x_future_changed, x_future_changed, x_future_changed, attn_mask=mask)

        # Change a SAME-group position (1, same group as 0) -- attendable from position 0 under the mask.
        x_samegroup_changed = x1.clone()
        x_samegroup_changed[:, 1, :] = torch.randn(1, d_model)
        out_samegroup_changed, _ = mha(x_samegroup_changed, x_samegroup_changed, x_samegroup_changed, attn_mask=mask)

    # No future-group leakage: positions 0, 1, 2 (all earlier-or-equal groups than 3) must be UNCHANGED when
    # only position 3's input changes.
    assert torch.allclose(out1[:, :3, :], out_future_changed[:, :3, :], atol=1e-6), "changing the latest-group position leaked into an earlier-group position's attention output"
    # Position 3 itself must change (it attends to itself).
    assert not torch.allclose(out1[:, 3, :], out_future_changed[:, 3, :], atol=1e-6)

    # Within-group full attention: changing position 1 (group 0) must change position 0's output (same group,
    # attendable per group[j] <= group[i]) -- proving the mask does NOT impose a spurious order within a group.
    assert not torch.allclose(out1[:, 0, :], out_samegroup_changed[:, 0, :], atol=1e-6), "expected full (non-causal) attention within the same group"


def test_transformer_sequence_encoder_accepts_optional_attn_mask_without_regression():
    torch.manual_seed(0)
    encoder = TransformerSequenceEncoder(input_size=6, hidden_size=16, num_layers=1, n_heads=2, dim_feedforward=32)
    encoder.eval()

    sequences = torch.randn(2, 5, 6)
    lengths = torch.tensor([5, 3])

    with torch.no_grad():
        out_default = encoder(sequences, lengths)
        out_explicit_none = encoder(sequences, lengths, attn_mask=None)

    assert out_default.shape == (2, 16)
    assert torch.allclose(out_default, out_explicit_none), "attn_mask=None (default) must be bit-identical to the pre-existing no-mask behaviour"


def test_transformer_sequence_encoder_with_group_causal_mask_produces_finite_output():
    import torch.nn.functional as F

    torch.manual_seed(0)
    encoder = TransformerSequenceEncoder(input_size=6, hidden_size=16, num_layers=1, n_heads=2, dim_feedforward=32)
    encoder.eval()

    batch_size, seq_len = 2, 5
    sequences = torch.randn(batch_size, seq_len, 6)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    group_ids = torch.tensor([0, 0, 1, 1, 2])
    raw_mask = group_causal_attention_mask(group_ids)  # (5, 5)
    # Pad with an always-attendable CLS row/column at index 0 (CLS pools the whole sequence).
    padded_mask = F.pad(raw_mask, (1, 0, 1, 0), value=0.0)  # (6, 6)

    with torch.no_grad():
        out = encoder(sequences, lengths, attn_mask=padded_mask)

    assert out.shape == (batch_size, 16)
    assert torch.all(torch.isfinite(out))
