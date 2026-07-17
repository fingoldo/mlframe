"""biz_value test for ``training.neural.group_causal_attention_mask``.

Claim: the mask correctly implements "full attention within a group, strict causal order across groups" --
proven directly at the attention-mechanism level (not just checking mask values): changing a position's input
in a STRICTLY LATER group must NOT change any earlier-group position's attention output (no future-group
leakage), while changing a position WITHIN the same group (or an earlier group, attendable per the causal
order) DOES change a later position's output.
"""

from __future__ import annotations

from typing import cast

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
    assert torch.allclose(out1[:, :3, :], out_future_changed[:, :3, :], atol=1e-6), (
        "changing the latest-group position leaked into an earlier-group position's attention output"
    )
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


def _causal_leakage_to_feat(vals: torch.Tensor, rng: torch.Generator, n_feat: int) -> torch.Tensor:
    """Embed scalar ``vals`` (n, k) as (n, k, n_feat) feature vectors: dim 0 carries the value, rest is noise."""
    n, k = vals.shape
    feat = torch.zeros(n, k, n_feat)
    feat[:, :, 0] = vals
    feat[:, :, 1:] = 0.01 * torch.randn(n, k, n_feat - 1, generator=rng)
    return feat


def _causal_leakage_make_batch(n: int, ood: bool, rng: torch.Generator, n_feat: int) -> tuple[torch.Tensor, torch.Tensor]:
    """5-position sequence: group 0 (2 causal-source events), group 1 (1 query anchor), group 2 (2 future events).

    ``target`` depends ONLY on group 0. In-distribution (``ood=False``), group 2's values are a low-noise copy
    of ``target`` (a leaky shortcut a bidirectional model can exploit) -- this mirrors a real causal-ML trap:
    "future" data that correlates with the label during training/backtest but is not honestly available (or not
    honestly correlated) at deployment. ``ood=True`` simulates deployment: group 2 becomes independent noise,
    decorrelated from ``target``, so any model that leaned on it degrades.
    """
    a = torch.rand(n, 2, generator=rng)
    target = a.mean(dim=1, keepdim=True) + 0.05 * torch.randn(n, 1, generator=rng)
    c = torch.rand(n, 2, generator=rng) if ood else target.expand(-1, 2) + 0.02 * torch.randn(n, 2, generator=rng)
    anchor_val = torch.rand(n, 1, generator=rng)  # dummy; carries no signal about target
    sequences = torch.cat(
        [
            _causal_leakage_to_feat(a, rng, n_feat),
            _causal_leakage_to_feat(anchor_val, rng, n_feat),
            _causal_leakage_to_feat(c, rng, n_feat),
        ],
        dim=1,
    )
    return sequences, target


def _causal_leakage_forward(encoder: TransformerSequenceEncoder, head: nn.Linear, sequences: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
    """Run the encoder's own submodules directly, pooling from the group-1 anchor position (index 2) instead of
    CLS -- CLS pooling is unsuitable for this claim because the encoder's documented CLS-mask padding convention
    (an always-0 CLS row/column) makes CLS attend to every group unconditionally, regardless of ``attn_mask``
    (verified empirically: a future-group edit still changes the CLS-pooled output under a group-causal mask).
    Pooling from a real sequence position exercises the identical production ops (input_projection, pos_encoder,
    transformer) while actually respecting the mask.
    """
    x = encoder.input_projection(sequences)
    x = encoder.pos_encoder(x)
    x = encoder.transformer(x, mask=attn_mask)
    return cast(torch.Tensor, head(x[:, 2, :]))


def _causal_leakage_train(
    attn_mask: torch.Tensor | None, seed: int, n_feat: int, num_steps: int = 100, batch: int = 32
) -> tuple[TransformerSequenceEncoder, nn.Linear]:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    encoder = TransformerSequenceEncoder(input_size=n_feat, hidden_size=12, num_layers=1, n_heads=2, dim_feedforward=24)
    head = nn.Linear(12, 1)
    encoder.train()
    head.train()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=8e-3)
    for _ in range(num_steps):
        sequences, target = _causal_leakage_make_batch(batch, ood=False, rng=rng, n_feat=n_feat)
        pred = _causal_leakage_forward(encoder, head, sequences, attn_mask)
        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    encoder.eval()
    head.eval()
    return encoder, head


def test_biz_val_group_causal_mask_prevents_trained_model_from_relying_on_future_group_leakage():
    """End-to-end training claim: a bidirectional (unmasked) TransformerSequenceEncoder-based regressor exploits
    a leaky future-group shortcut during training and degrades sharply when that shortcut is unavailable at
    deployment (OOD group), while the SAME architecture trained with ``group_causal_attention_mask`` cannot see
    the future group at all and is therefore provably invariant to it.
    """
    n_feat = 4
    group_ids = torch.tensor([0, 0, 1, 2, 2])  # anchor query sits in group 1; group 2 is strictly later
    causal_mask = group_causal_attention_mask(group_ids)

    encoder_masked, head_masked = _causal_leakage_train(causal_mask, seed=0, n_feat=n_feat)
    encoder_unmasked, head_unmasked = _causal_leakage_train(None, seed=0, n_feat=n_feat)

    def eval_mse(encoder: TransformerSequenceEncoder, head: nn.Linear, attn_mask: torch.Tensor | None, ood: bool) -> float:
        rng = torch.Generator().manual_seed(1999)
        sequences, target = _causal_leakage_make_batch(256, ood=ood, rng=rng, n_feat=n_feat)
        with torch.no_grad():
            pred = _causal_leakage_forward(encoder, head, sequences, attn_mask)
            return float(torch.nn.functional.mse_loss(pred, target).item())

    mse_masked_indist = eval_mse(encoder_masked, head_masked, causal_mask, ood=False)
    mse_masked_ood = eval_mse(encoder_masked, head_masked, causal_mask, ood=True)
    mse_unmasked_indist = eval_mse(encoder_unmasked, head_unmasked, None, ood=False)
    mse_unmasked_ood = eval_mse(encoder_unmasked, head_unmasked, None, ood=True)

    # Both models must actually learn the task in-distribution (rules out "unmasked just fails to train").
    assert mse_masked_indist < 0.02
    assert mse_unmasked_indist < 0.02

    # The causally-masked model structurally cannot see group 2 at all -- its OOD error must stay close to its
    # in-distribution error (measured ratio ~1.0x; 3x is a generous stability bound for a 256-sample eval).
    assert mse_masked_ood < mse_masked_indist * 3.0

    # The unmasked model learned to lean on the leaky future-group shortcut -- OOD error must degrade sharply
    # (measured ~23x on the original run; 5x set well below that to absorb seed variance while still proving
    # real degradation, not noise).
    assert mse_unmasked_ood > mse_unmasked_indist * 5.0

    # The core business claim: under the future/deployment distribution shift, the causally-masked model must
    # beat the unmasked one by a wide margin (measured ~8x on the original run; 3x set below that).
    assert mse_masked_ood < mse_unmasked_ood / 3.0


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
