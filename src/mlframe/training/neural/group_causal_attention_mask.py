"""``group_causal_attention_mask``: causal masking that treats same-group events as simultaneous.

Source: MLFRAME_IDEAS_production.md -- "container/group-aware attention masking for grouped-but-simultaneous
events in sequence models": when multiple events in a sequence share the same "container"/basket/timestamp,
their relative order within that group is arbitrary, so imposing a strict left-to-right causal order on them
(as a plain triangular mask does) injects a spurious ordering signal the model has no reason to trust. This
builds a mask where position ``i`` may attend to position ``j`` iff ``group[j] <= group[i]`` -- full (bidirectional)
attention WITHIN a group, strict causal ordering ACROSS groups, using the group index itself (not raw sequence
position) as the causal clock.

Confirmed via reuse-check: mlframe's ``training.neural._recurrent_arch.TransformerSequenceEncoder`` is a plain
BIDIRECTIONAL encoder-only pooling architecture with no causal mask at all (only a padding mask) -- this module
does not retrofit that encoder into an autoregressive one; it is a standalone, reusable mask-construction
primitive any attention layer's ``attn_mask=`` parameter can consume (PyTorch's standard convention: an additive
float mask, ``0.0`` where attention is allowed and ``-inf`` where it is blocked).
"""
from __future__ import annotations

import torch

__all__ = ["group_causal_attention_mask"]


def group_causal_attention_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """Build an additive attention mask: full attention within a group, strict causal order across groups.

    Parameters
    ----------
    group_ids : torch.Tensor
        ``(seq_len,)`` or ``(batch, seq_len)`` integer tensor giving each position's group/container/timestamp
        index. Positions sharing the same value are treated as SIMULTANEOUS (no relative-order constraint
        between them); a HIGHER group value is treated as strictly LATER than a lower one. Need not be
        contiguous or start at 0 -- only the relative ordering of values matters.

    Returns
    -------
    torch.Tensor
        ``(seq_len, seq_len)`` (unbatched input) or ``(batch, seq_len, seq_len)`` (batched input) additive
        float mask: ``mask[..., i, j] = 0.0`` if position ``j`` is attendable from position ``i`` (``group[j] <=
        group[i]``), else ``-inf``. Directly usable as ``attn_mask=`` for ``nn.MultiheadAttention`` /
        ``nn.TransformerEncoderLayer`` (which add this to the raw attention logits before softmax).
    """
    if group_ids.ndim not in (1, 2):
        raise ValueError(f"group_causal_attention_mask: group_ids must be 1-D or 2-D; got shape {tuple(group_ids.shape)}.")

    unbatched = group_ids.ndim == 1
    g = group_ids.unsqueeze(0) if unbatched else group_ids  # (batch, seq_len)

    # allowed[b, i, j] = True iff group[b, j] <= group[b, i] (j is attendable from i).
    allowed = g.unsqueeze(2) >= g.unsqueeze(1)  # (batch, seq_len, seq_len); dim1=i (query), dim2=j (key)

    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=group_ids.device)
    mask = mask.masked_fill(~allowed, float("-inf"))

    return mask.squeeze(0) if unbatched else mask
