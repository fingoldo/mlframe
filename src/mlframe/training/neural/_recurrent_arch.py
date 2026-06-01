"""Recurrent model architecture components."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Attention mechanism to aggregate variable-length RNN outputs.

    Learns importance weights for each timestep, producing a fixed-size
    context vector regardless of sequence length.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        rnn_output: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            rnn_output: (batch, seq_len, hidden) RNN outputs
            lengths: (batch,) original sequence lengths

        Returns:
            Context vector (batch, hidden)
        """
        batch_size, max_len, hidden_size = rnn_output.size()
        device = rnn_output.device

        # F-50 (2026-05-31): investigated rewrite via F.scaled_dot_product_attention.
        # Win would be 1.3-2x on Ampere+ Flash, but ONLY when seq_len > 128.
        # Tabular sequences here are typically short (<=64), and SDPA requires
        # explicit scale=1.0 override (default 1/sqrt(d) changes the softmax
        # temperature, breaking trained-weight reproducibility). Deferred until
        # we ship a long-seq usecase that measurably wins.
        scores = self.attention(rnn_output).squeeze(-1)  # (batch, seq_len)

        # Mask padded positions with -inf so softmax assigns them zero weight
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        # F-E (2026-05-31, audit follow-up): if EVERY position in a row is
        # masked (lengths[b] == 0), softmax(-inf, -inf, ..., -inf) is NaN
        # everywhere; bmm propagates NaN into the context vector and
        # silently poisons the MLPHead with NaN. Detect such rows, give
        # them a safe uniform-zero attention (so the bmm produces a zero
        # vector), and continue. This is semantically the "no valid input
        # -> no signal -> zero context" contract; downstream layers can
        # then learn whatever bias is appropriate. Pre-fix, the upstream
        # _encode_sequences clamped lengths to >= 1 to dodge an exception
        # but that path made position 0 valid -> attention scored on the
        # RNN's response to a padding token (F-I concern). The all-masked
        # case is the residual one this guard catches.
        all_masked = mask.all(dim=1)  # (batch,)
        # Replace -inf rows with zeros so softmax yields uniform weights
        # (1/max_len) which we then explicitly zero out below.
        if all_masked.any():
            scores = torch.where(
                all_masked.unsqueeze(1), torch.zeros_like(scores), scores,
            )

        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum over time
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)

        # F-E zero-out: rows that had no valid input get a zero context.
        if all_masked.any():
            context = context * (~all_masked).unsqueeze(1).to(context.dtype)

        return context


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to input embeddings since Transformers have no inherent notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        # Sinusoidal PE requires interleaved sin/cos pairs; d_model<2 has no cos
        # channel and silently degrades to single-component encoding. Refuse
        # rather than emit a half-encoded buffer the caller can't detect.
        if d_model < 2:
            raise ValueError(
                f"PositionalEncoding requires d_model >= 2 (got {d_model}); "
                "sinusoidal encoding needs sin+cos channel pair."
            )
        self.dropout = nn.Dropout(p=dropout)

        # Precompute sin/cos positional encodings (Vaswani et al., 2017)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for variable-length sequences.

    Projects input to hidden_size, applies positional encoding, then runs through TransformerEncoder layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # F-48 (2026-05-31): enable_nested_tensor=True lets PyTorch 2.x
        # strip padded positions BEFORE the attention call and feed a
        # NestedTensor to the SDPA Flash backend. 1.5-3x on batches with
        # high padding variance. Safe with our boolean src_key_padding_mask.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True,
        )

        # Learnable CLS token, pooled from output[:, 0] for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.hidden_size = hidden_size

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sequences with Transformer.

        Args:
            sequences: (batch, seq_len, input_size) padded sequences
            lengths: (batch,) original sequence lengths

        Returns:
            Context vector (batch, hidden_size) from CLS token
        """
        batch_size, max_len, _ = sequences.size()
        device = sequences.device

        x = self.input_projection(sequences)  # (batch, seq_len, hidden_size)

        # Prepend CLS token so its final state pools whole-sequence context
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_size)

        x = self.pos_encoder(x)

        # src_key_padding_mask convention: True = position is padding and must be ignored.
        # CLS token sits at index 0 and is always valid; lengths counts real tokens, so positions > lengths are padding.
        seq_positions = torch.arange(max_len + 1, device=device).unsqueeze(0)
        padding_mask = seq_positions > lengths.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        return x[:, 0, :]  # (batch, hidden_size) - CLS token output


class MLPHead(nn.Module):
    """
    MLP classification/regression head used by all input modes for the final output projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits/predictions."""
        return self.mlp(x)
