"""Recurrent model architecture components."""

from __future__ import annotations

from typing import Optional

import math
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

        # Compute attention scores
        scores = self.attention(rnn_output).squeeze(-1)  # (batch, seq_len)

        # Create mask for padded positions
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over valid positions
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)

        return context


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to input embeddings since Transformers
    have no inherent notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for variable-length sequences.

    Projects input to hidden_size, applies positional encoding,
    then runs through TransformerEncoder layers.
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

        # Project input to hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for classification (learnable)
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

        # Project to hidden size
        x = self.input_projection(sequences)  # (batch, seq_len, hidden_size)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_size)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for padded positions (True = ignore)
        seq_positions = torch.arange(max_len + 1, device=device).unsqueeze(0)
        padding_mask = seq_positions > lengths.unsqueeze(1)

        # Run through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Return CLS token representation
        return x[:, 0, :]  # (batch, hidden_size)


class MLPHead(nn.Module):
    """
    MLP classification/regression head.

    Used by all input modes for final output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, ...],
        output_size: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        layers: List[nn.Module] = []
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


# ----------------------------------------------------------------------------------------------------------------------------
# Lightning Module
# ----------------------------------------------------------------------------------------------------------------------------


