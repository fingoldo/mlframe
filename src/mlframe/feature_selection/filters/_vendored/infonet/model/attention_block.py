"""Vendored (infonet): pre-norm transformer blocks combining attention + a residual MLP feed-forward."""
from typing import Optional

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention


class CrossAttentionBlock(nn.Module):
    """Cross-attention (query ``x_q`` attends to context ``x_kv``) followed by a residual MLP, both with residual adds."""

    def __init__(self, q_dim: int, kv_dim: int, qk_out_dim: Optional[int], v_out_dim: Optional[int], heads: int, widening_factor: int = 1, dropout: float = 0.0):

        super().__init__()
        self.cross_attention = CrossAttention(q_dim=q_dim, kv_dim=kv_dim, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, heads=heads, dropout=dropout)

        self.mlp = MLP(q_dim, widening_factor, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """``x_q``: (..., q_dim) query tokens; ``x_kv``: (..., kv_dim) context tokens. Returns (..., q_dim)."""
        attention = self.cross_attention(x_q=x_q, x_kv=x_kv, attention_mask=attention_mask)
        attention = self.dropout(attention)
        x = x_q + attention
        # x = attention
        x = x + self.mlp(x)

        return x


class SelfAttentionBlock(nn.Module):
    """Self-attention over ``x`` followed by a residual MLP, both with residual adds."""

    def __init__(self, q_dim: int, qk_out_dim: Optional[int], v_out_dim: Optional[int], heads: int, widening_factor: int = 1, dropout: float = 0.0):

        super().__init__()
        self.self_attention = SelfAttention(q_dim=q_dim, qk_out_dim=qk_out_dim, v_out_dim=v_out_dim, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(q_dim, widening_factor, dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """``x``: (..., q_dim). Returns (..., q_dim)."""
        attention = self.self_attention(x_q=x, attention_mask=attention_mask)
        attention = self.dropout(attention)
        x = x + attention
        # x = attention
        x = x + self.mlp(x)

        return x


class MLP(nn.Module):
    """Pre-norm feed-forward block: LayerNorm -> Linear(widening_factor*hidden_dim) -> GELU -> Linear(hidden_dim) -> Dropout."""

    def __init__(self, hidden_dim: int, widening_factor: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, widening_factor * hidden_dim),
            nn.GELU(),
            nn.Linear(widening_factor * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """``x``: (..., hidden_dim). Returns (..., hidden_dim)."""
        return self.mlp(x)
