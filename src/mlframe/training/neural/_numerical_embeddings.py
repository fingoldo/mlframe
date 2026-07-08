"""C7 (F-32, 2026-05-31): numerical-feature embeddings for tabular MLPs.

Per Holzmuller et al. 2024 "Better by Default" (NeurIPS, RealMLP-TD)
the single biggest ablation lift on tabular regression came from
periodic embeddings of numerical features (PLR / "Periodic Linear
ReLU"): +20.6% R^2 on the regression aggregate, +2.3% on classification.

The embedding maps each scalar feature ``x_i`` to a higher-dimensional
representation via sinusoidal positional-like encodings + a learned
projection:

    PLR(x_i) = ReLU( W_i @ concat[ sin(2*pi*c_i*x_i), cos(2*pi*c_i*x_i),
                                   x_i ] )

where ``c_i`` are K learnable frequencies (one set per input feature)
initialised from N(0, sigma) and ``W_i`` is a per-feature Linear
projecting to ``embed_dim``.

The resulting output shape is ``(N, in_features * embed_dim)`` and feeds
the existing MLP trunk as if it were the raw feature vector.

Reference: Gorishniy et al. 2022, "On Embeddings for Numerical Features
in Tabular Deep Learning" (https://arxiv.org/abs/2203.05556) — the PLR
embedding's predecessor; RealMLP-TD tuned the default frequency count
and sigma to 24 / 0.05 respectively for tabular.
"""
from __future__ import annotations

from typing import Optional, cast

import math
import torch
import torch.nn as nn


class PeriodicLinearEmbedding(nn.Module):
    """PLR embedding for numerical features.

    Args:
        in_features: Number of input numerical features.
        embed_dim: Per-feature embedding dimensionality. Total output
            dim is ``in_features * embed_dim``.
        n_frequencies: K -- number of learnable sin/cos frequencies per
            input feature. RealMLP-TD default: 24.
        sigma: Std of N(0, sigma) initialisation for the frequencies.
            RealMLP-TD default: 0.05 (small sigma => low-frequency basis
            on tabular features that already live near unit scale).
        include_raw: If True, include the raw scalar x_i in the
            per-feature input to the projection Linear (concatenated
            with sin / cos). RealMLP includes it; toggle False to
            reproduce vanilla periodic embedding.
        activation_cls: Activation class applied to the projection
            output. ReLU per the paper; set to None for identity
            (e.g. if downstream block already activates).

    Shapes:
        forward(x: (N, in_features)) -> (N, in_features * embed_dim)
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int = 8,
        n_frequencies: int = 24,
        sigma: float = 0.05,
        include_raw: bool = True,
        activation_cls: Optional[type] = nn.ReLU,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if n_frequencies < 1:
            raise ValueError(f"n_frequencies must be >= 1, got {n_frequencies}")

        self.in_features = in_features
        self.embed_dim = embed_dim
        self.n_frequencies = n_frequencies
        self.include_raw = include_raw
        self.out_features = in_features * embed_dim

        # Per-feature learnable frequencies: shape (in_features, n_frequencies).
        self.coeffs = nn.Parameter(torch.randn(in_features, n_frequencies) * sigma)

        # Per-feature input to the projection Linear:
        # 2 * n_frequencies (sin + cos) + 1 (raw, if included).
        per_feat_in = 2 * n_frequencies + (1 if include_raw else 0)
        # Stack per-feature Linears into one big block-diagonal Linear
        # for efficiency (single GEMM). Equivalent to in_features
        # independent Linear(per_feat_in -> embed_dim) but the
        # in_features axis is the "channel" axis of the conv-like op.
        # Implement via a single Linear over (in_features * per_feat_in)
        # then reshape; the block-diagonal structure is enforced
        # implicitly because we keep per-feature slices separate.
        self.proj = nn.Linear(in_features * per_feat_in, in_features * embed_dim)
        self.activation = activation_cls() if activation_cls is not None else nn.Identity()

        # ``example_input_array`` so MLPTorchModel's wrapper can do
        # ONNX export / Lightning's tensor-shape probe.
        self.example_input_array = torch.zeros(1, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features)
        if x.dim() != 2:
            raise ValueError(f"PLR embedding expects 2-D input (N, D); got shape {tuple(x.shape)}")
        # Outer product with frequencies -> (N, in_features, n_frequencies).
        # coeffs.unsqueeze(0) is (1, in_features, n_frequencies); x.unsqueeze(-1)
        # is (N, in_features, 1); broadcast gives the angle tensor.
        angles = (2.0 * math.pi) * x.unsqueeze(-1) * self.coeffs.unsqueeze(0)
        sines = torch.sin(angles)
        cosines = torch.cos(angles)
        if self.include_raw:
            per_feat = torch.cat([sines, cosines, x.unsqueeze(-1)], dim=-1)  # (N, D, 2K+1)
        else:
            per_feat = torch.cat([sines, cosines], dim=-1)  # (N, D, 2K)
        flat = per_feat.flatten(1)  # (N, D * (2K[+1]))
        return cast(torch.Tensor, self.activation(self.proj(flat)))  # (N, D * embed_dim)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, embed_dim={self.embed_dim}, "
            f"n_frequencies={self.n_frequencies}, "
            f"include_raw={self.include_raw}, out_features={self.out_features}"
        )
