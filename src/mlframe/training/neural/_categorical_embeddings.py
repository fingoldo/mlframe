"""Learnable entity embeddings for categorical features in tabular MLPs.

Each categorical column is mapped through its own ``nn.Embedding`` table, trained end-to-end with the rest of the network (Guo & Berkhahn 2016,
"Entity Embeddings of Categorical Variables"). Unlike a target encoder (CatBoostEncoder / WoE) -- which collapses a category to a single scalar
derived from the target and so can only express a MONOTONE category->target relationship without leakage controls -- a learned embedding gives
each category a free ``embed_dim``-vector the trunk can combine non-linearly, so a non-monotone ``y = lookup[cat]`` mapping is recoverable.

Layout contract (mirrors how the estimator hands data in): ``forward(x)`` receives a single ``(N, D)`` float tensor whose FIRST ``len(cardinalities)``
columns are integer category codes (cast to long + clamped to ``[0, card]`` so an unseen / overflow code lands on the reserved last embedding row),
and whose remaining ``D - k`` columns are already-numeric features passed through untouched. The output concatenates every per-cat embedding with the
numeric block, giving width ``sum(embed_dims) + (D - k)`` exposed as ``out_features`` -- the downstream trunk consumes it as if it were the raw vector.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def default_embed_dim(cardinality: int) -> int:
    """Fastai-style embedding-dimension heuristic: ``min(50, round(1.6 * card**0.56))``.

    Grows sub-linearly with cardinality and caps at 50 so a very-high-card column doesn't blow up the parameter count. ``card`` here is the
    number of distinct categories (NOT counting the reserved unknown row); the floor of 1 keeps a binary / constant column at a usable width.
    """
    return max(1, min(50, round(1.6 * (max(1, cardinality) ** 0.56))))


class CategoricalEmbedding(nn.Module):
    """Per-categorical learnable entity embeddings + numeric passthrough.

    Args:
        cardinalities: Per-categorical category count (the number of distinct codes the estimator's factorizer assigned, NOT including the
            reserved unknown slot). One ``nn.Embedding(card + 1, d)`` is built per entry; row ``card`` is the reserved unknown/overflow row.
        embed_dim: Fixed per-cat embedding width. When ``None`` (default) each cat's width is the fastai heuristic ``default_embed_dim(card)``.
        padding_idx: Optional embedding row index whose vector is fixed at zero (passed through to every ``nn.Embedding``). ``None`` (default)
            leaves all rows trainable.

    Shapes:
        forward(x: (N, D)) -> (N, sum(embed_dims) + (D - k)) where ``k = len(cardinalities)``.
    """

    def __init__(
        self,
        cardinalities: list[int],
        embed_dim: Optional[int] = None,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not cardinalities:
            raise ValueError("cardinalities must be a non-empty list of per-cat category counts")
        for c in cardinalities:
            if not isinstance(c, (int,)) or isinstance(c, bool):
                raise TypeError(f"each cardinality must be an int, got {type(c).__name__} ({c!r})")
            if c < 1:
                raise ValueError(f"each cardinality must be >= 1, got {c!r}")
        if embed_dim is not None and embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1 when set, got {embed_dim!r}")

        self.cardinalities = list(cardinalities)
        self.n_cat = len(self.cardinalities)
        # +1 reserved row per table: an unseen-at-fit-time or out-of-range code clamps to index ``card`` (the last row).
        self.embed_dims = [
            (embed_dim if embed_dim is not None else default_embed_dim(c))
            for c in self.cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(card + 1, dim, padding_idx=padding_idx)
                for card, dim in zip(self.cardinalities, self.embed_dims)
            ]
        )
        # out_features is resolved lazily against the live input width in forward(): the numeric block is ``D - n_cat`` columns. We expose the
        # CAT contribution here; ``_numeric_width`` is filled on first forward so callers that need the exact total can read ``out_features``
        # after a forward, but generate_mlp passes the known total width via out_features computed from the fit-time D (see set_num_numeric).
        self._cat_out = int(sum(self.embed_dims))
        self._num_numeric: Optional[int] = None

        # ``example_input_array`` mirrors PeriodicLinearEmbedding so Lightning's tensor-shape probe / ONNX export has a concrete sample. Width
        # is just the cat block here (n_cat code columns); numeric columns, if any, are appended by the caller's real data at train time.
        self.example_input_array = torch.zeros(1, self.n_cat)

    def set_num_numeric(self, num_numeric: int) -> None:
        """Record the count of trailing numeric (non-cat) input columns so ``out_features`` reflects the full post-embedding width.

        Called by ``generate_mlp`` right after construction with ``D - k`` where ``D`` is the fit-time input width and ``k = n_cat``. This lets
        the trunk's first Linear / input Dropout / norm size to ``sum(embed_dims) + num_numeric`` without waiting for a forward pass.
        """
        if num_numeric < 0:
            raise ValueError(f"num_numeric must be >= 0, got {num_numeric!r}")
        self._num_numeric = int(num_numeric)
        self.example_input_array = torch.zeros(1, self.n_cat + self._num_numeric)

    @property
    def out_features(self) -> int:
        num_numeric = self._num_numeric if self._num_numeric is not None else 0
        return self._cat_out + num_numeric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"CategoricalEmbedding expects 2-D input (N, D); got shape {tuple(x.shape)}")
        if x.shape[1] < self.n_cat:
            raise ValueError(
                f"CategoricalEmbedding input has {x.shape[1]} columns but {self.n_cat} leading cat columns are expected"
            )

        outs = []
        for i, emb in enumerate(self.embeddings):
            # Cast to long for the embedding lookup; clamp so an unseen / negative / overflow code maps to the reserved last row (index card)
            # instead of raising IndexError. Codes arrive as float (the whole X tensor is one float dtype) holding integer values.
            codes = x[:, i].long().clamp_(0, self.cardinalities[i])
            outs.append(emb(codes))

        if x.shape[1] > self.n_cat:
            outs.append(x[:, self.n_cat:])
        return torch.cat(outs, dim=1)

    def extra_repr(self) -> str:
        return (
            f"n_cat={self.n_cat}, cardinalities={self.cardinalities}, "
            f"embed_dims={self.embed_dims}, num_numeric={self._num_numeric}, "
            f"out_features={self.out_features}"
        )
