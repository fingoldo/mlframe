"""``FixedSparseLinear``: dense layer with a fixed (non-stochastic) binary weight mask.

Source: 3rd_porto-seguro-safe-driver-prediction.md -- a dense layer regularized to "10% nonzero weights" by
multiplying the layer's weight matrix by a fixed binary (0/1) mask before every forward pass. Distinct from
dropout: the mask is FIXED at construction (not resampled per-batch), so it's a permanent structural sparsity
constraint on which weight connections can ever be nonzero, rather than a stochastic per-batch regularizer --
the author found a single sparse layer could match deeper architectures.

Standalone `nn.Module`, not wired into mlframe's Lightning-based NN estimator infra (`training/neural/base`,
`flat.py`) -- usable as a drop-in `nn.Linear` replacement inside any custom `nn.Sequential`/`nn.Module`
architecture, matching the precedent set earlier this session (swap-noise DAE, `MultiTaskAuxiliaryLossRegressor`)
of building small self-contained raw-PyTorch pieces when the full Lightning infra's fit/predict loop isn't
what's needed.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _build_importance_mask(
    importance: torch.Tensor,
    out_features: int,
    in_features: int,
    sparsity: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Build a fixed mask that keeps the top ``(1 - sparsity) * in_features`` inputs ranked by ``importance``.

    Every output row keeps the same set of input connections (the highest-ranked ones) -- there's no
    per-input importance signal that would justify varying the kept set per output neuron, so this
    concentrates all rows' capacity on the inputs a caller's importance signal says matter.
    """
    importance = importance.reshape(-1).to(torch.float32)
    if importance.numel() != in_features:
        raise ValueError(f"FixedSparseLinear: importance must have length in_features={in_features}, got {importance.numel()}")

    n_keep = max(1, round((1.0 - sparsity) * in_features))

    # Break ties among equal-importance entries deterministically-but-reproducibly via a tiny random jitter,
    # so a caller passing e.g. a coarse/binned importance signal doesn't get an arbitrary (argsort-order) bias.
    jitter = torch.rand(in_features, generator=generator) * 1e-9
    order = torch.argsort(importance + jitter, descending=True)
    keep_idx = order[:n_keep]

    mask = torch.zeros(out_features, in_features, dtype=torch.float32)
    mask[:, keep_idx] = 1.0
    return mask


class FixedSparseLinear(nn.Module):
    """A ``nn.Linear`` layer whose weight matrix is permanently masked to a fixed sparsity pattern.

    Parameters
    ----------
    in_features, out_features
        Same as ``nn.Linear``.
    sparsity
        Fraction of weight entries forced to (and kept at) zero, e.g. ``0.9`` for "10% nonzero weights".
    bias
        Whether to include a bias term (never masked).
    random_state
        Seed for the fixed mask's random layout. Ignored when ``importance`` is given and ties don't need
        breaking (kept for reproducible tie-breaking among equal-importance inputs).
    importance
        Optional length-``in_features`` per-input-feature importance score (e.g. mutual information,
        correlation with target, or any mlframe redundancy/MRMR ranking). When given, the mask is NOT drawn
        uniformly at random: every output neuron keeps the SAME top ``(1 - sparsity) * in_features`` input
        connections, ranked by ``importance`` (ties broken via ``random_state``). This turns the fixed mask
        from something the caller must hand-craft into a one-line call -- the actual adoption barrier for
        this layer -- and concentrates capacity on the inputs known to matter instead of spending it on
        connections to noise features a uniform-random mask would keep by chance. Leaving this ``None``
        reproduces the exact prior (uniform-random) mask construction, bit-for-bit, for a given
        ``random_state``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.9,
        bias: bool = True,
        random_state: Optional[int] = 42,
        importance: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= sparsity < 1.0):
            raise ValueError(f"FixedSparseLinear: sparsity must be in [0, 1), got {sparsity}")

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mask: torch.Tensor

        generator = torch.Generator().manual_seed(random_state) if random_state is not None else None
        keep_prob = 1.0 - sparsity

        if importance is None:
            mask = (torch.rand(out_features, in_features, generator=generator) < keep_prob).to(torch.float32)
        else:
            mask = _build_importance_mask(importance, out_features=out_features, in_features=in_features, sparsity=sparsity, generator=generator)

        self.register_buffer("mask", mask)

        with torch.no_grad():
            self.linear.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer with its fixed sparsity mask re-applied to the weight."""
        # Re-apply the mask every forward pass, not just at init: gradient updates would otherwise
        # reintroduce nonzero values at masked positions over training (the mask constrains the FORWARD
        # weight, not the gradient), so masking-at-init alone doesn't keep the sparsity pattern fixed.
        return nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)

    @property
    def actual_sparsity(self) -> float:
        """Fraction of weight entries currently exactly zero (should equal the constructor's ``sparsity``)."""
        return float((self.mask == 0).float().mean().item())


__all__ = ["FixedSparseLinear"]
