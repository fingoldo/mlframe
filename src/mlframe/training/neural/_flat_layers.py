"""Helpers carved out of ``flat`` to keep that module under its size budget."""

from __future__ import annotations


import logging
from enum import Enum, auto
from typing import Callable, Optional, cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLPNeuronsByLayerArchitecture(Enum):
    """Per-layer neuron-count progression pattern consumed by ``generate_mlp``.

    ``Constant`` keeps every hidden layer at ``first_layer_num_neurons``;
    ``Declining``/``Expanding`` scale each successive layer by
    ``consec_layers_neurons_ratio`` (down/up); ``ExpandingThenDeclining``
    grows to the middle layer then shrinks (a "bottleneck-inverted" shape);
    ``Autoencoder`` shrinks to the middle then grows back (the classic
    encoder/decoder bottleneck shape).
    """

    Constant = auto()
    Declining = auto()
    Expanding = auto()
    ExpandingThenDeclining = auto()
    Autoencoder = auto()


def get_valid_num_groups(num_channels: int, preferred_num_groups: int) -> int:
    """Finds the largest divisor of ``num_channels`` not exceeding ``preferred_num_groups``.

    ``nn.GroupNorm`` requires ``num_channels % num_groups == 0``; layer widths chosen by the
    architecture-progression logic are rarely divisible by an arbitrary preferred group count.
    Falls back to 1 group (LayerNorm-equivalent behaviour) when no larger divisor exists.
    """
    for g in range(preferred_num_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # Fallback to 1 (LayerNorm-like) if no divisor found


class _BoundedTanhOutput(nn.Module):
    """Bounded-range output head: ``tanh(x) * scale + center``.

    Wraps the last ``nn.Linear`` of a regression MLP to HARD-CAP the
    output to ``[center - scale, center + scale]``. Composes cleanly with
    ``_TTRWithEvalSetScaling``: when y is z-scored by the TTR transformer
    and ``scale``/``center`` are computed on the SCALED y the MLP sees at
    fit-time, the tanh window is in scaled space and the TTR's
    ``inverse_transform`` unwinds it back to raw-y space correctly.

    Why this fix is independent of the defensive TTR predict clip
    (which lives at ``_TTRWithEvalSetScaling.predict``): the TTR clip
    BOUNDS the damage from runaway predictions; this output activation
    PREVENTS the MLP's affine-composition from emitting them in the
    first place. The MLP gradient sees the bound during training and
    learns parameters that keep activations inside the window; the TTR
    clip only catches what slips through at inference.

    ``scale`` and ``center`` are registered as non-trainable BUFFERS so
    they (a) move to the right device with ``.to(device)`` calls,
    (b) save/load with state_dict, and (c) do not get updated by the
    optimizer (they are fixed properties of the train target).
    """

    # Declared here (register_buffer() below sets them) so mypy sees plain Tensors rather than
    # the register_buffer stub's broad Tensor|Module return.
    scale: torch.Tensor
    center: torch.Tensor

    def __init__(self, scale: float, center: float) -> None:
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(float(scale), dtype=torch.float32),
        )
        self.register_buffer(
            "center", torch.tensor(float(center), dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the bounded-tanh output head: ``tanh(x) * scale + center``."""
        # F-37 (2026-05-31): always use the separate-op form
        # ``tanh(x) * scale + center``. Pre-fix this had a ``if x.is_cuda:
        # return addcmul(...)`` data-dependent branch — measured 1.5-2.4x
        # on CUDA via the explicit addcmul fusion, but the branch
        # FRAGMENTS torch.compile's Inductor fusion of the output head
        # (per the 2026-05-31 torch.compile audit, Agent A finding #2).
        # The whole tanh + mul + add chain is exactly the pure-pointwise
        # pattern Inductor fuses into ONE Triton kernel automatically
        # when torch.compile is active, recovering the CUDA fusion win
        # without the data-dependent branch. The CPU trade is 5-20% on
        # tiny output heads (bench: (200000, 1) 0.78x, (4096, 64) 0.95x),
        # accepted because (a) it's a tiny absolute time, (b) compile is
        # the load-bearing perf lever now, (c) the previous CUDA branch
        # was opt-in to a fusion win only available without compile —
        # a no-win combo. autograd's tanh gradient (1 - tanh^2) is
        # preserved either way.
        return torch.tanh(x) * self.scale + self.center

    def extra_repr(self) -> str:
        """Reports the current ``scale``/``center`` for ``print(module)`` / ``repr``."""
        return f"scale={float(self.scale):.4g}, center={float(self.center):.4g}"


class _ResidualLinearBlock(nn.Module):
    """C6 (F-31, 2026-05-31): single residual block for tabular MLPs.

    Gorishniy 2021 ("Revisiting Deep Learning Models for Tabular Data")
    found that a properly-tuned residual MLP outperforms TabNet /
    NODE / TabTransformer on standard tabular benchmarks. The block is::

        x  ->  Linear(in, out)
            -> [BN] -> activation -> [Dropout]
            -> ADD skip(x)

    where ``skip(x)`` is either identity (when ``in == out``) or a
    bias-free 1-Linear projection (when ``in != out``) so the addition
    is shape-safe across the existing per-layer-width MLP architectures
    (Constant / Declining / Expanding / etc.).

    This is the lightweight ResNet-tabular variant the agent recommended
    as the highest-ROI architectural change (~30 LoC, zero new deps,
    drop-in to ``generate_mlp``).
    """

    # norm is one of BatchNorm1d / LayerNorm / Identity depending on the constructor flags below.
    norm: nn.Module

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_cls: Optional[Callable],
        dropout_prob: float,
        use_batchnorm: bool,
        use_layernorm_per_layer: bool,
        batch_norm_kwargs: dict,
        layer_norm_kwargs: dict,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if use_batchnorm:
            self.norm = nn.BatchNorm1d(out_dim, **batch_norm_kwargs)
        elif use_layernorm_per_layer:
            self.norm = nn.LayerNorm(out_dim, **layer_norm_kwargs)
        else:
            self.norm = nn.Identity()
        self.act = activation_cls() if activation_cls is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        # Skip projection: identity when dims match (parameter-free);
        # bias-free Linear when dims differ (smallest projection cost).
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs Linear -> [norm] -> activation -> [dropout], then adds the (identity or projected) skip connection."""
        return cast(torch.Tensor, self.dropout(self.act(self.norm(self.linear(x)))) + self.skip(x))

    def extra_repr(self) -> str:
        """Reports in/out widths and skip-connection kind (identity vs. bias-free linear projection)."""
        return f"in={self.linear.in_features}, out={self.linear.out_features}, " f"skip={'identity' if isinstance(self.skip, nn.Identity) else 'linear'}"
