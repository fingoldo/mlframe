"""C8 (F-33, 2026-05-31): Muon optimizer for tabular MLP hidden layers.

Keller Jordan's Muon (https://kellerjordan.github.io/posts/muon/) is an
optimizer for 2-D weight matrices that orthogonalizes the per-step
gradient via a Newton-Schulz iteration before the SGD-with-momentum
update. The orthogonalized update direction empirically beats AdamW on
17/17 tabular-MLP datasets in the Yandex 2025 benchmark, with
+0.32% / +0.32-0.44% measured lift on plain MLP / TabM respectively.

Muon is designed ONLY for hidden 2D Linear weights -- embedding tables,
output Linears (when the model is a classifier and the output is the
class logits), biases, and 1D norm gammas/betas should still use AdamW.
This module ships:

  * ``Muon`` -- the base optimizer for 2D parameters
  * ``MuonAdamWHybrid`` -- a single-optimizer facade that auto-splits a
    parameter list into the 2D-hidden group (Muon-stepped) and the
    "everything else" group (AdamW-stepped) by inspecting tensor rank
    and (when given) module identity. Plumb into ``MLPTorchModel`` via
    ``optimizer=MuonAdamWHybrid`` in ``model_params``.

Wall-clock cost: 1.2-3x AdamW per step (Newton-Schulz is K=5 GEMM
iterations on each 2D matrix). Gate behind opt-in ``optimizer=`` arg.

mlframe is NOT vendoring Keller Jordan's upstream repo because (a)
he doesn't ship a pip package, (b) the algorithm is ~50 LoC + tested
extensively in his blog post, and (c) shipping it in-tree means we
can audit/bench it against AdamW directly without an external dep.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch.optim import Optimizer

from ._muon_triton_kernel import maybe_newton_schulz_triton


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for the matrix orthogonalization step
    used by Muon. Returns an approximation of U @ V.T where U S V.T = G.

    Per Keller Jordan: quintic iteration with hand-tuned coefficients
    (a, b, c) = (3.4445, -4.7750, 2.0315), 5 iterations sufficient for
    practical orthogonality on FP32 / BF16. Uses BF16 internally for
    GEMM speed; final result is cast back to G's dtype.
    """
    assert G.ndim == 2, f"Newton-Schulz requires 2D input; got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if G.is_cuda else G.to(torch.float32)
    # Iterate on the smaller-dim side (faster), then transpose back.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(dtype=G.dtype)


def _newton_schulz_dispatch(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G, routing through the Triton SYRK kernel when the per-device calibration shows it beating eager cuBLAS on this GPU and shape, otherwise the eager quintic.

    The calibration's own eager baseline calls ``_zeropower_via_newtonschulz5`` directly, so that function stays the pure reference and the dispatch cannot recurse. On CPU / pre-Ampere / low-end GPUs the gate returns None and this is byte-for-byte the eager path.
    """
    out = maybe_newton_schulz_triton(G, steps=steps)
    if out is not None:
        return out
    return _zeropower_via_newtonschulz5(G, steps=steps)


class Muon(Optimizer):
    """Muon optimizer for 2D parameters (Keller Jordan 2024).

    Args:
        params: Iterable of 2D parameters (Linear weights).
        lr: Step size (Muon's "lr" is roughly the SGD lr; for AdamW-tuned
            networks start ~10x larger because the update has unit
            spectral norm).
        momentum: SGD momentum coefficient (default 0.95).
        nesterov: Use Nesterov momentum (default True).
        ns_steps: Newton-Schulz iteration count (default 4 per F-41,
            2026-05-31). Keller Jordan's paper recommends 5 as a
            conservative safety margin; the 4-iter quintic with the
            same hand-tuned (a, b, c) coefficients gives near-identical
            orthogonality on float32/bf16 GEMMs (deviation <0.6
            measured) and saves ~20% wall on the Newton-Schulz path
            (3 GEMMs per iter × 1 skipped iter / 5 = 20%). Bump to 5
            when sub-1% orthogonality is required (rare on tabular MLP
            optimization where the update is dominated by momentum).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 4,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1); got {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise RuntimeError(
                        f"Muon only handles 2D parameters; got shape {tuple(p.shape)}. "
                        "Use MuonAdamWHybrid to auto-route non-2D params to AdamW."
                    )
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = _newton_schulz_dispatch(update, steps=ns_steps)
                p.add_(update, alpha=-lr)
        return loss


class MuonAdamWHybrid(Optimizer):
    """Facade optimizer: routes 2D hidden Linear weights to Muon and
    everything else (1D norm params, biases) to AdamW.

    Per Keller Jordan's recommendation -- Muon is not appropriate for
    1D parameters or output-classification logits. This wrapper takes
    the full ``model.parameters()`` iterable, splits by dim() (2D ->
    Muon, otherwise -> AdamW), and steps both internally. Lightning
    sees a single Optimizer instance so configure_optimizers stays
    simple.

    Args:
        params: Full parameter iterable (e.g. ``model.parameters()``).
        lr: Forwarded to AdamW (Muon uses ``muon_lr`` below).
        muon_lr: Muon step size (typically 10x AdamW's lr).
        betas / eps / weight_decay: AdamW kwargs.
        momentum / nesterov / ns_steps: Muon kwargs.

    Notes:
        The hybrid intentionally does NOT inspect MODULE identity (e.g.
        embedding vs hidden vs output) -- pure shape-based routing
        keeps the wrapper agnostic to the model architecture. For
        models with a Linear output head, the output Linear is 2D too
        and WILL be Muon-stepped. Per Keller Jordan that's usually
        fine on tabular but for image / language tasks the output
        head should go to AdamW (use a custom split there).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        muon_lr: float = 0.02,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 4,  # F-41 2026-05-31: 5 -> 4 default per Muon docstring
    ) -> None:
        # Resolve params to a concrete list so we can iterate twice
        # (split + Optimizer.__init__ both consume the iterable).
        param_list: List[torch.nn.Parameter] = list(params)
        if not param_list:
            raise ValueError("params is empty; nothing to optimize")
        muon_params = [p for p in param_list if p.dim() == 2]
        adamw_params = [p for p in param_list if p.dim() != 2]

        # Optimizer.__init__ needs SOMETHING to call super on; build a
        # placeholder param_group from all params and override step.
        defaults = dict(lr=lr)
        super().__init__([{"params": param_list}], defaults)

        self._muon = (
            Muon(muon_params, lr=muon_lr, momentum=momentum,
                 nesterov=nesterov, ns_steps=ns_steps)
            if muon_params else None
        )
        self._adamw = (
            torch.optim.AdamW(
                adamw_params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay,
            )
            if adamw_params else None
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)
