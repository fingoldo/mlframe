"""F-63 (2026-05-31): Sharpness-Aware Minimization (Foret et al. 2020,
https://arxiv.org/abs/2010.01412).

SAM modifies the standard "minimize the loss at the current weights"
objective to "minimize the loss in the WORST-CASE neighbourhood of the
current weights." Concretely, every SGD step becomes two passes:

    Step 1 (ascent / perturbation):
        L1 = loss(theta)
        grad = dL1/dtheta
        epsilon = rho * grad / |grad|_2
        theta_perturbed = theta + epsilon

    Step 2 (descent at perturbed point):
        L2 = loss(theta_perturbed)
        grad' = dL2/dtheta
        theta = theta - lr * grad'
        # (restore: theta = theta_perturbed - epsilon = theta_orig, then
        #  optimizer step from theta_orig using grad')

The net effect is that the optimizer steps using a gradient evaluated
at a "neighbouring" point, biasing the trajectory toward FLAT minima
(low-curvature loss regions) which generalise better than sharp minima.

Foret et al. 2020 measured +0.8-1.5% on ImageNet over SGD-momentum;
follow-up work confirmed similar gains on tabular MLP (+0.3-0.7% in
the Yandex 2025 benchmark) and ViT/CNN.

Cost: 2 forward + 2 backward per training step = ~2x train wall vs
the base optimizer alone. Composition with cheap base optimizers
(SGD-momentum, AdamW) amortizes well; composition with Muon's heavy
Newton-Schulz orthogonalization makes the wrap pricier.

This module ships a clean implementation that:
  1. Inherits from torch.optim.Optimizer so Lightning treats it as a
     normal optimizer with state_dict round-trip support.
  2. Forwards param_groups + state to the wrapped base so schedulers
     and add_param_group work transparently.
  3. Exposes first_step / second_step explicit calls for users who
     want fine control (e.g. interleaving SAM with a custom training
     loop); also a one-shot ``step(closure)`` that does both stages.

Mlframe wiring: MLPTorchModel reads ``use_sam`` / ``sam_rho`` /
``sam_adaptive`` hparams (defaults False / 0.05 / False). When
use_sam=True, configure_optimizers wraps the base AdamW (or Muon, or
whatever the user provided) in SAM. The Lightning training_step needs a
closure-style call; we provide a Lightning-compatible ``optimizer_step``
override hook documented in this module's caller.

Adaptive SAM (Kwon 2021): scales the perturbation per-parameter by
that parameter's magnitude. ``sam_adaptive=True`` enables this; +0.2%
additional lift on most settings.
"""
from __future__ import annotations

from typing import Callable, List

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper.

    Args:
        base_optimizer: An ALREADY-initialised optimizer (AdamW, SGD,
            Muon, ...). SAM will call its ``step()`` after the perturbed
            backward pass.
        rho: Perturbation radius. Foret 2020 used 0.05 for ImageNet
            CNNs; 0.05-0.10 works on tabular MLP per follow-up papers.
        adaptive: When True, switch to Adaptive-SAM (Kwon 2021) which
            scales the per-parameter perturbation by |theta|. Often
            +0.2% over vanilla SAM at no extra cost.

    Notes on the Optimizer API:
        We do NOT call ``super().__init__`` because we forward
        ``param_groups`` to ``base_optimizer``. The standard Optimizer
        init expects a param iterable + defaults; both already live on
        the base. Lightning's optimizer-handling code calls
        ``optimizer.param_groups`` / ``state`` / ``step`` -- we
        intercept those.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
    ) -> None:
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError(f"base_optimizer must be torch.optim.Optimizer; got " f"{type(base_optimizer).__name__}")
        if rho <= 0:
            raise ValueError(f"rho must be > 0; got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        # Cache of pre-perturbation params (set by first_step, restored
        # by second_step). Keyed by param id; values are detached clones.
        self._param_backup: dict[int, torch.Tensor] = {}
        self.defaults = base_optimizer.defaults

    # ---- Optimizer protocol forwarding -------------------------------

    @property
    def param_groups(self) -> List[dict]:
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: List[dict]) -> None:
        self.base_optimizer.param_groups = value

    @property  # type: ignore[override]  # SAM wraps base_optimizer and intentionally delegates state as a property
    def state(self) -> dict:
        return self.base_optimizer.state

    @state.setter
    def state(self, value: dict) -> None:
        self.base_optimizer.state = value

    def add_param_group(self, param_group: dict) -> None:
        self.base_optimizer.add_param_group(param_group)

    def state_dict(self) -> dict:
        return {
            "base": self.base_optimizer.state_dict(),
            "rho": self.rho,
            "adaptive": self.adaptive,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict["base"])
        self.rho = state_dict.get("rho", self.rho)
        self.adaptive = state_dict.get("adaptive", self.adaptive)
        self._param_backup = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    # ---- SAM-specific two-step API ------------------------------------

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Concatenate per-param grad norms into one L2 norm. For
        adaptive SAM, scale each by |theta| before norming."""
        device = None
        norms = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if device is None:
                    device = p.grad.device
                g = p.grad
                if self.adaptive:
                    g = g * torch.abs(p.data)
                norms.append(g.norm(p=2).to(device))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Apply the ascent perturbation: theta <- theta + epsilon
        where epsilon = rho * grad / |grad|. Caller MUST have run a
        loss.backward() pass before this so each param.grad holds dL/dtheta.

        After first_step, the user runs a SECOND forward + backward at
        the perturbed weights, then calls ``second_step()``.
        """
        grad_norm = self._grad_norm()
        # Avoid div-by-zero: scale floor.
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Backup the current param value so second_step can
                # restore before stepping.
                self._param_backup[id(p)] = p.data.clone()
                g = p.grad
                e_w = (torch.abs(p.data) if self.adaptive else 1.0) * g * scale.to(p.data.device)
                p.data.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the pre-perturbation params, then run a single step
        of the base optimizer using the gradient from the perturbed
        forward+backward.
        """
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if id(p) in self._param_backup:
                    p.data.copy_(self._param_backup[id(p)])
        self._param_backup.clear()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):  # type: ignore[override]
        """One-shot SAM step. ``closure`` MUST be supplied and re-evaluate
        the loss + backward; this method calls it twice (once at the
        original params, once at the perturbed params).

        Lightning's automatic optimization passes a closure that wraps
        training_step -- so this is the right entry point for the
        Lightning integration.
        """
        if closure is None:
            raise RuntimeError(
                "SAM.step() requires a closure that re-runs loss + backward; " "got closure=None. Use first_step() / second_step() for " "explicit control."
            )
        # First pass: closure was already called (Lightning's contract).
        # Use the existing grads for the ascent perturbation.
        self.first_step(zero_grad=True)
        # Second pass: rerun forward + backward at perturbed weights.
        with torch.enable_grad():
            loss = closure()
        self.second_step(zero_grad=False)
        return loss
