"""``AdditiveDecompositionRegressor``: shared-trunk NN whose prediction IS the sum of named component heads.

Source: 3rd_champs-scalar-coupling.md -- "Spin-coupling value can be decomposed into four different terms
(fc, sd, pso, dso)... auxiliary target using contributions gave a high boost": when a target is a KNOWN
additive decomposition of named physical/domain components (even when component labels exist only in training
data), jointly predicting each component AND supervising their sum consistently helped in two independent
winning solutions (this one and the LANL earthquake writeup already implemented as
``multitask_auxiliary_loss.MultiTaskAuxiliaryLossRegressor``).

Distinct from ``MultiTaskAuxiliaryLossRegressor``: that class has a SEPARATE, freely-parameterized primary head
plus up to two fixed AUXILIARY heads used purely as regularizers (their own loss shapes the shared trunk, but
the primary head's output is not architecturally constrained to relate to them). This class has NO separate
primary head at all -- the primary prediction IS the sum of N NAMED component heads
(``primary_pred = sum(component_heads)``), a physically-constrained decomposition architecture, not a
regularization trick. Component supervision is OPTIONAL per component (a component with no label at fit time
still gets a head and contributes to the sum, just receives no direct supervision loss -- gradient reaches it
only via the primary-sum loss).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

_SUPPORTED_COMPONENT_CONSTRAINTS = ("non_negative",)


class AdditiveDecompositionRegressor(BaseEstimator, RegressorMixin):
    """Shared-trunk MLP whose output is the SUM of named component heads, jointly trained.

    Parameters
    ----------
    component_names
        Names of the additive components (e.g. ``("fc", "sd", "pso", "dso")``). The prediction is
        ``sum(component_head(x) for component in component_names)``.
    hidden_sizes
        Shared-trunk hidden layer widths.
    component_task_weight
        Weight applied to each SUPERVISED component's own loss in the joint sum (the primary sum's own loss
        against ``y_primary`` always has weight 1.0).
    component_constraints
        Optional ``{component_name: "non_negative"}`` -- when a component is marked ``"non_negative"``, its raw
        linear head output is passed through ``softplus`` (smooth, differentiable) before being summed into the
        primary prediction and before being compared to its own supervision target. Components with no entry
        here behave exactly as before (raw linear output, unchanged math) -- default ``None`` reproduces the
        pre-existing behavior bit-for-bit. ``"non_negative"`` is currently the only supported constraint kind.
    n_epochs, lr, batch_size
        Training configuration (full-batch Adam by default, ``batch_size=None``).
    random_state
        Seed for weight init.

    Attributes
    ----------
    train_losses_
        Per-epoch joint training loss, for diagnostics.
    """

    def __init__(
        self,
        component_names: Sequence[str],
        hidden_sizes: tuple = (32, 16),
        component_task_weight: float = 0.3,
        component_constraints: Optional[Dict[str, str]] = None,
        n_epochs: int = 300,
        lr: float = 0.01,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.component_names = component_names
        self.hidden_sizes = hidden_sizes
        self.component_task_weight = component_task_weight
        self.component_constraints = component_constraints
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_trunk_and_heads(self, n_features: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        layers = []
        prev = n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        trunk = nn.Sequential(*layers)
        component_heads = nn.ModuleDict({name: nn.Linear(prev, 1) for name in self.component_names})
        return trunk, component_heads

    def _validate_component_constraints(self) -> Dict[str, str]:
        constraints = self.component_constraints or {}
        unknown_components = set(constraints) - set(self.component_names)
        if unknown_components:
            raise ValueError(
                f"AdditiveDecompositionRegressor: component_constraints has unknown component(s) {sorted(unknown_components)}; expected a subset of {list(self.component_names)}."
            )
        unknown_kinds = {kind for kind in constraints.values() if kind not in _SUPPORTED_COMPONENT_CONSTRAINTS}
        if unknown_kinds:
            raise ValueError(f"AdditiveDecompositionRegressor: unsupported component_constraints value(s) {sorted(unknown_kinds)}; supported: {_SUPPORTED_COMPONENT_CONSTRAINTS}.")
        return constraints

    @staticmethod
    def _apply_component_constraint(name: str, raw_output, constraints: Dict[str, str]):
        # Deliberately untyped (torch.Tensor): an explicit annotation flips sum()'s overload resolution to
        # Tensor | int at the call sites below, since nn.Linear.__call__ itself is untyped (Any).
        # Unconstrained components (the default, no entry in ``constraints``) pass through unchanged --
        # this branch is what makes the no-constraint path bit-identical to the pre-existing behavior.
        kind = constraints.get(name)
        if kind is None:
            return raw_output
        import torch.nn.functional as F

        if kind == "non_negative":
            return F.softplus(raw_output)
        raise ValueError(f"AdditiveDecompositionRegressor: unsupported component constraint kind {kind!r} for component {name!r}.")

    def fit(
        self,
        X: np.ndarray,
        y_primary: np.ndarray,
        component_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> "AdditiveDecompositionRegressor":
        """Fit the additive decomposition.

        Parameters
        ----------
        X
            Feature matrix.
        y_primary
            The TRUE-units target (the known sum of the components).
        component_targets
            Optional ``{component_name: labels}`` -- components without a supplied label still get a head
            (contributing to the sum) but receive no direct supervision loss.
        """
        import torch
        import torch.nn as nn

        component_targets = component_targets or {}
        unknown = set(component_targets) - set(self.component_names)
        if unknown:
            raise ValueError(f"AdditiveDecompositionRegressor.fit: component_targets has unknown component(s) {sorted(unknown)}; expected a subset of {list(self.component_names)}.")
        constraints = self._validate_component_constraints()

        X_arr = np.asarray(X, dtype=np.float32)
        y_primary_arr = np.asarray(y_primary, dtype=np.float32).reshape(-1, 1)

        self.trunk_, self.component_heads_ = self._build_trunk_and_heads(X_arr.shape[1])

        params = list(self.trunk_.parameters()) + list(self.component_heads_.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        X_t = torch.from_numpy(X_arr)
        y_primary_t = torch.from_numpy(y_primary_arr)
        component_targets_t = {name: torch.from_numpy(np.asarray(labels, dtype=np.float32).reshape(-1, 1)) for name, labels in component_targets.items()}

        mse = nn.MSELoss()

        self.train_losses_ = []
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            hidden = self.trunk_(X_t)
            component_preds = {name: self._apply_component_constraint(name, self.component_heads_[name](hidden), constraints) for name in self.component_names}
            primary_pred = sum(component_preds.values())
            loss = mse(primary_pred, y_primary_t)
            for name, target_t in component_targets_t.items():
                loss = loss + self.component_task_weight * mse(component_preds[name], target_t)
            loss.backward()
            optimizer.step()
            self.train_losses_.append(float(loss.item()))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the primary target (sum of all component predictions)."""
        import torch

        constraints = self._validate_component_constraints()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            hidden = self.trunk_(X_t)
            primary_pred = sum(self._apply_component_constraint(name, self.component_heads_[name](hidden), constraints) for name in self.component_names)
        return np.asarray(primary_pred.numpy().ravel(), dtype=np.float64)

    def predict_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict each named component separately (diagnostic access to the decomposition)."""
        import torch

        constraints = self._validate_component_constraints()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            hidden = self.trunk_(X_t)
            return {
                name: np.asarray(self._apply_component_constraint(name, self.component_heads_[name](hidden), constraints).numpy().ravel(), dtype=np.float64)
                for name in self.component_names
            }


__all__ = ["AdditiveDecompositionRegressor"]
