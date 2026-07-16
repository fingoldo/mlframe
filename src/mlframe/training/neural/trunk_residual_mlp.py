"""``TrunkResidualMLPRegressor``: a shared trunk block skip-connected into EVERY deeper block.

Source: 3rd_jane-street-market-prediction.md -- a 49-layer MLP built from a trunk block (0) whose output is
skip-connected into every subsequent residual block 1..23. Distinct from mlframe's existing
`_ResidualLinearBlock` (`training/neural/flat.py`), which only skip-connects a block's own IMMEDIATE
predecessor (standard adjacent-layer ResNet skip): here ONE shared low-level representation is re-injected at
EVERY later block, not just handed down block-to-block -- as the network gets deep, adjacent-only skips still
let the original trunk signal dilute/degrade across many hops; re-injecting it directly at every block keeps
it available at full strength no matter how deep the tower gets.

Standalone `nn.Module` + a self-contained training loop, matching the `FixedSparseLinear`/`Tabular1DCNNRegressor`
precedent (raw-PyTorch pieces for niche architectures, not routed through mlframe's Lightning-based estimator
infra).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from torch import nn


class _TrunkInjectedBlock(nn.Module):
    """Residual block that re-injects the shared trunk representation into its own input every forward pass."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        """Add the trunk representation into ``x``, transform, and residual-add the result back onto ``x``."""
        # trunk is re-injected (added) at THIS block's input, on top of the running residual carry from x --
        # the block sees both "what the tower has computed so far" and "the original trunk representation",
        # not just the former (which is all an adjacent-only skip would give it this many hops downstream).
        h: torch.Tensor = self.act(self.norm(self.linear(x + trunk)))
        return x + h


class _TrunkResidualMLPModule(nn.Module):
    """Torch module: a shared trunk block whose output is skip-connected into every subsequent residual block."""

    def __init__(self, n_features: int, trunk_dim: int = 32, n_blocks: int = 6) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_features, trunk_dim), nn.LayerNorm(trunk_dim), nn.ReLU())
        self.blocks = nn.ModuleList([_TrunkInjectedBlock(trunk_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(trunk_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the trunk representation and pass it through every trunk-injected block, then the head."""
        trunk_out = self.trunk(x)
        h = trunk_out
        for block in self.blocks:
            h = block(h, trunk_out)
        out: torch.Tensor = self.head(h).squeeze(-1)
        return out


class TrunkResidualMLPRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible regressor: a shared trunk representation re-injected into every deeper block.

    Parameters
    ----------
    trunk_dim
        Width of the shared trunk representation (and every subsequent block).
    n_blocks
        Number of trunk-injected residual blocks stacked after the trunk.
    n_epochs
        Training epochs (full-batch Adam).
    learning_rate
        Adam learning rate.
    random_state
        Seed for model init and training.
    """

    def __init__(self, trunk_dim: int = 32, n_blocks: int = 6, n_epochs: int = 300, learning_rate: float = 0.01, random_state: int = 0) -> None:
        self.trunk_dim = trunk_dim
        self.n_blocks = n_blocks
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrunkResidualMLPRegressor":
        """Train the trunk-residual MLP module with full-batch Adam."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        torch.manual_seed(self.random_state)
        self.model_ = _TrunkResidualMLPModule(n_features=X_arr.shape[1], trunk_dim=self.trunk_dim, n_blocks=self.n_blocks)

        X_t = torch.from_numpy(X_arr)
        y_t = torch.from_numpy(y_arr)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(X_t), y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the fitted trunk-residual MLP module in eval mode."""
        X_arr = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.from_numpy(X_arr))
        return np.asarray(preds.numpy())

    # ------------------------------------------------------------------
    # Opt-in seed-ensemble: fit()/predict() above are untouched by any of this --
    # a caller who never invokes fit_seed_ensemble/predict_std gets bit-identical
    # single-seed behavior. Full-batch Adam training here is seed-sensitive (small
    # data, few epochs converge to different local optima per init) enough that a
    # single seed's point estimate can be noisy; a seed-ensemble trades K-fold
    # training cost for a lower-variance averaged prediction plus an epistemic
    # per-row std, without guessing K up front (see seed_ensemble_variance_curve).
    # ------------------------------------------------------------------

    def fit_seed_ensemble(self, X: np.ndarray, y: np.ndarray, n_seeds: int = 8, base_random_state: Optional[int] = None) -> "TrunkResidualMLPRegressor":
        """Fit ``n_seeds`` independently-seeded clones of this estimator on the SAME (X, y).

        Each clone differs only in ``random_state`` (init + training draw the same data). Also fits ``self``
        on the base ``random_state`` so ``predict``/``predict_ensemble_mean`` are both usable afterward.
        Stores the fitted clones in ``ensemble_members_`` (list, length ``n_seeds``).
        """
        if n_seeds < 1:
            raise ValueError(f"TrunkResidualMLPRegressor.fit_seed_ensemble: n_seeds must be >=1, got {n_seeds}.")
        seed0 = self.random_state if base_random_state is None else base_random_state
        rng = np.random.RandomState(seed0)
        seeds = rng.randint(0, np.iinfo(np.int32).max, size=n_seeds)

        self.fit(X, y)  # keeps self.model_/predict() usable at the base seed, same as single-seed usage.
        members: List["TrunkResidualMLPRegressor"] = []
        for seed in seeds:
            member = clone(self).set_params(random_state=int(seed))
            member.fit(X, y)
            members.append(member)
        self.ensemble_members_ = members
        return self

    def _ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Stack seed-member predictions -> (n_seeds, n_rows) float64 matrix."""
        members = getattr(self, "ensemble_members_", None)
        if not members:
            raise NotFittedError("TrunkResidualMLPRegressor: call fit_seed_ensemble before predict_ensemble_mean/predict_std.")
        return np.vstack([np.asarray(m.predict(X), dtype=np.float64) for m in members])

    def predict_ensemble_mean(self, X: np.ndarray) -> np.ndarray:
        """Seed-averaged point estimate across ``ensemble_members_`` (lower variance than any single seed)."""
        return np.asarray(self._ensemble_predictions(X).mean(axis=0))

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Per-row across-seed prediction std (population, ddof=0) -- an epistemic/seed-sensitivity signal."""
        return np.asarray(self._ensemble_predictions(X).std(axis=0, ddof=0))

    @staticmethod
    def seed_ensemble_variance_curve(
        X: np.ndarray,
        y: np.ndarray,
        X_eval: np.ndarray,
        k_values: Sequence[int] = (1, 2, 4, 8, 16),
        base_random_state: int = 0,
        **model_params: Any,
    ) -> Dict[str, List[float]]:
        """Diagnose the minimal seed-ensemble size ``K`` that stabilizes predictions, WITHOUT guessing.

        Fits ``max(k_values)`` seeded members once (reusing ``fit_seed_ensemble``) to get a converged estimate
        of the per-row across-seed std ``sigma`` (the noise floor a single extra seed contributes). The
        variance of a K-member ensemble MEAN is ``sigma^2 / K`` (standard-error-of-the-mean scaling for K
        exchangeable draws of the same noise source), so its std shrinks as ``sigma / sqrt(K)`` -- reported per
        requested K WITHOUT re-fitting per K. This is the quantity that actually stabilizes as K grows (unlike
        the raw across-member spread, which converges UP to sigma as K grows, not down): it shows diminishing
        returns (halving from K=1 to K=8 removes far more std than halving from K=8 to K=16), letting callers
        pick the smallest K past the knee instead of an arbitrary default.

        Returns ``{"k_values": [...], "mean_std": [...]}`` (``mean_std`` = mean-over-rows predicted std of the
        K-ensemble mean estimate), both sorted ascending by K.
        """
        k_sorted = sorted(set(int(k) for k in k_values))
        if any(k < 1 for k in k_sorted):
            raise ValueError(f"TrunkResidualMLPRegressor.seed_ensemble_variance_curve: all k_values must be >=1, got {k_values!r}.")
        k_max = k_sorted[-1]

        probe = TrunkResidualMLPRegressor(random_state=base_random_state, **model_params)
        probe.fit_seed_ensemble(X, y, n_seeds=k_max, base_random_state=base_random_state)
        preds = np.vstack([np.asarray(m.predict(X_eval), dtype=np.float64) for m in probe.ensemble_members_])  # (k_max, n_rows)
        sigma_row = preds.std(axis=0, ddof=0)  # per-row across-seed std, converged at k_max members.

        mean_std = [float((sigma_row / np.sqrt(k)).mean()) for k in k_sorted]
        return {"k_values": [float(k) for k in k_sorted], "mean_std": mean_std}


__all__ = ["TrunkResidualMLPRegressor"]
