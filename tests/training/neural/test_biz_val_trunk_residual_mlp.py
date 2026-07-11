"""biz_value test for ``training.neural.trunk_residual_mlp.TrunkResidualMLPRegressor``.

Source: 3rd_jane-street-market-prediction.md -- a 49-layer MLP built from a trunk block whose output is
skip-connected into every subsequent residual block, rather than only the standard adjacent-layer ResNet skip
mlframe's existing `_ResidualLinearBlock` provides. At real depth (20 blocks here, close to the source's own
23), a deep tower with NO skip connections at all collapses (vanishing/degenerate gradient flow) -- direct
trunk re-injection at every block keeps the original low-level representation available at full strength no
matter how deep the tower gets, recovering a target a no-skip deep MLP of the same depth cannot learn.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn

from mlframe.training.neural.trunk_residual_mlp import TrunkResidualMLPRegressor


def _make_data(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).astype(np.float32) + rng.normal(scale=0.3, size=n).astype(np.float32)
    return X, y


class _NoSkipDeepMLP(nn.Module):
    def __init__(self, n_features: int, width: int = 12, n_blocks: int = 15) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, width), nn.LayerNorm(width), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU()) for _ in range(n_blocks)])
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)  # NO skip connection at all -- the naive deep-MLP default.
        out: torch.Tensor = self.head(h).squeeze(-1)
        return out


def test_biz_val_trunk_residual_mlp_beats_no_skip_deep_mlp_at_depth():
    X, y = _make_data(n=300, n_features=10, seed=1)
    Xtr, Xte, ytr, yte = X[:220], X[220:], y[:220], y[220:]

    trunk_model = TrunkResidualMLPRegressor(trunk_dim=12, n_blocks=15, n_epochs=200, learning_rate=0.015, random_state=1).fit(Xtr, ytr)
    r2_trunk = float(r2_score(yte, trunk_model.predict(Xte)))

    torch.manual_seed(1)
    plain = _NoSkipDeepMLP(n_features=X.shape[1])
    optimizer = torch.optim.Adam(plain.parameters(), lr=0.015)
    X_t, y_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    plain.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(plain(X_t), y_t)
        loss.backward()
        optimizer.step()
    plain.eval()
    with torch.no_grad():
        pred_plain = plain(torch.from_numpy(Xte)).numpy()
    r2_plain = float(r2_score(yte, pred_plain))

    assert r2_trunk >= 0.5, f"expected the trunk-residual MLP to learn a strong held-out R2 at 15 blocks deep, got {r2_trunk:.4f}"
    assert r2_trunk > r2_plain + 0.3, f"expected the trunk-residual MLP to massively beat a same-depth no-skip MLP (which should collapse), got trunk={r2_trunk:.4f} plain={r2_plain:.4f}"


def test_trunk_residual_mlp_predict_shape():
    X, y = _make_data(n=100, n_features=5, seed=0)
    model = TrunkResidualMLPRegressor(trunk_dim=8, n_blocks=3, n_epochs=20, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
