"""biz_value test for ``training.neural.tabular_1dcnn.Tabular1DCNNRegressor``.

Source: 2nd_mechanisms-of-action-moa-prediction.md -- reshapes tabular features along a meaningful axis and
applies a 1D convolution to capture LOCAL higher-order interactions a plain MLP treats as unstructured. On a
synthetic with "pathway" groups of correlated features where the target depends on a LOCAL pairwise product
within each group, correlation-ordered 1D-CNN should out-perform a comparably-sized plain MLP at the same
training budget, since the CNN's local kernel forces it to learn compact reusable local patterns instead of
an unstructured full-pairwise weight matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import r2_score, roc_auc_score
from torch import nn

from mlframe.training.neural.tabular_1dcnn import Tabular1DCNNClassifier, Tabular1DCNNRegressor, correlation_order_features


def _make_pathway_local_interaction_data(n: int, n_pathways: int, pathway_size: int, seed: int):
    """Make pathway local interaction data."""
    rng = np.random.default_rng(seed)
    n_features = n_pathways * pathway_size
    X = np.zeros((n, n_features), dtype=np.float32)
    target = np.zeros(n, dtype=np.float32)
    for p in range(n_pathways):
        latent = rng.normal(size=n)
        for k in range(pathway_size):
            X[:, p * pathway_size + k] = latent + rng.normal(scale=0.3, size=n)
        target += X[:, p * pathway_size] * X[:, p * pathway_size + 1]  # local interaction within the pathway.
    y = target + rng.normal(scale=0.5, size=n)

    perm = rng.permutation(n_features)  # shuffle raw column order -- ordering must be RECOVERED, not given.
    return X[:, perm].astype(np.float32), y.astype(np.float32)


class _MLPBaseline(nn.Module):
    """Groups tests covering m l p baseline."""
    def __init__(self, n_features: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


def test_biz_val_1dcnn_beats_mlp_on_local_pathway_interactions():
    """Biz val 1dcnn beats mlp on local pathway interactions."""
    X, y = _make_pathway_local_interaction_data(n=300, n_pathways=6, pathway_size=4, seed=0)
    Xtr, Xte, ytr, yte = X[:200], X[200:], y[:200], y[200:]

    cnn = Tabular1DCNNRegressor(n_channels=8, kernel_size=3, n_epochs=200, learning_rate=0.02, random_state=0).fit(Xtr, ytr)
    r2_cnn = float(r2_score(yte, cnn.predict(Xte)))

    torch.manual_seed(0)
    mlp = _MLPBaseline(n_features=X.shape[1])
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.02)
    X_t, y_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    mlp.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(mlp(X_t), y_t)
        loss.backward()
        optimizer.step()
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(torch.from_numpy(Xte)).numpy()
    r2_mlp = float(r2_score(yte, pred_mlp))

    assert r2_cnn >= 0.85, f"expected the correlation-ordered 1D-CNN to recover most of the local-interaction signal, got r2={r2_cnn:.4f}"
    assert r2_cnn > r2_mlp + 0.08, f"expected the 1D-CNN to beat a comparably-sized MLP at the same training budget, got cnn={r2_cnn:.4f} mlp={r2_mlp:.4f}"


def _make_pathway_local_interaction_labels(n: int, n_pathways: int, pathway_size: int, seed: int):
    """Same "pathway" structure as the regression fixture, but the target is the SIGN of the local
    within-pathway product summed across pathways, turned into a binary label -- a plain MLP still has
    to discover the same local structure without the 1D-CNN's forced local kernel window.
    """
    rng = np.random.default_rng(seed)
    n_features = n_pathways * pathway_size
    X = np.zeros((n, n_features), dtype=np.float32)
    score = np.zeros(n, dtype=np.float32)
    for p in range(n_pathways):
        latent = rng.normal(size=n)
        for k in range(pathway_size):
            X[:, p * pathway_size + k] = latent + rng.normal(scale=0.3, size=n)
        score += X[:, p * pathway_size] * X[:, p * pathway_size + 1]
    noisy_score = score + rng.normal(scale=0.5, size=n)
    y = (noisy_score > np.median(noisy_score)).astype(np.int64)  # median split -> exactly balanced classes.

    perm = rng.permutation(n_features)  # shuffle raw column order -- ordering must be RECOVERED, not given.
    row_perm = rng.permutation(n)  # shuffle rows too, so a train/test slice isn't accidentally class-skewed.
    X, y = X[:, perm], y[row_perm]
    X = X[row_perm]
    return X.astype(np.float32), y


class _MLPClassifierBaseline(nn.Module):
    """Groups tests covering m l p classifier baseline."""
    def __init__(self, n_features: int, n_classes: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out: torch.Tensor = self.net(x)
        return out


def test_biz_val_1dcnn_classifier_beats_mlp_on_local_pathway_interactions():
    """Biz val 1dcnn classifier beats mlp on local pathway interactions."""
    X, y = _make_pathway_local_interaction_labels(n=400, n_pathways=6, pathway_size=4, seed=0)
    Xtr, Xte, ytr, yte = X[:280], X[280:], y[:280], y[280:]

    clf = Tabular1DCNNClassifier(n_channels=8, kernel_size=3, n_epochs=200, learning_rate=0.02, random_state=0).fit(Xtr, ytr)
    proba_cnn = clf.predict_proba(Xte)[:, 1]
    auc_cnn = float(roc_auc_score(yte, proba_cnn))

    torch.manual_seed(0)
    mlp = _MLPClassifierBaseline(n_features=X.shape[1], n_classes=2)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.02)
    X_t, y_t = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    mlp.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(mlp(X_t), y_t)
        loss.backward()
        optimizer.step()
    mlp.eval()
    with torch.no_grad():
        proba_mlp = torch.softmax(mlp(torch.from_numpy(Xte)), dim=-1)[:, 1].numpy()
    auc_mlp = float(roc_auc_score(yte, proba_mlp))

    naive_auc = 0.5  # majority/coin-flip baseline for a balanced binary target.
    assert auc_cnn >= 0.80, f"expected the correlation-ordered 1D-CNN classifier to recover most of the local-interaction signal, got auc={auc_cnn:.4f}"
    assert auc_cnn > naive_auc + 0.25, f"expected the classifier to clearly beat the naive coin-flip baseline, got auc={auc_cnn:.4f}"
    assert (
        auc_cnn > auc_mlp + 0.05
    ), f"expected the 1D-CNN classifier to beat a comparably-sized MLP at the same training budget, got cnn={auc_cnn:.4f} mlp={auc_mlp:.4f}"


def test_biz_val_1dcnn_classifier_predict_matches_argmax_proba():
    """Biz val 1dcnn classifier predict matches argmax proba."""
    X, y = _make_pathway_local_interaction_labels(n=120, n_pathways=4, pathway_size=4, seed=2)
    clf = Tabular1DCNNClassifier(n_channels=8, kernel_size=3, n_epochs=50, learning_rate=0.02, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    preds = clf.predict(X)
    expected = clf.classes_[np.argmax(proba, axis=-1)]
    assert np.array_equal(preds, expected)


def test_correlation_order_features_groups_correlated_columns_adjacently():
    """Correlation order features groups correlated columns adjacently."""
    rng = np.random.default_rng(1)
    n = 200
    latent_a = rng.normal(size=n)
    latent_b = rng.normal(size=n)
    X = np.column_stack(
        [
            latent_a + rng.normal(scale=0.1, size=n),
            latent_b + rng.normal(scale=0.1, size=n),
            latent_a + rng.normal(scale=0.1, size=n),
            latent_b + rng.normal(scale=0.1, size=n),
        ]
    ).astype(np.float32)

    order = correlation_order_features(X)
    # columns 0 and 2 (both latent_a) must end up adjacent in the returned order, likewise 1 and 3 (latent_b).
    pos = {col: i for i, col in enumerate(order)}
    assert abs(pos[0] - pos[2]) == 1
    assert abs(pos[1] - pos[3]) == 1
