"""Differentiable "Schrodinger features" subset search (PyTorch, optional).

A binary feature is relaxed into a continuous gate ``g_j = sigmoid(logit_j) in (0, 1)``: the feature
is partly in / partly out of the coalition. The relaxed proxy margin is
``base[i] + sum_j g_j * phi[i, j]``, which is differentiable in the logits, so Adam can slide
features in/out to minimise a differentiable proxy loss (MSE for regression, BCE on sigmoid(margin)
for classification) plus an L1 sparsity penalty that pushes toward compact subsets.

Because the objective we actually care about is the *discrete* subset loss, we do not trust the final
relaxed gates blindly: along the optimisation trajectory we snapshot the thresholded subset
(``g_j > 0.5``) and score its TRUE (discrete) proxy loss with the numpy objective, collecting those
as candidates. The research found this approach gave only "limited success", so it is an opt-in
backend, not the default. Lazy torch import; a private ``torch.Generator`` keeps RNG isolated.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss, resolve_metric


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def gradient_top_n(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric=None,
    n_iter: int = 400,
    lr: float = 0.1,
    l1: float = 0.01,
    snapshot_every: int = 5,
    random_state: int = 0,
    device: str = "cpu",
    top_n: int = 30,
) -> list[tuple[float, tuple[int, ...]]]:
    import torch

    metric = resolve_metric(classification, metric)
    phi_np = np.ascontiguousarray(phi, dtype=np.float64)
    base_np = np.ascontiguousarray(base, dtype=np.float64)
    y_np = np.ascontiguousarray(y, dtype=np.float64)
    f = phi_np.shape[1]

    gen = torch.Generator(device=device).manual_seed(int(random_state))
    phi_t = torch.as_tensor(phi_np, device=device)
    base_t = torch.as_tensor(base_np, device=device)
    y_t = torch.as_tensor(y_np, device=device)
    logits = (0.1 * torch.randn(f, generator=gen, dtype=torch.float64, device=device)).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)

    candidates: dict[tuple[int, ...], float] = {}

    def record(idx_tuple):
        if not idx_tuple:
            return
        key = tuple(sorted(idx_tuple))
        if key not in candidates:
            candidates[key] = proxy_loss(coalition_margin(phi_np, base_np, list(key)), y_np, metric)

    for it in range(n_iter):
        opt.zero_grad()
        gates = torch.sigmoid(logits)
        margin = base_t + (phi_t * gates).sum(dim=1)
        if classification:
            p = torch.sigmoid(margin)
            data_loss = torch.nn.functional.binary_cross_entropy(p, y_t)
        else:
            data_loss = torch.mean((margin - y_t) ** 2)
        loss = data_loss + l1 * gates.sum()
        loss.backward()
        opt.step()
        if it % snapshot_every == 0:
            with torch.no_grad():
                on = torch.sigmoid(logits) > 0.5
                idx = tuple(int(j) for j in torch.nonzero(on, as_tuple=False).ravel().tolist())
                record(idx)
    # Final snapshot + the top-k gates as a fallback subset.
    with torch.no_grad():
        g = torch.sigmoid(logits).cpu().numpy()
    record(tuple(int(j) for j in np.flatnonzero(g > 0.5)))
    record(tuple(int(j) for j in np.argsort(-g)[: max(1, int((g > 0.5).sum()))]))

    items = [(v, k) for k, v in candidates.items() if np.isfinite(v)]
    items.sort(key=lambda t: t[0])
    return items[:top_n]
