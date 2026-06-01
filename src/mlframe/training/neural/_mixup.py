"""F-68 (2026-05-31): Mixup augmentation for tabular MLP training.

Mixup (Zhang et al. 2018, https://arxiv.org/abs/1710.09412) creates
synthetic training samples by linearly interpolating pairs of inputs
AND their targets. For each training batch:

    lam ~ Beta(alpha, alpha)
    idx = permute(batch_size)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_mixed = lam * y + (1 - lam) * y[idx]     (regression / soft labels)
    OR    loss = lam * CE(pred, y) + (1 - lam) * CE(pred, y[idx])   (classification)

The Beta distribution with alpha < 1 concentrates mass near 0 and 1,
giving a mild interpolation; alpha=0.2-0.4 is the published tabular
sweet spot (RealMLP-TD 2024 ablation: +0.6% regression / +0.3%
classification on Yandex tabular benchmark).

Why it works on tabular:
  * Acts as a soft label-smoothing + input-jitter regulariser
  * Encourages linear behaviour between training examples
  * Free wins on data-limited regimes (n < 10k); diminishing on n > 1M

Why off by default in mlframe:
  * Mixup HURTS on very-low-noise regimes where the interpolated
    targets become inconsistent with the underlying function (e.g.
    pure linear regression with eps=0). Users should A/B before
    enabling. RealMLP-TD lists it as "task-dependent".
  * Composes oddly with class-weight loss (the per-sample weight tied
    to y_a is applied to the mixed batch); needs care.
  * 2x batch memory for the permuted view (avoidable via in-place +
    careful indexing) but the obvious implementation is cleaner.

Wiring:
  * MLPTorchModel reads ``use_mixup`` / ``mixup_alpha`` (defaults
    False / 0.2). When use_mixup=True, training_step mixes the batch
    BEFORE the forward and computes the appropriate mixup loss
    according to ``task_type``.
  * Mixup applies ONLY to the training_step. Validation, test, and
    predict use unmodified inputs (matches torchvision Mixup
    convention).

Sample-weight semantics under mixup:
  * Per-sample weights are NOT mixed; the weight tied to y_a is used
    for the combined-loss numerator. This preserves user intent
    (rare-class-upweighting still applies to the dominant-sample
    half of each pair) but isn't quite the unbiased estimator. Users
    who need strict unbiased weighted-mixup loss can supply a custom
    loss_fn that handles it.
"""
from __future__ import annotations

from typing import Tuple

import torch


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    sample_weight: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute a Mixup-perturbed batch.

    Args:
        x: Feature tensor of shape (N, ...).
        y: Target tensor of shape (N,) or (N, K).
        alpha: Beta-distribution concentration parameter. Higher alpha
            means stronger mixing. Common range: 0.1-0.4 for tabular.
            Must be > 0.
        sample_weight: Optional (N,) per-sample weights. Returned as-is
            (the weight tied to the original index is preserved; see
            module docstring for the semantic).

    Returns:
        x_mixed: lam * x + (1-lam) * x[idx]
        y_a: original y (same as input)
        y_b: y[idx] (permuted)
        lam: scalar in (0, 1)

    Notes:
        The caller computes the appropriate task-specific loss outside
        this function. For regression / multilabel:
            loss = loss_fn(pred, lam * y_a + (1 - lam) * y_b)
        For classification (CE / NLL on class indices):
            loss = lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)
    """
    if alpha <= 0:
        raise ValueError(f"mixup alpha must be > 0; got {alpha}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x batch dim {x.shape[0]} != y batch dim {y.shape[0]}"
        )

    # Sample lam ~ Beta(alpha, alpha) on the same device for zero CPU sync.
    # torch.distributions.Beta is OK but allocates a Distribution object
    # per call; the raw .beta_ kernel is cheaper. Fallback to torch.empty
    # + uniform-rejection isn't worth the complexity here.
    lam_t = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    # Clamp to [eps, 1-eps] to avoid degenerate "no mix at all" / "fully
    # swapped" which would not contribute regularisation but would
    # introduce sample-weight bias.
    lam = float(lam_t.clamp(min=1e-3, max=1.0 - 1e-3).item())

    idx = torch.randperm(x.shape[0], device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[idx]
    y_a = y
    y_b = y[idx]
    return x_mixed, y_a, y_b, lam
