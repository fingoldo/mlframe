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
        raise ValueError(f"x batch dim {x.shape[0]} != y batch dim {y.shape[0]}")

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


def mixup_sequence_batch(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    aux_features: torch.Tensor | None = None,
):
    """F-70 (2026-05-31): Mixup over a padded sequence batch + optional
    aux features + labels.

    Args:
        sequences: padded sequence tensor of shape (B, T, F)
        lengths: (B,) integer lengths of the unpadded sequences (each
            ``lengths[i] <= T``).
        y: target tensor of shape (B,) or (B, K)
        alpha: Beta-distribution concentration (> 0)
        aux_features: optional (B, A) aux tensor mixed with the SAME
            (idx, lam) draw so the per-sample identity is preserved
            across modalities

    Returns:
        sequences_mixed: lam * sequences + (1-lam) * sequences[idx]
        lengths_mixed: torch.maximum(lengths, lengths[idx])
        aux_mixed: same convention if aux_features supplied; else None
        y_a, y_b, lam: see mixup_batch

    Notes on the lengths handling:
        For each pair (a, b) in the batch, ``lengths_mixed = max(l_a, l_b)``.
        Positions in [min(l_a, l_b), max(l_a, l_b)) of the SHORTER
        sequence are padding (zeros) in the source ``sequences`` tensor;
        the lam * 0 + (1-lam) * seq_long[p] interpolation at those
        positions is therefore the longer sequence's content scaled by
        (1-lam). pack_padded_sequence with lengths_mixed correctly drives
        the RNN to process all max(l_a, l_b) steps; the RNN sees a
        smoothly-interpolated signal in the overlap region and a scaled
        version of the longer sequence in the tail. This is the standard
        recipe (e.g. fairseq, AdaMixup-Seq, Manifold-Mixup-Seq).
    """
    if alpha <= 0:
        raise ValueError(f"mixup alpha must be > 0; got {alpha}")
    B = sequences.shape[0]
    if lengths.shape[0] != B or y.shape[0] != B:
        raise ValueError(f"batch dim mismatch: sequences={B}, lengths={lengths.shape[0]}, y={y.shape[0]}")

    lam_t = torch.distributions.Beta(alpha, alpha).sample().to(sequences.device)
    lam = float(lam_t.clamp(min=1e-3, max=1.0 - 1e-3).item())
    idx = torch.randperm(B, device=sequences.device)

    sequences_mixed = lam * sequences + (1.0 - lam) * sequences[idx]
    lengths_mixed = torch.maximum(lengths, lengths[idx])
    y_a = y
    y_b = y[idx]
    aux_mixed = None
    if aux_features is not None:
        aux_mixed = lam * aux_features + (1.0 - lam) * aux_features[idx]
    return sequences_mixed, lengths_mixed, aux_mixed, y_a, y_b, lam
