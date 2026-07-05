"""Loss + metric mixin carved out of ``_flat_torch_module.MLPTorchModel``."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..base import to_numpy_safe

logger = logging.getLogger("mlframe.training.neural.flat")


class _LossMixin:
    """Batch unpacking, sample-weighted loss, and metric computation for ``MLPTorchModel``."""

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Unpack batch into features, labels, and optional sample_weight."""
        if isinstance(batch, dict):
            return batch["features"], batch["labels"], batch.get("sample_weight")
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                return batch[0], batch[1], None
        raise TypeError(f"Unexpected batch format: {type(batch).__name__}")

    def _loss_unreduced(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample loss via self.loss_fn with reduction='none' when available.

        Most torch.nn loss modules expose a ``reduction`` attribute; toggling it
        round-trip preserves the loss type (BCE stays BCE, MSE stays MSE) so
        sample weighting doesn't silently swap loss semantics.
        """
        # Module loss (CrossEntropyLoss / BCEWithLogitsLoss / MSELoss / ...):
        # set reduction='none', call, restore.
        if hasattr(self.loss_fn, "reduction"):
            prev = self.loss_fn.reduction
            try:
                self.loss_fn.reduction = "none"
                return self.loss_fn(predictions, labels)
            finally:
                self.loss_fn.reduction = prev
        # Functional / lambda loss: try kwarg, fall back to CE/MSE shape-guess.
        try:
            return self.loss_fn(predictions, labels, reduction="none")
        except TypeError:
            if predictions.dim() == 2 and predictions.shape[1] > 1:
                return F.cross_entropy(predictions, labels, reduction="none")
            return F.mse_loss(predictions, labels, reduction="none")

    def _compute_weighted_loss(self, predictions: torch.Tensor, labels: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute loss with optional sample weighting.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            sample_weight: Optional per-sample weights

        Returns:
            Scalar loss tensor
        """
        # Defensive shape alignment: torch's MSELoss / L1Loss / SmoothL1Loss
        # emit a UserWarning ("Using a target size (torch.Size([N])) that is
        # different to the input size (torch.Size([N, 1]))") AND apply
        # incorrect broadcasting when predictions and labels disagree on a
        # singleton trailing dim. training_step + validation_step squeeze
        # raw_predictions in the common (N, 1) case, but other call paths
        # (test fixtures, ranker, custom loss_fn) can still enter here
        # mismatched. Squeeze the singleton dim on either side so the loss
        # function never sees a broadcasting collision.
        if predictions.dim() == 2 and predictions.shape[-1] == 1 and labels.dim() == 1:
            predictions = predictions.squeeze(-1)
        elif labels.dim() == 2 and labels.shape[-1] == 1 and predictions.dim() == 1:
            labels = labels.squeeze(-1)

        if sample_weight is None:
            return self.loss_fn(predictions, labels)

        # Honour self.loss_fn if it supports reduction='none' (every
        # torch.nn.*Loss does); the previous hard-coded CE/MSE branch silently
        # switched a BCE binary classifier (with sample_weight) to MSE,
        # producing a degenerate gradient.
        loss_unreduced = self._loss_unreduced(predictions, labels)

        # torch.dot fuses mul+sum into one kernel; ~1.7-2.2x faster than
        # (a*b).sum() across N=256-16384 on CPU. 1-D fast path covers the
        # common case (per-sample weights, scalar-per-sample loss); fall
        # back to broadcast-mul for any unexpected shape.
        weight_sum = sample_weight.sum()
        if loss_unreduced.dim() == 1 and sample_weight.dim() == 1 and loss_unreduced.shape == sample_weight.shape:
            raw = torch.dot(loss_unreduced, sample_weight)
        else:
            # Multilabel BCE / multiclass per-sample losses produce a 2-D
            # (B, K) loss tensor while sample_weight is 1-D (B,); torch
            # broadcasts (B,) as (1, B) which collides with the (B, K)
            # right-aligned shape (B vs K mismatch). Reshape 1-D weights to
            # (B, 1) so they broadcast across labels / classes uniformly.
            # Fuzz c0052 (multilabel + MLP + uniform weights) crashed here
            # before the unsqueeze: "tensor a (3) must match tensor b (30928)".
            if loss_unreduced.dim() == 2 and sample_weight.dim() == 1 and sample_weight.shape[0] == loss_unreduced.shape[0]:
                sample_weight_b = sample_weight.unsqueeze(1)
            else:
                sample_weight_b = sample_weight
            raw = (loss_unreduced * sample_weight_b).sum()
        # Safe divide avoids the prior ``if weight_sum > 0`` Python branch (a
        # forced GPU->CPU sync ~30 us per call) without losing the all-zero-
        # weight semantic: when weight_sum is 0, raw is also 0 (sum(loss * 0)),
        # so 0 / clamp(0, 1e-12) = 0 -- the same zero loss the legacy branch
        # returned. Bench at n=50k (CUDA): 299 us -> 267 us per call (~10%),
        # bit-identical for both non-degenerate and all-zero-weight cases.
        weighted_loss = raw / torch.clamp(weight_sum, min=1e-12)
        # Emit a ONCE-per-LightningModule WARN on the first all-zero-weight
        # batch. Pre-fix the safe-divide path silently returned 0 loss + 0
        # gradient with no log message -- the model trained on nothing and the
        # operator saw a flat val curve with no clue why. The check costs one
        # GPU->CPU sync (weight_sum.item()) per training_step UNTIL the warning
        # fires, after which the flag short-circuits everything. The sync is
        # only paid when sample_weight is not None (opt-in path).
        if not getattr(self, "_warned_zero_weight_batch", False):
            if float(weight_sum.detach().item()) < 1e-12:
                logger.warning(
                    "All-zero sample_weight batch in MLP training_step "
                    "(batch_idx=%d): loss=0, gradient=0, model receives "
                    "no learning signal from this step. Suppressing "
                    "subsequent warnings for this LightningModule. If "
                    "the entire fit has zero-weight batches the model "
                    "will not learn -- check the sample_weight pipeline.",
                    getattr(self, "_current_batch_idx", -1),
                )
                self._warned_zero_weight_batch = True
        return weighted_loss

    def compute_metrics(self, predictions_and_labels: List[Tuple[torch.Tensor, torch.Tensor]], prefix: str = "val") -> None:
        """
        Compute and log all metrics given raw predictions and labels.
        Optimized: compute argmax, softmax, CPU/numpy only if needed, once each.

        Args:
            predictions_and_labels: List of (raw_predictions, labels) tuples from each batch (on CPU)
            prefix: Logging prefix ('train' or 'val')
        """
        raw_predictions, labels = zip(*predictions_and_labels)
        raw_predictions = torch.cat(raw_predictions)
        labels = torch.cat(labels)

        need_argmax = any(m.requires_argmax for m in self.metrics)
        need_softmax = any(m.requires_probs for m in self.metrics)
        need_cpu = any(m.requires_cpu for m in self.metrics)

        # argmax along the class dim only makes sense for multi-class K>1 logits (shape (N, K)).
        # Regression (dim==1) or multilabel BCE (each label independent) would silently emit
        # garbage from raw_predictions.argmax(dim=1); guard explicitly.
        _is_multiclass = raw_predictions.dim() == 2 and raw_predictions.shape[1] > 1 and self.task_type != "multilabel"

        preds_dict = {}
        if need_argmax:
            if _is_multiclass:
                preds_dict["argmax"] = raw_predictions.argmax(dim=1)
            elif self.task_type == "multilabel":
                # Multilabel: per-label thresholded predictions at 0.5 (each output independent binary).
                preds_dict["argmax"] = (torch.sigmoid(raw_predictions) >= 0.5).long()
            elif self.task_type == "binary":
                # Binary single-output: threshold sigmoid at 0.5 -> {0, 1}.
                preds_dict["argmax"] = (torch.sigmoid(raw_predictions).squeeze(-1) >= 0.5).long()
            else:
                # Regression: argmax has no meaning; skip metric below.
                preds_dict["argmax"] = None
        if need_softmax:
            if _is_multiclass:
                preds_dict["softmax"] = F.softmax(raw_predictions, dim=1)
            elif self.task_type == "multilabel":
                preds_dict["softmax"] = torch.sigmoid(raw_predictions)
            elif self.task_type == "binary":
                # Binary single-output: convert (N, 1) sigmoid -> (N, 2)
                # column-stacked [P(y=0), P(y=1)] so the same probabilistic
                # metric (ICE, etc.) the multiclass path uses keeps working.
                p1 = torch.sigmoid(raw_predictions).reshape(-1)
                p0 = 1.0 - p1
                preds_dict["softmax"] = torch.stack([p0, p1], dim=1)
            else:
                preds_dict["softmax"] = raw_predictions

        labels_cpu = None
        if need_cpu:
            # cpu=False: tensors are already on CPU thanks to the per-step .cpu() in *_step.
            labels_cpu = to_numpy_safe(labels, cpu=False)

        # CPU-numpy memoisation keyed by tagged source (argmax / softmax / raw) NOT id(preds).
        # id() values can be reused after garbage collection across loop iterations and silently
        # return the wrong tensor's cached numpy view.
        cpu_cache: dict[str, np.ndarray] = {}

        for metric in self.metrics:
            if metric.requires_argmax:
                preds = preds_dict["argmax"]
                preds_tag = "argmax"
            elif metric.requires_probs:
                preds = preds_dict["softmax"]
                preds_tag = "softmax"
            else:
                preds = raw_predictions
                preds_tag = "raw"

            if preds is None:
                # argmax requested but logits aren't multi-class; skip silently to avoid garbage.
                continue

            if metric.requires_cpu:
                if preds_tag not in cpu_cache:
                    cpu_cache[preds_tag] = to_numpy_safe(preds, cpu=False)
                preds_np = cpu_cache[preds_tag]
                labels_np = labels_cpu
            else:
                preds_np = preds
                labels_np = labels

            try:
                value = metric.fcn(y_true=labels_np, y_score=preds_np)
                self.log(
                    f"{prefix}_{metric.name}",
                    value,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            except Exception:
                logger.exception("Failed to compute metric %s_%s", prefix, metric.name)
