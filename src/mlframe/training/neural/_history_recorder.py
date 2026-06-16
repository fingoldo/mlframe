"""Lightning callback that records per-epoch train/val metric history in the booster-compatible
``evals_result_`` shape.

The per-model training-curve chart (``training/reporting/_reporting._render_training_curves``,
default-ON via ``ReportingConfig.training_curves``) auto-emits for any fitted model exposing
``evals_result_`` -- lgb/xgb/cb already do. Neural (lightning) estimators expose no such history, so the
training-curve panel (with the early-stop vline + wasted-post-ES shading) was missing from neural reports.
This recorder accumulates ``{split: {metric: [...]}}`` across epochs so the neural estimator can expose
the same accessor and get the same chart for free, no reporting-side change.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class TrainingHistoryRecorder(Callback):
    """Accumulate per-epoch train/val metrics into the booster ``evals_result_`` shape + track best epoch.

    Both splits are recorded at validation-epoch-end from ``trainer.callback_metrics`` (which holds the
    latest train AND val values), so the ``train`` / ``val`` lists stay length-aligned -- one append per
    validated epoch. The Lightning sanity-check val pass (before training) is skipped so it does not inject
    a spurious leading point. ``mode`` follows the monitored metric's direction (``"min"`` for losses).
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min") -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.evals_result_: dict[str, dict[str, list]] = {}
        self.best_iteration_: Optional[int] = None
        self._best = float("inf") if mode == "min" else float("-inf")

    def _record_split(self, metrics) -> None:
        for key, value in metrics.items():
            k = str(key)
            if k.startswith("val_"):
                split, metric = "val", k[len("val_"):]
            elif k.startswith("train_"):
                split, metric = "train", k[len("train_"):]
            else:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            self.evals_result_.setdefault(split, {}).setdefault(metric, []).append(v)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        metrics = dict(trainer.callback_metrics)
        self._record_split(metrics)
        cur = metrics.get(self.monitor)
        if cur is None:
            return
        try:
            cur = float(cur)
        except (TypeError, ValueError):
            return
        better = cur < self._best if self.mode == "min" else cur > self._best
        if better:
            self._best = cur
            self.best_iteration_ = int(getattr(trainer, "current_epoch", 0))


class IterationMetricsRecorder(Callback):
    """Capture the FULL target-type metric suite per validation epoch into ``iteration_metrics_``.

    Accumulates the concatenated val predictions + labels at each validation epoch (same pattern as
    ``AggregatingValidationCallback``) and feeds them to ``mlframe.metrics.compute_all_metrics``. The marginal cost
    is only the (cheap, numba-kernel) metric suite because the predictions are already materialised each epoch --
    hence per-iteration capture is default-ON for neural (unlike boosters, which must re-predict val per round).

    ``iteration_metrics_`` is ``{epoch -> {metric_name -> float}}``; the trainer stamps it onto the estimator after
    fit. The Lightning sanity-check val pass is skipped so it injects no spurious leading point.
    """

    def __init__(self, target_type: str, n_classes: Optional[int] = None) -> None:
        super().__init__()
        self.target_type = str(target_type)
        self.n_classes = n_classes
        self.iteration_metrics_: dict[int, dict[str, float]] = {}
        self._batched_predictions: list = []
        self._batched_labels: list = []

    @staticmethod
    def _extract(outputs):
        """Pull (predictions, labels) tensors from the validation-step output across the flat (dict) and recurrent
        (tuple) module shapes. Returns (None, None) when the shape is unrecognised."""
        if isinstance(outputs, dict):
            return outputs.get("raw_predictions", outputs.get("predictions")), outputs.get("labels")
        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            return outputs[0], outputs[1]
        return None, None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        predictions, labels = self._extract(outputs)
        if predictions is None or labels is None:
            return
        self._batched_predictions.append(predictions)
        self._batched_labels.append(labels)

    def _logits_to_score(self, logits: np.ndarray) -> np.ndarray:
        """Map raw network outputs to probabilities for classification, leaving regression outputs untouched.

        Classification validation outputs are LOGITS; the metric suite expects [0,1] probabilities. Apply softmax
        for a (N, K>=2) head and sigmoid for a (N,) / (N, 1) binary head; regression passes through unchanged.
        """
        if "regression" in self.target_type:
            return logits
        a = np.asarray(logits, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] >= 2:
            a = a - a.max(axis=1, keepdims=True)
            e = np.exp(a)
            return e / e.sum(axis=1, keepdims=True)
        flat = a[:, 0] if a.ndim == 2 else a
        return 1.0 / (1.0 + np.exp(-flat))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        preds, labels = self._batched_predictions, self._batched_labels
        self._batched_predictions, self._batched_labels = [], []
        if getattr(trainer, "sanity_checking", False) or not preds:
            return
        from mlframe.metrics import compute_all_metrics

        try:
            logits = torch.concat(preds).detach().cpu().float().numpy()
            y_true = torch.concat(labels).detach().cpu().numpy()
            y_score = self._logits_to_score(logits)
            epoch = int(getattr(trainer, "current_epoch", 0))
            self.iteration_metrics_[epoch] = compute_all_metrics(
                y_true, y_score, target_type=self.target_type, n_classes=self.n_classes
            )
        except Exception as exc:  # never abort training on a metric-capture failure
            logger.warning("neural iteration-metrics capture failed: %s", exc, exc_info=False)
