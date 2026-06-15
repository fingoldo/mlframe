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
