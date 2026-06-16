"""PyTorch Lightning callbacks for the mlframe neural base.

Carved out of ``base.py`` to keep the parent below the 1k-line monolith threshold. The parent re-exports these classes so existing imports keep working unchanged; class identity preserved bit-for-bit.
"""
from __future__ import annotations

import logging
import operator  # picklable comparison functions (needed by ddp_spawn strategy)
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateFinder,
    ModelCheckpoint,
)
from pyutilz.pythonlib import get_parent_func_args, store_params_in_object


logger = logging.getLogger(__name__)


class NetworkGraphLoggingCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        pl_module.logger.log_graph(model=pl_module)


class AggregatingValidationCallback(Callback):

    def __init__(self, metric_name: str, metric_fcn: object, on_epoch: bool = True, on_step: bool = False, prog_bar: bool = True):
        # Forward to Lightning's Callback base so any future state it sets in __init__ is populated (currently a no-op).
        super().__init__()
        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)
        self.init_accumulators()

    def init_accumulators(self):
        self.batched_predictions = []
        self.batched_labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        predictions, labels = outputs
        self.batched_labels.append(labels)
        self.batched_predictions.append(predictions)

    def on_validation_epoch_end(self, trainer, pl_module):
        labels = torch.concat(self.batched_labels).detach().cpu().numpy()
        predictions = torch.concat(self.batched_predictions).detach().cpu().float().numpy()
        metric_value = self.metric_fcn(y_true=labels, y_score=predictions)
        pl_module.log(name="val_" + self.metric_name, value=metric_value, on_epoch=self.on_epoch, on_step=self.on_step, prog_bar=True)
        self.init_accumulators()


class ValLossDivergenceCallback(Callback):
    """Detect catastrophic val-loss divergence DURING training.

    Identity-MLP-style collapses on group-aware test splits often show
    up as val_loss growing wildly within the first few epochs while
    train_loss decreases (model fits train, extrapolates badly on val
    because val groups differ from train). The post-predict
    ``regression-collapse-sensor`` catches this AFTER fit, but operators
    pay the full training budget first.

    This callback WARNs at val-epoch-end when ``current_val / initial_val
    >= divergence_factor`` (default 100x). The warning includes the
    epoch number and current value, giving operators a chance to abort
    OR signal that the model is unrecoverable so the checkpoint loader
    later doesn't surprise them with a near-dummy fit.

    No automatic stop -- early-stopping already handles the
    "no-improvement" case; this callback addresses the "actively
    getting worse by orders of magnitude" case.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        divergence_factor: float = 100.0,
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._divergence_factor = float(divergence_factor)
        self._initial_value: Optional[float] = None
        self._warned: bool = False

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._warned:
            return
        metrics = trainer.callback_metrics
        if self._monitor not in metrics:
            return
        try:
            current = float(metrics[self._monitor])
        except (TypeError, ValueError):
            return
        if not (current == current) or current <= 0:  # NaN / non-positive guard
            return
        if self._initial_value is None:
            # Skip epoch 0's metric -- often noisy before warm-up converges. Latch on epoch 1's value as the baseline.
            if trainer.current_epoch <= 0:
                return
            self._initial_value = current
            return
        if current >= self._initial_value * self._divergence_factor:
            logger.warning(
                "[mlp-val-divergence] epoch %d: %s=%.4g is %.1fx the "
                "initial baseline %.4g (>%.0fx threshold). The model is "
                "actively diverging on validation -- expect the loaded "
                "best checkpoint to be a near-dummy / collapsed fit "
                "(observed in prod: Identity-MLP on group-OOD: "
                "val_loss climbed to ~1.1 from initial ~0.04 in <20 epochs "
                "before stopping at a noise-floor checkpoint). Pick a "
                "real nonlinearity, drop ``use_layernorm`` on already-"
                "z-scored inputs, or let composite-target discovery "
                "produce a bounded-residual target.",
                trainer.current_epoch, self._monitor,
                current, current / self._initial_value,
                self._initial_value, self._divergence_factor,
            )
            self._warned = True


class MonotonicDeclineStopCallback(Callback):
    """Monotonic strict-decline overfitting stop for Lightning, COMPLEMENTARY to ``EarlyStopping``.

    At every validation-epoch-end it feeds ``trainer.callback_metrics[monitor]`` to the shared
    ``MonotonicDeclineStopper``: once the monitored val metric has STRICTLY worsened for
    ``patience`` consecutive epochs since the global best, it sets ``trainer.should_stop = True``.
    A new global best, a plateau, or a bounce-up resets the streak. This is the confident-
    overfitting signal that fires faster than (and alongside) the patience-based ``EarlyStopping``
    -- whichever stop fires first wins. ``BestEpochModelCheckpoint`` still restores the global-best
    epoch's weights, so an early monotonic stop keeps the right model.

    ``mode`` must match the monitored metric's direction (``"min"`` for val_loss / val_RMSE / val_ICE,
    ``"max"`` for AUC-style). ``patience=None`` disables the callback.
    """

    def __init__(self, monitor: str = "val_loss", patience: Optional[int] = 7, mode: str = "min") -> None:
        super().__init__()
        # Lazy import keeps the estimators package free of any lightning/torch coupling.
        from mlframe.estimators.early_stopping_monotonic import MonotonicDeclineStopper

        self._monitor = monitor
        self._stopper = MonotonicDeclineStopper(patience, mode=mode)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._stopper.enabled:
            return
        metrics = trainer.callback_metrics
        if self._monitor not in metrics:
            return
        try:
            current = float(metrics[self._monitor])
        except (TypeError, ValueError):
            return
        if self._stopper.update(current):
            logger.info(
                "[monotonic-decline] stopping at epoch %d: %s strictly worsened for %d "
                "consecutive epochs since the best (confident overfitting).",
                trainer.current_epoch, self._monitor, self._stopper.streak,
            )
            trainer.should_stop = True


class BestEpochModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint that tracks the epoch of the best model according to the monitored metric.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", **kwargs):
        super().__init__(monitor=monitor, mode=mode, **kwargs)
        self.best_epoch: Optional[int] = None
        self.best_score: Optional[float] = None

        # operator module used for pickling support
        if mode == "min":
            self.monitor_op = operator.lt
            self.best_score = float("inf")
        elif mode == "max":
            self.monitor_op = operator.gt
            self.best_score = float("-inf")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logger.info("Initialized BestEpochModelCheckpoint with monitor=%s, mode=%s", monitor, mode)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Update best_epoch after each validation step if current metric improves.
        """
        super().on_validation_end(trainer, pl_module)

        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"Monitor metric '{self.monitor}' not found in callback_metrics.")
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch
            # Also set on pl_module for DDP synchronization
            pl_module.best_epoch = self.best_epoch
            # DEBUG (was INFO): per-epoch best-model bump produces O(epochs) log spam (~20-40 lines per MLP fit) without surfacing any actionable signal -- the final best_epoch lands in the model and the "Loaded weights from epoch N" line at end-of-fit reports it.
            logger.debug("New best model at epoch %s with %s=%.4f", self.best_epoch, self.monitor, self.best_score)


class PeriodicLearningRateFinder(LearningRateFinder):
    def __init__(self, period: int, *args, **kwargs):
        if not isinstance(period, int) or isinstance(period, bool):
            raise TypeError(f"period must be an int, got {type(period).__name__}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period!r}")
        super().__init__(*args, **kwargs)
        self.period = period

    def on_train_epoch_start(self, trainer, pl_module):
        if (trainer.current_epoch % self.period) == 0 or trainer.current_epoch == 0:
            print(f"Finding optimal learning rate. Current rate={pl_module.learning_rate}")
            self.lr_find(trainer, pl_module)
            print(f"Set learning rate to {pl_module.learning_rate}")
