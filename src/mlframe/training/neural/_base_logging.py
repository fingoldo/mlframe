"""Logging filters + scope-local warning suppression + metric helpers for the PyTorch Lightning neural base.

Carved out of ``base.py`` to keep the parent below the 1k-line monolith threshold. Side-effect imports (the ``_LIGHTNING_NOISE_FILTER`` attach loop) run at this module's import time; the parent re-exports the symbols so existing imports keep working unchanged.
"""
from __future__ import annotations

import logging
import warnings as _warnings
from contextlib import contextmanager as _contextmanager
from typing import Callable

from pydantic import BaseModel
from sklearn.metrics import root_mean_squared_error

logger = logging.getLogger(__name__)


# Silence the trio of INFO bullets Lightning emits on every trainer init ("GPU available: True ... TPU/IPU/HPU available: False ...
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]"). With composite-target discovery + multi-model suites these emit 5+ lines per fit,
# drowning the real signal. Useful messages from the same loggers (``Time limit reached``, ``Metric val_MSE improved``, ``Loading
# best model``) are PRESERVED via a substring filter rather than blanket WARNING bump.
class _LightningRankZeroNoiseFilter(logging.Filter):
    """Drop the device-availability bullets that Lightning emits on every trainer init; let everything else through."""

    _PATTERNS = (
        "GPU available",
        "TPU available",
        "IPU available",
        "HPU available",
        "LOCAL_RANK:",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Drop the record (return False) iff its message matches one of the device-availability noise patterns; every other record passes through unchanged."""
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


_LIGHTNING_NOISE_FILTER = _LightningRankZeroNoiseFilter()
for _quiet_logger_name in (
    "lightning.pytorch.utilities.rank_zero",
    "lightning.pytorch.accelerators.cuda",
):
    _quiet_logger = logging.getLogger(_quiet_logger_name)
    # Idempotent attach: skip if already filtered on a prior module import.
    if not any(isinstance(f, _LightningRankZeroNoiseFilter) for f in _quiet_logger.filters):
        _quiet_logger.addFilter(_LIGHTNING_NOISE_FILTER)


# Scoped Lightning DataLoader warning suppressor. Lightning's data_connector emits "does not have many workers which may be a bottleneck" for every train/val/predict DataLoader; the recommendation is wrong for mlframe (num_workers > 0 pickles the full polars/pandas frame into every worker -- catastrophic on 100+ GB frames per the bench at _benchmarks/bench_dataloader_workers.py). Suppress at call sites, NOT at module import time (was poisoning the process-global filter for every importer of this neural base).
@_contextmanager
def suppress_lightning_workers_warning():
    """Scope-local suppression of Lightning's num_workers DataLoader warning.
    Wrap the trainer.fit() / trainer.predict() invocations in this neural base."""
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=r".*does not have many workers which may be a bottleneck.*",
            category=UserWarning,
        )
        yield


def _rmse_metric(y_true, y_score):
    """Wrapper for root_mean_squared_error that accepts y_score parameter name."""
    return root_mean_squared_error(y_true=y_true, y_pred=y_score)


class MetricSpec(BaseModel):
    """Declarative spec for a single training/validation metric: its callable plus flags describing what input shape it expects (argmax labels vs. probabilities) and whether it must run on CPU."""

    name: str
    fcn: Callable  # the metric function
    requires_argmax: bool = False  # True if metric wants predicted class labels
    requires_probs: bool = False  # True if metric wants probabilities (softmax)
    requires_cpu: bool = True  # True if metric should run on CPU (sklearn), False if GPU-compatible
