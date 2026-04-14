"""Stop-file driven training callbacks for major ML libraries.

The `stop_file(fpath)` factory returns a zero-arg predicate that becomes True
as soon as a file at `fpath` exists on disk — a cheap cooperative-shutdown
mechanism used across long-running training jobs.

Each framework-specific callback is a thin shim that checks this predicate on
every iteration and triggers that framework's early-stopping hook.

Library imports are guarded: the module imports cleanly even when the target
library is not installed; instantiating the corresponding class will then
raise ImportError with a helpful message.
"""

from __future__ import annotations

import os
from typing import Callable


def stop_file(fpath: str) -> Callable[[], bool]:
    """Return a predicate that returns True iff a file at `fpath` exists."""
    return lambda: os.path.exists(fpath)


# ----------------------------------------------------------------------------------------------------------------------------
# CatBoost
# ----------------------------------------------------------------------------------------------------------------------------

try:
    import catboost  # noqa: F401

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


if _HAS_CATBOOST:

    class CatBoostStopFileCallback:
        """CatBoost callback that stops training when a stop-file appears on disk."""

        def __init__(self, fpath: str):
            self.fpath = fpath
            self._check = stop_file(fpath)

        def after_iteration(self, info):  # pragma: no cover — exercised in smoke test
            # CatBoost convention: return False to stop training.
            return not self._check()

else:  # pragma: no cover

    class CatBoostStopFileCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError("catboost is not installed; cannot use CatBoostStopFileCallback")


# ----------------------------------------------------------------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------------------------------------------------------------

try:
    import lightgbm  # noqa: F401

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


if _HAS_LIGHTGBM:

    class LightGBMStopFileCallback:
        """LightGBM callback that raises EarlyStopException when a stop-file appears."""

        def __init__(self, fpath: str):
            self.fpath = fpath
            self._check = stop_file(fpath)
            # LightGBM expects callbacks to expose this attribute.
            self.order = 20

        def __call__(self, env):  # pragma: no cover
            if self._check():
                import lightgbm as _lgb

                raise _lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

else:  # pragma: no cover

    class LightGBMStopFileCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError("lightgbm is not installed; cannot use LightGBMStopFileCallback")


# ----------------------------------------------------------------------------------------------------------------------------
# XGBoost
# ----------------------------------------------------------------------------------------------------------------------------

try:
    import xgboost as _xgb  # noqa: F401

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


if _HAS_XGBOOST:
    import xgboost as _xgb_mod

    class XGBoostStopFileCallback(_xgb_mod.callback.TrainingCallback):
        """XGBoost callback: returns True from after_iteration when a stop-file appears."""

        def __init__(self, fpath: str):
            super().__init__()
            self.fpath = fpath
            self._check = stop_file(fpath)

        def after_iteration(self, model, epoch, evals_log):  # pragma: no cover
            # XGBoost convention: return True to stop training.
            return bool(self._check())

else:  # pragma: no cover

    class XGBoostStopFileCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError("xgboost is not installed; cannot use XGBoostStopFileCallback")


# ----------------------------------------------------------------------------------------------------------------------------
# PyTorch Lightning
# ----------------------------------------------------------------------------------------------------------------------------

try:
    import pytorch_lightning as _pl  # noqa: F401

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False


if _HAS_LIGHTNING:
    import pytorch_lightning as _pl_mod

    class LightningStopFileCallback(_pl_mod.callbacks.Callback):
        """PyTorch-Lightning callback that sets trainer.should_stop when a stop-file appears."""

        def __init__(self, fpath: str):
            super().__init__()
            self.fpath = fpath
            self._check = stop_file(fpath)

        def on_train_epoch_end(self, trainer, pl_module):  # pragma: no cover
            if self._check():
                trainer.should_stop = True

else:  # pragma: no cover

    class LightningStopFileCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError("pytorch_lightning is not installed; cannot use LightningStopFileCallback")


__all__ = [
    "stop_file",
    "CatBoostStopFileCallback",
    "LightGBMStopFileCallback",
    "XGBoostStopFileCallback",
    "LightningStopFileCallback",
]
