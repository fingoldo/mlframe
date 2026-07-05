"""Per-round full-metric-suite capture callbacks for boosters (lgb / xgb / cb).

Each callback re-predicts the val set at the current boosting round via the booster's NATIVE per-iteration
prediction (``num_iteration`` / ``iteration_range`` / ``ntree_end``), feeds ``(y_val, val_score)`` to
``mlframe.metrics.compute_all_metrics`` for the model's ``target_type``, and accumulates the resulting
``{metric_name -> float}`` dict into ``iteration_metrics_[round]``. The trajectory feeds meta-learning /
HPO-from-early-observation (predict final holdout from the first K rounds, prune bad configs early).

Re-predicting val every round is the cost driver, so capture is STRIDE-sampled: only every ``stride``-th round is
captured, plus the final round (and the round count starts at the first round so round 0 is always present). The
shim wires these only when ``capture_iteration_metrics`` is enabled (default OFF for boosters -- see
``TrainingBehaviorConfig``). Mirrors the wiring pattern of ``monotonic_decline``'s booster callbacks.

The raw ``X_val`` is required for prediction: lgb / cb val carriers (binned Dataset / Pool) and the xgb val DMatrix
can be predicted directly, but lgb's binned Dataset cannot, so the lgb callback holds the raw ``X_val`` frame.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _should_capture(round_idx: int, stride: int) -> bool:
    return stride <= 1 or (round_idx % stride) == 0


def _store(iteration_metrics: dict, round_idx: int, y_true, y_score, target_type: str, n_classes: Optional[int]) -> None:
    from mlframe.metrics import compute_all_metrics

    try:
        iteration_metrics[int(round_idx)] = compute_all_metrics(y_true, y_score, target_type=target_type, n_classes=n_classes)
    except Exception as exc:  # never let metric capture abort a training run
        logger.warning("iteration-metrics capture failed at round %d: %s", round_idx, exc, exc_info=False)


class LGBIterationMetricsCallback:
    """LightGBM callback: capture the full val metric suite every ``stride`` rounds.

    LightGBM's binned val Dataset cannot be re-predicted, so the raw ``X_val`` frame is held here and scored via
    ``env.model.predict(X_val, num_iteration=round+1)``. ``order`` runs after metric eval (mirrors the native ES
    callback ordering) so the round index matches the booster's recorded history.
    """

    order = 35
    before_iteration = False

    def __init__(self, X_val, y_val, target_type: str, *, stride: int = 1, n_classes: Optional[int] = None) -> None:
        self.X_val = X_val
        self.y_val = np.asarray(y_val)
        self.target_type = str(target_type)
        self.stride = max(1, int(stride))
        self.n_classes = n_classes
        self.iteration_metrics_: dict[int, dict[str, float]] = {}
        self._last_round = -1

    def __call__(self, env) -> None:
        round_idx = env.iteration - env.begin_iteration
        self._last_round = round_idx
        is_last = env.iteration == (env.end_iteration - 1)
        if not (_should_capture(round_idx, self.stride) or is_last):
            return
        score = env.model.predict(self.X_val, num_iteration=env.iteration + 1)
        _store(self.iteration_metrics_, round_idx, self.y_val, score, self.target_type, self.n_classes)


def make_xgb_iteration_metrics_callback(
    dval: Any, y_val: npt.ArrayLike, target_type: str, *, stride: int = 1, n_classes: Optional[int] = None
) -> Optional[Any]:
    """Build an XGBoost ``TrainingCallback`` capturing the full val metric suite every ``stride`` rounds.

    Factory so ``xgboost`` is imported lazily (module stays importable without xgboost). The val DMatrix ``dval``
    can be predicted directly via ``model.predict(dval, iteration_range=(0, epoch+1))``. Returns ``None`` when
    xgboost is unavailable. The callback exposes ``iteration_metrics_`` after fit.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return None

    class _XGBIterationMetrics(xgb.callback.TrainingCallback):
        _is_mlframe_iteration_metrics = True

        def __init__(self) -> None:
            super().__init__()
            self.iteration_metrics_: dict[int, dict[str, float]] = {}
            self._stride = max(1, int(stride))
            self._n_classes = n_classes
            self._target_type = str(target_type)

        def after_iteration(self, model, epoch, evals_log) -> bool:
            if not (_should_capture(epoch, self._stride)):
                # last round handled in after_training; capture stride rounds here.
                return False
            score = model.predict(dval, iteration_range=(0, epoch + 1))
            _store(self.iteration_metrics_, epoch, y_val, score, self._target_type, self._n_classes)
            return False

        def after_training(self, model):
            # Always capture the final round even if stride skipped it (the best/last round is the meta-learning anchor).
            try:
                last = int(model.num_boosted_rounds()) - 1
            except Exception:
                last = max(self.iteration_metrics_, default=-1)
            if last >= 0 and last not in self.iteration_metrics_:
                score = model.predict(dval, iteration_range=(0, last + 1))
                _store(self.iteration_metrics_, last, y_val, score, self._target_type, self._n_classes)
            return model

    return _XGBIterationMetrics()


class CBIterationMetricsCallback:
    """CatBoost callback: capture the full val metric suite every ``stride`` iterations.

    CatBoost calls ``after_iteration(info)`` per iteration; ``info.iteration`` is the (1-based) round. The val
    ``Pool`` is held here so the current-round score comes from ``model.predict(pool, ntree_end=info.iteration)``
    (probabilities via ``prediction_type``). When the build does not expose the model on ``info`` the callback
    degrades to no capture for that round rather than raising.
    """

    def __init__(self, val_pool, y_val, target_type: str, *, stride: int = 1, n_classes: Optional[int] = None) -> None:
        self.val_pool = val_pool
        self.y_val = np.asarray(y_val)
        self.target_type = str(target_type)
        self.stride = max(1, int(stride))
        self.n_classes = n_classes
        self.iteration_metrics_: dict[int, dict[str, float]] = {}

    def after_iteration(self, info) -> bool:
        it = int(getattr(info, "iteration", 0))
        round_idx = it - 1  # normalise to 0-based to match lgb / xgb
        if round_idx < 0 or not _should_capture(round_idx, self.stride):
            return True
        model = getattr(info, "model", None)
        if model is None:
            return True
        score = self._predict(model, it)
        if score is not None:
            _store(self.iteration_metrics_, round_idx, self.y_val, score, self.target_type, self.n_classes)
        return True

    def _predict(self, model, ntree_end: int) -> Optional[Any]:
        try:
            is_reg = "regression" in self.target_type
            if is_reg:
                return model.predict(self.val_pool, ntree_end=ntree_end)
            proba = model.predict_proba(self.val_pool, ntree_end=ntree_end)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        except Exception as exc:
            logger.debug("CBIterationMetricsCallback predict failed at ntree_end=%d: %s", ntree_end, exc)
            return None


__all__ = [
    "LGBIterationMetricsCallback",
    "make_xgb_iteration_metrics_callback",
    "CBIterationMetricsCallback",
]
