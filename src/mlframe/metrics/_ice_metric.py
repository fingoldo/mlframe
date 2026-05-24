"""ICE (Integral Calibration Error) metric for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import ICE, compute_probabilistic_multiclass_error``
imports continue to work.

What lives here:
  - ``compute_probabilistic_multiclass_error`` (multi-class fastpath +
    legacy per-class loop)
  - ``ICE`` class (CatBoost ``eval_metric`` adapter with optional
    calibration-plot side effect)
  - ``_install_catboost_sklearn_clone_patch`` and the install call so
    CatBoost+ICE survives ``sklearn.base.clone()``
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

# Helpers used inside the metric body live in core.py / sibling modules.
# ``fast_brier_score_loss`` / ``fast_precision`` / CB logits-to-probs all live
# in ``core.py``; we import them lazily inside the function/method bodies
# (rather than at module top) to dodge the ``metrics -> core -> _ice_metric
# -> core`` import cycle that an eager ``from . import core`` would trigger
# (and that ``tests/test_meta/test_no_import_cycles.py`` flags as a hard fail).
from ._classification_report import (
    fast_calibration_report,
    _batch_per_class_ice_kernel,
    fast_ice_only,
)

logger = logging.getLogger(__name__)


def compute_probabilistic_multiclass_error(
    y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_score: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence],
    labels: np.ndarray = None,
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.8,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    verbose: bool = False,
    ndigits: int = 4,
    multilabel: bool = False,
    **kwargs,  # scorer can pass kwargs like {'needs_proba': True, 'needs_threshold': False}
):
    """Given a sequence of per-class probabilities (predicted by some model), and ground truth targets,
    computes weighted sum of per-class errors.
    Supports several error estimation methods: "multicrit", "brier_score", "precision".
    If number of classes is only 2, skips class 0 as it's fully complementary to class 1.

    ``multilabel=True``: y_true is a 2D (n_samples, n_classes) indicator matrix, each column
    treated as an independent binary target. Single-label (``y_true == class_id``) comparison
    is wrong for multilabel data because a sample can carry multiple positives simultaneously.

    NOT threaded across classes. Benched against a ``ThreadPoolExecutor.map``
    variant fanning each class to its own thread: ``par/seq`` ratio
    2.65-10.65x SLOWER across (N, K) shapes from (10k, 3) to (1M, 5). Reason:
    per-class work is small (each inner kernel already releases GIL, but
    Python-level fanout overhead dominates) AND the inner kernels
    (fast_brier_score_loss, fast_ice_only) already auto-dispatch to par at
    N>=100k, so threading over classes layers concurrency on top of per-
    kernel concurrency and produces oversubscription. Bench preserved at
    ``profiling/bench_multiclass_error_parallel.py``.
    """

    # Bad method slipped past under -O when this was an ``assert``; the elif
    # chain then returned a default value for the wrong metric definition.
    if method not in ("multicrit", "brier_score", "precision"):
        raise ValueError(
            f"method must be 'multicrit', 'brier_score', or 'precision'; "
            f"got {method!r}."
        )

    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.to_numpy()
    if labels is not None and isinstance(labels, (pd.Series, pd.DataFrame)):
        labels = labels.to_numpy()

    if isinstance(y_score, Sequence):
        probs = y_score
    else:
        if len(y_score.shape) == 1:
            y_score = np.vstack([1 - y_score, y_score]).T
        probs = [y_score[:, i] for i in range(y_score.shape[1])]

    # Auto-detect multilabel from shape: a 2D y_true with width matching probs count is
    # an indicator matrix; caller can also set ``multilabel=True`` explicitly.
    # Object-dtype-of-arrays (``pl.List`` -> pandas roundtrip) presents as 1-D
    # but each cell is a per-row label vector - stack to 2-D so the shape
    # check below activates the multilabel branch correctly. Surfaced 3-way
    # fuzz c0000 / c0008 (cb / multilabel target) - without the stack, the
    # ``y_true == class_id`` fall-through raised ``truth value of array
    # ambiguous`` on the cell-array comparison.
    if (
        isinstance(y_true, np.ndarray)
        and y_true.dtype == object
        and y_true.ndim == 1
        and y_true.shape[0] > 0
    ):
        _first = y_true[0]
        if hasattr(_first, "shape") or (
            hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))
        ):
            try:
                y_true = np.stack([np.asarray(c) for c in y_true], axis=0)
            except Exception as _e_stack:
                # Silent pass was the prior behaviour: stack failure left
                # y_true as object-of-arrays, then the multilabel auto-detect
                # branch below couldn't pick it up and the metric routed
                # through the wrong code path silently. DEBUG-log (not WARN,
                # the multilabel detection is best-effort) so the trail
                # exists for triage.
                logger.debug(
                    "compute_probabilistic_multiclass_error: y_true stack "
                    "failed (%s); multilabel auto-detect may misroute.",
                    _e_stack,
                )
    if not multilabel and isinstance(y_true, np.ndarray) and y_true.ndim == 2 and y_true.shape[1] == len(probs):
        multilabel = True
        logger.debug("compute_probabilistic_multiclass_error: detected multilabel y_true shape, enabling multilabel mode.")

    total_error = 0.0
    weights_sum = 0

    # Batched-numba fastpath. When the hot path applies (method='multicrit'
    # AND not verbose AND probs convert to a clean (N, K) float64 stack),
    # process all K classes in one numba dispatch via
    # ``_batch_per_class_ice_kernel``. Eliminates the per-class
    # Python->numba transition overhead that dominated the K-class
    # Python loop (cProfile attributed ~60 ms / call avg, of which
    # ~30 ms was Python glue for K=3 classes). Bit-exact equivalent
    # of the legacy fast_ice_only K-loop -- verified in
    # ``profiling/bench_compute_multiclass_error.py``.
    if method == "multicrit" and not verbose:
        # Build the set of class_ids to evaluate (binary case skips 0).
        _class_ids = [
            c for c in range(len(probs))
            if not (len(probs) == 2 and c == 0 and not multilabel)
        ]
        try:
            # Stack y_pred_NK. column_stack on a single-element list still
            # allocates a fresh (N, 1) array; cheaper to ``.reshape(-1, 1)``
            # on the already-contiguous column. Binary classification
            # (the iter#5 hot path) hits this K=1 case on every LGB eval
            # callback, so the saving is real (~5s on 400K-row 200-iter LGB).
            _y_pred_cols = []
            for _c in _class_ids:
                _yp = probs[_c]
                if isinstance(_yp, pl.Series):
                    _yp = _yp.to_numpy()
                elif isinstance(_yp, (pd.Series,)):
                    # ``.to_numpy()`` materialises nullable Int/Float dtypes as object-free ndarrays; ``.values`` returns an ExtensionArray that
                    # silently breaks downstream ``np.ascontiguousarray(_, dtype=np.float64)`` for pandas nullable columns.
                    _yp = _yp.to_numpy()
                _y_pred_cols.append(np.ascontiguousarray(_yp, dtype=np.float64))
            if len(_y_pred_cols) == 1:
                _y_pred_NK = _y_pred_cols[0].reshape(-1, 1)
            else:
                _y_pred_NK = np.column_stack(_y_pred_cols)
            # Stack y_true_NK (indicator matrix)
            _y_true_cols = []
            for _c in _class_ids:
                if multilabel:
                    _yt = y_true[:, _c]
                elif labels is not None:
                    _yt = y_true == labels[_c]
                else:
                    _yt = y_true == _c
                if isinstance(_yt, pl.Series):
                    _yt = _yt.cast(pl.Int8).to_numpy()
                elif isinstance(_yt, (pd.Series,)):
                    # ``.to_numpy()`` round-trips pandas nullable BooleanArray to a real bool ndarray; ``.values.astype(np.int8)`` on a nullable
                    # boolean returns object-dtype and silently mis-casts None entries to 1 -- the bug the audit caught at metric line 189.
                    _yt = _yt.to_numpy().astype(np.int8)
                else:
                    _yt = np.ascontiguousarray(_yt, dtype=np.int8)
                _y_true_cols.append(_yt)
            if len(_y_true_cols) == 1:
                _y_true_NK = _y_true_cols[0].reshape(-1, 1)
            else:
                _y_true_NK = np.column_stack(_y_true_cols)
            # Single-dispatch batched kernel
            ice_per_class = _batch_per_class_ice_kernel(
                _y_true_NK, _y_pred_NK, nbins,
                bool(use_weighted_calibration),
                float(mae_weight), float(std_weight), float(brier_loss_weight),
                float(roc_auc_weight), float(pr_auc_weight),
                float(min_roc_auc), float(roc_auc_penalty),
            )
            # Reduce with per-class weights
            for _k, _cid in enumerate(_class_ids):
                if weight_by_class_npositives:
                    weight = int(_y_true_NK[:, _k].sum())
                else:
                    weight = 1
                total_error += float(ice_per_class[_k]) * weight
                weights_sum += weight
            if weights_sum > 0:
                total_error /= weights_sum
            else:
                logger.warning(
                    "compute_probabilistic_multiclass_error: sum of per-class weights is 0; returning NaN."
                )
                total_error = float("nan")
            return total_error
        except Exception as _exc:
            # Defensive: any kernel / stacking issue falls through to the
            # legacy per-class Python loop below. Log at DEBUG so the path
            # transition stays visible during dev but doesn't spam INFO.
            logger.debug(
                "_batch_per_class_ice_kernel fastpath failed (%s); falling back to per-class loop.",
                _exc,
            )

    for class_id in range(len(probs)):

        if len(probs) == 2 and class_id == 0 and not multilabel:
            continue

        # Get prediction and ground truth

        y_pred = probs[class_id]
        if multilabel:
            # Indicator column for this class; each row is an independent binary label.
            correct_class = y_true[:, class_id]
        elif labels is not None:
            correct_class = y_true == labels[class_id]
        else:
            correct_class = y_true == class_id

        if isinstance(correct_class, (pd.Series, np.ndarray)):
            correct_class = correct_class.astype(np.int8)
        elif isinstance(correct_class, pl.Series):
            correct_class = correct_class.cast(pl.Int8).to_numpy()

        # Compute class error. When verbose=False (the fairness fan-out
        # hot path), take the ICE-only / brier-only fastpath and skip the
        # log_loss + precision/recall/f1 + title work that
        # ``fast_calibration_report`` does for its reporting callers.
        # Bit-exact equivalent -- see ``bench_ice_only.py``.

        if method == "multicrit":
            if verbose:
                brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, metrics_string, fig = fast_calibration_report(
                    y_true=correct_class, y_pred=y_pred, use_weights=use_weighted_calibration, nbins=nbins,
                    show_plots=False, verbose=False,
                    mae_weight=mae_weight, std_weight=std_weight, brier_loss_weight=brier_loss_weight,
                    roc_auc_weight=roc_auc_weight, pr_auc_weight=pr_auc_weight,
                    min_roc_auc=min_roc_auc, roc_auc_penalty=roc_auc_penalty,
                )
                logger.info("\t class_id=%s, %s", class_id, metrics_string)
                class_error = ice
            else:
                class_error = fast_ice_only(
                    y_true=correct_class, y_pred=y_pred, nbins=nbins, use_weights=use_weighted_calibration,
                    mae_weight=mae_weight, std_weight=std_weight, brier_loss_weight=brier_loss_weight,
                    roc_auc_weight=roc_auc_weight, pr_auc_weight=pr_auc_weight,
                    min_roc_auc=min_roc_auc, roc_auc_penalty=roc_auc_penalty,
                )
        elif method == "brier_score":
            # Only brier_loss is used -- skip binning/AUC/ICE entirely.
            from .core import fast_brier_score_loss  # lazy: import-cycle avoidance, see module top
            class_error = fast_brier_score_loss(y_true=correct_class, y_prob=y_pred)
            if verbose:
                logger.info(f"\t class_id={class_id}, brier_loss={class_error:.{ndigits}f}")
        elif method == "precision":
            from .core import fast_precision  # lazy: import-cycle avoidance, see module top
            class_error = fast_precision(y_true=correct_class, y_pred=(y_pred >= 0.5).astype(np.int8), zero_division=0)

        # Assign weights

        if weight_by_class_npositives:
            weight = correct_class.sum()
        else:
            weight = 1

        total_error += class_error * weight
        weights_sum += weight

    # Guard against div-by-zero when every per-class weight was 0 (e.g.
    # weight_by_class_npositives with all-negative y_true, or empty probs).
    # Previously propagated 0/0 -> NaN silently.
    if weights_sum > 0:
        total_error /= weights_sum
    else:
        logger.warning("compute_probabilistic_multiclass_error: sum of per-class weights is 0; returning NaN.")
        total_error = float("nan")

    if verbose:
        logger.info(f"method={method}, data size={len(correct_class):_} mean_class_error={total_error:.{ndigits}f}")

    return total_error


class ICE:
    """Custom probabilistic prediction error metric balancing predictive power with calibration.
    Can regularly create a calibration plot.
    """

    def __init__(
        self,
        metric: Callable,
        higher_is_better: bool,
        calibration_plot_period: int = 0,
        max_arr_size: int = 0,
    ) -> None:

        # save params
        store_params_in_object(obj=self, params=get_parent_func_args())

        self.nruns = 0

    def is_max_optimal(self):
        return self.higher_is_better

    def __sklearn_clone__(self):
        """Identity clone for sklearn's ``clone()`` (sklearn >= 1.3).

        ICE has no fit-state worth re-initialising on clone; sharing the
        same instance is harmless. Returning ``self`` here is necessary
        but not sufficient for the CB+ICE clone bind: ``CatBoost``'s
        ``__init__`` always deep-copies ``eval_metric`` internally, so
        the sklearn clone assertion (``new_obj.get_params()[name] is
        param``) still trips. The ``__sklearn_clone__`` patch installed
        below on ``CatBoostClassifier`` / ``CatBoostRegressor`` closes
        that gap by returning a CB instance with explicit identity-shared
        ``eval_metric`` instead of going through the parametric-check
        path.
        """
        return self

    def evaluate(self, approxes, target, weight):
        output_weight = 1  # weight is not used

        # to avoid expensive train set metric evaluation, we simply return 0 for any input larger than max_arr_size
        if self.max_arr_size and len(approxes[0]) > self.max_arr_size:
            return 0, output_weight

        # Convert CatBoost logits to probabilities using numba-optimized functions
        from .core import cb_logits_to_probs_binary, cb_logits_to_probs_multiclass  # lazy: import-cycle, see module top
        if len(approxes) == 1:
            # Binary classification
            probs_2d = cb_logits_to_probs_binary(approxes[0])
            probs = probs_2d  # Shape: (n_samples, 2)
            class_id = 1
            y_pred = probs_2d[:, 1]  # For plotting
        else:
            # Multiclass: stack approxes into 2D array (n_classes, n_samples)
            logits_2d = np.vstack(approxes)
            probs_2d = cb_logits_to_probs_multiclass(logits_2d)
            probs = probs_2d  # Shape: (n_samples, n_classes)
            class_id = len(approxes) - 1
            y_pred = probs_2d[:, class_id]  # For plotting

        total_error = self.metric(y_true=target, y_score=probs)

        self.nruns += 1

        # Additional visualization of training process (for the last class_id) is possible.

        if self.calibration_plot_period and (self.calibration_plot_period > 0 and (self.nruns % self.calibration_plot_period == 0)):
            y_true = (target == class_id).astype(np.int8)
            brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, metrics_string, fig = fast_calibration_report(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{len(approxes[0]):_} records of class {class_id}, integral error={total_error:.4f}, nruns={self.nruns:_}\r\n",
                use_weights=True,
                verbose=False,
            )
            logger.info(metrics_string)

        return total_error, output_weight

    def get_final_error(self, error, weight):
        return error


# -----------------------------------------------------------------------------
# CatBoost compat: ``__sklearn_clone__`` patch for CB+ICE eval_metric clone bug
# -----------------------------------------------------------------------------
# CatBoost's ``__init__`` deep-copies its ``eval_metric`` argument
# internally, then ``get_params(deep=False)`` returns the deep-copied
# instance. ``sklearn.base.clone()``'s parametric path verifies that
# ``new_object.get_params()[name] is param`` for every parameter -- and
# this fails for CB+ICE because ``param`` (post-clone-of-ICE) is the
# original ICE while ``new_object.get_params()['eval_metric']`` is a CB-
# internal deep copy. Identity is destroyed by CB, not by ICE.
#
# Touching ICE.__deepcopy__ to return self DID restore identity for CB's
# internal copy but BROKE pickle: sharing ICE across the pre-pickle
# ``copy.deepcopy(model)`` retains a CatBoost-internal numba JIT
# cyfunction reference (``_cpu_jit_method_wrap.<locals>.new_method``)
# that dill cannot serialize.
#
# The clean fix lives at the CB level: install ``__sklearn_clone__`` on
# ``CatBoostClassifier`` / ``CatBoostRegressor`` that reuses the same
# ``eval_metric`` instance for the cloned estimator, bypassing
# ``_clone_parametrized``'s identity check entirely. ICE retains its
# default deepcopy/pickle behaviour, so save/load is unaffected.
def _install_catboost_sklearn_clone_patch() -> None:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        return

    def _cb_sklearn_clone(self):
        # Reuse the SAME eval_metric instance to bypass CB's internal
        # deep-copy of eval_metric on __init__ that breaks identity.
        params = self.get_params(deep=False)
        eval_metric = getattr(self, "_init_params", {}).get("eval_metric")
        if eval_metric is None:
            eval_metric = params.get("eval_metric")
        cls = type(self)
        # Strip eval_metric from params before re-init so we can attach
        # the original instance after construction (CB's __init__ would
        # deep-copy it otherwise).
        params_no_em = {k: v for k, v in params.items() if k != "eval_metric"}
        new = cls(**params_no_em)
        if eval_metric is not None:
            # Re-attach by direct assignment -- CB stores eval_metric in
            # ``_init_params`` and the public param map.
            try:
                new._init_params["eval_metric"] = eval_metric
            except (AttributeError, KeyError):
                pass
            try:
                new.set_params(eval_metric=eval_metric)
            except Exception:
                pass
        return new

    for cls in (CatBoostClassifier, CatBoostRegressor):
        if not hasattr(cls, "__sklearn_clone__"):
            cls.__sklearn_clone__ = _cb_sklearn_clone


_install_catboost_sklearn_clone_patch()
