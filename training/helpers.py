"""
Training helper functions and callback classes.

This module contains helper utilities:
- parse_catboost_devices: GPU device parsing for CatBoost
- get_training_configs: Training configuration factory
- get_trainset_features_stats: Compute training set statistics (pandas)
- get_trainset_features_stats_polars: Compute training set statistics (polars)
- UniversalCallback: Base callback class for training monitoring
- LightGBMCallback, XGBoostCallback, CatBoostCallback: Model-specific callbacks
"""

import logging
import psutil
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Optional, Dict, List, Callable, Sequence, Any, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

# NOTE: torch + mlframe.lightninglib are imported lazily inside `get_training_configs`
# (only needed for MLP configs). Top-level import cost ~2-3s — avoided for CB/LGB/XGB-only runs.
import lightgbm as lgb

import xgboost as xgb
from xgboost.callback import TrainingCallback

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from pyutilz.system import get_gpuinfo_gpu_info, tqdmu, get_own_memory_usage
from pyutilz.pythonlib import get_parent_func_args, store_params_in_object
from mlframe.metrics import (
    compute_probabilistic_multiclass_error,
    robust_mlperf_metric,
    ICE,
)

from .utils import get_numeric_columns, get_categorical_columns

logger = logging.getLogger(__name__)


# Constant - CUDA availability
try:
    from numba.cuda import is_available as is_cuda_available

    CUDA_IS_AVAILABLE = is_cuda_available()
except (ImportError, AttributeError, ModuleNotFoundError):
    CUDA_IS_AVAILABLE = False


# =============================================================================
# Multi-output (multiclass + multilabel) dispatch helpers — 2026-04-24
# =============================================================================
#
# Probability-surface contract: every classification estimator's
# ``predict_proba`` is canonicalised to ``(N, K)`` shape regardless of
# source — sklearn binary returns ``(N, 2)``, ``MultiOutputClassifier``
# returns ``List[(N, 2)]``, CB native ``MultiLogloss`` returns ``(N, K)``
# already. The canonicalizer + decision-rule pair below wraps that
# heterogeneity behind two pure functions used at every site that
# previously hard-coded ``probs[:, 1]`` (4 sites in core.py, 3 in
# evaluation.py, 2 in automl.py).


def _canonical_predict_proba_shape(probs, classes_=None):
    """Force a classifier's ``predict_proba`` output into ``(N, K)`` shape.

    Handles every form a classifier might return:
    - ``np.ndarray`` shape ``(N, K)`` (sklearn binary, CB native, etc.) → pass-through
    - ``List[np.ndarray]`` of length K, each shape ``(N, 2)`` (``MultiOutputClassifier``)
      → stack the class-1 column from each label, returning ``(N, K)``.
      Per-estimator ``classes_`` may be ``[0]`` only (constant label column);
      in that case, emit a column of zeros for that label rather than raise.
    - ``np.ndarray`` shape ``(N,)`` (1-D sigmoid output) → stack as ``[1-p, p]``
      to ``(N, 2)``.

    Parameters
    ----------
    probs
        Whatever ``model.predict_proba(X)`` returned.
    classes_
        Per-estimator ``classes_`` when ``probs`` is the list-form returned
        by ``MultiOutputClassifier`` (one ``classes_`` per output). Used to
        detect constant-label columns and emit zeros for them. Optional;
        if not provided we infer that any ``(N, 1)`` per-label array is a
        constant column and emit zeros.

    Returns
    -------
    np.ndarray of shape ``(N, K)``, dtype float64.
    """
    if isinstance(probs, list):
        # MultiOutputClassifier: List[(N, 2)] OR List[(N, 1)] for constant cols.
        cols = []
        for j, sub in enumerate(probs):
            sub = np.asarray(sub)
            if sub.ndim == 2 and sub.shape[1] >= 2:
                # Standard binary output for this label — take the class-1 column.
                # If classes_ provided and last class is the "positive" class,
                # use that; else assume class index 1.
                pos_idx = sub.shape[1] - 1
                if classes_ is not None and j < len(classes_):
                    cls = np.asarray(classes_[j])
                    # Pick the column corresponding to label "1" if present,
                    # else fall back to the last column.
                    pos_idx = (
                        int(np.where(cls == 1)[0][0])
                        if cls.size > 1 and (cls == 1).any()
                        else sub.shape[1] - 1
                    )
                cols.append(sub[:, pos_idx])
            elif sub.ndim == 2 and sub.shape[1] == 1:
                # Constant label column: estimator never saw class 1 in training.
                # Emit zeros (the estimator will always predict 0).
                cols.append(np.zeros(sub.shape[0], dtype=np.float64))
            elif sub.ndim == 1:
                cols.append(sub.astype(np.float64))
            else:
                raise ValueError(
                    f"_canonical_predict_proba_shape: unexpected per-label "
                    f"shape {sub.shape} at index {j}"
                )
        return np.column_stack(cols).astype(np.float64)

    arr = np.asarray(probs)
    if arr.ndim == 1:
        # 1-D sigmoid → [1-p, p]
        return np.column_stack([1.0 - arr, arr]).astype(np.float64)
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"_canonical_predict_proba_shape: unsupported probs shape {arr.shape}"
    )


def _predict_from_probs(probs_NK, target_type, classes_=None, threshold=0.5):
    """Decision rule converting an (N, K) probability matrix to predictions.

    Parameters
    ----------
    probs_NK : np.ndarray (N, K)
        Output of ``_canonical_predict_proba_shape``.
    target_type : TargetTypes
        Determines the decision rule:
        - BINARY:    threshold-based on column 1 → (N,) labels via classes_
        - MULTICLASS: argmax over K → (N,) labels via classes_
        - MULTILABEL: per-label threshold → (N, K) binary matrix
        - REGRESSION: not supported (caller bug); raises.
    classes_ : np.ndarray (K,), optional
        Class labels for binary/multiclass mapping. If None, returns
        integer class indices.
    threshold : float OR np.ndarray (K,)
        Decision threshold for BINARY and MULTILABEL. Per-label vector
        is permitted for MULTILABEL (label-cost-sensitive thresholds).

    Returns
    -------
    np.ndarray
        BINARY/MULTICLASS: shape ``(N,)``
        MULTILABEL:        shape ``(N, K)`` binary {0, 1}
    """
    from .configs import TargetTypes

    arr = np.ascontiguousarray(probs_NK)
    if arr.ndim != 2:
        raise ValueError(
            f"_predict_from_probs expects (N, K); got shape {arr.shape}"
        )

    if target_type == TargetTypes.BINARY_CLASSIFICATION:
        thr = float(threshold) if np.ndim(threshold) == 0 else float(threshold[-1])
        # Column 1 is the positive-class probability for sklearn binary.
        pos = arr[:, -1] if arr.shape[1] >= 2 else arr[:, 0]
        idx = (pos >= thr).astype(np.int8)
        if classes_ is not None:
            cls = np.asarray(classes_)
            return cls[idx]
        return idx

    if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
        idx = np.argmax(arr, axis=1)
        if classes_ is not None:
            cls = np.asarray(classes_)
            return cls[idx]
        return idx

    if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
        # Per-label thresholds: scalar broadcasts; (K,) array applied
        # column-wise. NaN-safe: treats NaN probabilities as "below threshold".
        thr = np.asarray(threshold, dtype=np.float64)
        if thr.ndim == 0:
            thr = np.broadcast_to(thr, (arr.shape[1],))
        out = np.zeros_like(arr, dtype=np.int8)
        np.greater_equal(arr, thr, out=out, where=~np.isnan(arr))
        return out

    raise ValueError(
        f"_predict_from_probs: target_type {target_type!r} is not a "
        "classification type (REGRESSION has no decision rule)."
    )


def _classif_objective_kwargs(flavor, target_type, n_classes):
    """Library-specific ``Classifier(**kwargs)`` injection for the target type.

    Returns the dict to splat into the classifier's ``__init__`` so the
    underlying library trains the right loss / objective. For multilabel
    via wrapper (XGB/LGB/HGB/Linear) returns empty — wrapping is handled
    separately via ``_maybe_wrap_multilabel``.

    Parameters
    ----------
    flavor : str
        One of ``"catboost"``, ``"xgboost"``, ``"lightgbm"``, ``"hgb"``,
        ``"linear"``. (Strategy-side overrides will eventually replace
        this stringly-typed dispatch — see ``ModelPipelineStrategy.
        get_classif_objective_kwargs`` for the OOP version.)
    target_type : TargetTypes
        Drives the objective selection.
    n_classes : int
        Required for MULTICLASS (XGB, LGB need ``num_class=K``) and
        MULTILABEL (CatBoost: number of label outputs).

    Returns
    -------
    dict
        Keyword arguments to pass to the classifier constructor.
    """
    from .configs import TargetTypes

    if target_type == TargetTypes.BINARY_CLASSIFICATION:
        return {
            "catboost": {},  # CB auto-detects from y dtype
            "xgboost": {"objective": "binary:logistic"},
            "lightgbm": {"objective": "binary"},
            "hgb": {},
            "linear": {},
        }.get(flavor, {})

    if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
        return {
            "catboost": {"loss_function": "MultiClass"},
            "xgboost": {"objective": "multi:softprob", "num_class": n_classes},
            "lightgbm": {"objective": "multiclass", "num_class": n_classes},
            "hgb": {},  # sklearn HistGradientBoostingClassifier auto-detects
            "linear": {"multi_class": "multinomial", "solver": "lbfgs"},
        }.get(flavor, {})

    if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
        # CatBoost has native multilabel via MultiLogloss (returns (N, K) directly).
        # Other libraries get an empty dict here — the OvR wrapper in
        # _maybe_wrap_multilabel handles their dispatch.
        if flavor == "catboost":
            return {"loss_function": "MultiLogloss"}
        return {}

    return {}


def _maybe_wrap_multilabel(estimator, target_type, multilabel_config=None,
                           strategy_supports_native_multilabel=False, n_labels=None):
    """Multilabel dispatch: native (CB) vs MultiOutputClassifier vs ChainEnsemble.

    Decision tree (when ``target_type == MULTILABEL_CLASSIFICATION``):
      1. ``multilabel_config.strategy == "native"`` AND
         ``strategy_supports_native_multilabel`` → return estimator unchanged
         (the strategy already injected the right ``loss_function`` via
         ``_classif_objective_kwargs``)
      2. ``multilabel_config.strategy == "native"`` AND NOT supported →
         raise (user explicitly asked for native; fail loud)
      3. ``multilabel_config.strategy == "auto"`` AND ``strategy_supports_
         native_multilabel`` → return estimator unchanged (CB native path)
      4. ``multilabel_config.strategy == "chain"`` → ``_ChainEnsemble`` of
         ``n_chains`` chains
      5. otherwise (``"auto"`` falling through, or ``"wrapper"``) →
         ``MultiOutputClassifier(estimator, n_jobs=...)``

    For non-multilabel target_types, returns ``estimator`` unchanged.

    Parameters
    ----------
    estimator
        The base classifier (already configured via
        ``_classif_objective_kwargs``).
    target_type : TargetTypes
    multilabel_config : MultilabelDispatchConfig, optional
        Defaults to ``MultilabelDispatchConfig()`` (auto strategy, n_chains=3).
    strategy_supports_native_multilabel : bool
        From ``ModelPipelineStrategy.supports_native_multilabel``.
        True only for ``CatBoostStrategy`` today.
    n_labels : int, optional
        Number of labels (K). Required for the chain path.

    Returns
    -------
    The estimator (possibly wrapped).
    """
    from .configs import TargetTypes, MultilabelDispatchConfig

    if target_type != TargetTypes.MULTILABEL_CLASSIFICATION:
        return estimator

    cfg = multilabel_config if multilabel_config is not None else MultilabelDispatchConfig()
    strat = cfg.strategy

    if strat == "native":
        if strategy_supports_native_multilabel:
            return estimator
        raise ValueError(
            f"MultilabelDispatchConfig.strategy='native' but the underlying "
            f"strategy ({type(estimator).__name__}) does not support native "
            f"multilabel. Use strategy='wrapper' or 'auto' instead."
        )

    if strat == "auto" and strategy_supports_native_multilabel:
        return estimator

    if strat == "chain":
        if n_labels is None:
            raise ValueError("multilabel chain strategy requires n_labels")
        return _build_classifier_chain_ensemble(
            estimator, n_labels=n_labels,
            n_chains=cfg.n_chains, seeds=cfg.chain_seeds,
            order_strategy=cfg.chain_order_strategy,
            user_orders=cfg.chain_order_user,
            cv=cfg.cv,
        )

    # Default + "wrapper" + "auto"-without-native: MultiOutputClassifier
    from sklearn.multioutput import MultiOutputClassifier
    n_jobs = cfg.wrapper_n_jobs
    if n_jobs == "auto":
        # Avoid nested-parallelism thrashing when inner estimator already
        # uses n_jobs=-1 (most GBMs do). Use up to half of CPUs, cap at K.
        import os
        cpu = os.cpu_count() or 1
        n_jobs = min(n_labels or cpu, max(1, cpu // 2))
    return MultiOutputClassifier(estimator, n_jobs=n_jobs)


def _compute_chain_orders(n_labels, n_chains, order_strategy="random",
                          user_orders=None, seeds=None, y=None):
    """Return ``n_chains`` orderings of ``range(n_labels)`` per the strategy.

    Strategies:
    - ``"random"``: ``np.random.RandomState(seed).permutation(n_labels)``
      per chain. Default.
    - ``"by_frequency"``: rare-first ordering of labels (heuristic — start
      with the easiest-to-predict-conditionally label, refine downstream).
      Requires ``y`` (training target). Same ordering for all chains; no
      randomisation, so n_chains > 1 makes the ensemble redundant.
    - ``"user"``: take ``user_orders`` as a list of orderings.

    Returns
    -------
    list[np.ndarray] of length n_chains
    """
    if order_strategy == "user":
        if user_orders is None or len(user_orders) != n_chains:
            raise ValueError(
                f"chain_order_strategy='user' requires chain_order_user with "
                f"{n_chains} orderings"
            )
        return [np.asarray(o, dtype=int) for o in user_orders]

    if order_strategy == "by_frequency":
        if y is None:
            raise ValueError("chain_order_strategy='by_frequency' requires y")
        freq = np.asarray(y).sum(axis=0)
        order = np.argsort(freq)  # rare-first
        return [order.copy() for _ in range(n_chains)]

    # random
    if seeds is None:
        seeds = list(range(n_chains))
    return [np.random.RandomState(s).permutation(n_labels) for s in seeds]


class _ChainEnsemble:
    """Soft-voting ensemble of ``ClassifierChain`` instances for multilabel.

    sklearn's ``VotingClassifier(soft)`` does NOT accept multilabel y
    (raises ``ValueError: multilabel-indicator is not supported``). This
    class is a minimal hand-rolled equivalent for multilabel:
    fit each chain on (X, y), then ``predict_proba(X)`` averages the
    per-chain ``(N, K)`` outputs.

    Empirical lift (sklearn docs ``plot_classifier_chain_yeast``):
    +2-5% Jaccard over ``MultiOutputClassifier`` on correlated labels;
    +cost is 3-5× training (one fit per chain).

    Parameters
    ----------
    base_estimator
        Underlying binary classifier (LR, XGBClassifier with binary
        objective, etc.). Cloned per chain via ``sklearn.base.clone``.
    n_labels : int
    n_chains : int
    seeds : list[int] | None
    order_strategy : str
    user_orders : list[list[int]] | None
    cv : int | None
        ``ClassifierChain.cv``. Default 5 — cross-validates chain
        features to avoid training-data leak.
    """

    def __init__(self, base_estimator, n_labels, *, n_chains=3, seeds=None,
                 order_strategy="random", user_orders=None, cv=5):
        from sklearn.base import clone
        from sklearn.multioutput import ClassifierChain

        self.base_estimator = base_estimator
        self.n_labels = n_labels
        self.n_chains = n_chains
        self.seeds = seeds if seeds is not None else list(range(n_chains))
        self.cv = cv
        # Compute orders eagerly (without y for non-by_frequency); resolve at fit
        # for by_frequency.
        self._orders_resolved = None
        if order_strategy != "by_frequency":
            self._orders_resolved = _compute_chain_orders(
                n_labels, n_chains, order_strategy=order_strategy,
                user_orders=user_orders, seeds=self.seeds,
            )
        self._order_strategy = order_strategy
        self._user_orders = user_orders
        self.chains_ = [
            ClassifierChain(
                clone(base_estimator),
                order=(self._orders_resolved[i].tolist()
                       if self._orders_resolved is not None else None),
                cv=cv,
                random_state=self.seeds[i],
            )
            for i in range(n_chains)
        ]

    def fit(self, X, y, **fit_params):
        from sklearn.multioutput import ClassifierChain
        from sklearn.base import clone

        if self._orders_resolved is None:
            # by_frequency — needs y
            self._orders_resolved = _compute_chain_orders(
                self.n_labels, self.n_chains,
                order_strategy=self._order_strategy,
                user_orders=self._user_orders, seeds=self.seeds, y=y,
            )
            # Rebuild chains with the now-resolved orders.
            self.chains_ = [
                ClassifierChain(
                    clone(self.base_estimator),
                    order=self._orders_resolved[i].tolist(),
                    cv=self.cv, random_state=self.seeds[i],
                )
                for i in range(self.n_chains)
            ]

        for chain in self.chains_:
            chain.fit(X, y, **fit_params)
        # Mirror sklearn estimator API.
        self.classes_ = self.chains_[0].classes_
        return self

    def predict_proba(self, X):
        # Each chain returns (N, K). Average them.
        per_chain = [chain.predict_proba(X) for chain in self.chains_]
        stacked = np.stack(per_chain, axis=0)  # (n_chains, N, K)
        return stacked.mean(axis=0)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int8)

    def __sklearn_is_fitted__(self):
        return all(hasattr(c, "estimators_") for c in self.chains_)


def _build_classifier_chain_ensemble(base_estimator, n_labels, *,
                                      n_chains=3, seeds=None,
                                      order_strategy="random",
                                      user_orders=None, cv=5):
    """Convenience factory for ``_ChainEnsemble`` (the public dispatch entry).

    See ``_ChainEnsemble`` for parameter semantics.
    """
    return _ChainEnsemble(
        base_estimator, n_labels=n_labels, n_chains=n_chains, seeds=seeds,
        order_strategy=order_strategy, user_orders=user_orders, cv=cv,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# GPU Device Parsing
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def parse_catboost_devices(devices: str, all_gpus: list = None) -> List[Dict]:
    """
    Parses a GPU devices string and returns a list of GPU info dicts
    corresponding to the specified device indices.

    Parameters
    ----------
    devices : str
        A string specifying device indices. Formats supported:
          - "0"             (single GPU)
          - "0:1:3"         (multiple GPUs)
          - "0-3"           (range of GPUs, inclusive)

    Returns
    -------
    list[dict]
        Filtered list of GPU info dictionaries.
    """

    if not all_gpus:
        all_gpus = get_gpuinfo_gpu_info()

    if not devices:
        return all_gpus

    # Parse the devices string
    device_indices = []
    try:
        if "-" in devices:  # range format
            parts = devices.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range format '{devices}'. Expected 'start-end' (e.g., '0-3')")
            start, end = parts
            start_int, end_int = int(start), int(end)
            if start_int > end_int:
                raise ValueError(f"Invalid range '{devices}': start ({start_int}) > end ({end_int})")
            device_indices = list(range(start_int, end_int + 1))
        elif ":" in devices:  # multiple specific GPUs
            device_indices = [int(x) for x in devices.split(":")]
        else:  # single GPU
            device_indices = [int(devices)]
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid device specification '{devices}'. Must contain integers only.") from e
        raise

    # Validate indices
    max_index = len(all_gpus) - 1
    invalid = [i for i in device_indices if i < 0 or i > max_index]
    if invalid:
        raise ValueError(f"Invalid GPU indices {invalid}. Available range: 0-{max_index}")

    # Filter GPU list
    filtered_gpus = [gpu for gpu in all_gpus if gpu["index"] in device_indices]
    return filtered_gpus


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Configuration Factory
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_training_configs(
    iterations: int = 5000,
    early_stopping_rounds: Optional[int] = 0,
    validation_fraction: float = 0.1,
    use_explicit_early_stopping: bool = True,
    has_time: bool = True,
    has_gpu: bool = None,
    subgroups: dict = None,
    learning_rate: float = 0.1,
    def_regr_metric: str = "MAE",
    def_classif_metric: str = "AUC",
    # 2026-04-24: target_type-aware classifier objective injection.
    # When target_type is BINARY_CLASSIFICATION (default), the existing
    # binary objectives ("binary:logistic" / "binary" etc.) are kept.
    # For MULTICLASS / MULTILABEL, _classif_objective_kwargs replaces
    # them with the right native dispatch ("multi:softprob"+num_class,
    # "MultiLogloss", etc.).
    target_type: Optional[Any] = None,  # TargetTypes; None = legacy binary
    n_classes: int = 2,
    catboost_custom_classif_metrics: Optional[Sequence] = None,
    catboost_custom_regr_metrics: Optional[Sequence] = None,
    random_seed: Optional[int] = None,
    verbose: int = 0,
    # ----------------------------------------------------------------------------------------------------------------------------
    # probabilistic errors
    # ----------------------------------------------------------------------------------------------------------------------------
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
    # ----------------------------------------------------------------------------------------------------------------------------
    # robustness parameters for early stopping metric
    # ----------------------------------------------------------------------------------------------------------------------------
    robustness_num_ts_splits: int = 0,  # 0 = disabled, >0 = number of consecutive time splits
    robustness_std_coeff: float = 0.1,  # multiplier for std penalty
    robustness_greater_is_better: bool = False,  # False for ICE (lower is better)
    # ----------------------------------------------------------------------------------------------------------------------------
    # model-specific params
    # ----------------------------------------------------------------------------------------------------------------------------
    cb_kwargs: dict = None,
    hgb_kwargs: dict = None,
    lgb_kwargs: dict = None,
    xgb_kwargs: dict = None,
    mlp_kwargs: dict = None,
    ngb_kwargs: dict = None,
    # ----------------------------------------------------------------------------------------------------------------------------
    # featureselectors
    # ----------------------------------------------------------------------------------------------------------------------------
    rfecv_kwargs: dict = None,
) -> tuple:
    """Returns comparable training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for getting the feeling of what ML models are capable of for a particular task.
    """

    if has_gpu is None:
        has_gpu = CUDA_IS_AVAILABLE

    # Initialize mutable defaults
    if catboost_custom_classif_metrics is None:
        catboost_custom_classif_metrics = ["AUC", "BrierScore", "PRAUC"]
    if catboost_custom_regr_metrics is None:
        catboost_custom_regr_metrics = ["RMSE", "MAPE"]

    # Initialize kwargs dicts with defaults, making copies to avoid mutating caller's dicts
    if cb_kwargs is None:
        cb_kwargs = dict(verbose=0)
    else:
        cb_kwargs = cb_kwargs.copy()  # Don't mutate caller's dict
    if lgb_kwargs is None:
        lgb_kwargs = dict(verbose=-1)
    else:
        lgb_kwargs = lgb_kwargs.copy()  # Don't mutate caller's dict
    if xgb_kwargs is None:
        xgb_kwargs = dict(verbosity=0)
    else:
        xgb_kwargs = xgb_kwargs.copy()  # Don't mutate caller's dict
    if hgb_kwargs is None:
        hgb_kwargs = dict(verbose=0)
    else:
        hgb_kwargs = hgb_kwargs.copy()
    if mlp_kwargs is None:
        mlp_kwargs = dict()
    else:
        mlp_kwargs = mlp_kwargs.copy()
    if ngb_kwargs is None:
        ngb_kwargs = dict(verbose=True)
    else:
        ngb_kwargs = ngb_kwargs.copy()

    # None = disabled (don't pass to model fit at all); 0 = auto (iterations // 3); int = as-is.
    early_stopping_disabled = early_stopping_rounds is None
    if not early_stopping_disabled and not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 3)

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    # Build defaults, then let caller's kwargs override any of them
    # via .update(). Using **cb_kwargs for merge crashes when the
    # caller passes a key that's already in the defaults dict
    # (TypeError: got multiple values).
    CB_GENERAL_PARAMS = dict(
        iterations=iterations,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type="GPU" if has_gpu else "CPU",
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
    )
    CB_GENERAL_PARAMS.update(cb_kwargs)

    CB_CLASSIF = CB_GENERAL_PARAMS.copy()
    CB_CLASSIF.update({"eval_metric": def_classif_metric})
    # NOTE: custom_metric breaks sklearn.clone() - CatBoost modifies this param after init.
    # TODO: Raise issue at https://github.com/catboost/catboost/issues
    # "custom_metric": tuple(catboost_custom_classif_metrics or [])

    CB_REGR = CB_GENERAL_PARAMS.copy()
    CB_REGR.update({"eval_metric": def_regr_metric})
    # NOTE: custom_metric breaks sklearn.clone() - CatBoost modifies this param after init.
    # TODO: Raise issue at https://github.com/catboost/catboost/issues
    # "custom_metric": tuple(catboost_custom_regr_metrics or [])

    HGB_GENERAL_PARAMS = dict(
        max_iter=iterations,
        learning_rate=learning_rate,
        early_stopping=True,
        validation_fraction=(None if use_explicit_early_stopping else validation_fraction),
        n_iter_no_change=early_stopping_rounds,
        categorical_features="from_dtype",
        random_state=random_seed,
    )
    HGB_GENERAL_PARAMS.update(hgb_kwargs)

    XGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,  # affects model size heavily when high cardinality cat features r present!
        tree_method="hist",
        device="cuda" if has_gpu else "cpu",
        n_jobs=psutil.cpu_count(logical=False),
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
    )
    XGB_GENERAL_PARAMS.update(xgb_kwargs)

    XGB_GENERAL_CLASSIF = XGB_GENERAL_PARAMS.copy()
    XGB_GENERAL_CLASSIF.update({"objective": "binary:logistic", "eval_metric": neg_ovr_roc_auc_score})

    # 2026-04-24: target_type-aware objective injection. For non-binary
    # classification target types, replace the binary defaults with the
    # native multi-output objective. Binary path is a no-op (helper
    # returns the same kwargs as the explicit defaults above).
    from .configs import TargetTypes as _TT

    _resolved_tt = target_type if target_type is not None else _TT.BINARY_CLASSIFICATION
    if _resolved_tt.is_classification and not _resolved_tt.is_binary:
        # Non-binary classification: inject native objective per library.
        cb_obj = _classif_objective_kwargs("catboost", _resolved_tt, n_classes)
        xgb_obj = _classif_objective_kwargs("xgboost", _resolved_tt, n_classes)
        lgb_obj = _classif_objective_kwargs("lightgbm", _resolved_tt, n_classes)
        if cb_obj:
            CB_CLASSIF.update(cb_obj)
            # 2026-04-24 Session 6: when loss_function=MultiLogloss (multilabel),
            # CB REJECTS eval_metric='AUC' with "metric AUC and loss MultiLogloss
            # are incompatible". Override to HammingLoss for MultiLogloss,
            # Accuracy for MultiClass. Caller can still override via cb_kwargs.
            if cb_obj.get("loss_function") == "MultiLogloss":
                CB_CLASSIF["eval_metric"] = "HammingLoss"
            elif cb_obj.get("loss_function") == "MultiClass":
                CB_CLASSIF["eval_metric"] = "Accuracy"
        if xgb_obj:
            # For multiclass, multi:softprob conflicts with binary metric.
            # Strip the binary eval_metric — caller can re-set if needed.
            XGB_GENERAL_CLASSIF.update(xgb_obj)
            # XGB multiclass eval_metric: mlogloss aligns with multi:softprob.
            # (binary binary_logloss / AUC don't apply.)
            if xgb_obj.get("objective") == "multi:softprob":
                XGB_GENERAL_CLASSIF["eval_metric"] = "mlogloss"
        if lgb_obj:
            # LGB_GENERAL_PARAMS gets the multiclass objective too — it has
            # no separate _CLASSIF variant currently.
            pass  # applied to LGB after LGB_GENERAL_PARAMS is built (below)
        # NOTE: _mlframe_target_type metadata tag was historically attached
        # here but REMOVED 2026-04-24 Session 6 — CatBoostClassifier init
        # raises TypeError on unknown kwargs, blocking the entire multilabel
        # path. Downstream observability (which lib + target_type) is covered
        # by the per-model model_schemas metadata record populated in
        # core.py around the fit call. Adding a side-channel tag here was
        # a premature optimisation that forked a 4-year-stable init API
        # contract for a diagnostic that's available elsewhere.

    def integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, verbose: bool = False) -> float:
        """Compute integral calibration error for probabilistic predictions.

        Wraps compute_probabilistic_multiclass_error with the outer function's
        configuration parameters (method, weights, etc.).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_score : np.ndarray
            Predicted probabilities.
        verbose : bool, default=False
            If True, print calibration error info.

        Returns
        -------
        float
            The computed calibration error (lower is better).
        """
        err = compute_probabilistic_multiclass_error(
            y_true=y_true,
            y_score=y_score,
            method=method,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        if verbose:
            print(len(y_true), "integral_calibration_error=", err)
        return err

    def make_robust_ts_metric(
        metric_fn,
        num_splits: int,
        std_coeff: float,
        greater_is_better: bool,
        min_samples_per_split: int = 100,
        ensure_enough_classes: bool = False,
        verbose: int = 0,
    ):
        """Wrap a metric to evaluate across consecutive time splits.

        Returns mean(metric_values) ± std(metric_values) * std_coeff
        where ± is + if greater_is_better=False (penalize variance for minimization)
              and - if greater_is_better=True (penalize variance for maximization)
        """

        def robust_metric(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):
            n = len(y_true)

            # Fallback 1: Not enough data for any splits
            if n < min_samples_per_split:
                if verbose:
                    logger.info(f"make_robust_ts_metric: n={n} < min_samples_per_split={min_samples_per_split}, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Compute actual number of splits we can do
            actual_splits = min(num_splits, n // min_samples_per_split)

            # Fallback 2: Can only do 1 split
            if actual_splits <= 1:
                if verbose:
                    logger.info(f"make_robust_ts_metric: actual_splits={actual_splits} <= 1, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Split into consecutive intervals
            split_size = n // actual_splits
            values = []

            for i in range(actual_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < actual_splits - 1 else n

                y_true_split = y_true[start_idx:end_idx]
                y_score_split = y_score[start_idx:end_idx]

                # Skip split if not enough samples
                if len(y_true_split) < min_samples_per_split:
                    if verbose:
                        logger.info(f"make_robust_ts_metric: split {i} skipped, len={len(y_true_split)} < {min_samples_per_split}")
                    continue

                # Skip split if single class (classification only)
                if ensure_enough_classes and len(np.unique(y_true_split)) < 2:
                    if verbose:
                        logger.info(f"make_robust_ts_metric: split {i} skipped, single class in y_true")
                    continue

                val = metric_fn(y_true_split, y_score_split, *args, **kwargs)
                if not np.isnan(val):
                    values.append(val)

            # Fallback 3: No valid splits computed
            if len(values) == 0:
                if verbose:
                    logger.info("make_robust_ts_metric: no valid splits, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Fallback 4: Only one valid split
            if len(values) == 1:
                if verbose:
                    logger.info(f"make_robust_ts_metric: only 1 valid split, returning {values[0]:.6f}")
                return values[0]

            mean_val = np.mean(values)
            std_val = np.std(values)

            if verbose:
                logger.info(f"make_robust_ts_metric: {len(values)} splits, mean={mean_val:.6f}, std={std_val:.6f}")

            # Penalize high variance
            if greater_is_better:
                # For maximization: subtract std penalty (lower result = worse)
                return mean_val - std_val * std_coeff
            else:
                # For minimization: add std penalty (higher result = worse)
                return mean_val + std_val * std_coeff

        return robust_metric

    if subgroups:

        def final_integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):  # partial won't work with xgboost
            return robust_mlperf_metric(
                y_true,
                y_score,
                *args,
                metric=integral_calibration_error,
                higher_is_better=False,
                subgroups=subgroups,
                **kwargs,
            )

    else:
        final_integral_calibration_error = integral_calibration_error

    # Apply robustness wrapper if enabled
    if robustness_num_ts_splits > 0:
        final_integral_calibration_error = make_robust_ts_metric(
            final_integral_calibration_error,
            num_splits=robustness_num_ts_splits,
            std_coeff=robustness_std_coeff,
            greater_is_better=robustness_greater_is_better,
            ensure_enough_classes=True,  # ICE is for classification
            verbose=verbose,
        )

    def fs_and_hpt_integral_calibration_error(*args, verbose: bool = True, **kwargs):
        err = compute_probabilistic_multiclass_error(
            *args,
            **kwargs,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        return err

    XGB_CALIB_CLASSIF = XGB_GENERAL_CLASSIF.copy()
    XGB_CALIB_CLASSIF.update({"eval_metric": final_integral_calibration_error})

    def lgbm_integral_calibration_error(y_true, y_score):
        metric_name = "integral_calibration_error"
        value = final_integral_calibration_error(y_true, y_score)
        higher_is_better = False
        return metric_name, value, higher_is_better

    CB_CALIB_CLASSIF = CB_CLASSIF.copy()
    # 2026-04-24 Session 6: ICE custom-metric only works for single-target
    # CB objectives (binary/multiclass). For MultiLogloss (multilabel), CB
    # asserts the custom metric inherits from MultiTargetCustomMetric. Until
    # we ship a multi-target ICE variant, fall back to HammingLoss for
    # multilabel — same as CB_CLASSIF (so calibrated path == base path).
    if _resolved_tt.is_classification and not _resolved_tt.is_binary and CB_CLASSIF.get("loss_function") == "MultiLogloss":
        # eval_metric already set to HammingLoss above; keep it.
        pass
    else:
        CB_CALIB_CLASSIF.update({"eval_metric": ICE(metric=final_integral_calibration_error, higher_is_better=False, max_arr_size=0)})

    LGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        early_stopping_rounds=early_stopping_rounds,
        device_type="cuda" if has_gpu else "cpu",
        random_state=random_seed,
        # histogram_pool_size=16384,
    )
    LGB_GENERAL_PARAMS.update(lgb_kwargs)
    # Target-type-aware objective for LGB (no separate _CLASSIF variant).
    if _resolved_tt.is_classification and not _resolved_tt.is_binary:
        _lgb_obj = _classif_objective_kwargs("lightgbm", _resolved_tt, n_classes)
        if _lgb_obj:
            LGB_GENERAL_PARAMS.update(_lgb_obj)
            LGB_GENERAL_PARAMS["_mlframe_target_type"] = str(_resolved_tt.value)

    NGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
    )
    NGB_GENERAL_PARAMS.update(ngb_kwargs)

    mlp_trainer_params: dict = dict(
        devices=1,  # Always use single device by default to avoid multi-GPU complexity
        # ----------------------------------------------------------------------------------------------------------------------
        # Runtime:
        # ----------------------------------------------------------------------------------------------------------------------
        min_epochs=1,
        max_epochs=iterations,
        max_time={"days": 0, "hours": 0, "minutes": 30},
        # max_steps=1,
        # ----------------------------------------------------------------------------------------------------------------------
        # Intervals:
        # ----------------------------------------------------------------------------------------------------------------------
        check_val_every_n_epoch=1,
        # val_check_interval=val_check_interval,
        # log_every_n_steps=log_every_n_steps,
        # ----------------------------------------------------------------------------------------------------------------------
        # Flags:
        # ----------------------------------------------------------------------------------------------------------------------
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        # ----------------------------------------------------------------------------------------------------------------------
        # Precision & accelerators:
        # ----------------------------------------------------------------------------------------------------------------------
        precision="32-true",
        num_nodes=1,
        # ----------------------------------------------------------------------------------------------------------------------
        # Logging:
        # ----------------------------------------------------------------------------------------------------------------------
        default_root_dir="logs",
    )

    if mlp_kwargs:
        mlp_trainer_params.update(mlp_kwargs.get("trainer_params", {}))

    # Lazy imports — only paid when MLP configs are actually being built.
    import torch
    import torch.nn.functional as F
    from mlframe.lightninglib import MLPTorchModel, TorchDataModule

    # Default loss function and dtype (classification)
    loss_fn = F.cross_entropy
    labels_dtype = torch.int64

    mlp_model_params = dict(
        loss_fn=loss_fn,
        learning_rate=1e-3,
        l1_alpha=0.0,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
    )
    if mlp_kwargs:
        mlp_model_params.update(mlp_kwargs.get("model_params", {}))

    mlp_dataloader_params = dict(
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=None,
        persistent_workers=False,
        batch_size=1024,
        shuffle=False,
    )
    if mlp_kwargs:
        mlp_dataloader_params.update(mlp_kwargs.get("dataloader_params", {}))

    mlp_datamodule_params = dict(
        read_fcn=None, data_placement_device=None, features_dtype=torch.float32, labels_dtype=labels_dtype, dataloader_params=mlp_dataloader_params
    )
    if mlp_kwargs:
        mlp_datamodule_params.update(mlp_kwargs.get("datamodule_params", {}))

    MLP_GENERAL_PARAMS = dict(
        model_class=MLPTorchModel,
        model_params=mlp_model_params,
        datamodule_class=TorchDataModule,
        datamodule_params=mlp_datamodule_params,  # includes dataloader_params
        trainer_params=mlp_trainer_params,
        use_swa=mlp_kwargs.get("use_swa", False) if mlp_kwargs else False,
        swa_params=(
            mlp_kwargs.get("swa_params", dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4))
            if mlp_kwargs
            else dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4)
        ),
        tune_params=mlp_kwargs.get("tune_params", False) if mlp_kwargs else False,
        float32_matmul_precision=mlp_kwargs.get("float32_matmul_precision", None) if mlp_kwargs else None,
        early_stopping_rounds=early_stopping_rounds,
    )

    if rfecv_kwargs is None:
        rfecv_kwargs = {}
    else:
        rfecv_kwargs = rfecv_kwargs.copy()

    cv = rfecv_kwargs.get("cv")
    if not cv:
        if has_time:
            cv = TimeSeriesSplit(n_splits=rfecv_kwargs.get("cv_n_splits", 3))
            logger.info(f"Using TimeSeriesSplit for RFECV...")
        else:
            cv = None
        rfecv_kwargs["cv"] = cv

    if "cv_n_splits" in rfecv_kwargs:
        del rfecv_kwargs["cv_n_splits"]

    COMMON_RFECV_PARAMS = dict(
        early_stopping_rounds=early_stopping_rounds,
        cv=cv,
        cv_shuffle=not has_time,
    )
    COMMON_RFECV_PARAMS.update(rfecv_kwargs)

    # If ES is disabled (early_stopping_rounds=None), strip the key from every per-model
    # constructor-params dict so backends don't register an ES callback.
    # - LGB: omitted from constructor → LightGBMSklearn skips ES on fit
    # - XGB: omitted from constructor → no early_stopping_rounds passed
    # - CB:  omitted → CatBoost runs full iterations (no od_type)
    # - HGB: replace n_iter_no_change with iterations+1 so ES condition never trips
    if early_stopping_disabled:
        for _params in (CB_GENERAL_PARAMS, CB_REGR, CB_CLASSIF, CB_CALIB_CLASSIF,
                        LGB_GENERAL_PARAMS, XGB_GENERAL_PARAMS,
                        XGB_GENERAL_CLASSIF, XGB_CALIB_CLASSIF,
                        MLP_GENERAL_PARAMS, COMMON_RFECV_PARAMS):
            _params.pop("early_stopping_rounds", None)
        # HGB uses early_stopping=True + n_iter_no_change; force ES off explicitly
        HGB_GENERAL_PARAMS["early_stopping"] = False
        HGB_GENERAL_PARAMS.pop("n_iter_no_change", None)

    return SimpleNamespace(
        integral_calibration_error=integral_calibration_error,
        final_integral_calibration_error=final_integral_calibration_error,
        lgbm_integral_calibration_error=lgbm_integral_calibration_error,
        fs_and_hpt_integral_calibration_error=fs_and_hpt_integral_calibration_error,
        CB_GENERAL_PARAMS=CB_GENERAL_PARAMS,
        CB_REGR=CB_REGR,
        CB_CLASSIF=CB_CLASSIF,
        CB_CALIB_CLASSIF=CB_CALIB_CLASSIF,
        HGB_GENERAL_PARAMS=HGB_GENERAL_PARAMS,
        LGB_GENERAL_PARAMS=LGB_GENERAL_PARAMS,
        XGB_GENERAL_PARAMS=XGB_GENERAL_PARAMS,
        XGB_GENERAL_CLASSIF=XGB_GENERAL_CLASSIF,
        XGB_CALIB_CLASSIF=XGB_CALIB_CLASSIF,
        COMMON_RFECV_PARAMS=COMMON_RFECV_PARAMS,
        MLP_GENERAL_PARAMS=MLP_GENERAL_PARAMS,
        NGB_GENERAL_PARAMS=NGB_GENERAL_PARAMS,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Set Feature Statistics
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_trainset_features_stats(train_df: pd.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables"""
    res = {}
    num_cols = get_numeric_columns(train_df)
    if num_cols:
        if len(num_cols) == train_df.shape[1]:
            res["min"] = train_df.min(axis=0)
            res["max"] = train_df.max(axis=0)
        else:
            # TypeError: Categorical is not ordered for operation min. you can use .as_ordered() to change the Categorical to an ordered one.
            res["min"] = pd.Series({col: train_df[col].min() for col in num_cols})
            res["max"] = pd.Series({col: train_df[col].max() for col in num_cols})

    cat_cols = get_categorical_columns(train_df, include_string=False)
    if cat_cols:
        cat_vals = {}
        for col in tqdmu(cat_cols, desc="cat vars stats", leave=False):
            unique_vals = train_df[col].unique()
            if not max_ncats_to_track or (len(unique_vals) <= max_ncats_to_track):
                cat_vals[col] = unique_vals
        res["cat_vals"] = cat_vals
    return res


def get_trainset_features_stats_polars(train_df: pl.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables using Polars.

    Uses lazy mode and selectors for parallel computation.

    Args:
        train_df: Polars DataFrame
        max_ncats_to_track: Max unique values to track for categorical columns

    Returns:
        dict with "min", "max" (as pd.Series) and "cat_vals" (dict of arrays)
    """

    res = {}
    lf = train_df.lazy()

    # Compute numeric min/max and categorical n_unique in a single parallel select
    stats = lf.select(
        # Numeric: min and max
        cs.numeric().min().name.suffix("__min"),
        cs.numeric().max().name.suffix("__max"),
        # Categorical: n_unique to filter before getting unique values
        cs.by_dtype(pl.String, pl.Categorical).n_unique().name.suffix("__n_unique"),
    ).collect()

    # Extract numeric stats
    if len(stats.columns) > 0:
        mins = {}
        maxs = {}
        for col in stats.columns:
            if col.endswith("__min"):
                orig_col = col[:-5]
                mins[orig_col] = stats[col][0]
            elif col.endswith("__max"):
                orig_col = col[:-5]
                maxs[orig_col] = stats[col][0]

        if mins:
            res["min"] = pd.Series(mins)
        if maxs:
            res["max"] = pd.Series(maxs)

    # Extract categorical columns that are under the threshold
    cat_cols_to_fetch = []
    for col in stats.columns:
        if col.endswith("__n_unique"):
            orig_col = col[:-10]
            n_unique = stats[col][0]
            if not max_ncats_to_track or n_unique <= max_ncats_to_track:
                cat_cols_to_fetch.append(orig_col)

    # Get unique values for qualifying categorical columns
    if cat_cols_to_fetch:
        cat_vals = {}
        for col in cat_cols_to_fetch:
            cat_vals[col] = lf.select(pl.col(col).unique()).collect()[col].to_numpy()
        res["cat_vals"] = cat_vals

    return res


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Callback Classes for Training Monitoring
# -----------------------------------------------------------------------------------------------------------------------------------------------------


class UniversalCallback:
    def __init__(
        self,
        time_budget_mins: Optional[float] = None,
        reporting_interval_mins: Optional[float] = 1.0,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        monitor_dataset: Optional[str] = None,
        monitor_metric: Optional[str] = None,
        mode: Optional[str] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        ndigits: int = 6,
        verbose: int = 1,
    ) -> None:

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        self.start_time = None
        self.best_metric = None
        self.first_iteration = True
        self.iterations_since_improvement = 0
        self.metric_history: Dict[str, Dict[str, List[float]]] = {}
        self.stop_flag = stop_flag if stop_flag is not None else lambda: False

        # Call super().__init__() to ensure proper MRO chain initialization.
        # For XGBoostCallback(UniversalCallback, TrainingCallback), this calls
        # TrainingCallback.__init__(), which is required by XGBoost >= 2.x ABC checks.
        super().__init__()

        if self.verbose > 0:
            logger.info(
                "UniversalCallback initialized with params: "
                f"time_budget_mins={time_budget_mins}, patience={patience}, min_delta={min_delta}, "
                f"monitor_dataset={monitor_dataset}, monitor_metric={monitor_metric}, mode={mode}"
            )

    def on_start(self) -> None:
        self.start_time = timer()
        if self.verbose > 0:
            self.last_reporting_ts = self.start_time
            logger.info(f"Training started. Timer initiated. RAM usage {get_own_memory_usage():.1f}GB.")

    def update_history(self, metrics_dict: Dict[str, Dict[str, float]]) -> None:
        for dataset in metrics_dict:
            if dataset not in self.metric_history:
                self.metric_history[dataset] = {}
            for metric, value in metrics_dict[dataset].items():
                self.metric_history[dataset].setdefault(metric, []).append(value)
        if self.verbose > 1:
            logger.debug(f"Updated metric history: {metrics_dict}")

    def derive_mode(self, metric_name: str) -> str:
        known_metric_modes = {
            "auc": "max",
            "accuracy": "max",
            "acc": "max",
            "f1": "max",
            "map": "max",
            "ndcg": "max",
            "ice": "min",
            "mae": "min",
            "mse": "min",
            "mape": "min",
            "rmse": "min",
            "logloss": "min",
            "error": "min",
            "loss": "min",
        }

        name = metric_name.lower()
        for key, default_mode in known_metric_modes.items():
            if key == name:
                return default_mode
        if "score" in name or "auc" in name or "accuracy" in name:
            return "max"
        elif "loss" in name or "error" in name:
            return "min"
        elif name.endswith("e"):
            return "min"
        else:
            logger.warning(f"Unsure about correct optimization mode for metric={name}, using min for now.")
            return "min"  # fallback default

    def set_default_monitor_metric(self, metrics_dict: Dict[str, Dict[str, float]]) -> None:
        if self.monitor_dataset not in metrics_dict:
            raise ValueError(f"Monitor dataset '{self.monitor_dataset}' not found in metrics.")
        available_metrics = list(metrics_dict[self.monitor_dataset].keys())
        logger.info(f"available_metrics={available_metrics}")
        for preferred in ["ICE", "integral_calibration_error", "auc", "AUC"]:
            if preferred in available_metrics:
                self.monitor_metric = preferred
                break
        else:
            self.monitor_metric = available_metrics[0]
        self.mode = self.derive_mode(self.monitor_metric)
        if self.verbose > 0:
            logger.info(f"Auto-selected monitor_metric: {self.monitor_metric}, mode: {self.mode}")

    def _get_state(self, current_value: float) -> str:
        return f"iter={self.iter:_}, {self.monitor_dataset} {self.monitor_metric}: current={current_value:.{self.ndigits}f}, best={self.best_metric:.{self.ndigits}f} @{self.best_iter:_}. RAM usage {get_own_memory_usage():.1f}GB."

    def should_stop(self) -> bool:
        cur_ts = timer()
        if self.time_budget_mins is not None and self.start_time is not None:

            elapsed = cur_ts - self.start_time
            if elapsed > self.time_budget_mins * 60:
                if self.verbose > 0:
                    logger.info(f"Stopping early due to time budget exceeded ({elapsed:.2f} sec).")
                return True

        if self.stop_flag():
            if self.verbose > 0:
                logger.info("Stopping early due to external stop flag.")
            return True

        if self.monitor_dataset in self.metric_history and self.monitor_metric in self.metric_history[self.monitor_dataset]:
            history = self.metric_history[self.monitor_dataset][self.monitor_metric]
            if history:
                current_value = history[-1]
                if self.best_metric is None:
                    self.iter = 0
                    self.best_iter = self.iter
                    self.best_metric = current_value
                    self.iterations_since_improvement = 0
                    if self.verbose > 0:
                        logger.info(f"Initial metric value: {current_value:.{self.ndigits}f}")
                        self.last_reporting_ts = cur_ts
                else:
                    self.iter += 1
                    improved = (self.mode == "min" and current_value < self.best_metric - self.min_delta) or (
                        self.mode == "max" and current_value > self.best_metric + self.min_delta
                    )
                    # Pre-compute reporting condition (used in both branches)
                    should_report = self.verbose > 0 and (
                        not self.reporting_interval_mins or (cur_ts - self.last_reporting_ts) >= self.reporting_interval_mins * 60
                    )
                    if improved:
                        self.best_iter = self.iter
                        self.best_metric = current_value
                        self.iterations_since_improvement = 0
                    else:
                        self.iterations_since_improvement += 1
                    if should_report:
                        logger.info(self._get_state(current_value=current_value))
                        self.last_reporting_ts = cur_ts
                    if self.patience is not None and self.iterations_since_improvement >= self.patience:
                        if self.verbose > 0:
                            logger.info(
                                f"Stopping early due to no improvement for {self.iterations_since_improvement} iterations. {self._get_state(current_value=current_value)}"
                            )
                            self.last_reporting_ts = cur_ts
                        return True
        return False


class LightGBMCallback(UniversalCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "valid_0"

    def __call__(self, env: lgb.callback.CallbackEnv) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False

        metrics_dict = {}
        for dataset, metric, value, _ in env.evaluation_result_list:
            metrics_dict.setdefault(dataset, {})[metric] = value
        self.update_history(metrics_dict)

        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)
        if self.should_stop():
            if hasattr(self, "best_iter"):
                best_iter = self.best_iter
            else:
                best_iter = 0
            raise lgb.callback.EarlyStopException(best_iter, [(dataset, metric, self.best_metric, False)])


class XGBoostCallback(UniversalCallback, TrainingCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "validation_0"

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Dict[str, List[float]]]) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False
        metrics_dict = {dataset: {metric: values[-1] for metric, values in metric_dict.items()} for dataset, metric_dict in evals_log.items()}

        self.update_history(metrics_dict)

        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)

        if self.should_stop():
            if hasattr(self, "best_iter"):
                best_iter = self.best_iter
            else:
                best_iter = 0
            model.set_attr(best_score=self.best_metric, best_iteration=best_iter)
            return True


class CatBoostCallback(UniversalCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "validation"

    def after_iteration(self, info: Any) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False

        metrics_dict = {dataset: {metric: values[-1] for metric, values in metric_dict.items()} for dataset, metric_dict in info.metrics.items()}
        self.update_history(metrics_dict)

        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)
        return not self.should_stop()


__all__ = [
    "parse_catboost_devices",
    "get_training_configs",
    "get_trainset_features_stats",
    "get_trainset_features_stats_polars",
    "UniversalCallback",
    "LightGBMCallback",
    "XGBoostCallback",
    "CatBoostCallback",
    "CUDA_IS_AVAILABLE",
]
