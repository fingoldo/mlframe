"""Classification helpers extracted from ``helpers.py``.

Probability post-processing, multilabel wrapping, classifier chains.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator, clone, is_classifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

logger = logging.getLogger(__name__)

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
            # ``multi_class`` kwarg was deprecated in sklearn 1.7 and removed
            # in 1.8 (passing it raises TypeError on LogisticRegression.__init__).
            # LR auto-detects multi-class from ``y`` since 1.5; explicit
            # ``solver='lbfgs'`` is the only kwarg still meaningful here
            # (defaults are fine but 'lbfgs' is the recommended multinomial solver).
            "linear": {"solver": "lbfgs"},
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


class _ChainEnsemble(ClassifierMixin, BaseEstimator):
    # sklearn 1.x requires ClassifierMixin to come BEFORE BaseEstimator
    # in the MRO so that ClassifierMixin.__sklearn_tags__ correctly
    # propagates ``estimator_type='classifier'`` via ``super()``. With the
    # reverse order, get_tags() (and is_classifier()) return False, and
    # downstream dispatchers (mlframe's report_model_perf) route through
    # the regression report — visible in the fuzz suite as
    # "x and y can be no greater than 2D" matplotlib errors.

    """Soft-voting ensemble of ``ClassifierChain`` instances for multilabel.

    sklearn's ``VotingClassifier(soft)`` does NOT accept multilabel y
    (raises ``ValueError: multilabel-indicator is not supported``). This
    class is a minimal hand-rolled equivalent for multilabel:
    fit each chain on (X, y), then ``predict_proba(X)`` averages the
    per-chain ``(N, K)`` outputs.

    Empirical lift (sklearn docs ``plot_classifier_chain_yeast``):
    +2-5% Jaccard over ``MultiOutputClassifier`` on correlated labels;
    +cost is 3-5× training (one fit per chain).

    Inherits BaseEstimator + ClassifierMixin so it survives
    ``sklearn.base.clone`` (CalibratedClassifierCV, RFECV, GridSearchCV
    all clone their inner estimator before fitting). All __init__ params
    are stored verbatim on self for sklearn's get_params introspection;
    chain construction is deferred to ``fit`` so cloning produces an
    unfitted estimator with the same hyperparameters.

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

    def __init__(self, base_estimator=None, n_labels=None, n_chains=3, seeds=None,
                 order_strategy="random", user_orders=None, cv=5):
        # All params stored as plain attributes -- sklearn BaseEstimator
        # reads them back via get_params() for cloning. Defaults given to
        # `base_estimator` / `n_labels` so that sklearn dispatchers that
        # introspect via `inspect.signature(_ChainEnsemble)` (RFECV,
        # CalibratedClassifierCV's clone path, sklearn.tests probes) do not
        # raise `TypeError: missing 2 required positional arguments` on a
        # bare introspection call. `fit` still raises on None to keep the
        # contract loud at use-time.
        self.base_estimator = base_estimator
        self.n_labels = n_labels
        self.n_chains = n_chains
        self.seeds = seeds
        self.order_strategy = order_strategy
        self.user_orders = user_orders
        self.cv = cv

    def fit(self, X, y, **fit_params):
        # Validate required state at fit-time (defaults on __init__ are for
        # sklearn introspection compliance; actually fitting with None
        # base_estimator / n_labels is a contract violation we surface
        # explicitly rather than letting `clone(None)` raise a confusing
        # AttributeError deep inside sklearn.).
        if self.base_estimator is None:
            raise ValueError("_ChainEnsemble.fit: base_estimator must be set")
        if self.n_labels is None:
            raise ValueError("_ChainEnsemble.fit: n_labels must be set")

        # Resolve seeds + per-chain orders lazily at fit time so that
        # cloning (which calls __init__ with the params get_params returned)
        # produces a fresh, unfitted estimator without inheriting cached
        # state from the parent.
        seeds_resolved = (
            self.seeds if self.seeds is not None
            else list(range(self.n_chains))
        )
        # by_frequency needs y; other strategies can be resolved without it.
        orders_resolved = _compute_chain_orders(
            self.n_labels, self.n_chains,
            order_strategy=self.order_strategy,
            user_orders=self.user_orders, seeds=seeds_resolved,
            y=y if self.order_strategy == "by_frequency" else None,
        )
        self.chains_ = [
            ClassifierChain(
                clone(self.base_estimator),
                order=orders_resolved[i].tolist(),
                cv=self.cv, random_state=seeds_resolved[i],
            )
            for i in range(self.n_chains)
        ]

        # ClassifierChain.fit in sklearn 1.x rejects extra kwargs unless
        # `enable_metadata_routing=True` is set globally. Callers commonly
        # pass eval_set / eval_metric / X_val / y_val for inner-model early
        # stopping — those make no sense for ClassifierChain (the chain
        # cross-validates internally via cv=) and can't be routed cleanly.
        # Drop them; the inner estimator's hyperparameters are already
        # baked in via clone(), which is the only thing fit_params used to
        # influence in this codepath.
        for chain in self.chains_:
            chain.fit(X, y)
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
        # chains_ is created in fit(); pre-fit (e.g. on a freshly-cloned
        # instance) the attribute won't exist yet.
        chains = getattr(self, "chains_", None)
        if not chains:
            return False
        return all(hasattr(c, "estimators_") for c in chains)



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


