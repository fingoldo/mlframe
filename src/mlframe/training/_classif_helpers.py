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

    Public alias: ``mlframe.training.canonical_predict_proba_shape`` (re-exported
    without the leading underscore; that name carries the stability contract).

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

    Public alias: ``mlframe.training.predict_from_probs`` (re-exported without
    the leading underscore; that name carries the stability contract).

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
        # Wave 21 P1: if a buggy estimator emits NaN probas, np.argmax picks
        # the NaN column -> silent misclassification for that row. Detect
        # any all-NaN row and WARN; nanargmax raises if a row is all-NaN
        # which loudly surfaces the data corruption.
        if np.any(~np.isfinite(arr)):
            _bad_rows = int(np.sum(~np.isfinite(arr).all(axis=1)))
            if _bad_rows > 0:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "argmax on probabilistic predictions: %d row(s) contain "
                    "non-finite probabilities; using nanargmax (will raise "
                    "on any all-NaN row).", _bad_rows,
                )
            idx = np.nanargmax(arr, axis=1)
        else:
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
        # OPT-5 (2026-05-23): pick n_jobs based on expected per-label work,
        # not just n_labels. joblib's loky backend pays ~500ms per worker
        # spawn (pickle X, fork process, import estimator deps); for small K
        # (3-4) AND small N (<50k) this overhead exceeds the parallel win.
        # Bench (HistGradientBoostingClassifier max_iter=5):
        #   n=50000  K= 3: parallel=6.24s  manual-loop=0.54s (11.5x SLOWER parallel)
        #   n=200000 K= 3: parallel=3.37s  manual-loop=8.01s (2.4x parallel win)
        #   n=50000  K=10: parallel=6.68s  manual-loop=3.82s (1.7x SLOWER parallel)
        #   n=200000 K=10: parallel=6.47s  manual-loop=7.53s (~equal)
        #   n=50000  K=50: parallel=2.54s  manual-loop=6.71s (2.6x parallel win)
        # Pre-fix the unconditional ``min(K, cpu//2)`` always spawned the
        # worker pool, paying spawn overhead even when K=3+small-n means
        # sequential wins. The new rule routes tiny workloads to n_jobs=1.
        # SAVING: c0023 iter190 attributed 90s to time.sleep waiting on
        # joblib workers for K=3 multilabel at 200k -- expected ~10-30s
        # saved depending on inner estimator's per-label fit time.
        n_jobs = _auto_wrapper_n_jobs(n_labels=n_labels)
    return MultiOutputClassifier(estimator, n_jobs=n_jobs)


def _auto_wrapper_n_jobs(n_labels: Optional[int]) -> int:
    """OPT-5 (2026-05-23): smart ``wrapper_n_jobs='auto'`` resolver.

    Returns 1 (sequential, no joblib spawn overhead) when the workload is
    too small to amortize loky's process-fork + pickle-X cost (~500ms /
    worker). Returns ``min(K, cpu//2)`` for large workloads where parallel
    actually wins.

    This is a heuristic on n_labels alone -- a more sophisticated form
    would also factor in n_rows + inner-estimator class. Conservative
    choice: prefer sequential whenever K <= 4 (covers c0023 / c0095 /
    c0062 / c0123 multilabel shapes observed in fuzz). For K >= 5 the
    spawn cost amortises across enough labels to recover parallel win.

    Override via ``MultilabelDispatchConfig.wrapper_n_jobs=<int>`` -- the
    auto path only activates on the string ``"auto"`` default.
    """
    if n_labels is None or n_labels <= 4:
        return 1
    import os
    cpu = os.cpu_count() or 1
    return min(n_labels, max(1, cpu // 2))


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
        #
        # NaN guard: sklearn's ClassifierChain wraps the base estimator
        # but calls ``_validate_data(X, ensure_all_finite=True)`` BEFORE
        # delegating to the base. Even NaN-tolerant bases (HGB / LGB /
        # XGB / CB) cannot rescue NaN cells from this pre-check, and the
        # raise is opaque ("ClassifierChain does not accept missing
        # values encoded as NaN natively. ..."). When X carries any NaN
        # cells, impute them column-wise with the per-column median
        # before fit; for non-numeric or empty columns fall back to 0.
        # WARNed loud so the operator sees the silent imputation.
        # Surfaced 2026-05-20 by fuzz combo c0029 (inject_inf_nan=True,
        # cb_hgb_mlp, multilabel_classification chain dispatch).
        _x_for_fit = X
        try:
            import numpy as _np
            import pandas as _pd
            _arr = _x_for_fit.to_numpy() if isinstance(_x_for_fit, _pd.DataFrame) else _np.asarray(_x_for_fit)
            if _arr.dtype.kind == "f" and not _np.all(_np.isfinite(_arr[~_np.isnan(_arr)])):
                pass  # inf already gone here; nan handled below
            _has_nan = (_arr.dtype.kind == "f") and bool(_np.isnan(_arr).any())
        except Exception:
            _has_nan = False
        if _has_nan:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "_ChainEnsemble.fit: input X contains NaN cells; sklearn "
                "ClassifierChain refuses them pre-base-estimator. "
                "Imputing column-wise median (fallback 0 for all-NaN cols) "
                "before chain fit. Upstream preprocessing should fill these "
                "earlier - see PreprocessingConfig.fix_infinities / impute."
            )
            if isinstance(_x_for_fit, _pd.DataFrame):
                # Shallow copy: only float columns carrying NaN are imputed below; deep-copying a 100+ GB frame to fill a few columns OOMs. ``deep=False`` shares untouched buffers, caller frame unmutated.
                _x_for_fit = _x_for_fit.copy(deep=False)
                for _c in _x_for_fit.columns:
                    _s = _x_for_fit[_c]
                    if _s.dtype.kind == "f" and _s.isna().any():
                        _med = _s.median(skipna=True)
                        _x_for_fit[_c] = _s.fillna(0.0 if _np.isnan(_med) else _med)
            else:
                _x_for_fit = _arr.copy()
                for _j in range(_x_for_fit.shape[1]):
                    _col = _x_for_fit[:, _j]
                    _mask = _np.isnan(_col)
                    if _mask.any():
                        _finite = _col[~_mask]
                        _fill = 0.0 if _finite.size == 0 else float(_np.median(_finite))
                        _x_for_fit[_mask, _j] = _fill
        for chain in self.chains_:
            chain.fit(_x_for_fit, y)
        # Mirror sklearn estimator API.
        self.classes_ = self.chains_[0].classes_
        # Prime the predict-time NaN guard with TRAIN-frame stats so a
        # NaN-bearing predict frame is imputed with train medians/scale rather
        # than refused (NanGuardNotPrimedError) or fit-on-the-predict-frame
        # (test-set leak). The chain imputes its OWN fit X above, but the outer
        # guard (_apply_nan_guard) had no persisted imputer/scaler. Surfaced by
        # fuzz combo c0146 (multilabel chain + inject_inf_nan). Best-effort: a
        # priming edge-case must never block an otherwise-successful fit.
        try:
            from ._predict_guards import prime_nan_guard_stats
            prime_nan_guard_stats(self, X)
        except Exception as _e:  # pragma: no cover - defensive
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "_ChainEnsemble.fit: NaN-guard priming failed (%s); a NaN-bearing "
                "predict frame may still raise. Non-fatal.", _e,
            )
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


