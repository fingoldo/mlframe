"""Feature-importance helpers carved out of evaluation.py.

Hosts the 5 internal FI sources + 2 public entry points
(get_model_feature_importances / plot_model_feature_importances)
to drop the parent evaluation.py below the 1k-LOC monolith threshold.
evaluation re-exports the public symbols so existing
from mlframe.training.evaluation import plot_model_feature_importances
imports keep working.

Internal helpers stay private (underscore-prefixed) so the re-export at
the bottom of evaluation only surfaces the public-API surface.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from mlframe.feature_selection.importance import plot_feature_importance

logger = logging.getLogger(__name__)

# FI bar-plot figsize. Half the perf-chart figsize so the FI panels
# don't dominate the suite report. Originally defined in
# evaluation.py at module top; carved alongside its only consumer.
DEFAULT_FI_FIGSIZE = (7.5, 2.5)

# Permutation-importance fallback caps. Used when the inner estimator has
# neither ``feature_importances_`` nor ``coef_`` (PyTorch-Lightning MLP,
# Keras nets, sklearn pipelines wrapping such regressors). We cap the
# subsample size and the number of shuffles per feature so the cost stays
# bounded on a 4M-row TVT regression with 200+ features (worst case
# ~PERM_N_SAMPLES * len(columns) * PERM_N_REPEATS predict calls).
_PERM_FI_MAX_SAMPLES: int = 5000
_PERM_FI_N_REPEATS: int = 3
_PERM_FI_RANDOM_STATE: int = 0


def _unwrap_estimator_chain(model: Any) -> Any:
    """Walk standard sklearn / mlframe wrappers to the inner estimator.

    Handles ``Pipeline`` (final step), ``TransformedTargetRegressor``
    (``.regressor_`` / ``.regressor``), and meta-estimator attributes
    (``estimator_``, ``base_estimator_``). Stops as soon as we hit
    something that exposes a native FI / coef attribute -- the caller's
    permutation fallback uses ``model`` as-is so unwrapping past a
    PyTorch regressor would break the predict signature.
    """
    seen = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "feature_importances_") or hasattr(cur, "coef_"):
            return cur
        if isinstance(cur, Pipeline):
            if not cur.steps:
                return cur
            cur = cur.steps[-1][1]
            continue
        for attr in ("regressor_", "regressor", "estimator_", "base_estimator_", "best_estimator_"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and id(nxt) not in seen:
                cur = nxt
                break
        else:
            return cur
    return cur


def _permutation_feature_importances(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_repeats: int = _PERM_FI_N_REPEATS,
    max_samples: int = _PERM_FI_MAX_SAMPLES,
    random_state: int = _PERM_FI_RANDOM_STATE,
    return_std: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Permutation importance for estimators without native FI / coef.

    Falls back to ``sklearn.inspection.permutation_importance`` on a
    capped row sample. Returns ``None`` if the model can't predict on
    the supplied X (wrong column set, optional dep missing) so the
    caller logs + omits the chart rather than crashing the whole
    report.

    ``return_std=True`` returns ``(mean, std)`` so the FI chart can draw
    per-feature error-bar whiskers; ``std`` is the per-repeat dispersion
    from ``permutation_importance``. The None failure case becomes
    ``(None, None)`` to keep the tuple shape stable for the caller.
    """
    # 2026-05-27: ensemble aggregator entries (EnsARITHM/HARM/MEDIAN/...)
    # arrive here with ``model=None`` because the per-member voting
    # logic doesn't expose a sklearn-style ``predict`` boundary. sklearn
    # permutation_importance then raises "estimator parameter must be an
    # object implementing 'fit'. Got None instead." Short-circuit with a
    # DEBUG-level note (was WARN, which spammed the log 6 times per
    # target -- once per ensemble flavour).
    _fail = (lambda: (None, None)) if return_std else (lambda: None)
    if model is None:
        logger.debug("permutation_importance skipped: model is None (ensemble " "aggregator without sklearn-style estimator surface).")
        return _fail()
    try:
        from sklearn.inspection import permutation_importance
    except Exception:
        logger.warning("permutation_importance unavailable; skipping FI for non-native estimator.")
        return _fail()
    try:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    except Exception as exc:
        logger.debug("permutation FI: X/y coercion to ndarray failed; skipping FI: %r", exc, exc_info=True)
        return _fail()
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[0] != y_arr.shape[0]:
        return _fail()
    n = X_arr.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X_arr[idx]
        y_sub = y_arr[idx]
    else:
        X_sub = X_arr
        y_sub = y_arr
    # Adaptive scorer: PyTorch-Lightning / Keras / custom predict-only
    # wrappers usually lack a sklearn-style ``.score(X, y)`` method, so
    # ``permutation_importance`` would raise "estimator does not have a
    # 'score' method". Supplying an explicit callable that dispatches
    # to r2_score (regression) or accuracy_score (classification) based
    # on predict's dtype keeps both paths working without the caller
    # threading a task hint.
    def _adaptive_scorer(estimator, X, y):
        try:
            # CatBoost.predict() flips its input ndarray to read-only; sklearn reuses one X_permuted buffer and
            # shuffles it in place across n_repeats, so predicting on it directly makes the next shuffle raise
            # "assignment destination is read-only". Predict on a private copy to keep sklearn's buffer writeable.
            preds = estimator.predict(np.array(X, copy=True) if isinstance(X, np.ndarray) else X)
        except Exception as exc:
            logger.debug("permutation FI: adaptive scorer predict failed; scoring as -inf: %r", exc, exc_info=True)
            return -np.inf
        preds_arr = np.asarray(preds)
        y_arr_local = np.asarray(y)
        if preds_arr.dtype.kind == "f" and y_arr_local.dtype.kind in {"f", "i", "u"}:
            from mlframe.metrics.core import fast_r2_score
            try:
                return float(fast_r2_score(y_arr_local, preds_arr))
            except Exception as exc:
                logger.debug("permutation FI: fast_r2_score scoring failed; scoring as -inf: %r", exc, exc_info=True)
                return -np.inf
        from mlframe.metrics.core import accuracy_ratio
        try:
            return float(accuracy_ratio(y_arr_local, preds_arr))
        except Exception as exc:
            logger.debug("permutation FI: accuracy_ratio scoring failed; scoring as -inf: %r", exc, exc_info=True)
            return -np.inf
    # Threading backend (NOT loky / multiprocessing):
    # ``profiling/bench_permutation_fi_nn.py`` (2026-05-26) measured a
    # ~1.4x median speedup across 6 NN-FI scenarios when
    # ``permutation_importance`` runs inside ``joblib.parallel_backend
    # ('threading')`` vs n_jobs=1. The same sweep showed loky/process
    # backend was 3-4x SLOWER for PyTorch estimators on Windows
    # (worker spawn ~2s each × cpu_count + per-task model pickle
    # dominates wall time; n_jobs=-1 with loky on a 200-feature
    # 5k-row task: 18.2s vs 5.7s baseline). Threading wins because
    # PyTorch matmul releases the GIL in its C++ kernels and there is
    # no model / X pickling cost. RSS overhead measured at <50 MB on
    # the same shape.
    try:
        import joblib
        with joblib.parallel_backend("threading", n_jobs=-1):
            result = permutation_importance(
                model, X_sub, y_sub,
                scoring=_adaptive_scorer,
                n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
            )
    except Exception as exc:
        logger.warning("permutation_importance failed (%s); skipping FI.", exc)
        return _fail()
    mean = np.asarray(result.importances_mean, dtype=np.float64)
    if return_std:
        std = np.asarray(getattr(result, "importances_std", np.zeros_like(mean)), dtype=np.float64)
        return mean, std
    return mean


def _cuda_batched_permutation_importance(
    net,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_repeats: int = _PERM_FI_N_REPEATS,
    max_samples: int = _PERM_FI_MAX_SAMPLES,
    chunk_size: int = 16,
    random_state: int = _PERM_FI_RANDOM_STATE,
    return_std: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """GPU-batched permutation importance for a torch.nn.Module.

    Strategy: ship X to CUDA once, then for each (feature, repeat)
    permute column j ON DEVICE, predict on the full chunked batch in
    one forward pass, decode scores. Bench wins 2-4x over the
    threading-backend permutation when n_features >= 50 and n_repeats
    >= 2 (warmup amortisation).

    Constraints / when to skip:
    - Requires the net to accept the raw feature tensor (skips models
      with non-torch preprocessing in front).
    - Loses on tiny n_features / n_repeats (warmup not amortised).
    - Returns None on any error so callers fall back to the CPU path.

    bench-attempt-rejected (2026-05-26, profiling/bench_cuda_perm_-
    variants.py on GTX 1050 Ti / 4GB): tried three optimisations on
    top of this kernel, none reliably beat current. Speedups vs
    current across 6 scenarios:
      V2 (torch.randperm on device, no CPU rng + H2D for indices):
         0.65-1.26x; lost in 4/6. Launching one randperm kernel per
         (feature, repeat) has more overhead than the ~40 KB H2D
         the optimisation was meant to avoid.
      V3 (adaptive chunk_size from torch.cuda.mem_get_info):
         0.89-1.11x; lost or tied in 5/6. On a 4 GB GPU the adaptive
         number ends up close to 16 anyway; bigger chunks (capped 64)
         buy almost nothing.
      V4 (gather: stack k*n_repeats permutations as one matrix and
         write columns via fancy indexing): 0.63-1.29x; mixed.
         Wins biggest on the moderate-n_repeats / moderate-n_features
         band, regresses elsewhere because the per-slot ``arange``
         allocation overhead exceeds the gather-vs-write win at
         high n_repeats.
      V5 (V2 + V3 + V4 combined): also mixed -- the three
         optimisations do not stack.
    Conclusion: current ``chunk_size=16`` + CPU rng + per-(feature,
    repeat) write is the best simple implementation on the bench
    GPU. Re-bench on different hardware before re-attempting.
    """
    _fail = (lambda: (None, None)) if return_std else (lambda: None)
    try:
        import torch
    except Exception:
        return _fail()
    if not torch.cuda.is_available():
        return _fail()
    try:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    except Exception as exc:
        logger.debug("cuda-batched FI: X/y coercion to ndarray failed; falling back: %r", exc, exc_info=True)
        return _fail()
    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.shape[0]:
        return _fail()
    n, n_features = X_arr.shape
    if n > max_samples:
        rng_sub = np.random.default_rng(random_state)
        idx_sub = rng_sub.choice(n, size=max_samples, replace=False)
        X_arr = X_arr[idx_sub]
        y_arr = y_arr[idx_sub]
        n = max_samples
    rng = np.random.default_rng(random_state)
    device = torch.device("cuda")
    try:
        net = net.to(device)
        net.eval()
        X_t = torch.as_tensor(np.ascontiguousarray(X_arr), dtype=torch.float32, device=device)
        with torch.no_grad():
            baseline_pred = net(X_t).reshape(-1).cpu().numpy().astype(np.float64)
    except Exception as exc:
        logger.warning("cuda-batched FI setup failed (%s); falling back.", exc)
        return _fail()
    from mlframe.metrics.core import fast_r2_score
    baseline_score = float(fast_r2_score(y_arr, baseline_pred))
    importances = np.zeros(n_features, dtype=np.float64)
    importances_std = np.zeros(n_features, dtype=np.float64)
    try:
        with torch.no_grad():
            for chunk_start in range(0, n_features, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_features)
                k = chunk_end - chunk_start
                batched = X_t.repeat(k * n_repeats, 1)
                for slot, j in enumerate(range(chunk_start, chunk_end)):
                    for r in range(n_repeats):
                        offset = (slot * n_repeats + r) * n
                        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long, device=device)
                        batched[offset : offset + n, j] = X_t[perm, j]
                preds = net(batched).reshape(-1).cpu().numpy().astype(np.float64)
                for slot, j in enumerate(range(chunk_start, chunk_end)):
                    scores = np.empty(n_repeats, dtype=np.float64)
                    for r in range(n_repeats):
                        offset = (slot * n_repeats + r) * n
                        scores[r] = fast_r2_score(y_arr, preds[offset : offset + n])
                    importances[j] = baseline_score - scores.mean()
                    # baseline_score is constant, so the per-repeat importance dispersion is just std(scores).
                    importances_std[j] = float(scores.std())
    except Exception as exc:
        logger.warning("cuda-batched FI inner loop failed (%s); falling back.", exc)
        return _fail()
    finally:
        try:
            net.to("cpu")
            torch.cuda.empty_cache()
        except Exception:
            pass
    if return_std:
        return importances, importances_std
    return importances


# Captum IntegratedGradients sample cap. Bench (profiling/bench_mlp_fi_methods.py)
# at n_samples=500 + n_steps=20 matched permutation recall@10=1.00 across the
# 5k/200, 2k/500, 1k/50 scenarios with ~5x speedup on the 500-feature case
# (0.835s vs 5.020s) and ~0.4-0.8x on small ones (overhead-dominated). The
# 500-row cap keeps the per-attribute backward-pass cost bounded on 4M-row
# test splits without losing accuracy (IG converges by row sub-sampling +
# averaging absolute attributions).
_CAPTUM_IG_N_SAMPLES: int = 500
_CAPTUM_IG_N_STEPS: int = 20

# CUDA-batched permutation thresholds. Bench (profiling/bench_permutation_fi_nn.py,
# GTX 1050 Ti, 2026-05-26):
#   5k x 200 features, n_repeats=10: A=11.43s, D=7.81s (1.46x),
#                                    F=2.91s (3.93x faster than A)
#   1k x 200 features, n_repeats=10: A=3.23s,  F=1.44s (2.25x)
#   2k x 500 features, n_repeats=5:  A=10.23s, F=7.13s (1.43x)
# CUDA wins biggest with n_repeats >= 3 (warmup cost amortised) and
# moderate-to-large n_features (per-call overhead matters more than
# raw matmul). Below ``_CUDA_PERM_MIN_FEATURES`` features the warmup
# is not amortised and threading wins.
_CUDA_PERM_MIN_FEATURES: int = 50
_CUDA_PERM_MIN_REPEATS: int = 2


def _torch_module_from_model(model: Any):
    """Locate the ``torch.nn.Module`` core inside a sklearn / Lightning /
    pipeline wrapper. Returns the module or None when not a torch model.
    """
    try:
        import torch.nn as nn
    except Exception:
        return None
    seen: set[int] = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, nn.Module):
            return cur
        # Lightning regressor stashes the trained net on ``network``.
        for attr in ("network", "module", "model_", "net"):
            inner = getattr(cur, attr, None)
            if isinstance(inner, nn.Module):
                return inner
        # Walk the standard wrapper chain.
        nxt = None
        for attr in ("regressor_", "regressor", "estimator_", "base_estimator_"):
            inner = getattr(cur, attr, None)
            if inner is not None and id(inner) not in seen:
                nxt = inner
                break
        if nxt is None and isinstance(cur, Pipeline):
            if cur.steps:
                nxt = cur.steps[-1][1]
        cur = nxt
    return None


def _first_layer_weight_importance(net) -> Optional[np.ndarray]:
    """Ultra-fast ``|W1|.sum(axis=hidden)`` proxy. NO X/y, NO predict
    calls. Reliable only when no feature-mixing layer precedes the
    first ``nn.Linear``; BatchNorm / LayerNorm / Dropout in front are
    fine (they don't mix features).

    Bench (2026-05-26 profiling/bench_mlp_fi_methods.py): ~10000x faster
    than permutation, but recall@10 of true informative features
    dropped to 0.40 on the 5k/200 scenario -- BN gamma/beta
    parameters effectively re-weight features and absorb signal from
    the raw |W| view. Useful as a CHEAP HINT (e.g. for a quick
    pre-screen) but NOT as a primary FI source. Wired behind opt-in
    flag.
    """
    try:
        import torch.nn as nn
    except Exception:
        return None
    if isinstance(net, nn.Sequential):
        layers = list(net.children())
    else:
        layers = list(net.modules())
    for layer in layers:
        if isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, nn.Identity)):
            continue
        if isinstance(layer, nn.Linear):
            return layer.weight.detach().abs().cpu().numpy().sum(axis=0).astype(np.float64)
        if hasattr(layer, "weight"):
            return None  # an unrecognised feature-mixing layer is in front
    return None


def _captum_integrated_gradients_importance(
    net, X: np.ndarray | pd.DataFrame,
    *,
    n_samples: int = _CAPTUM_IG_N_SAMPLES,
    n_steps: int = _CAPTUM_IG_N_STEPS,
    random_state: int = 0,
) -> Optional[np.ndarray]:
    """IntegratedGradients attribution averaged over a row subsample.

    Returns per-input absolute attribution. Bench matched permutation
    recall@10=1.00 on synthetic with known ground truth (200, 500, 50
    feature scenarios). Requires the model to be a single torch
    Module that accepts the raw feature tensor (Lightning regressors
    expose this via ``.network``).
    """
    try:
        import torch
        from captum.attr import IntegratedGradients
    except Exception:
        return None
    try:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    except Exception:
        return None
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        return None
    n = X_arr.shape[0]
    if n > n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=n_samples, replace=False)
        X_sub = X_arr[idx]
    else:
        X_sub = X_arr
    net.eval()
    try:
        ig = IntegratedGradients(net)
        X_t = torch.as_tensor(np.asarray(X_sub), dtype=torch.float32)
        baseline = torch.zeros_like(X_t)
        attrs = ig.attribute(X_t, baselines=baseline, n_steps=n_steps)
    except Exception as exc:
        logger.warning("captum IntegratedGradients failed (%s); skipping.", exc)
        return None
    return attrs.detach().abs().mean(axis=0).cpu().numpy().astype(np.float64)


def _collapse_coef(coef: np.ndarray) -> np.ndarray:
    """Collapse a (possibly 2-D) ``coef_`` to a per-feature importance vector.

    Binary classifiers / single-target regressors have ``coef_`` shape ``(n,)``
    or ``(1, n)`` -- return the single signed row as-is. Multiclass / multi-
    target models have shape ``(n_classes, n)``; aggregate as ``mean(|coef|,
    axis=0)`` so a feature important for ANY class ranks high. Pre-fix code took
    ``coef[-1, :]`` (last class only), silently discarding every other class.
    """
    if coef.ndim == 1:
        return coef
    if coef.shape[0] == 1:
        return coef[0, :]
    return np.abs(coef).mean(axis=0)


def get_model_feature_importances(
    model: Any,
    columns: Sequence[str],
    return_df: bool = False,
    X: np.ndarray | pd.DataFrame | None = None,
    y: np.ndarray | pd.Series | None = None,
    nn_fi_method: str = "auto",
    return_std: bool = False,
) -> Optional[Union[np.ndarray, pd.DataFrame]] | Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract feature importances from a trained model.

    Native FI sources, in precedence:
      1. ``feature_importances_`` (tree-based: CatBoost / XGBoost / LightGBM / sklearn).
      2. ``coef_`` (linear / ridge / lasso).
      3. Permutation importance (sklearn ``permutation_importance``) when
         the inner estimator exposes neither attribute -- e.g.
         ``PytorchLightningRegressor`` / Keras nets / any custom
         predict-only wrapper. Requires ``X`` + ``y`` from the caller.

    Walks through ``Pipeline``, ``TransformedTargetRegressor``, and the
    common meta-estimator attribute chain (``regressor_`` /
    ``estimator_`` / ``base_estimator_`` / ``best_estimator_``) to find
    the innermost FI-bearing estimator. If none of them carry FI, the
    OUTER model is used for permutation (so the full predict pipeline,
    including target standardisation, is respected).

    Parameters
    ----------
    model : Any
        Trained model (possibly wrapped). The permutation fallback
        always calls ``predict`` on the OUTER model.
    columns : Sequence[str]
        Feature column names.
    return_df : bool, default=False
        If True, return a DataFrame with feature names and importances.
    X, y : array-like, optional
        Required for permutation-importance fallback. Pass the test (or
        validation) feature matrix + target. When omitted, the
        fallback is skipped and ``None`` is returned for non-native
        estimators.

    Returns
    -------
    np.ndarray, pd.DataFrame, or None
        Feature importances array (signed for ``coef_``, non-negative
        for tree FI / permutation), or DataFrame (if return_df=True),
        or None when no source is available.
    """
    inner = _unwrap_estimator_chain(model)
    feature_importances: Optional[np.ndarray] = None
    # Per-feature dispersion (permutation / CUDA-permutation only); native tree-gain / coef have none.
    feature_importances_std: Optional[np.ndarray] = None
    if hasattr(inner, "feature_importances_"):
        feature_importances = np.asarray(inner.feature_importances_)
    elif hasattr(inner, "coef_"):
        feature_importances = _collapse_coef(np.asarray(inner.coef_))
    elif (
        # MultiOutputClassifier / MultiOutputRegressor wrap a base estimator
        # one-per-label; the wrapper exposes ``estimators_`` (the fitted list)
        # but NOT the aggregate ``feature_importances_`` / ``coef_``.
        # Without this branch every multilabel CB / XGB / LGB combo falls
        # through to permutation_importance per (model, split) and pays many
        # seconds where the child's native FI would aggregate in
        # milliseconds. c0008 profile (2026-05-28, multilabel cb+hgb @200k):
        # 35.8 s cumtime / 26 calls of ``_permutation_feature_importances``;
        # CB-multilabel branches would have aggregated natively.
        #
        # iter577 (2026-05-30): two follow-up upgrades over the
        # iter562 b21eaf3c version of this branch.
        #
        # (A) ROBUST AGGREGATOR. The original ``np.mean`` over signed
        # ``coef_`` entries can produce a misleadingly small (or zero)
        # aggregate when two labels carry strong-but-opposite-sign
        # weights for the same feature -- the mean cancels them out
        # and a genuinely-important feature gets pushed to the bottom
        # of the report. For ``coef_`` aggregation we now use
        # ``np.median(np.abs(per_child), axis=0)``: |coef| is the
        # standard signed-coefficient feature-importance proxy in
        # sklearn (LogisticRegression.coef_ etc. all report |coef|
        # in feature_importances_ when available), and the median is
        # outlier-robust to a single label whose magnitude is several
        # orders larger than its siblings (an XOR-style label or a
        # near-constant label). For ``feature_importances_``
        # aggregation (already non-negative; CB / XGB / LGB) we keep
        # mean because sign-cancellation is impossible. See
        # ``tests/training/test_multioutput_fi_robust_aggregator_biz_value.py``
        # for the regression-multilabel biz-value demonstration.
        #
        # (B) HGB-MULTILABEL PER-CHILD PERMUTATION FALLBACK. The
        # earlier b21eaf3c version returned None whenever any child
        # lacked both ``feature_importances_`` and ``coef_`` (the
        # sklearn ``HistGradientBoostingClassifier`` /
        # ``...Regressor`` case -- no native FI at any sklearn
        # version). The code then fell through to a wrapper-level
        # permutation_importance which pays full predict overhead on
        # the multilabel WRAPPER (calls each child then stacks the
        # output, then sklearn scoring expects 2-D y). The per-child
        # ladder below runs ``_permutation_feature_importances`` on
        # each native-FI-less child with the 1-D ``y[:, j]`` slice
        # instead, returning a per-label FI that aggregates through
        # the same robust median path. Native FI children
        # contribute via the cheap path; per-child permutation
        # children contribute via the (still slow) sklearn path --
        # but at least the result is now a real aggregate and a
        # mixed CB+HGB multilabel combo gets the CB labels via
        # cheap-native and only HGB labels pay permutation cost.
        hasattr(inner, "estimators_")
        and isinstance(getattr(inner, "estimators_", None), (list, tuple))
        and len(inner.estimators_) > 0
    ):
        per_child: list[np.ndarray] = []
        kinds: list[str] = []  # "native_fi" | "native_coef" | "permutation"
        children = list(inner.estimators_)
        # Pass 1: collect native FI / coef contributions, mark missing.
        needs_perm: list[int] = []
        for j, child in enumerate(children):
            if hasattr(child, "feature_importances_"):
                per_child.append(np.asarray(child.feature_importances_))
                kinds.append("native_fi")
            elif hasattr(child, "coef_"):
                per_child.append(_collapse_coef(np.asarray(child.coef_)))
                kinds.append("native_coef")
            else:
                # Placeholder; filled in pass 2 if X+y are available.
                per_child.append(None)  # type: ignore[arg-type]
                kinds.append("permutation")
                needs_perm.append(j)
        # Pass 2: per-child permutation for HGB-like children. Requires
        # a 2-D y so we can slice ``y[:, j]`` for each label.
        if needs_perm and X is not None and y is not None:
            try:
                y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            except Exception:
                y_arr = None
            if y_arr is not None and y_arr.ndim == 2 and y_arr.shape[1] >= len(children):
                for j in needs_perm:
                    child_fi = _permutation_feature_importances(children[j], X, y_arr[:, j])
                    if child_fi is not None:
                        per_child[j] = np.asarray(child_fi)
        # Drop any child whose permutation also failed (None placeholder).
        # Filter (kinds, per_child) in lockstep so the indices stay aligned.
        _filtered = [(k, fi) for k, fi in zip(kinds, per_child) if fi is not None]
        kinds = [k for k, _ in _filtered]
        per_child = [fi for _, fi in _filtered]
        # Aggregate when shapes line up. For ``coef_`` contributions
        # use ``median(abs(...))`` (robust to sign cancellation + outlier
        # label magnitudes). For ``feature_importances_`` / permutation
        # (already non-negative) use mean.
        if per_child and all(fi.shape == per_child[0].shape for fi in per_child):
            if any(k == "native_coef" for k in kinds):
                # |coef|-median: sign-canceling-safe + outlier-robust.
                stacked = np.stack(per_child, axis=0)
                feature_importances = np.median(np.abs(stacked), axis=0)
            else:
                feature_importances = np.mean(per_child, axis=0)
    if feature_importances is None and not hasattr(inner, "feature_importances_") and not hasattr(inner, "coef_"):
        # Non-native source: try the NN-specific paths first when the
        # model is a torch Module wrapper. ``nn_fi_method``:
        #   * "auto"            -> Captum IG if available, else permutation.
        #                          When CUDA + torch model + n_features
        #                          >= 50, the permutation fallback uses
        #                          the GPU-batched kernel (~3x speedup).
        #   * "captum"          -> Captum IG only; None if unavailable.
        #   * "first_layer"     -> ``|W1|.sum(axis=hidden)`` proxy
        #                          (ultra-fast but recall@10 only 40-90%
        #                          on bench; opt-in for quick-screen).
        #   * "permutation"     -> Force CPU sklearn permutation fallback.
        #   * "permutation_cuda"-> Force the CUDA-batched kernel
        #                          (skips when no CUDA / not a torch model).
        net = _torch_module_from_model(model)
        if nn_fi_method == "first_layer" and net is not None:
            feature_importances = _first_layer_weight_importance(net)
        elif nn_fi_method in ("auto", "captum") and net is not None and X is not None:
            feature_importances = _captum_integrated_gradients_importance(net, X)
            if feature_importances is None and nn_fi_method == "auto" and y is not None:
                # Captum unavailable -> try CUDA-batched permutation when
                # n_features justifies the warmup amortisation.
                _n_feats = X.shape[1] if hasattr(X, "shape") and len(X.shape) == 2 else None
                if net is not None and _n_feats is not None and _n_feats >= _CUDA_PERM_MIN_FEATURES:
                    feature_importances, feature_importances_std = _cuda_batched_permutation_importance(net, X, y, return_std=True)
                if feature_importances is None:
                    feature_importances, feature_importances_std = _permutation_feature_importances(model, X, y, return_std=True)
        elif nn_fi_method == "permutation_cuda" and net is not None and X is not None and y is not None:
            feature_importances, feature_importances_std = _cuda_batched_permutation_importance(net, X, y, return_std=True)
            if feature_importances is None:
                logger.info("CUDA-batched FI unavailable; falling back to threading permutation.")
                feature_importances, feature_importances_std = _permutation_feature_importances(model, X, y, return_std=True)
        elif X is not None and y is not None:
            feature_importances, feature_importances_std = _permutation_feature_importances(model, X, y, return_std=True)

    if feature_importances is not None:
        feature_importances = np.asarray(feature_importances, dtype=np.float64)
        # Length mismatch -> the proxy applied to the wrong layer
        # (e.g. an embedding-prefixed net). Don't return mis-sized FI.
        if columns is not None and len(columns) > 0 and feature_importances.size != len(columns):
            logger.warning(
                "FI length mismatch: %d values vs %d columns; skipping.",
                feature_importances.size, len(columns),
            )
            return (None, None) if return_std else None
        if feature_importances_std is not None:
            feature_importances_std = np.asarray(feature_importances_std, dtype=np.float64)
            if feature_importances_std.shape != feature_importances.shape:
                feature_importances_std = None
        if return_df:
            feature_importances = pd.DataFrame({"feature": columns, "importance": feature_importances})

    if return_std:
        return feature_importances, feature_importances_std
    return feature_importances


def plot_model_feature_importances(
    model: Any,
    columns: Sequence[str],
    model_name: Optional[str] = None,
    num_factors: int = 15,
    figsize: Tuple[int, int] = DEFAULT_FI_FIGSIZE,
    positive_fi_only: bool = False,
    show_plots: bool = True,
    plot_file: str = "",
    max_zero_fi_to_plot: int = 4,
    X: np.ndarray | pd.DataFrame | None = None,
    y: np.ndarray | pd.Series | None = None,
    nn_fi_method: str = "auto",
) -> Optional[np.ndarray]:
    """
    Plot feature importances for a trained model.

    Extracts and visualizes feature importances as a bar chart.

    Parameters
    ----------
    model : Any
        Trained model with extractable feature importances.
    columns : Sequence[str]
        Feature column names.
    model_name : str, optional
        Title for the plot.
    num_factors : int, default=10
        Maximum number of features to display. Reduced from 40 to 10 so
        plots stay scannable on common feature counts; override via
        ``reporting_config.fi_top_n`` in ``train_mlframe_models_suite``.
    figsize : tuple, default=(15, 10)
        Figure size for the plot.
    positive_fi_only : bool, default=False
        If True, only show features with positive importance.
    plot_file : str, default=""
        Path for saving the plot.

    Returns
    -------
    np.ndarray or None
        Feature importances array, or None if extraction failed.
    """
    feature_importances, feature_importances_std = get_model_feature_importances(
        model=model, columns=columns, X=X, y=y, nn_fi_method=nn_fi_method, return_std=True,
    )

    if feature_importances is not None:
        try:
            plot_feature_importance(
                feature_importances=feature_importances,
                columns=columns,
                kind=model_name,
                figsize=figsize,
                plot_file=plot_file,
                positive_fi_only=positive_fi_only,
                n=num_factors,
                show_plots=show_plots,
                max_zero_fi_to_plot=max_zero_fi_to_plot,
                importances_std=feature_importances_std,
            )
        except (ValueError, AttributeError, IndexError, TypeError):
            logger.warning("Could not plot feature importances. Maybe data shape changed within a pipeline?", exc_info=True)

        return feature_importances
