"""``make_outlier_detector``: named factory for swappable outlier-detector backends.

mlframe's existing outlier machinery (``preprocessing.outliers.reject_outliers``,
``compute_outlier_detector_score``) already accepts ANY duck-typed detector object (``.fit``/``.predict``), but
has no NAMED selection layer -- ``IsolationForest`` is the only built-in default, hardwired at the call site,
with no ``method="lof"``/``"ecod"`` string dispatch a caller could reach for instead. This factory is that
missing selection layer: it returns a fresh, unfitted detector instance by name, usable anywhere the existing
functions' ``model=``/``detector=`` params already accept a detector object.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from mlframe.models.ensembling.selection import rank_average_blend

_METHODS = ("isolation_forest", "lof", "ecod")


def make_outlier_detector(method: str = "isolation_forest", *, random_state: int = 0, **kwargs: Any) -> Any:
    """Return a fresh, unfitted outlier detector by name.

    Parameters
    ----------
    method
        ``"isolation_forest"`` (sklearn ``IsolationForest``, tree-based, scales well to high dimensions),
        ``"lof"`` (sklearn ``LocalOutlierFactor``, density-based, effective when outliers are local/regional
        rather than globally extreme), or ``"ecod"`` (pyod ``ECOD`` -- Empirical Cumulative Distribution-based
        Outlier Detection, parameter-free tail-probability scoring; requires the optional ``pyod`` dependency,
        imported lazily so the rest of mlframe stays usable without it).
    random_state
        Forwarded to ``IsolationForest``/``ECOD``; ignored by ``LocalOutlierFactor`` (deterministic, no seed).
    **kwargs
        Extra constructor kwargs forwarded to the underlying detector class.

    Returns
    -------
    Any
        An unfitted detector instance exposing ``.fit``/``.predict`` (IsolationForest/ECOD: +1 inlier / -1
        outlier convention; LOF in its default ``novelty=False`` mode exposes ``.fit_predict`` with the same
        convention -- callers using ``preprocessing.outliers.reject_outliers``'s ``model=`` param should
        instantiate LOF with ``novelty=True`` if they need a separate ``.fit`` + ``.predict`` split).
    """
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(random_state=random_state, **kwargs)
    if method == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(**kwargs)
    if method == "ecod":
        try:
            from pyod.models.ecod import ECOD
        except ImportError as exc:
            raise ImportError("make_outlier_detector: method='ecod' requires the optional 'pyod' dependency ('pip install pyod').") from exc
        return ECOD(**kwargs)
    raise ValueError(f"make_outlier_detector: unknown method {method!r}; choose one of {_METHODS}.")


def _fit_anomaly_score(method: str, X: Any, *, random_state: int, detector_kwargs: Mapping[str, Any]) -> np.ndarray:
    """Fit one named detector on ``X`` and return a per-row anomaly score (higher = MORE anomalous).

    Both backends natively score higher = more NORMAL (``IsolationForest.decision_function`` / LOF's
    ``negative_outlier_factor_``), so this negates to a common "higher = more anomalous" orientation before
    the caller rank-blends across backends.
    """
    if method == "ecod":
        raise ValueError(
            "make_ensemble_outlier_scores: method='ecod' is not supported -- pyod's ECOD.decision_function "
            "already returns higher=more-anomalous, the opposite sign convention of decision_function/"
            "negative_outlier_factor_, so it cannot share this helper's uniform negate-to-anomalous step."
        )
    detector = make_outlier_detector(method, random_state=random_state, **detector_kwargs)
    if method == "lof":
        detector.fit_predict(X)
        raw = detector.negative_outlier_factor_
    else:
        detector.fit(X)
        raw = detector.decision_function(X)
    return -np.asarray(raw, dtype=np.float64)


def make_ensemble_outlier_scores(
    X: Any,
    methods: Sequence[str] = ("isolation_forest", "lof"),
    *,
    weights: Optional[Sequence[float]] = None,
    random_state: int = 0,
    detector_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> np.ndarray:
    """Fit multiple named detectors on ``X`` and rank-average-blend their anomaly scores into one.

    Addresses the scaling gap the biz_value test for ``make_outlier_detector`` surfaced: IsolationForest and
    LOF disagree on ranking (Spearman rho~0.61) because they catch different outlier SHAPES -- IsolationForest
    (global tree-partition isolation depth) is strong on globally extreme points, LOF (local k-NN density
    ratio) is strong on points that are only anomalous relative to their local neighborhood. Combining both
    catches both shapes in one pass, at the cost of paying LOF's O(n*k) k-NN search once (the ensemble is only
    as fast as its slowest member -- use ``method="lof"``/``"isolation_forest"`` alone when n is large and only
    one outlier shape matters).

    Blending reuses ``mlframe.models.ensembling.selection.rank_average_blend`` (built for combining
    heterogeneously-scaled model scores) rather than averaging raw scores, since IsolationForest's
    ``decision_function`` and LOF's ``negative_outlier_factor_`` live on unrelated numeric scales.

    Parameters
    ----------
    X
        Row-major feature matrix, forwarded to each detector's ``.fit``/``.fit_predict``.
    methods
        Detector names to combine, from ``"isolation_forest"``/``"lof"`` (both always available). Default
        pair is the always-available sklearn combo; extend to 3+ methods freely.
    weights
        Per-method non-negative weights (length ``len(methods)``), forwarded to ``rank_average_blend``.
        Defaults to uniform.
    random_state
        Forwarded to every detector that accepts it (ignored by LOF).
    detector_kwargs
        Optional ``{method: {kwarg: value}}`` overrides, e.g. ``{"lof": {"n_neighbors": 35}}``.

    Returns
    -------
    np.ndarray
        Blended per-row anomaly score, shape ``(n_rows,)``, higher = more anomalous.
    """
    if len(methods) < 2:
        raise ValueError(f"make_ensemble_outlier_scores: need >=2 methods to ensemble; got {methods!r}.")
    kwargs_by_method = detector_kwargs or {}
    stacked = np.vstack([_fit_anomaly_score(method, X, random_state=random_state, detector_kwargs=kwargs_by_method.get(method, {})) for method in methods])
    return rank_average_blend(stacked, weights=weights)


__all__ = ["make_outlier_detector", "make_ensemble_outlier_scores"]
