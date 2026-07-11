"""``make_outlier_detector``: named factory for swappable outlier-detector backends.

mlframe's existing outlier machinery (``preprocessing.outliers.reject_outliers``,
``compute_outlier_detector_score``) already accepts ANY duck-typed detector object (``.fit``/``.predict``), but
has no NAMED selection layer -- ``IsolationForest`` is the only built-in default, hardwired at the call site,
with no ``method="lof"``/``"ecod"`` string dispatch a caller could reach for instead. This factory is that
missing selection layer: it returns a fresh, unfitted detector instance by name, usable anywhere the existing
functions' ``model=``/``detector=`` params already accept a detector object.
"""
from __future__ import annotations

from typing import Any

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


__all__ = ["make_outlier_detector"]
