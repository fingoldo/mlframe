"""Self-contained module-level helpers for ``CompositeTargetEstimator``, carved out of ``_estimator.py`` to keep that
module under the 1000-LOC house limit. Re-imported into ``_estimator`` so call sites are unchanged."""
from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

try:
    # sklearn's canonical "does this estimator's fit accept <param>" check.
    # Available since sklearn 0.x; guarded so a stripped install still imports.
    from sklearn.utils.validation import has_fit_parameter as _sk_has_fit_parameter
except ImportError:  # pragma: no cover - sklearn always ships this
    _sk_has_fit_parameter = None


def _callable_accepts_param(fn: Callable[..., Any], name: str) -> bool:
    """True when ``fn`` declares a parameter ``name`` or accepts ``**kwargs``.

    Used to signature-GATE optional ``sample_weight`` pass-through instead of the catch-all ``except TypeError`` retry
    pattern. The retry pattern is wrong because a ``TypeError`` raised DEEP inside a fit that *does* accept
    ``sample_weight`` (a bad dtype, a shape mismatch, a downstream library bug) is mis-attributed to "no sample_weight
    support" -> the estimator is then silently re-fit UNWEIGHTED, dropping the weighting the caller asked for with zero
    diagnostics. Gating on the declared signature only swallows the genuine "this fit has no sample_weight parameter"
    case and lets every real error propagate.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Builtins / C-extensions without an introspectable signature: be permissive (assume the param is accepted) so
        # we never silently drop weighting on an estimator we could not introspect; a real "no such param" TypeError
        # then surfaces loudly to the caller.
        return True
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _estimator_fit_accepts_sample_weight(estimator: Any) -> bool:
    """Signature-gate for an sklearn-style estimator's ``fit(..., sample_weight=)``.

    Prefers sklearn's ``has_fit_parameter`` (handles metadata-routing / delegated estimators); falls back to
    introspecting ``estimator.fit``.
    """
    if _sk_has_fit_parameter is not None:
        try:
            return bool(_sk_has_fit_parameter(estimator, "sample_weight"))
        except Exception:  # pragma: no cover - defensive; fall through  # nosec B110 - best-effort/optional path, no module logger
            pass
    fit_fn = getattr(estimator, "fit", None)
    if fit_fn is None:
        return False
    return _callable_accepts_param(fit_fn, "sample_weight")


def _carry_forward_fill(arr: "np.ndarray", keep: "np.ndarray") -> "np.ndarray":
    """Return a copy of ``arr`` (1-D) with the rows where ``keep`` is False replaced by the last preceding kept value
    (carry-forward); any leading not-kept rows back-fill from the first kept value.

    Called with ``keep = np.isfinite(arr)`` to keep a time-recurrent forward well-defined and row-position-preserving
    across a domain-filtered gap: a non-finite recurrent input would otherwise poison the convolution / EWMA / window on
    the neighbouring valid rows. Carry-forward is the standard time-series gap-fill and matches the missing-value anchor
    the recurrent forwards already use. When every row is kept the result equals the input.
    """
    a = np.asarray(arr, dtype=np.float64).reshape(-1).copy()
    n = a.size
    if n == 0 or bool(keep.all()):
        return a
    # Forward-fill the source index: idx[i] = last position <= i that is kept.
    idx = np.where(keep, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    # Leading not-kept rows (idx still -1) back-fill from the first kept row.
    first_kept = int(np.argmax(keep)) if bool(keep.any()) else 0
    idx[idx < 0] = first_kept
    return a[idx]
