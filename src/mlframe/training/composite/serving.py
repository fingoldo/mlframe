"""Lightweight, dependency-light serving export for a fitted ``CompositeTargetEstimator``.

The fitted wrapper carries a fitted sklearn inner model + the transform registry
+ the conformal / streaming machinery. At SERVE time none of that is needed for a
plain point prediction: the inverse of the composite target is a pure numpy
function of the inner's RAW (T-scale) prediction, the base column(s), and a small
dict of fitted scalars (alpha / beta / clip envelope / eps / ...).

:func:`export_serving_spec` distils a fitted estimator into a plain
JSON-serialisable dict capturing exactly that. :func:`load_serving_spec` rebuilds
a pure-callable ``predict(X_base_matrix, inner_raw_pred) -> y`` that applies the
SAME T-clip -> inverse -> y-clip + domain-fallback the live ``predict`` does, using
ONLY numpy + the spec dict -- no sklearn, no transform registry, no estimator class.

The INNER model is NOT captured: it must be served separately (its own booster
file / ONNX / pickle) and its raw T-scale prediction passed into the rebuilt
callable. This keeps the spec tiny + JSON-clean and decouples the inverse maths
(stable, dependency-light) from the heavy model artefact (framework-specific).

Only transforms whose inverse is a closed-form numpy expression are covered (the
additive-residual + simple-ratio family); transforms whose inverse needs the
registry (PCHIP splines, rolling-median recurrence, per-group lookups, ...) raise a
clear :class:`NotImplementedError` at export time so a caller never ships a spec
that would silently mis-predict at serve time.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np

__all__ = [
    "SERVING_SPEC_VERSION",
    "LIGHTWEIGHT_TRANSFORMS",
    "export_serving_spec",
    "load_serving_spec",
]

# Bump when the spec schema changes in a way that breaks old loaders.
SERVING_SPEC_VERSION = 1

# Transforms whose inverse is a pure closed-form numpy expression of
# (t_hat, base, fitted_params) -- reproducible WITHOUT the registry. Each maps
# to a key in ``_INVERSE_TABLE`` below. Transforms NOT listed here (PCHIP
# monotonic_residual, rolling/ewma/frac recurrence, per-bin median lookups,
# grouped per-row alpha, unary y-only clips, chained transforms, ...) raise
# NotImplementedError at export: their inverse needs registry-resident state /
# logic that the lightweight table deliberately does not reimplement.
LIGHTWEIGHT_TRANSFORMS: tuple[str, ...] = (
    "diff",
    "additive_residual",
    "linear_residual",
    "linear_residual_robust",
    "theilsen_residual",
    "linear_residual_multi",
    "ratio",
    "logratio",
)

# Whether each lightweight transform consumes a base column. ``True`` for every
# entry above (all are base-dependent); kept explicit so the loader can size a
# zeros placeholder for any future base-free addition without guessing.
_REQUIRES_BASE: dict[str, bool] = {name: True for name in LIGHTWEIGHT_TRANSFORMS}

# Whether the transform takes a MULTI-column base matrix (n, K) rather than a
# single (n,) base vector. Only ``linear_residual_multi`` does.
_MULTI_BASE: dict[str, bool] = {name: False for name in LIGHTWEIGHT_TRANSFORMS}
_MULTI_BASE["linear_residual_multi"] = True


# ----------------------------------------------------------------------
# Numpy-only inverse table. Each function mirrors EXACTLY the registry inverse
# for the named transform (cross-checked against transforms/{simple,linear}.py)
# so the rebuilt predict is bit-identical to estimator.predict on the same raw
# inner prediction. base is 1-D (n,) for single-base transforms, 2-D (n, K) for
# linear_residual_multi.
# ----------------------------------------------------------------------

def _inv_diff(t_hat: np.ndarray, base: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    # transforms/simple.py::_diff_inverse -> t + base
    return np.asarray(t_hat + base)


def _inv_additive_residual(t_hat: np.ndarray, base: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    # _additive_residual_inverse -> t + base + beta
    return np.asarray(t_hat + base + float(p.get("beta", 0.0)))


def _inv_linear_residual(t_hat: np.ndarray, base: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    # _linear_residual_inverse -> t + alpha*base + beta. Shared by linear_residual,
    # linear_residual_robust, theilsen_residual (all return {alpha, beta}).
    return t_hat + float(p["alpha"]) * base + float(p["beta"])


def _inv_linear_residual_multi(t_hat: np.ndarray, base: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    # _linear_residual_multi_inverse -> t + base @ alphas + beta.
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alphas = np.asarray(p["alphas"], dtype=np.float64)
    if base.shape[1] != alphas.size:
        raise ValueError(f"linear_residual_multi serving: base has {base.shape[1]} columns " f"but fitted alphas has {alphas.size} entries")
    # Normalise to a canonical C-contiguous layout before the matvec: BLAS dgemv rounds an (n,K)@(K,)
    # product differently for C- vs F-ordered bases (~1 ULP), and the predict path always feeds a
    # C-contiguous base (via the soft-shrink guard's ascontiguousarray), so this keeps serving byte-identical.
    return t_hat + (np.ascontiguousarray(base, dtype=np.float64) @ alphas) + float(p["beta"])


def _inv_ratio(t_hat: np.ndarray, base: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    # _ratio_inverse -> t * safe_base, with the SAME eps-floor as the forward so
    # the round-trip is exact on in-domain near-zero base rows.
    eps = float(p["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return np.asarray(t_hat * safe_base)


def _inv_logratio(t_hat: np.ndarray, base: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    # _logratio_inverse -> base * exp(softcap(t)) centred on median_t.
    median_t = float(p["median_t"])
    mad = float(p["mad_eff"])
    k = float(p["soft_cap_k"])
    cap = k * mad
    t_capped = np.clip(t_hat, median_t - cap, median_t + cap)
    return np.asarray(base * np.exp(t_capped))


_INVERSE_TABLE: dict[str, Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray]] = {
    "diff": _inv_diff,
    "additive_residual": _inv_additive_residual,
    "linear_residual": _inv_linear_residual,
    "linear_residual_robust": _inv_linear_residual,
    "theilsen_residual": _inv_linear_residual,
    "linear_residual_multi": _inv_linear_residual_multi,
    "ratio": _inv_ratio,
    "logratio": _inv_logratio,
}

# Per-transform base-side domain predicate (numpy-only mirror of the registry
# domain_check with y=None). Rows failing this route to the fallback at serve
# time, exactly as the live predict path does. Default: finite base.
def _domain_finite(base: np.ndarray) -> np.ndarray:
    return np.asarray(np.isfinite(base) if base.ndim == 1 else np.all(np.isfinite(base), axis=1))


def _domain_ratio(base: np.ndarray) -> np.ndarray:
    # _ratio_domain (y=None) -> finite & |base| > 0.
    return np.asarray(np.isfinite(base) & (np.abs(base) > 0))


def _domain_logratio(base: np.ndarray) -> np.ndarray:
    # _logratio_domain (y=None) -> finite & base > 0.
    return np.asarray(np.isfinite(base) & (base > 0))


_DOMAIN_TABLE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ratio": _domain_ratio,
    "logratio": _domain_logratio,
}


# ----------------------------------------------------------------------
# Spec keys that must survive json round-trip. fitted_params_ may carry numpy
# scalars / arrays (alphas list, bin_edges ndarray); _jsonify coerces to plain
# python so json.dumps never trips on a numpy type or a non-finite float.
# ----------------------------------------------------------------------

def _jsonify(value: Any) -> Any:
    """Recursively coerce numpy scalars / arrays + non-finite floats to JSON-safe python.

    json.dumps cannot serialise np.float64 / np.ndarray and emits the
    non-standard ``Infinity`` / ``NaN`` tokens for inf / nan. We map non-finite
    floats to sentinel strings so a strict JSON consumer round-trips them, and
    _de_jsonify reverses the mapping on load.
    """
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isinf(f):
            return "__inf__" if f > 0 else "__-inf__"
        if math.isnan(f):
            return "__nan__"
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _de_jsonify(value: Any) -> Any:
    """Reverse :func:`_jsonify`: restore inf / -inf / nan sentinel strings to floats."""
    if isinstance(value, dict):
        return {k: _de_jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_de_jsonify(v) for v in value]
    if value == "__inf__":
        return float("inf")
    if value == "__-inf__":
        return float("-inf")
    if value == "__nan__":
        return float("nan")
    return value


def export_serving_spec(estimator: Any) -> Dict[str, Any]:
    """Distil a fitted ``CompositeTargetEstimator`` into a JSON-serialisable serving spec.

    The returned dict carries everything the lightweight serve path needs to
    reproduce ``predict`` WITHOUT sklearn or the transform registry: the
    transform name, the base column name(s), the fitted params (alpha / beta /
    eps / clip envelope / median_t / ...), the fallback strategy, and the
    schema version. The INNER model is intentionally excluded -- serve it
    separately and feed its raw T-scale prediction into the
    :func:`load_serving_spec` callable.

    Raises
    ------
    ValueError
        If ``estimator`` is not fitted (no ``fitted_params_``).
    NotImplementedError
        If ``transform_name`` is not in :data:`LIGHTWEIGHT_TRANSFORMS` -- its
        inverse needs registry-resident logic the lightweight table does not
        reimplement; serve it via the full estimator instead.
    """
    transform_name = getattr(estimator, "transform_name", None)
    fitted_params = getattr(estimator, "fitted_params_", None)
    if fitted_params is None:
        raise ValueError("export_serving_spec: estimator is not fitted (no fitted_params_). " "Call fit() / from_fitted_inner() first.")
    if transform_name not in _INVERSE_TABLE:
        raise NotImplementedError(
            f"export_serving_spec: transform '{transform_name}' has no "
            f"lightweight numpy inverse. Supported: {sorted(_INVERSE_TABLE)}. "
            "Serve this composite via the full CompositeTargetEstimator instead "
            "(its inverse needs the transform registry: PCHIP spline / rolling "
            "recurrence / per-group lookup / unary y-transform / chain)."
        )

    # Resolve the base column name(s). Prefer the estimator's own resolver so the
    # base_columns / base_column priority matches fit / predict exactly.
    resolver = getattr(estimator, "_resolve_base_columns", None)
    if callable(resolver):
        base_columns: tuple[str, ...] = tuple(resolver())
    else:
        bc = getattr(estimator, "base_columns", None)
        base_columns = tuple(bc) if bc else ((getattr(estimator, "base_column", "") or "",) if getattr(estimator, "base_column", "") else ())

    spec: Dict[str, Any] = {
        "spec_version": SERVING_SPEC_VERSION,
        "kind": "composite_target_serving_spec",
        "transform_name": transform_name,
        "base_columns": list(base_columns),
        "requires_base": _REQUIRES_BASE.get(transform_name, True),
        "multi_base": _MULTI_BASE.get(transform_name, False),
        "fallback_predict": getattr(estimator, "fallback_predict", "y_train_median"),
        "fitted_params": _jsonify(dict(fitted_params)),
    }
    return spec


def load_serving_spec(
    spec: Dict[str, Any],
    inner_predict: Callable[[Any], np.ndarray] | None = None,
) -> Callable[..., np.ndarray]:
    """Rebuild a pure-callable y-scale predict from a serving spec.

    Parameters
    ----------
    spec
        A dict produced by :func:`export_serving_spec` (optionally round-tripped
        through ``json.dumps`` / ``json.loads``).
    inner_predict
        Optional caller-supplied hook that maps a feature matrix ``X`` to the
        inner model's RAW (T-scale) prediction. The INNER model is served
        separately (its own booster file / ONNX / pickle); this hook is the only
        coupling to it. When provided, the returned callable accepts ``X`` and
        calls ``inner_predict(X)`` internally. When ``None``, the caller passes
        the already-computed raw prediction directly.

    Returns
    -------
    predict : callable
        ``predict(base, inner_raw_pred) -> y``. ``base`` is the base-column
        matrix: a 1-D ``(n,)`` array (single-base transforms) or a 2-D
        ``(n, K)`` array (``linear_residual_multi``). ``inner_raw_pred`` is the
        inner model's T-scale prediction ``(n,)``. When ``inner_predict`` was
        supplied at load time, the SECOND positional arg may instead be the raw
        feature matrix ``X`` -- it is routed through ``inner_predict`` first.

        Applies T-clip -> domain-aware inverse -> y-clip + fallback using ONLY
        numpy + the spec dict; bit-identical to ``CompositeTargetEstimator.predict``
        on the same raw inner prediction.

    Raises
    ------
    NotImplementedError
        If the spec's transform has no lightweight numpy inverse (should not
        happen for a spec produced by :func:`export_serving_spec`, but guards a
        hand-edited / forward-compatible spec).
    """
    spec = _de_jsonify(dict(spec))
    transform_name = spec.get("transform_name")
    inverse_fn = _INVERSE_TABLE.get(transform_name)
    if inverse_fn is None:
        raise NotImplementedError(f"load_serving_spec: transform '{transform_name}' has no lightweight " f"numpy inverse. Supported: {sorted(_INVERSE_TABLE)}.")
    params: dict[str, Any] = dict(spec.get("fitted_params", {}))
    fallback_predict = spec.get("fallback_predict", "y_train_median")
    multi_base = bool(spec.get("multi_base", False))
    domain_fn = _DOMAIN_TABLE.get(transform_name, _domain_finite)

    # Pre-resolve the clip envelope + the finite fallback constant once at load
    # so the hot predict path does only array work.
    t_clip_low = float(params.get("t_clip_low", float("-inf")))
    t_clip_high = float(params.get("t_clip_high", float("+inf")))
    y_clip_low = float(params.get("y_clip_low", float("-inf")))
    y_clip_high = float(params.get("y_clip_high", float("+inf")))
    _med = params.get("y_train_median", 0.0)
    fallback_const = float(_med) if np.isfinite(_med) else 0.0

    def predict(base: Any, inner_raw_pred: Any) -> np.ndarray:
        # When an inner_predict hook was supplied, the 2nd arg is the raw feature
        # matrix X; route it through the hook to obtain the T-scale prediction.
        if inner_predict is not None:
            t_hat = np.asarray(inner_predict(inner_raw_pred), dtype=np.float64).reshape(-1)
        else:
            t_hat = np.asarray(inner_raw_pred, dtype=np.float64).reshape(-1)

        base_arr = np.asarray(base, dtype=np.float64)
        if multi_base and base_arr.ndim == 1:
            base_arr = base_arr.reshape(-1, 1)

        # 1) T-clip BEFORE inverse (mirror _apply_t_clip). No-op when bounds
        # are non-finite; only clips when at least one bound is finite.
        if math.isfinite(t_clip_low) or math.isfinite(t_clip_high):
            t_hat = np.clip(t_hat, t_clip_low, t_clip_high)

        # 2) Base-side domain mask (mirror _compute_base_domain_ok with y=None).
        domain_ok = np.asarray(domain_fn(base_arr), dtype=bool)

        # 3) Domain-aware inverse with fallback (mirror _inverse_with_fallback).
        if domain_ok.all():
            y_hat = np.asarray(inverse_fn(t_hat, base_arr, params), dtype=np.float64).reshape(-1)
        else:
            mask = domain_ok if base_arr.ndim == 1 else domain_ok[:, None]
            base_safe = np.where(mask, base_arr, 1.0)
            y_valid = np.asarray(inverse_fn(t_hat, base_safe, params), dtype=np.float64).reshape(-1)
            y_hat = np.full(t_hat.shape, np.nan, dtype=np.float64)
            y_hat[domain_ok] = y_valid[domain_ok]
            if fallback_predict == "y_train_median":
                y_hat[~domain_ok] = fallback_const
            elif fallback_predict != "nan":
                raise ValueError(f"load_serving_spec: unknown fallback_predict " f"'{fallback_predict}'; choose 'y_train_median' or 'nan'.")

        # 4) Non-finite guard on domain-valid rows (mirror the general guard).
        nonfinite = ~np.isfinite(y_hat)
        if nonfinite.any() and fallback_predict == "y_train_median":
            y_hat[nonfinite] = fallback_const

        # 5) Post-inverse y-clip (mirror predict()). np.clip leaves NaN as NaN,
        # matching the 'nan' fallback path.
        if math.isfinite(y_clip_low) or math.isfinite(y_clip_high):
            y_hat = np.clip(y_hat, y_clip_low, y_clip_high)
        return y_hat

    return predict
