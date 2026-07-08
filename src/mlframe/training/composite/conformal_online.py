"""Adaptive Conformal Inference (ACI) for ``CompositeTargetEstimator``.

Split-conformal (``conformal.py``) freezes one radius from a held-out set and
re-uses it forever. That radius is valid only while the test rows stay
exchangeable with the calibration rows -- under DRIFT (the residual scale grows
over time) the frozen band silently under-covers. ACI (Gibbs & Candes, 2021)
makes the level itself a feedback controller that tracks the realised coverage:

    err_t       = 1[ y_t not in interval_t ]            (1 = miss, 0 = hit)
    alpha_{t+1} = clip( alpha_t + gamma * (alpha - err_t),  0, 1 )

When recent misses outnumber ``alpha`` the controller LOWERS ``alpha_t`` (wider
band -> fewer misses); when it over-covers it RAISES ``alpha_t`` (tighter band).
The long-run miss rate then tracks the target ``alpha`` regardless of how the
residual distribution drifts -- a frozen split-conformal band cannot. With
``gamma=0`` the update is inert and ACI reduces to static split-conformal.

Per-step radius. Each step's radius is the empirical ``(1-alpha_t)`` quantile of
a rolling buffer of recent absolute residuals (the same nonconformity score as
split-conformal, just over a sliding window so the radius itself tracks scale
drift on top of the alpha controller). The buffer is the last ``buffer_n``
residuals; the quantile uses the finite-sample ``ceil((m+1)(1-a))`` rank so a
short buffer yields a valid (possibly ``+inf``) radius rather than a too-tight
one.

State lives in trailing-underscore runtime fields (``self._aci_state_``) so
``sklearn.clone`` / pickle stay clean -- the controller is streaming state, not
a constructor hyperparameter, exactly like the rolling-refit buffer.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _sorted_quantile_radius(r_sorted: Sequence[float], m: int, alpha: float) -> float:
    """Finite-sample ``(1-alpha)`` quantile from an already-sorted buffer.

    ``r_sorted`` holds the ``m`` finite non-negative abs-residuals in ascending
    order; the radius is the plain ``ceil((m+1)(1-alpha))`` rank (no
    interpolation), so reading ``r_sorted[rank-1]`` is bit-identical to
    ``np.sort(r)[rank-1]`` on the same float values. ``alpha`` is the adaptive
    ``alpha_t``; the saturations match ``_rolling_quantile_radius`` exactly.
    """
    if m == 0:
        return float("inf")
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0
    rank = math.ceil((m + 1) * (1.0 - alpha))
    if rank > m:
        return float("inf")
    return float(r_sorted[rank - 1])


def _rolling_quantile_radius(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample ``(1-alpha)`` quantile of absolute residuals in the buffer.

    Mirrors ``conformal.conformal_quantile`` but operates on the live rolling
    buffer. ``alpha`` here is the ADAPTIVE ``alpha_t`` (possibly driven to 0 or
    1 by the controller); both saturations are handled: ``alpha_t<=0`` -> the
    band must cover everything (``+inf``), ``alpha_t>=1`` -> a zero-width band
    (radius 0, the controller has decided to stop covering).
    """
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    r = r[np.isfinite(r)]
    m = int(r.size)
    if m == 0:
        return float("inf")
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0
    rank = math.ceil((m + 1) * (1.0 - alpha))
    if rank > m:
        return float("inf")
    r_sorted = np.sort(r)
    return float(r_sorted[rank - 1])


def _aci_default_state(alpha: float, gamma: float, buffer_n: int) -> Dict[str, Any]:
    """Fresh ACI controller state dict (target alpha, current alpha_t, buffer)."""
    return {
        "alpha_target": float(alpha),
        "alpha_t": float(alpha),
        "gamma": float(gamma),
        "buffer_n": int(buffer_n),
        "residuals": [],  # rolling abs-residual buffer (FIFO, last buffer_n)
        "residuals_sorted": [],  # same values kept ascending (bisect.insort) for O(1)-index quantile
        "errors": [],  # rolling 0/1 miss history (for rolling coverage)
        "n_seen": 0,
        "n_miss": 0,
        "last_radius": float("inf"),
    }


def init_aci(self, alpha: float = 0.1, gamma: float = 0.05, buffer_n: int = 500, warmup_residuals: Optional[np.ndarray] = None) -> "Any":
    """Initialise / reset the online ACI controller on the wrapper.

    Parameters
    ----------
    alpha
        Target miss rate; long-run coverage tracks ``1 - alpha``.
    gamma
        Controller step size. ``0`` makes the controller inert (static
        split-conformal on the rolling buffer); larger ``gamma`` reacts faster
        to drift but is noisier. Typical 0.01 .. 0.1.
    buffer_n
        Rolling-buffer length for the per-step quantile radius and the rolling
        coverage estimate. Older residuals are evicted FIFO.
    warmup_residuals
        Optional held-out absolute (or signed -- abs is taken) residuals to seed
        the buffer so the very first ``predict_interval_online`` already has a
        meaningful radius instead of ``+inf``. Pass the calibration-split
        residuals here for a warm start.

    State is stored under ``self._aci_state_`` (trailing underscore -> runtime
    only, ignored by ``sklearn.clone``). Returns ``self``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"init_aci: alpha must be in (0, 1), got {alpha!r}")
    if gamma < 0.0:
        raise ValueError(f"init_aci: gamma must be >= 0, got {gamma!r}")
    if buffer_n < 1:
        raise ValueError(f"init_aci: buffer_n must be >= 1, got {buffer_n!r}")
    state = _aci_default_state(alpha, gamma, buffer_n)
    if warmup_residuals is not None:
        warm = np.abs(np.asarray(warmup_residuals, dtype=np.float64).reshape(-1))
        warm = warm[np.isfinite(warm)]
        if warm.size:
            state["residuals"] = list(warm[-buffer_n:])
            state["residuals_sorted"] = sorted(state["residuals"])
            state["last_radius"] = _sorted_quantile_radius(
                state["residuals_sorted"], len(state["residuals_sorted"]), state["alpha_t"],
            )
    self._aci_state_ = state
    return self


def _aci_radius(state: Dict[str, Any]) -> float:
    """Current radius = rolling ``(1-alpha_t)`` quantile of the buffer."""
    r_sorted = state["residuals_sorted"]
    return _sorted_quantile_radius(r_sorted, len(r_sorted), state["alpha_t"])


def predict_interval_online(self, X, clip: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` using the CURRENT adaptive radius.

    Uses the live ``alpha_t`` + rolling buffer (no per-call argument): the band
    is ``predict(X) +/- radius_t``. Call :meth:`update_conformal` after each
    observed ``(x, y)`` to let the controller adapt. Requires a prior
    :meth:`init_aci`.

    The band is clipped to the wrapper's train envelope (same as
    ``predict_interval``) unless ``clip=False`` (diagnostics / coverage checks
    that want the raw band).
    """
    state = getattr(self, "_aci_state_", None)
    if state is None:
        raise RuntimeError("predict_interval_online: ACI not initialised. Call init_aci(alpha, " "gamma) first.")
    radius = _aci_radius(state)
    state["last_radius"] = radius
    point = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    lower = point - radius
    upper = point + radius
    if clip:
        params = getattr(self, "fitted_params_", {}) or {}
        lo_b = params.get("y_clip_low", float("-inf"))
        hi_b = params.get("y_clip_high", float("inf"))
        lower = np.clip(lower, lo_b, hi_b)
        upper = np.clip(upper, lo_b, hi_b)
    return lower, upper


def _aci_step(state: Dict[str, Any], residual: float, in_interval: bool) -> None:
    """Apply one ACI update given a single observed residual + hit/miss.

    Pushes the abs-residual into the rolling buffer, records the 0/1 miss, and
    runs the alpha controller ``alpha_{t+1} = clip(alpha_t + gamma*(alpha-err), 0, 1)``.
    """
    buffer_n = state["buffer_n"]
    ar = abs(float(residual))
    if np.isfinite(ar):
        buf = state["residuals"]
        buf_sorted = state["residuals_sorted"]
        buf.append(ar)
        bisect.insort(buf_sorted, ar)
        # Evict oldest (FIFO) once over capacity; drop its exact value from the
        # sorted mirror so the index-based quantile stays bit-identical to a
        # full ``np.sort`` of the live FIFO window.
        if len(buf) > buffer_n:
            n_evict = len(buf) - buffer_n
            for evicted in buf[:n_evict]:
                del buf_sorted[bisect.bisect_left(buf_sorted, evicted)]
            del buf[:n_evict]
    err = 0.0 if in_interval else 1.0
    state["errors"].append(err)
    if len(state["errors"]) > buffer_n:
        del state["errors"][: len(state["errors"]) - buffer_n]
    state["n_seen"] += 1
    state["n_miss"] += int(err)
    gamma = state["gamma"]
    alpha_t = state["alpha_t"] + gamma * (state["alpha_target"] - err)
    # Keep alpha_t in [0, 1]; the radius helper maps the saturations to
    # +inf / 0 so the band stays valid at the boundary.
    state["alpha_t"] = float(min(1.0, max(0.0, alpha_t)))


def update_conformal(self, x: Any, y: Any) -> Dict[str, Any]:
    """Observe one (or a batch of) ``(x, y)`` and advance the ACI controller.

    For each row: form the interval at the CURRENT ``alpha_t``, check whether
    the true ``y`` fell inside, push the residual into the rolling buffer, and
    update ``alpha_t`` by the ACI rule. ``x`` may be a single-row or multi-row
    frame/array; ``y`` a scalar or 1-D array of the matching length. Rows are
    consumed in order (online), so a multi-row call is equivalent to that many
    sequential single-row updates.

    Returns a small diagnostic dict (current ``alpha_t``, ``radius``, rolling
    ``coverage``, counts). Requires a prior :meth:`init_aci`.
    """
    state = getattr(self, "_aci_state_", None)
    if state is None:
        raise RuntimeError("update_conformal: ACI not initialised. Call init_aci(alpha, gamma) " "first.")
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y_arr.shape[0]
    if n == 0:
        return get_aci_state(self)
    # One y-scale prediction per row. ``predict`` accepts the same frame flavour
    # the wrapper was fit on; we never materialise / convert the frame here.
    point = np.asarray(self.predict(x), dtype=np.float64).reshape(-1)
    if point.shape[0] != n:
        raise ValueError(f"update_conformal: predict produced {point.shape[0]} rows but y has " f"{n} -- caller passed misaligned (x, y).")
    for i in range(n):
        # Interval formed with the radius CURRENT as of this row (pre-update),
        # then the controller steps for the next row -- the online contract.
        radius = _aci_radius(state)
        yi = y_arr[i]
        in_interval = bool(np.isfinite(yi) and (point[i] - radius) <= yi <= (point[i] + radius))
        residual = yi - point[i]
        _aci_step(state, residual, in_interval)
    state["last_radius"] = _aci_radius(state)
    return get_aci_state(self)


def get_aci_state(self) -> Dict[str, Any]:
    """Diagnostic snapshot of the ACI controller (no heavy buffers copied out)."""
    state = getattr(self, "_aci_state_", None)
    if state is None:
        return {"initialised": False}
    errors = state["errors"]
    rolling_cov = (1.0 - float(np.mean(errors))) if errors else float("nan")
    lifetime_cov = 1.0 - state["n_miss"] / state["n_seen"] if state["n_seen"] else float("nan")
    return {
        "initialised": True,
        "alpha_target": state["alpha_target"],
        "alpha_t": state["alpha_t"],
        "gamma": state["gamma"],
        "buffer_n": state["buffer_n"],
        "buffer_fill": len(state["residuals"]),
        "radius": state["last_radius"],
        "rolling_coverage": rolling_cov,
        "lifetime_coverage": lifetime_cov,
        "n_seen": state["n_seen"],
        "n_miss": state["n_miss"],
    }
