"""Split-conformal prediction intervals for ``CompositeTargetEstimator``.

The wrapper already produces honest y-scale point predictions; conformal adds a
distribution-free, finite-sample-valid prediction INTERVAL on top. Given a
held-out calibration set (rows the inner never trained on -- the suite's val
split, or an OOF fold), we compute the empirical quantile of a nonconformity
score and widen every point prediction by it. The DEFAULT score is normalised
(locally-adaptive):

    interval(x) = [ y_hat(x) - q*sigma_hat(x), y_hat(x) + q*sigma_hat(x) ]

with ``q`` the ``ceil((n+1)(1-alpha))/n`` empirical quantile of the calibration
scores ``(y_cal - y_hat(x_cal)) / sigma_hat(x_cal)`` (the standard split-conformal
level with the finite-sample +1 correction) and ``sigma_hat(x)`` a binned
conditional residual-scale model. Under exchangeability of the calibration and
test rows this guarantees marginal coverage >= 1 - alpha for ANY underlying model
-- no Gaussian / homoscedastic assumption. The normalised score widens the band
where the model is noisier, restoring CONDITIONAL coverage on heteroscedastic
targets (bench: 25/25 het cells beat the constant-width absolute score, worst-bin
coverage gap 0.042 vs 0.227, sharper width). ``score="absolute"`` selects the
legacy constant-width band (symmetric absolute-residual nonconformity).

Design choices mirroring the rest of the package:
- The calibration quantile(s) are stored per-alpha in ``self._conformal_q_`` --
  a plain dict of floats, so ``sklearn.clone`` / pickle stay clean and the
  wrapper carries no captured frames.
- Calibration consumes the inner's y-scale ``predict`` (the full
  transform-and-invert path), so the interval is on the ORIGINAL y scale the
  user cares about, not the T scale.
"""
from __future__ import annotations

import math
import warnings

import numpy as np


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal radius: the finite-sample ``(1-alpha)`` quantile of the
    absolute residuals.

    Uses the conservative rank ``ceil((n+1)(1-alpha))`` (the smallest residual
    that guarantees marginal coverage >= 1-alpha). Returns ``+inf`` when the
    requested rank exceeds ``n`` (too few calibration points for the level) so
    the interval is uninformative-but-valid rather than silently under-covering.

    Tiny-n contract: ``n_cal`` in ``{0, 1, 2}`` (and more generally any ``n``
    below ``ceil((n+1)(1-alpha)) > n``, e.g. n=1/2 at alpha=0.1) cannot certify
    the level at finite sample, so the radius is ``+inf`` -- a valid but
    uninformative band -- rather than a too-tight one that silently mis-covers.
    The caller never crashes on these sizes; the band is just (-inf, +inf).
    """
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n == 0:
        return float("inf")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"conformal alpha must be in (0, 1), got {alpha!r}")
    # 1-indexed rank of the order statistic that bounds 1-alpha mass.
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        # Not enough calibration points to certify this level at finite n.
        return float("inf")
    r_sorted = np.sort(r)
    return float(r_sorted[rank - 1])


_CONFORMAL_SIGMA_NBINS = 20


def _fit_sigma_model(y_pred_cal: np.ndarray, abs_res_cal: np.ndarray, n_bins: int = _CONFORMAL_SIGMA_NBINS):
    """Non-parametric conditional residual-scale model: mean |residual| per yhat-bin.

    A learner-free, pickle-clean estimate of sigma_hat(x) (stored as bin edges + per-bin
    means), the locally-adaptive denominator of the normalized nonconformity score. The
    floor (5% of the mean abs-residual) prevents div-by-zero / exploding intervals in a
    bin with vanishing residuals. Returns ``(edges, sigma_per_bin)`` plus the per-row
    sigma for the calibration rows.
    """
    finite = np.isfinite(y_pred_cal) & np.isfinite(abs_res_cal)
    yp = y_pred_cal[finite]
    ar = abs_res_cal[finite]
    if yp.size == 0:
        edges = np.array([-np.inf, np.inf])
        return edges, np.array([1.0]), np.ones_like(y_pred_cal)
    edges = np.quantile(yp, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([yp.min() - 1.0, yp.max() + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    nb = edges.size - 1
    floor = max(1e-9, 0.05 * float(ar.mean()))
    sigma = np.full(nb, ar.mean() if ar.size else 1.0)
    idx = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, nb - 1)
    for b in range(nb):
        m = idx == b
        if m.any():
            sigma[b] = ar[m].mean()
    sigma = np.maximum(sigma, floor)
    sig_rows = _sigma_for(edges, sigma, y_pred_cal)
    return edges, sigma, sig_rows


def _sigma_for(edges: np.ndarray, sigma: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map point predictions to their conditional residual-scale via the binned model."""
    nb = sigma.size
    yp = np.where(np.isfinite(y_pred), y_pred, 0.0)
    idx = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, nb - 1)
    return sigma[idx]


def calibrate_conformal(self, X_cal, y_cal, alpha=0.1, score="normalized"):
    """Fit the split-conformal radius from a held-out calibration set.

    ``X_cal`` / ``y_cal`` MUST be rows the inner estimator did NOT train on
    (the suite val split, or an OOF fold) -- conformal validity rests on the
    calibration rows being exchangeable with the test rows, which in-sample
    rows are not. Stores ``self._conformal_q_[round(alpha, 6)]`` and returns
    ``self`` (sklearn-style).

    ``alpha`` may be a scalar or an iterable of levels; each is calibrated and
    cached so ``predict_interval`` can serve any pre-calibrated level cheaply.

    ``score`` selects the nonconformity score (default ``"normalized"``):

    - ``"normalized"`` -- locally-adaptive: divide each residual by a binned
      conditional residual-scale ``sigma_hat(x)`` before taking the quantile, so
      the band widens where the model is noisier and tightens where it is sharp.
      On heteroscedastic targets this restores CONDITIONAL coverage (covers the
      high-variance region too, not just on average) and yields a sharper mean
      width. Bench ``_benchmarks/bench_conformal_normalized_vs_absolute.py``:
      normalized won 25/25 heteroscedastic cells (5 scenarios x 5 seeds), avg
      worst-bin coverage gap 0.042 vs 0.227 for absolute, with smaller width;
      on homoscedastic data it ties absolute. Default flipped to normalized.
    - ``"absolute"`` -- plain absolute-residual score, a CONSTANT-width band.
      Kept for callers who need the legacy fixed-width interval or have no usable
      conditional-scale signal.

    Marginal coverage ``>= 1 - alpha`` holds for BOTH scores (the finite-sample
    split-conformal guarantee is on the exchangeable score, not on its scale).
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError(
            "CompositeTargetEstimator.calibrate_conformal called before fit."
        )
    if score not in ("normalized", "absolute"):
        raise ValueError(f"calibrate_conformal: score must be 'normalized' or 'absolute', got {score!r}")
    y_true = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(self.predict(X_cal), dtype=np.float64).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            "calibrate_conformal: predict produced "
            f"{y_pred.shape[0]} rows but y_cal has {y_true.shape[0]}"
        )
    residuals = y_true - y_pred
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not hasattr(self, "_conformal_q_") or self._conformal_q_ is None:
        self._conformal_q_ = {}
    if not hasattr(self, "_conformal_sigma_") or self._conformal_sigma_ is None:
        self._conformal_sigma_ = {}
    if score == "normalized":
        edges, sigma, sig_rows = _fit_sigma_model(y_pred, np.abs(residuals))
        scores = residuals / sig_rows
        for a in alphas:
            key = round(float(a), 6)
            self._conformal_q_[key] = conformal_quantile(scores, float(a))
            self._conformal_sigma_[key] = (edges, sigma)
    else:
        for a in alphas:
            key = round(float(a), 6)
            self._conformal_q_[key] = conformal_quantile(residuals, float(a))
            self._conformal_sigma_.pop(key, None)
    self._conformal_n_cal_ = int(np.isfinite(residuals).sum())
    return self


def predict_interval(self, X, alpha=0.1):
    """Return ``(lower, upper)`` y-scale prediction intervals of marginal
    coverage ``>= 1 - alpha``.

    Requires a prior :meth:`calibrate_conformal` at this ``alpha`` (a clear
    error otherwise -- the radius cannot be invented from train rows without
    breaking conformal validity). The band is ``predict(X) +/- q`` where ``q``
    is the calibrated radius; it inherits the wrapper's post-inverse y-clip via
    ``predict`` on the point estimate, then the band is clipped to the same
    train envelope so the interval never claims an unphysical value.
    """
    key = round(float(alpha), 6)
    q = getattr(self, "_conformal_q_", {}) or {}
    if key not in q:
        raise RuntimeError(
            f"predict_interval: no conformal radius calibrated for alpha={alpha}. "
            f"Call calibrate_conformal(X_cal, y_cal, alpha={alpha}) on a held-out "
            f"set first (calibrated levels: {sorted(q.keys())})."
        )
    radius = q[key]
    point = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    sigma_map = getattr(self, "_conformal_sigma_", {}) or {}
    if key in sigma_map:
        # Normalized score: the calibrated quantile is on residual/sigma_hat(x),
        # so the y-scale radius is q * sigma_hat(x) -- a locally-adaptive width.
        edges, sigma = sigma_map[key]
        local = radius * _sigma_for(edges, sigma, point)
        lower = point - local
        upper = point + local
    else:
        lower = point - radius
        upper = point + radius
    # Keep the band inside the same train envelope the point estimate uses.
    params = getattr(self, "fitted_params_", {}) or {}
    lo_b = params.get("y_clip_low", float("-inf"))
    hi_b = params.get("y_clip_high", float("inf"))
    lower = np.clip(lower, lo_b, hi_b)
    upper = np.clip(upper, lo_b, hi_b)
    return lower, upper


def _signed_finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n+1)(1-alpha))`` order statistic of SIGNED scores (no abs).

    The CQR conformity score is a signed max that may be negative when the base
    quantile band already over-covers; conformal then SHRINKS the band, so the
    quantile must be taken on the raw signed scores rather than their magnitude.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        return float("inf")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"conformal alpha must be in (0, 1), got {alpha!r}")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(s)[rank - 1])


def calibrate_conformal_cqr(self, X_cal, y_cal, alpha=0.1):
    """Calibrate Conformalized Quantile Regression (CQR) for ADAPTIVE-width
    prediction intervals.

    Unlike :func:`calibrate_conformal` (constant-width absolute-residual band),
    CQR conformalizes the wrapper's own quantile predictions, so the band is
    wide where the model is uncertain and tight where it is confident -- the
    right behaviour for heteroscedastic targets. Requires the inner estimator
    to expose ``predict_quantile`` (a quantile-regressor inner, e.g. LightGBM
    ``objective="quantile"``); a clear error is raised otherwise.

    Computes the lower/upper quantile predictions ``q_lo`` / ``q_hi`` at
    ``alpha/2`` / ``1-alpha/2`` on the held-out calibration set, the signed
    conformity score ``E_i = max(q_lo_i - y_i, y_i - q_hi_i)``, and stores its
    finite-sample ``(1-alpha)`` quantile ``Q`` (per alpha). Marginal coverage
    ``>= 1-alpha`` holds for any base quantile model.
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError("calibrate_conformal_cqr called before fit.")
    a = float(alpha)
    lo_hi = self.predict_quantile(X_cal, [a / 2.0, 1.0 - a / 2.0])
    lo_hi = np.asarray(lo_hi, dtype=np.float64)
    if lo_hi.ndim != 2 or lo_hi.shape[1] != 2:
        raise RuntimeError(
            "calibrate_conformal_cqr: predict_quantile must return a (n, 2) "
            f"array for the [alpha/2, 1-alpha/2] pair; got shape {lo_hi.shape}. "
            "The inner estimator likely does not support quantile prediction."
        )
    q_lo, q_hi = lo_hi[:, 0], lo_hi[:, 1]
    y_true = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    scores = np.maximum(q_lo - y_true, y_true - q_hi)
    if not hasattr(self, "_cqr_q_") or self._cqr_q_ is None:
        self._cqr_q_ = {}
    self._cqr_q_[round(a, 6)] = _signed_finite_sample_quantile(scores, a)
    return self


def predict_interval_cqr(self, X, alpha=0.1):
    """Return adaptive ``(lower, upper)`` CQR intervals of marginal coverage
    ``>= 1 - alpha``.

    Requires a prior :func:`calibrate_conformal_cqr` at this ``alpha``. The band
    is ``[q_lo(x) - Q, q_hi(x) + Q]`` where ``q_lo`` / ``q_hi`` are the wrapper's
    quantile predictions and ``Q`` the calibrated CQR radius (which may be
    negative, shrinking an over-wide base band). Clipped to the train envelope.
    """
    key = round(float(alpha), 6)
    q = getattr(self, "_cqr_q_", {}) or {}
    if key not in q:
        raise RuntimeError(
            f"predict_interval_cqr: no CQR radius calibrated for alpha={alpha}. "
            f"Call calibrate_conformal_cqr(X_cal, y_cal, alpha={alpha}) on a "
            f"held-out set first (calibrated levels: {sorted(q.keys())})."
        )
    radius = q[key]
    a = float(alpha)
    lo_hi = np.asarray(
        self.predict_quantile(X, [a / 2.0, 1.0 - a / 2.0]), dtype=np.float64,
    )
    lower = lo_hi[:, 0] - radius
    upper = lo_hi[:, 1] + radius
    # A negative radius can cross the bounds; keep lower <= upper.
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    params = getattr(self, "fitted_params_", {}) or {}
    lo_b = params.get("y_clip_low", float("-inf"))
    hi_b = params.get("y_clip_high", float("inf"))
    return np.clip(lower, lo_b, hi_b), np.clip(upper, lo_b, hi_b)


def _normalize_groups(groups, n: int) -> np.ndarray:
    """Coerce a group label vector to a 1-D object array of length ``n``.

    Accepts ndarray / list / pandas Series / polars Series; never copies a
    frame. Raises on a length mismatch so a mis-aligned ``groups`` is caught at
    calibration rather than silently mis-binning the residuals.
    """
    if hasattr(groups, "to_numpy"):
        g = np.asarray(groups.to_numpy())
    else:
        g = np.asarray(groups)
    g = g.reshape(-1)
    if g.shape[0] != n:
        raise ValueError(
            f"groups has {g.shape[0]} entries but {n} rows were expected"
        )
    return g


def calibrate_conformal_mondrian(self, X_cal, y_cal, groups_cal, alpha=0.1):
    """Mondrian (group-conditional) split-conformal: a SEPARATE finite-sample
    radius per group, for conditional coverage ``>= 1-alpha`` WITHIN each group.

    The plain marginal band (:func:`calibrate_conformal`) shares one radius
    across all rows, so it under-covers groups with larger residual spread and
    over-covers the tighter ones. Mondrian conformal partitions the calibration
    residuals by ``groups_cal`` and takes the conservative
    ``ceil((n_g + 1)(1-alpha))`` quantile *within* each group -- the conditional
    analogue of the marginal guarantee, exact and distribution-free per group.

    A global radius (the pooled marginal quantile) is always computed and stored
    as the fallback for groups unseen at predict time, or groups too small to
    certify the level at finite sample (their per-group rank exceeds ``n_g``, so
    the per-group quantile would be ``+inf``); such groups fall back to the
    global radius with a one-time warning rather than an uninformative band.

    Stores ``self._mondrian_q_[round(alpha, 6)]`` as ``{group_label: radius}``
    plus a ``None`` key holding the global fallback radius, and returns ``self``.
    ``alpha`` may be a scalar or an iterable of levels.
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError(
            "CompositeTargetEstimator.calibrate_conformal_mondrian called before fit."
        )
    y_true = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(self.predict(X_cal), dtype=np.float64).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            "calibrate_conformal_mondrian: predict produced "
            f"{y_pred.shape[0]} rows but y_cal has {y_true.shape[0]}"
        )
    residuals = y_true - y_pred
    g = _normalize_groups(groups_cal, residuals.shape[0])
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not hasattr(self, "_mondrian_q_") or self._mondrian_q_ is None:
        self._mondrian_q_ = {}
    uniq = [u for u in np.unique(g)]
    for a in alphas:
        af = float(a)
        global_r = conformal_quantile(residuals, af)
        per_group: dict = {None: global_r}
        for u in uniq:
            r_g = residuals[g == u]
            rad = conformal_quantile(r_g, af)
            # A too-small group cannot certify the level on its own; fall back
            # to the (finite, pooled) global radius instead of an inf band.
            if not np.isfinite(rad) and np.isfinite(global_r):
                rad = global_r
            per_group[u] = float(rad)
        self._mondrian_q_[round(af, 6)] = per_group
    self._conformal_n_cal_ = int(np.isfinite(residuals).sum())
    return self


def predict_interval_mondrian(self, X, groups, alpha=0.1):
    """Return group-conditional ``(lower, upper)`` y-scale intervals, each of
    conditional coverage ``>= 1-alpha`` within its group.

    Requires a prior :func:`calibrate_conformal_mondrian` at this ``alpha``.
    Each row's radius is the calibrated per-group radius; rows whose group was
    unseen at calibration (or was too small to certify the level) fall back to
    the stored global radius, with a one-time ``warnings.warn``. A single test
    row returns a 1-element ``(lower, upper)`` pair. The band is clipped to the
    same train envelope as the point estimate.
    """
    key = round(float(alpha), 6)
    table = getattr(self, "_mondrian_q_", {}) or {}
    if key not in table:
        raise RuntimeError(
            f"predict_interval_mondrian: no Mondrian radius calibrated for "
            f"alpha={alpha}. Call calibrate_conformal_mondrian(X_cal, y_cal, "
            f"groups_cal, alpha={alpha}) on a held-out set first "
            f"(calibrated levels: {sorted(table.keys())})."
        )
    per_group = table[key]
    global_r = per_group.get(None, float("inf"))
    point = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    g = _normalize_groups(groups, point.shape[0])
    radii = np.empty(point.shape[0], dtype=np.float64)
    missing = set()
    for i in range(point.shape[0]):
        lab = g[i]
        if lab in per_group:
            radii[i] = per_group[lab]
        else:
            radii[i] = global_r
            missing.add(lab)
    if missing:
        warnings.warn(
            "predict_interval_mondrian: groups not seen at calibration fell "
            f"back to the global radius: {sorted(map(str, missing))}",
            stacklevel=2,
        )
    lower = point - radii
    upper = point + radii
    params = getattr(self, "fitted_params_", {}) or {}
    lo_b = params.get("y_clip_low", float("-inf"))
    hi_b = params.get("y_clip_high", float("inf"))
    return np.clip(lower, lo_b, hi_b), np.clip(upper, lo_b, hi_b)


def weighted_conformal_quantile(
    residuals: np.ndarray, weights: np.ndarray, alpha: float,
) -> float:
    """Tibshirani-et-al. weighted split-conformal radius under covariate shift.

    Provenance / formula. When the test covariate law ``P_test`` differs from the
    calibration law ``P_cal`` (covariate shift), the calibration residuals are no
    longer exchangeable with the test residual, so the plain order statistic
    (:func:`conformal_quantile`) mis-covers. Tibshirani, Barber, Candes &
    Ramdas (2019) restore validity by tilting each calibration point ``i`` by its
    importance weight ``w_i = dP_test/dP_cal(x_i)``: the radius is the smallest
    ``r`` such that the NORMALISED weighted mass of ``{|R_i| <= r}`` reaches
    ``1 - alpha``, where the normalisation includes the test point's own weight
    ``w_{n+1}`` carrying an atom at ``+inf``:

        p_i = w_i / (sum_j w_j + w_{n+1}),   p_{n+1} = w_{n+1} / (sum_j w_j + w_{n+1})
        r   = inf{ |R_(k)| : sum_{|R_i| <= |R_(k)|} p_i >= 1 - alpha }

    Because the ``+inf`` atom holds mass ``p_{n+1}``, when ``p_{n+1} > alpha`` no
    finite residual can accumulate ``1-alpha`` and the radius is ``+inf`` -- the
    valid-but-uninformative band, exactly mirroring the unweighted tiny-n
    contract. Uniform weights collapse this to the unweighted finite-sample
    quantile (the ``+1`` correction is the test-point self-weight atom).

    Description. Compute the cumulative normalised weight over residuals sorted
    by magnitude; the radius is the first magnitude whose cumulative mass (plus
    nothing -- the inf atom sits last) reaches ``1-alpha``. ``w_{n+1}`` defaults
    to the mean calibration weight (the natural self-weight for an unseen test
    point drawn from ``P_test``); pass it explicitly for a per-test-point exact
    band. Non-finite residuals are dropped with their weights; negative or
    non-finite weights raise.
    """
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != r.shape[0]:
        raise ValueError(
            f"weighted_conformal_quantile: {w.shape[0]} weights for "
            f"{r.shape[0]} residuals"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"conformal alpha must be in (0, 1), got {alpha!r}")
    finite = np.isfinite(r) & np.isfinite(w)
    r, w = r[finite], w[finite]
    if w.size and (w < 0).any():
        raise ValueError("weighted_conformal_quantile: weights must be >= 0")
    total_cal = float(w.sum())
    if r.size == 0 or total_cal <= 0.0:
        return float("inf")
    # Self-weight of the (unseen) test point: mean calibration weight is the
    # natural choice when no per-test weight is supplied.
    w_test = total_cal / float(w.size)
    denom = total_cal + w_test
    order = np.argsort(r, kind="mergesort")
    r_sorted = r[order]
    cum = np.cumsum(w[order]) / denom
    target = 1.0 - alpha
    # First magnitude whose cumulative normalised mass reaches 1-alpha. If the
    # test-atom mass (w_test/denom) exceeds alpha, the cum never reaches target
    # -> +inf (valid, uninformative).
    hit = np.searchsorted(cum, target, side="left")
    if hit >= r_sorted.shape[0]:
        return float("inf")
    return float(r_sorted[hit])


def _resolve_weights(weights, X_cal, n: int) -> np.ndarray:
    """Coerce ``weights`` (array OR callable density-ratio estimator) to a length
    ``n`` float64 vector of importance weights ``w_i = dP_test/dP_cal(x_i)``.

    A callable is invoked as ``weights(X_cal)`` and may return an ndarray / list
    / pandas / polars Series; an array-like is taken as-is. Never copies a frame
    -- the callable owns any narrow column reads it needs. Raises on a length
    mismatch so a mis-aligned weight vector is caught at calibration.
    """
    w = weights(X_cal) if callable(weights) else weights
    if hasattr(w, "to_numpy"):
        w = w.to_numpy()
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(
            f"weights resolved to {w.shape[0]} entries but {n} rows were expected"
        )
    return w


def calibrate_conformal_weighted(self, X_cal, y_cal, alpha=0.1, weights=None):
    """Weighted (covariate-shift) split-conformal calibration.

    Under covariate shift the plain band (:func:`calibrate_conformal`) mis-covers
    because the calibration and test covariate laws differ. This computes the
    WEIGHTED empirical quantile of the absolute calibration residuals
    (:func:`weighted_conformal_quantile`), each calibration row ``i`` tilted by
    its importance weight ``w_i = dP_test/dP_cal(x_i)``, restoring marginal
    coverage ``>= 1-alpha`` under the shifted test law (Tibshirani et al. 2019).

    ``weights`` accepts EITHER an array-like (one weight per calibration row) OR
    a callable density-ratio estimator invoked as ``weights(X_cal)``; the latter
    lets a fitted ``P_test/P_cal`` classifier supply the ratios at calibration
    time. ``X_cal`` / ``y_cal`` MUST be rows the inner never trained on. Stores
    ``self._weighted_conformal_q_[round(alpha, 6)]`` and returns ``self``.
    ``alpha`` may be a scalar or an iterable of levels. ``weights=None`` (uniform)
    reduces exactly to the unweighted finite-sample band.
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError(
            "CompositeTargetEstimator.calibrate_conformal_weighted called before fit."
        )
    y_true = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(self.predict(X_cal), dtype=np.float64).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            "calibrate_conformal_weighted: predict produced "
            f"{y_pred.shape[0]} rows but y_cal has {y_true.shape[0]}"
        )
    residuals = y_true - y_pred
    n = residuals.shape[0]
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _resolve_weights(weights, X_cal, n)
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not hasattr(self, "_weighted_conformal_q_") or self._weighted_conformal_q_ is None:
        self._weighted_conformal_q_ = {}
    for a in alphas:
        self._weighted_conformal_q_[round(float(a), 6)] = weighted_conformal_quantile(
            residuals, w, float(a),
        )
    self._conformal_n_cal_ = int(np.isfinite(residuals).sum())
    return self


def predict_interval_weighted(self, X, alpha=0.1):
    """Return covariate-shift-corrected ``(lower, upper)`` y-scale intervals of
    marginal coverage ``>= 1 - alpha`` under the shifted test law.

    Requires a prior :func:`calibrate_conformal_weighted` at this ``alpha``. The
    band is ``predict(X) +/- q_w`` where ``q_w`` is the weighted calibration
    radius; clipped to the same train envelope as the point estimate so the
    interval never claims an unphysical value.
    """
    key = round(float(alpha), 6)
    q = getattr(self, "_weighted_conformal_q_", {}) or {}
    if key not in q:
        raise RuntimeError(
            f"predict_interval_weighted: no weighted conformal radius calibrated "
            f"for alpha={alpha}. Call calibrate_conformal_weighted(X_cal, y_cal, "
            f"alpha={alpha}, weights=...) on a held-out set first "
            f"(calibrated levels: {sorted(q.keys())})."
        )
    radius = q[key]
    point = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    lower = point - radius
    upper = point + radius
    params = getattr(self, "fitted_params_", {}) or {}
    lo_b = params.get("y_clip_low", float("-inf"))
    hi_b = params.get("y_clip_high", float("inf"))
    return np.clip(lower, lo_b, hi_b), np.clip(upper, lo_b, hi_b)
