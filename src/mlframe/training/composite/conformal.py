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


def _conformal_internal_split(n_cal: int, time_ordering=None) -> tuple[np.ndarray, np.ndarray]:
    """Two-way calibration split for the normalized score's sigma_hat fit / calibrate halves.

    Returns ``(fit_idx, cal_idx)`` -- positional indices into the calibration rows.
    ``fit_idx`` fits sigma_hat; ``cal_idx`` computes the calibrated normalized scores.

    Random by default (exchangeable cross-sectional data). When ``time_ordering`` is
    truthy the split is BLOCKED in time: the earlier time-block fits sigma_hat and the
    later block calibrates, so the calibration never uses a future fold's scale to judge
    a past row (which a random split would, breaking the >= 1-alpha guarantee on temporally
    drifting data). ``time_ordering`` may be a 1-D sort key (rows are ordered by it) or
    ``True`` to take the rows as already time-ordered.
    """
    half = n_cal // 2
    if time_ordering is None or time_ordering is False:
        rng = np.random.default_rng(0)
        perm = rng.permutation(n_cal)
        return perm[:half], perm[half:]
    if time_ordering is True:
        order = np.arange(n_cal)
    else:
        key = np.asarray(time_ordering).ravel()
        if key.shape[0] != n_cal:
            raise ValueError(
                f"time_ordering has {key.shape[0]} entries but {n_cal} calibration rows were expected"
            )
        order = np.argsort(key, kind="mergesort")  # stable: ties keep input order
    return order[:half], order[half:]


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
    # Single grouped pass instead of nb full-array masked sweeps: two O(n) bincounts
    # over the bin index replace the per-bin ``ar[idx==b].mean()`` loop. Empty bins keep
    # the global-mean default; non-empty bins get sum/count (~1e-15 reduction-order vs the
    # masked mean, a sigma width-scale, never a selection score).
    counts = np.bincount(idx, minlength=nb)
    sums = np.bincount(idx, weights=ar, minlength=nb)
    nonempty = counts > 0
    sigma[nonempty] = sums[nonempty] / counts[nonempty]
    sigma = np.maximum(sigma, floor)
    sig_rows = _sigma_for(edges, sigma, y_pred_cal)
    return edges, sigma, sig_rows


def _sigma_for(edges: np.ndarray, sigma: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map point predictions to their conditional residual-scale via the binned model."""
    nb = sigma.size
    yp = np.where(np.isfinite(y_pred), y_pred, 0.0)
    idx = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, nb - 1)
    return sigma[idx]


def calibrate_conformal(self, X_cal, y_cal, alpha=0.1, score="normalized", time_ordering=None):
    """Fit the split-conformal radius from a held-out calibration set.

    ``X_cal`` / ``y_cal`` MUST be rows the inner estimator did NOT train on
    (the suite val split, or an OOF fold) -- conformal validity rests on the
    calibration rows being exchangeable with the test rows, which in-sample
    rows are not. Stores ``self._conformal_q_[round(alpha, 6)]`` and returns
    ``self`` (sklearn-style).

    ``time_ordering`` (1-D array of a sort key, or ``True`` to take the rows
    as already time-ordered) flags TEMPORAL calibration data. Exchangeability
    -- the assumption the >= 1-alpha guarantee rests on -- is broken by
    temporal drift, and a RANDOM internal split (used by the normalized score
    to keep sigma_hat independent of the calibrated scores) leaks future-fold
    information into the sigma_hat fit, under-covering. When set, the internal
    split becomes a BLOCKED one (the earlier time-block fits sigma_hat, the
    later block calibrates), so the calibration honours the arrow of time. It
    defaults on automatically when the estimator was fit on data flagged
    temporal (``self._is_temporal_`` / a stored ``time_ordering``).

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
        # Split the calibration set: fit sigma_hat on one half, compute the
        # normalized scores on the OTHER half. Fitting sigma_hat on the same
        # residuals it then normalizes makes the score artificially small in
        # well-fit bins (sigma_hat already saw those residuals), so the quantile
        # under-covers -- the band is anti-conservative. A clean two-way split
        # keeps sigma_hat and the calibrated scores independent, restoring the
        # exchangeability the >= 1-alpha guarantee rests on.
        n_cal = residuals.shape[0]
        # Default the temporal flag on when the fitted estimator carries one, so a
        # temporal model does not silently get a random (exchangeability-breaking) split.
        time_ord = time_ordering
        if time_ord is None:
            time_ord = getattr(self, "_is_temporal_", None) or getattr(self, "_time_ordering_", None)
        fit_idx, cal_idx = _conformal_internal_split(n_cal, time_ord)
        half = fit_idx.size
        if half >= 2 and cal_idx.size >= 1:
            edges, sigma, _ = _fit_sigma_model(y_pred[fit_idx], np.abs(residuals[fit_idx]))
            sig_rows = _sigma_for(edges, sigma, y_pred[cal_idx])
            scores = residuals[cal_idx] / sig_rows
        else:
            # Too few calibration rows to split; fall back to the pooled fit (the
            # tiny-n band is uninformative-but-valid via conformal_quantile's inf rank).
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


def _mondrian_ood_radius(certified_radii: list, global_r: float, alpha: float) -> float:
    """Conservative interval radius for OOD (unseen or uncertifiable) groups.

    An unseen test group is exchangeable with the CALIBRATION groups, so its required
    width is estimated by a GROUP-LEVEL conformal upper quantile of the per-group radii:
    treating each certified group's own radius as an exchangeable draw, the
    ``ceil((G+1)(1-alpha))`` order statistic covers a NEW group's width at ``>= 1-alpha``
    ACROSS groups. This is MEASURED from the calibration groups' own radii (no magic
    constant) and floored at the pooled radius, so it only ever INFLATES the legacy
    global fallback -- which under-covers exactly the OOD groups whose residual spread
    exceeds the pooled bulk. With too few certified groups to certify at the group level
    the widest observed group radius is used; with no certified group at all, the pooled
    radius (nothing better is measurable).
    """
    radii = np.asarray([r for r in certified_radii if np.isfinite(r)], dtype=np.float64)
    if radii.size == 0:
        return float(global_r)
    # radii are non-negative -> conformal_quantile's abs is a no-op; it returns the
    # ceil((G+1)(1-alpha)) order statistic (a conservative upper quantile of group radii).
    q = conformal_quantile(radii, alpha)
    if not np.isfinite(q):
        q = float(radii.max())
    if np.isfinite(global_r):
        q = max(q, float(global_r))
    return float(q)


def calibrate_conformal_mondrian(self, X_cal, y_cal, groups_cal, alpha=0.1):
    """Mondrian (group-conditional) split-conformal: a SEPARATE finite-sample
    radius per group, for conditional coverage ``>= 1-alpha`` WITHIN each group.

    The plain marginal band (:func:`calibrate_conformal`) shares one radius
    across all rows, so it under-covers groups with larger residual spread and
    over-covers the tighter ones. Mondrian conformal partitions the calibration
    residuals by ``groups_cal`` and takes the conservative
    ``ceil((n_g + 1)(1-alpha))`` quantile *within* each group -- the conditional
    analogue of the marginal guarantee, exact and distribution-free per group.

    For groups UNSEEN at predict time (or seen but too small to certify the level
    at finite sample -- their per-group rank exceeds ``n_g``, so the per-group
    quantile would be ``+inf``) the plain pooled radius UNDER-covers exactly the
    OOD groups that most need a wide band. Instead the fallback is OOD-adaptive: a
    conservative upper quantile of the calibration groups' own radii, floored at
    the pooled radius (:func:`_mondrian_ood_radius`), so unseen/uncertifiable
    groups get an inflated width measured from the between-group dispersion rather
    than the central pooled value. The exact per-SEEN-CERTIFIED-group radius is
    unchanged (bit-identical). Gate the inflation off with
    ``self.conformal_ood_adaptive = False`` (legacy raw-global fallback).

    Stores ``self._mondrian_q_[round(alpha, 6)]`` as ``{group_label: radius}``
    plus a ``None`` key holding the pooled global radius (both bit-identical to
    the pre-OOD path), and alongside it ``self._mondrian_ood_[round(alpha, 6)]``
    (the OOD-adaptive fallback radius) and ``self._mondrian_uncertified_[...]``
    (the set of seen-but-too-small labels). Returns ``self``. ``alpha`` may be a
    scalar or an iterable of levels.
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
    if not hasattr(self, "_mondrian_ood_") or self._mondrian_ood_ is None:
        self._mondrian_ood_ = {}
    if not hasattr(self, "_mondrian_uncertified_") or self._mondrian_uncertified_ is None:
        self._mondrian_uncertified_ = {}
    # Group the residuals into contiguous per-group blocks ONCE (factorize O(n) + a single
    # stable argsort) instead of building a fresh ``g == u`` boolean mask over all n residuals
    # for every unique group on every alpha (the old O(G*n)-per-alpha sweep). ``order`` sorts
    # rows by group code; ``starts``/``stops`` index each group's slice of ``res_by_group``.
    # The stable sort keeps each block in original row order, and conformal_quantile sorts
    # internally, so the per-group radius is bit-identical to the masked slice.
    import pandas as pd
    codes, uniq = pd.factorize(g, sort=True, use_na_sentinel=False)
    counts = np.bincount(codes, minlength=uniq.shape[0])
    order = np.argsort(codes, kind="stable")
    res_by_group = residuals[order]
    starts = np.zeros(uniq.shape[0], dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    stops = starts + counts
    for a in alphas:
        af = float(a)
        global_r = conformal_quantile(residuals, af)
        per_group: dict = {None: global_r}
        certified_radii: list = []  # own radii of groups that certified on their own rows
        uncertified: set = set()    # seen labels too small to certify -> OOD fallback at predict
        for j in range(uniq.shape[0]):
            r_g = res_by_group[starts[j]:stops[j]]
            own = conformal_quantile(r_g, af)
            if np.isfinite(own):
                per_group[uniq[j]] = float(own)
                certified_radii.append(float(own))
            else:
                # A too-small group cannot certify the level on its own; store the
                # (finite, pooled) global radius so ``_mondrian_q_`` stays bit-identical
                # to the pre-OOD path, but mark the label so predict routes it to the
                # OOD-adaptive width and flags it low-confidence.
                per_group[uniq[j]] = float(global_r) if np.isfinite(global_r) else float(own)
                uncertified.add(uniq[j])
        key = round(af, 6)
        self._mondrian_q_[key] = per_group
        self._mondrian_ood_[key] = _mondrian_ood_radius(certified_radii, global_r, af)
        self._mondrian_uncertified_[key] = frozenset(uncertified)
    self._conformal_n_cal_ = int(np.isfinite(residuals).sum())
    return self


def _record_mondrian_ood(self, key, ood_flags: np.ndarray) -> None:
    """Surface the OOD fallback rate into ``runtime_stats_`` (the summary the report /
    model card already read) plus a per-alpha snapshot, so the do-not-deploy verdict --
    'fraction of predict rows on OOD groups' -- is available downstream without recompute.
    """
    n = int(ood_flags.shape[0])
    n_ood = int(ood_flags.sum())
    frac = (n_ood / n) if n else 0.0
    rs = getattr(self, "runtime_stats_", None)
    if isinstance(rs, dict):
        rs["mondrian_ood_rows"] = rs.get("mondrian_ood_rows", 0) + n_ood
        rs["mondrian_pred_rows"] = rs.get("mondrian_pred_rows", 0) + n
        tot = rs["mondrian_pred_rows"]
        rs["mondrian_ood_fraction"] = (rs["mondrian_ood_rows"] / tot) if tot else 0.0
    snap = getattr(self, "_mondrian_ood_summary_", None)
    if not isinstance(snap, dict):
        snap = {}
        self._mondrian_ood_summary_ = snap
    snap[key] = {"n_rows": n, "n_ood": n_ood, "fraction_ood": frac}


def mondrian_ood_summary(self, alpha: float = 0.1) -> dict:
    """Do-not-deploy / low-confidence verdict aggregate for Mondrian intervals.

    Returns ``{"n_rows", "n_ood", "fraction_ood"}`` for the MOST RECENT
    :func:`predict_interval_mondrian` call at this ``alpha``: how many rows landed
    on a group that was unseen at calibration or too small to certify (so their
    band came from the OOD fallback and their point estimate is low-confidence --
    the model is extrapolating). Zeros before any predict. The running rate across
    all predicts also lives in ``runtime_stats_['mondrian_ood_fraction']``.
    """
    key = round(float(alpha), 6)
    snap = getattr(self, "_mondrian_ood_summary_", {}) or {}
    return dict(snap.get(key, {"n_rows": 0, "n_ood": 0, "fraction_ood": 0.0}))


def predict_interval_mondrian(self, X, groups, alpha=0.1, return_ood=False):
    """Return group-conditional ``(lower, upper)`` y-scale intervals, each of
    conditional coverage ``>= 1-alpha`` within its group.

    Requires a prior :func:`calibrate_conformal_mondrian` at this ``alpha``. Each
    SEEN-and-CERTIFIED row's radius is its calibrated per-group radius (unchanged).
    Rows whose group was unseen at calibration OR was too small to certify the
    level fall back to the OOD-ADAPTIVE radius (a conservative upper quantile of
    the calibration groups' own radii; :func:`_mondrian_ood_radius`) rather than
    the raw pooled radius that under-covers OOD groups, and are FLAGGED as
    low-confidence. Unseen labels also raise a one-time ``warnings.warn``. Set
    ``self.conformal_ood_adaptive = False`` to restore the legacy raw-global
    fallback (still flagged). A single test row returns a 1-element pair. The band
    is clipped to the same train envelope as the point estimate.

    ``return_ood=True`` returns ``(lower, upper, ood_flags)`` where ``ood_flags``
    is a boolean per-row mask, True exactly on rows served by the OOD fallback;
    the fraction is also recorded in ``runtime_stats_`` and via
    :func:`mondrian_ood_summary`.
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
    # OOD-adaptive fallback radius (measured between-group upper quantile), gated ON by
    # default (corrective mechanism). Absent stats (hand-set table / legacy pickle) or the
    # gate off -> fall back to the raw pooled radius, bit-identical to the pre-OOD path.
    ood_adaptive = bool(getattr(self, "conformal_ood_adaptive", True))
    ood_map = getattr(self, "_mondrian_ood_", {}) or {}
    ood_radius = ood_map.get(key, global_r) if ood_adaptive else global_r
    uncertified = (getattr(self, "_mondrian_uncertified_", {}) or {}).get(key, frozenset())
    point = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    g = _normalize_groups(groups, point.shape[0])
    # Vectorized per-group radius gather: hash-based factorize (O(n), C-level, no object
    # sort) maps rows to unique-label codes, the dict lookup runs once per UNIQUE label
    # (not once per row), and a single code-gather assigns the radius to every row --
    # replaces the O(n) Python loop, identical result. factorize beats np.unique here
    # because the labels are an object array (np.unique would object-sort, ~4x slower).
    import pandas as pd
    # use_na_sentinel=False keeps a NaN label as its own category (code, not -1) so the
    # ``lab in per_group`` test runs on it -- matching the loop, where a NaN group falls
    # through to the global radius rather than silently gathering the last unique radius.
    codes, uniq = pd.factorize(g, sort=False, use_na_sentinel=False)
    radius_per_uniq = np.empty(uniq.shape[0], dtype=np.float64)
    ood_per_uniq = np.zeros(uniq.shape[0], dtype=bool)
    missing = set()
    for j, lab in enumerate(uniq):
        if lab in per_group and lab not in uncertified:
            # Seen + certified on its own calibration rows: exact per-group radius (unchanged).
            radius_per_uniq[j] = per_group[lab]
        else:
            # Unseen OR seen-but-uncertifiable (too small): OOD fallback, flagged low-confidence.
            radius_per_uniq[j] = ood_radius
            ood_per_uniq[j] = True
            if lab not in per_group:
                missing.add(lab)
    radii = radius_per_uniq[codes]
    ood_flags = ood_per_uniq[codes]
    if missing:
        warnings.warn(
            "predict_interval_mondrian: groups not seen at calibration fell back to the "
            f"OOD-adaptive global radius: {sorted(map(str, missing))}",
            stacklevel=2,
        )
    _record_mondrian_ood(self, key, ood_flags)
    lower = point - radii
    upper = point + radii
    params = getattr(self, "fitted_params_", {}) or {}
    lo_b = params.get("y_clip_low", float("-inf"))
    hi_b = params.get("y_clip_high", float("inf"))
    lower = np.clip(lower, lo_b, hi_b)
    upper = np.clip(upper, lo_b, hi_b)
    if return_ood:
        return lower, upper, ood_flags
    return lower, upper


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
