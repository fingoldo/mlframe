"""WAIC / LOO-style information-criterion validation of transform choice.

An ADDITIONAL ranking signal for composite-target discovery, never a replacement
for the MI-gain / tiny-CV-RMSE gates that already decide which ``(base,
transform)`` specs survive. The motivation: MI-gain and in-sample fit can TIE two
transforms while one of them (``A``) genuinely generalises and the other (``B``)
has merely memorised the tiny screening sample. A widely-applicable-information-
criterion score -- the expected pointwise OUT-OF-FOLD predictive density of the
tiny-CV residuals, penalised for effective complexity -- separates them: the
overfit candidate pays for its across-fold instability and ranks below the
genuinely-generalising one.

This module is deliberately SELF-CONTAINED. It does not rewire the discovery
loop; it exposes pure scoring functions the caller MAY consult when the config
flag ``transform_waic_validation_enabled`` is set:

* :func:`waic_from_oof_residuals` -- the pure information-criterion kernel. Given
  per-fold out-of-fold residual arrays it returns a :class:`WaicScore` with the
  pointwise log predictive density (``elpd``), the effective-complexity penalty
  (``p_eff``), and the final ``waic`` (higher = better generalisation).
* :func:`compute_transform_waic` -- a tiny self-contained K-fold pass that fits a
  cheap model on a candidate target and returns its :class:`WaicScore`. Uses only
  sklearn + numpy so it never reaches into the heavyweight rerank machinery.
* :func:`rank_transforms_by_waic` -- a convenience wrapper returning a per-name
  ``{transform: WaicScore}`` map the discovery loop can fold into its ranking.

No frame copies: the caller passes already-extracted ``float64`` numpy arrays
(the standard ``extract_column_array`` narrow-column pull), so a 100 GB polars
frame never materialises here.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Floor on a fold's residual variance so the Gaussian log-density stays finite
# when a tiny model fits a fold almost perfectly (variance -> 0 would blow the
# log-lik to +inf and let an overfit candidate win by construction). Scaled by
# the target variance at call time; this is the relative floor.
_REL_VAR_FLOOR = 1e-6
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class WaicScore:
    """Result of a WAIC-style transform validation.

    ``waic`` is the headline number the caller ranks on -- the expected pointwise
    out-of-fold log predictive density (``elpd``) minus the effective-complexity
    penalty (``p_eff``). HIGHER is better (better out-of-fold generalisation per
    unit of effective complexity). All three are on the natural-log density scale.

    ``n_points`` is how many OOF residuals fed the score; ``n_folds`` how many
    folds contributed. ``valid`` is False when the inputs were too small / all
    non-finite to score (the caller should then ignore this candidate's WAIC and
    fall back to the MI-gain ranking alone).
    """

    waic: float
    elpd: float
    p_eff: float
    n_points: int
    n_folds: int
    valid: bool

    def __bool__(self) -> bool:  # truthiness == "did we get a usable score".
        return self.valid


_INVALID = WaicScore(
    waic=float("-inf"), elpd=float("-inf"), p_eff=float("nan"),
    n_points=0, n_folds=0, valid=False,
)


def waic_from_oof_residuals(
    fold_residuals: Sequence[np.ndarray],
    *,
    target_scale: Optional[float] = None,
    fold_scale_residuals: Optional[Sequence[np.ndarray]] = None,
) -> WaicScore:
    """Pure WAIC-style score from per-fold out-of-fold residual arrays.

    ``fold_residuals[k]`` is the 1-D array of ``y_true - y_pred`` for the rows
    that were HELD OUT in fold ``k`` (so every residual is an honest out-of-fold
    prediction error, never an in-sample one). The score models each fold's
    residuals as zero-mean Gaussian with that fold's own residual variance and
    sums the pointwise log predictive density:

        lpd_i = -0.5 * (log(2*pi*sigma_k^2) + r_i^2 / sigma_k^2)   (i in fold k)

    ``elpd`` is the MEAN pointwise lpd (per-point so it is comparable across
    candidates with different surviving row counts). The effective-complexity
    penalty ``p_eff`` is the across-fold dispersion of each fold's mean pointwise
    lpd: a transform that overfits the screen fits some folds far better than
    others, so its per-fold lpd is unstable -- exactly the WAIC ``p`` notion of
    "effective number of parameters" estimated from the variance of the pointwise
    log-likelihood. The returned ``waic = elpd - p_eff`` therefore rewards a
    transform that predicts held-out rows well AND consistently, and penalises
    one that wins on one fold's screen but not the rest.

    ``target_scale`` (typically ``std(y_target)``) sets the relative variance
    floor so a near-perfect fold fit cannot send the log-density to ``+inf``;
    when omitted it is inferred from the pooled residual scale.

    ``fold_scale_residuals[k]`` (when supplied) is the TRAIN-fold residual array
    used to estimate fold ``k``'s Gaussian variance. Estimating ``sigma_k^2``
    from the held-out residuals themselves self-normalises the density to each
    transform's own held-out scale: a transform with large but well-calibrated
    held-out errors gets the SAME per-point density as one with tiny errors, so
    the OOF-accuracy signal cancels out and WAIC stops ranking on generalisation.
    Estimating the variance from the train-fold residuals (a scale the held-out
    points did not see) makes the held-out log-density actually measure how well
    each transform predicts unseen rows. Falls back to the held-out residuals
    only when no train-fold scale is provided (legacy callers).

    Returns an invalid :class:`WaicScore` (``valid=False``) when fewer than two
    folds carry any finite residual -- the across-fold penalty is undefined with
    a single fold, so the caller must not rank on it.

    LABEL CAVEAT: this is NOT the canonical WAIC of Watanabe / Gelman. True WAIC needs the per-sample
    posterior predictive density and uses ``p_waic = sum_i Var_posterior(log p(y_i))`` -- the variance of
    the pointwise log-likelihood OVER THE POSTERIOR DRAWS. We have no posterior; this is a WAIC-FLAVOURED
    cross-validation proxy: ``elpd`` is the mean out-of-fold Gaussian log-density and ``p_eff`` is the
    ACROSS-FOLD dispersion of per-fold mean lpd, a coarse fold-granularity stand-in for the posterior
    variance term. It is a useful relative tie-breaker between transforms, not a calibrated information
    criterion -- do not compare its magnitude against a true WAIC / LOO-IC from a Bayesian fit.
    """
    clean: list[np.ndarray] = []
    scale_resid: list[Optional[np.ndarray]] = []
    have_scale = fold_scale_residuals is not None
    for j, arr in enumerate(fold_residuals):
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size >= 2:  # need >=2 points to estimate the fold variance.
            clean.append(a)
            if have_scale and j < len(fold_scale_residuals):
                sr = np.asarray(fold_scale_residuals[j], dtype=np.float64).ravel()
                sr = sr[np.isfinite(sr)]
                scale_resid.append(sr if sr.size >= 2 else None)
            else:
                scale_resid.append(None)
    if len(clean) < 2:
        return _INVALID

    pooled = np.concatenate(clean)
    if target_scale is not None and math.isfinite(target_scale) and target_scale > 0:
        scale = float(target_scale)
    else:
        scale = float(np.std(pooled)) or 1.0
    var_floor = (scale * scale) * _REL_VAR_FLOOR

    per_point_lpd: list[np.ndarray] = []
    fold_mean_lpd = np.empty(len(clean), dtype=np.float64)
    n_points = 0
    for k, r in enumerate(clean):
        # Fold variance about zero (the residual mean is ~0 by construction; using
        # the raw second moment keeps a biased fold that systematically mispredicts
        # from hiding its bias inside a re-centred variance). The variance is
        # estimated from the TRAIN-fold residuals when available so the held-out
        # density is NOT self-normalised to its own scale (see docstring).
        scale_arr = scale_resid[k] if scale_resid[k] is not None else r
        sigma2 = max(float(np.mean(scale_arr * scale_arr)), var_floor)
        lpd = -0.5 * (_LOG_2PI + math.log(sigma2) + (r * r) / sigma2)
        per_point_lpd.append(lpd)
        fold_mean_lpd[k] = float(np.mean(lpd))
        n_points += r.size

    elpd = float(np.mean(np.concatenate(per_point_lpd)))
    # Effective complexity: across-fold dispersion of the per-fold mean lpd. A
    # ddof=1 sample variance over the folds is the WAIC-2 "variance of the
    # pointwise log-likelihood" estimator collapsed to the fold granularity we
    # have (one held-out predictive surface per fold).
    p_eff = float(np.var(fold_mean_lpd, ddof=1))
    waic = elpd - p_eff
    return WaicScore(
        waic=waic, elpd=elpd, p_eff=p_eff,
        n_points=n_points, n_folds=len(clean), valid=True,
    )


def _oof_residuals_kfold(
    y_target: np.ndarray,
    x_matrix: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
    model_factory: Optional[Callable[[], object]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect per-fold out-of-fold AND train-fold residuals from a cheap K-fold pass.

    Self-contained: builds a small model (``model_factory`` or a default tiny
    gradient booster / ridge fallback), fits on K-1 folds, predicts BOTH the
    held-out fold and the train portion, and returns ``(oof_residuals,
    train_residuals)`` aligned per fold. The train-fold residuals supply the
    Gaussian scale for the held-out density so it is not self-normalised (see
    :func:`waic_from_oof_residuals`). No global state, no rerank machinery -- the
    only dependency is sklearn's ``KFold``.
    """
    from sklearn.model_selection import KFold

    y = np.asarray(y_target, dtype=np.float64).ravel()
    x = np.asarray(x_matrix, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y, x = y[finite], x[finite]
    n = y.size
    if n < n_folds * 4:  # too small for an honest K-fold split.
        return [], []

    factory = model_factory if model_factory is not None else _default_tiny_model
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    residuals: list[np.ndarray] = []
    train_residuals: list[np.ndarray] = []
    for tr, va in kf.split(x):
        try:
            model = factory()
            model.fit(x[tr], y[tr])
            pred = np.asarray(model.predict(x[va]), dtype=np.float64).ravel()
            pred_tr = np.asarray(model.predict(x[tr]), dtype=np.float64).ravel()
        except Exception as err:  # a single fold failing must not kill the score.
            logger.debug("WAIC fold fit failed: %s", err)
            continue
        residuals.append(y[va] - pred)
        train_residuals.append(y[tr] - pred_tr)
    return residuals, train_residuals


def _default_tiny_model():
    """A cheap, dependency-light regressor for the WAIC OOF pass.

    Prefers a small LightGBM (matches the rest of discovery's tiny models); falls
    back to a shallow sklearn ``HistGradientBoostingRegressor`` and finally to
    ``Ridge`` so the WAIC helper works with only sklearn installed.
    """
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=60, num_leaves=15, learning_rate=0.1,
            n_jobs=1, verbose=-1,
        )
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _eval_waic.py:250: %s", e)
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15)
    except Exception:
        from sklearn.linear_model import Ridge

        return Ridge()


def compute_transform_waic(
    y_target: np.ndarray,
    x_matrix: np.ndarray,
    *,
    n_folds: int = 4,
    random_state: int = 0,
    model_factory: Optional[Callable[[], object]] = None,
) -> WaicScore:
    """WAIC-style validation score for ONE candidate target on the screen sample.

    ``y_target`` is the candidate composite target ``T = transform(y, base)`` (or
    raw ``y``) and ``x_matrix`` the feature matrix used to predict it -- both as
    already-extracted ``float64`` numpy arrays (no frame copy). Runs a single
    cheap K-fold pass (see :func:`_oof_residuals_kfold`) to gather honest
    out-of-fold residuals, then scores them with :func:`waic_from_oof_residuals`.

    Returns an invalid :class:`WaicScore` (``valid=False``) when the sample is too
    small for an honest fit -- the caller should then drop WAIC from this
    candidate's ranking rather than treat ``-inf`` as a real score.
    """
    fold_residuals, train_fold_residuals = _oof_residuals_kfold(
        y_target, x_matrix,
        n_folds=n_folds, random_state=random_state, model_factory=model_factory,
    )
    if len(fold_residuals) < 2:
        return _INVALID
    scale = float(np.std(np.asarray(y_target, dtype=np.float64).ravel()))
    return waic_from_oof_residuals(
        fold_residuals, target_scale=scale, fold_scale_residuals=train_fold_residuals,
    )


def rank_transforms_by_waic(
    candidates: Sequence[tuple[str, np.ndarray]],
    x_matrix: np.ndarray,
    *,
    n_folds: int = 4,
    random_state: int = 0,
    model_factory: Optional[Callable[[], object]] = None,
) -> dict[str, WaicScore]:
    """Per-transform WAIC map the discovery loop MAY consult as a ranking signal.

    ``candidates`` is a sequence of ``(name, y_target)`` pairs -- one per surviving
    ``(base, transform)`` spec -- sharing the same feature ``x_matrix``. Returns a
    ``{name: WaicScore}`` dict; names whose sample was too small map to an invalid
    score so the caller can skip them. This NEVER decides selection on its own: it
    is an extra signal to break ties / down-rank overfit candidates that MI-gain
    alone cannot separate.
    """
    out: dict[str, WaicScore] = {}
    for name, y_target in candidates:
        out[name] = compute_transform_waic(
            y_target, x_matrix,
            n_folds=n_folds, random_state=random_state,
            model_factory=model_factory,
        )
    return out
