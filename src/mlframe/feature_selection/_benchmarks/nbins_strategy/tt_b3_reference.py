"""Reference implementation of the model-free TARGET-transform recommender (wave 1 consensus, written and re-tested by wave-2 agent B3, 2026-11-24).

Question answered without training the downstream model: which monotone unary transform f of the regression target y lets a LINEAR (or additive-spline) head predict y better, and is
"identity" good enough? Only numpy is used; the function is deterministic for a seed and every step is vectorised over the candidate transforms.

Algorithm (``recommend_target_transform``)
------------------------------------------
1. Candidates by the RANGE of y (``_valid_names``): identity, cbrt, signed_sqrt, signed_square, asinh(y / std), Yeo-Johnson (lambda fitted) always; log and log1p only for y > 0 / y >= 0;
   reciprocal ``-1/y`` only for single-signed y; logit only for y inside (0, 1); Box-Cox only on request. cbrt / signed_sqrt have an infinite derivative at 0 and are dropped when more than 0.2 percent
   of y is exactly 0 (a tie, not a small value: the Jacobian would reward the atom).
2. Rows are split at random into halves A and B. The design is D = [1, X] (``head="linear"``) or [1, X, natural cubic spline terms of every continuous column] (``head="spline"``, additive mean).
   The closed-form ridge solution ``beta = (D_A'D_A + ridge I)^-1 D_A' f(y_A)`` is computed ONCE and applied to the whole matrix Z = [f_1(y), ..., f_K(y)] with a single matrix product.
3. Score on B, for all candidates at once, the Jacobian-corrected profile log-likelihood per row ``LL_f = -0.5 log MSE_B(f) + mean_B log|f'(y)|`` (the Box-Cox likelihood with covariates; MSE_B is the
   held-out mean squared residual of the head on f(y)). Its first term is the Gaussian information 0.5 log(1/(1-R2_f)) between f(y) and X, the Jacobian makes the values comparable across f
   (raw R2 on f(y) is not: an exploding inverse inflates it). The gain is ``LL_f - LL_identity``. A rank MI is the same for every monotone f and cannot select; an MI on equal-width bins is biased.
   Yeo-Johnson (and Box-Cox) lambda is fitted on A by the profile likelihood on a 17-point grid refined by a parabola; a Yeo-Johnson with |lambda - 1| < 0.1 is the identity and is not a candidate.
4. Null: ``n_null`` sign-flip residual bootstraps from the identity head fitted on all rows (``y* = clip(fit + s r, min y, max y)``, s = random signs: keeps heteroscedasticity and heavy tails,
   keeps y inside the domain of every candidate). Each draw reruns steps 2-3 including the lambda fits and records the MAXIMUM gain over the non-identity candidates. The threshold tau is the
   ceil((1 - alpha)(n_null + 1))-th smallest draw (a finite-sample valid permutation p-value, no fixed margin); the identity is rejected only when the observed best gain exceeds tau.
5. Every candidate whose own gain exceeds tau is significant (the max statistic controls the family-wise error). They are ordered by gain; a candidate within ``tie_se`` paired standard errors of the
   best is a tie and the SIMPLEST family wins (log < cbrt, signed_sqrt, log1p, reciprocal, logit < asinh, signed_square < Box-Cox, Yeo-Johnson).
6. Second gate (needed because the Gaussian likelihood also rewards a better tail fit that does not reduce the error of the mean): the head fitted on f(y_A) is inverted (prediction clipped to the training
   range of f(y) +- 5 percent, Duan smearing generalised to any monotone f: ``mean_k inv(zhat + r_k)`` over 100 residual quantiles; the median ``inv(zhat)`` for ``metric="mae"``) and its error on B on the ORIGINAL
   scale must not exceed that of the identity head. The first candidate of the order that passes wins (``fallthrough=True``); otherwise the function abstains with the identity and says why.

Result: ``TargetTransformChoice`` (name, params, LL gain, tau, p-value, held-out error before and after, abstain reason, the usable forward / clipped-inverse pair). ``analyse_target_transforms`` returns
the full table (``decide`` turns it into a choice for a given metric), useful for the metadata of ``MRMR.target_transform_``.

Porting notes for MRMR
----------------------
* Input: the numeric columns the estimator already holds (at most ``max_rows`` rows are used, a seeded subsample) and the raw target; the target given to the estimator is never changed, only the verdict is
  stored in ``target_transform_`` (name, params, ll_gain, null_threshold, p_value, err_identity, err_chosen, reason; ``TargetTransformChoice.transform`` carries the clipped inverse for the recipe).
* Cost (measured in the B3 run, one thread, CPU shared with other jobs so an upper bound): 0.6 s at 1500 rows x 5..40 columns, 4 s at 15000 rows with ``n_null=40`` of which 0.4 s is the analysis itself
  and the rest the null (about 0.1 s per draw); spline head 5..8 s. ``n_null`` must be at least 19 for alpha = 0.05. The equal-width-MI + ridge-gate pipeline it replaces costs 0.13..1.3 s.
* Heads: valid for linear / additive heads. On trees the picks are neutral to slightly harmful (B3: HistGB RMSE worse than identity by more than 2 percent in 11-12 percent of cases at p=5), so do not
  apply the recipe to a GBDT head.
* Measured corrections to the wave-1 text (found by fresh fixtures): (a) the clip of the prediction on the transformed scale must not be widened toward the asymptote of a bounded-image transform
  (Yeo-Johnson with lambda < 0 or > 2, Box-Cox, reciprocal), a 5-10 percent margin gave test errors of 1e9..1e10 times the identity; (b) the "atom at 0" rule must test exact ties, not small values
  (cube-like targets have a density spike at 0 and lost cbrt in 25 percent of the seeds).
* Known failure modes (B3 report, section 6): lambda at the edge of [-1, 3] means the family is too small (neg_skew, log_comp: 13 percent of those picks are worse than identity); the density-optimal
  Yeo-Johnson lambda is not the error-optimal one (``tt_b3_ext.polish_lambda`` repairs it: good picks 88 -> 93.5 percent on the p=40 blocks); a linear head on a nonlinear mean prefers a milder
  transform than the likelihood does (log_comp); RMSE of heavy-tailed targets at 750 held-out rows is noisy (hetero_log at n=2000).
* float64 is used on purpose: the gains to resolve are 1e-3 of a nat; float32 normal equations were not tested.
* Not covered: classification, non-monotone transforms, targets with fewer than ``min_distinct`` distinct values (abstains), NaN rows (dropped).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FittedTransform", "TargetTransformChoice", "TransformAnalysis", "analyse_target_transforms", "decide", "recommend_target_transform", "build_transform", "DEFAULT_CANDIDATES"]

DEFAULT_CANDIDATES: Tuple[str, ...] = ("identity", "cbrt", "signed_sqrt", "signed_square", "asinh", "log", "log1p", "reciprocal", "logit", "yeo_johnson")
_ALL_CANDIDATES = (*DEFAULT_CANDIDATES, "box_cox")
_PARAMETRIC = ("yeo_johnson", "box_cox")
_GRID = {"yeo_johnson": np.linspace(-1.0, 3.0, 17), "box_cox": np.linspace(-2.0, 2.0, 17)}
_COMPLEXITY = {"identity": 0, "log": 1, "cbrt": 2, "signed_sqrt": 2, "log1p": 2, "reciprocal": 2, "logit": 2, "asinh": 3, "signed_square": 3, "box_cox": 4, "yeo_johnson": 4}
_MARGIN = 0.05
_N_SMEAR = 100
_TINY = 1e-300
_YJ_COLLAPSE = 0.1
_ATOM_FRACTION = 0.002


# ------------------------------------------------------------------------------------------------ transforms: forward, log-Jacobian log|f'|, inverse
def _fixed_forward(name: str, y: np.ndarray, s: float) -> np.ndarray:
    if name == "identity":
        return y
    if name == "cbrt":
        return np.cbrt(y)
    if name == "signed_sqrt":
        return np.sign(y) * np.sqrt(np.abs(y))
    if name == "signed_square":
        return np.sign(y) * y * y
    if name == "asinh":
        return np.arcsinh(y / s)
    if name == "log":
        return np.log(y)
    if name == "log1p":
        return np.log1p(y)
    if name == "reciprocal":
        return -1.0 / y
    if name == "logit":
        c = np.clip(y, 1e-9, 1.0 - 1e-9)
        return np.log(c / (1.0 - c))
    raise ValueError(name)


def _fixed_logjac(name: str, y: np.ndarray, s: float) -> np.ndarray:
    ay = np.maximum(np.abs(y), _TINY)
    if name == "identity":
        return np.zeros_like(y)
    if name == "cbrt":
        return -math.log(3.0) - (2.0 / 3.0) * np.log(ay)
    if name == "signed_sqrt":
        return -math.log(2.0) - 0.5 * np.log(ay)
    if name == "signed_square":
        return math.log(2.0) + np.log(ay)
    if name == "asinh":
        return -0.5 * np.log(s * s + y * y)
    if name == "log":
        return -np.log(y)
    if name == "log1p":
        return -np.log1p(y)
    if name == "reciprocal":
        return -2.0 * np.log(ay)
    if name == "logit":
        c = np.clip(y, 1e-9, 1.0 - 1e-9)
        return -np.log(c) - np.log1p(-c)
    raise ValueError(name)


def _fixed_inverse(name: str, z: np.ndarray, s: float) -> np.ndarray:
    if name == "identity":
        return z
    if name == "cbrt":
        return z**3
    if name == "signed_sqrt":
        return np.sign(z) * z * z
    if name == "signed_square":
        return np.sign(z) * np.sqrt(np.abs(z))
    if name == "asinh":
        return s * np.sinh(z)
    if name == "log":
        return np.exp(z)
    if name == "log1p":
        return np.expm1(z)
    if name == "reciprocal":
        return -1.0 / z
    if name == "logit":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(name)


def _param_forward(name: str, y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """(n,) x (L,) -> (n, L): Yeo-Johnson or Box-Cox of ``y`` for every lambda at once (expm1 / log1p forms, exact limit at lambda -> 0)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    small = np.abs(lam) < 1e-8
    if name == "box_cox":
        ly = np.log(y)[:, None]
        out = np.expm1(ly * np.where(small, 1.0, lam)[None, :]) / np.where(small, 1.0, lam)[None, :]
        out[:, small] = ly
        return out
    pos = y >= 0
    out = np.empty((len(y), len(lam)))
    for mask, ex, sign in ((pos, lam, 1.0), (~pos, 2.0 - lam, -1.0)):
        if not mask.any():
            continue
        ly = np.log1p(np.abs(y[mask]))[:, None]
        sm = np.abs(ex) < 1e-8
        exs = np.where(sm, 1.0, ex)
        blk = np.expm1(ly * exs[None, :]) / exs[None, :]
        blk[:, sm] = ly
        out[mask] = sign * blk
    return out


def _param_logjac(name: str, y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    if name == "box_cox":
        return np.log(y)[:, None] * (lam - 1.0)[None, :]
    return (np.sign(y) * np.log1p(np.abs(y)))[:, None] * (lam - 1.0)[None, :]


def _param_inverse(name: str, z: np.ndarray, lam: float) -> np.ndarray:
    eps = 1e-12
    if name == "box_cox":
        if abs(lam) < 1e-8:
            return np.exp(z)
        return np.maximum(lam * z + 1.0, eps) ** (1.0 / lam)
    out = np.empty_like(z)
    pos = z >= 0
    zp, zn = z[pos], z[~pos]
    out[pos] = np.expm1(zp) if abs(lam) < 1e-8 else np.maximum(zp * lam + 1.0, eps) ** (1.0 / lam) - 1.0
    out[~pos] = -np.expm1(-zn) if abs(lam - 2.0) < 1e-8 else 1.0 - np.maximum(1.0 - (2.0 - lam) * zn, eps) ** (1.0 / (2.0 - lam))
    return out


def _safe_bounds(name: str, z_train: np.ndarray, margin: float, lam: Optional[float] = None) -> Tuple[float, float]:
    """Clip range for a prediction on the transformed scale before inverting: the training range of f(y) widened by ``margin``. No widening toward an asymptote: reciprocal never crosses zero,
    and Yeo-Johnson / Box-Cox with a bounded image (lambda < 0 above, lambda > 2 / > 0 below) are not extended on that side (the inverse is unbounded at the image edge)."""
    lo, hi = float(z_train.min()), float(z_train.max())
    r = hi - lo
    zlo, zhi = lo, hi
    lo, hi = lo - margin * r, hi + margin * r
    if name == "reciprocal":
        if z_train.max() < 0:
            hi = min(hi, zhi / 2.0)
        else:
            lo = max(lo, zlo / 2.0)
    elif name == "yeo_johnson" and lam is not None:
        if lam < 0.0:
            hi = zhi
        if lam > 2.0:
            lo = zlo
    elif name == "box_cox" and lam is not None:
        if lam < 0.0:
            hi = zhi
        elif lam > 0.0:
            lo = zlo
    return lo, hi


@dataclasses.dataclass(frozen=True)
class FittedTransform:
    """A fitted monotone transform with a safe inverse (clipped to the training range of f(y); optional smearing over residuals on the transformed scale)."""

    name: str
    params: Dict[str, float]
    scale: float
    z_lo: float
    z_hi: float

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.name in _PARAMETRIC:
            return _param_forward(self.name, y.ravel(), np.array([self.params["lambda"]]))[:, 0].reshape(y.shape)
        return _fixed_forward(self.name, y, self.scale)

    def _inv(self, z: np.ndarray) -> np.ndarray:
        if self.name in _PARAMETRIC:
            return _param_inverse(self.name, z, self.params["lambda"])
        return _fixed_inverse(self.name, z, self.scale)

    def inverse(self, z: np.ndarray, residuals: Optional[np.ndarray] = None) -> np.ndarray:
        """Back to the original scale; ``residuals`` (transformed-scale residuals of the model) switch on generalised Duan smearing (mean prediction), otherwise the median prediction."""
        z = np.asarray(z, dtype=np.float64)
        with np.errstate(all="ignore"):
            if residuals is None:
                return self._inv(np.clip(z, self.z_lo, self.z_hi))
            q = np.quantile(residuals, (np.arange(_N_SMEAR) + 0.5) / _N_SMEAR)
            return self._inv(np.clip(z[..., None] + q, self.z_lo, self.z_hi)).mean(-1)


def build_transform(name: str, y_ref: np.ndarray, lam: Optional[float] = None) -> FittedTransform:
    """Fitted transform ``name`` with the clip range taken from ``y_ref`` (the training target); ``lam`` is required for Yeo-Johnson / Box-Cox."""
    y_ref = np.asarray(y_ref, dtype=np.float64).ravel()
    s = float(np.std(y_ref)) + 1e-12
    if name in _PARAMETRIC:
        z = _param_forward(name, y_ref, np.array([lam]))[:, 0]
        params = {"lambda": float(lam)}
    else:
        z = _fixed_forward(name, y_ref, s)
        params = {}
    lo, hi = _safe_bounds(name, z, _MARGIN, lam)
    return FittedTransform(name, params, s, lo, hi)


# ------------------------------------------------------------------------------------------------ designs and closed-form ridge folds
def _nat_spline(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (Harrell's restricted cubic spline, k-2 nonlinear columns) for knots ``t``."""
    k = len(t)
    d = t[-1] - t[-2]
    out = np.empty((len(x), k - 2))
    p3 = lambda v: np.maximum(v, 0.0) ** 3  # noqa: E731
    for j in range(k - 2):
        out[:, j] = (p3(x - t[j]) - (p3(x - t[k - 2]) * (t[k - 1] - t[j]) - p3(x - t[k - 1]) * (t[k - 2] - t[j])) / d) / (t[-1] - t[0]) ** 2
    return out


def _design(X: np.ndarray, head: str) -> np.ndarray:
    """Standardised columns of X (+ spline terms of every column with >= 10 distinct values for ``head="spline"``); constant columns are dropped. No intercept (folds centre)."""
    sd = X.std(0)
    keep = sd > 1e-12
    X = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    cols = [X]
    if head == "spline":
        for j in range(X.shape[1]):
            x = X[:, j]
            if np.unique(x).size < 10:
                continue
            t = np.unique(np.quantile(x, [0.05, 0.35, 0.65, 0.95]))
            if len(t) == 4:
                cols.append(_nat_spline(x, t))
    D = np.hstack(cols)
    sd = D.std(0)
    return (D[:, sd > 1e-12] - D[:, sd > 1e-12].mean(0)) / sd[sd > 1e-12]


@dataclasses.dataclass
class _Fold:
    """Closed-form ridge fitted on rows ``tr`` and applied to rows ``ev`` (centred design blocks, projection matrix P = (G + ridge I)^-1 D_tr')."""

    tr: np.ndarray
    ev: np.ndarray
    P: np.ndarray
    Dtr: np.ndarray
    Dev: np.ndarray

    @staticmethod
    def make(D: np.ndarray, tr: np.ndarray, ev: np.ndarray, ridge: float) -> "_Fold":
        mu = D[tr].mean(0)
        Dtr, Dev = D[tr] - mu, D[ev] - mu
        G = Dtr.T @ Dtr
        G[np.diag_indices_from(G)] += ridge * len(tr)
        return _Fold(tr, ev, np.linalg.solve(G, Dtr.T), Dtr, Dev)

    def fit(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(zbar, beta) of the head for every column of Z (n, K) using the training rows."""
        zbar = Z[self.tr].mean(0)
        return zbar, self.P @ (Z[self.tr] - zbar)


# ------------------------------------------------------------------------------------------------ candidates and scoring of all of them at once
@dataclasses.dataclass
class _Spec:
    fixed: List[str]
    param: List[str]
    scale: float

    @property
    def names(self) -> List[str]:
        return self.fixed + self.param


def _valid_names(y: np.ndarray, wanted: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(valid names, notes): candidates whose domain contains the range of ``y``; identity always first."""
    ymin, ymax = float(y.min()), float(y.max())
    float(np.std(y)) + 1e-12
    ok = {
        "identity": True, "cbrt": True, "signed_sqrt": True, "signed_square": True, "asinh": True, "yeo_johnson": True,
        "log": ymin > 0.0, "log1p": ymin >= 0.0, "box_cox": ymin > 0.0, "reciprocal": ymin > 0.0 or ymax < 0.0, "logit": ymin > 0.0 and ymax < 1.0,
    }
    notes: List[str] = []
    if float(np.mean(y == 0.0)) > _ATOM_FRACTION:
        ok["cbrt"] = ok["signed_sqrt"] = False
        notes.append("atom at 0: cbrt and signed_sqrt dropped (infinite derivative at 0)")
    names = ["identity"] + [n for n in wanted if n != "identity" and ok.get(n, False)]
    return names, notes


def _build(y: np.ndarray, spec: _Spec) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    Zf = np.column_stack([_fixed_forward(n, y, spec.scale) for n in spec.fixed])
    Jf = np.column_stack([_fixed_logjac(n, y, spec.scale) for n in spec.fixed])
    grids = {n: (_param_forward(n, y, _GRID[n]), _param_logjac(n, y, _GRID[n])) for n in spec.param}
    return Zf, Jf, grids


def _pick_lambda(fold: _Fold, name: str, Zg: np.ndarray, Jg: np.ndarray) -> float:
    """lambda maximising the in-sample profile log-likelihood on the training rows of ``fold`` (grid + parabola); also used for the final recipe on all rows."""
    zbar, beta = fold.fit(Zg)
    e = Zg[fold.tr] - (zbar + fold.Dtr @ beta)
    ll = -0.5 * np.log(np.maximum((e * e).mean(0), _TINY)) + Jg[fold.tr].mean(0)
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    g = _GRID[name]
    i = int(np.argmax(ll))
    if 0 < i < len(g) - 1:
        a, b, c = ll[i - 1], ll[i], ll[i + 1]
        den = a - 2.0 * b + c
        if den < 0.0:
            return float(np.clip(g[i] + 0.5 * (g[1] - g[0]) * (a - c) / den, g[i - 1], g[i + 1]))
    return float(g[i])


def _score(fold: _Fold, spec: _Spec, y: np.ndarray, cols, want_rows: bool):
    """Held-out profile LL of every candidate: (ll (K,), rows (n_ev, K) or None, lam (K,), usable (K,)); K follows ``spec.names``."""
    Zf, Jf, grids = cols
    lam: List[float] = [np.nan] * len(spec.fixed)
    use = [True] * len(spec.fixed)
    Zc, Jc = [Zf], [Jf]
    for n in spec.param:
        Zg, Jg = grids[n]
        lv = _pick_lambda(fold, n, Zg, Jg)
        lam.append(lv)
        use.append(not (n == "yeo_johnson" and abs(lv - 1.0) < _YJ_COLLAPSE))
        Zc.append(_param_forward(n, y, np.array([lv])))
        Jc.append(_param_logjac(n, y, np.array([lv])))
    Z, J = np.hstack(Zc), np.hstack(Jc)
    zbar, beta = fold.fit(Z)
    e = Z[fold.ev] - (zbar + fold.Dev @ beta)
    mse = np.maximum((e * e).mean(0), _TINY)
    rows = -0.5 * np.log(mse) - 0.5 * (e * e / mse - 1.0) + J[fold.ev]
    ll = rows.mean(0)
    use = np.array(use) & np.isfinite(ll)
    return np.where(use, ll, -np.inf), (rows if want_rows else None), np.array(lam), use


def _score_all(folds: Sequence[_Fold], spec: _Spec, y: np.ndarray, want_rows: bool):
    cols = _build(y, spec)
    out = [_score(f, spec, y, cols, want_rows) for f in folds]
    ll = np.mean([o[0] for o in out], axis=0)
    rows = np.vstack([o[1] for o in out]) if want_rows else None
    lam = np.nanmean([o[2] for o in out], axis=0) if len(out) > 1 else out[0][2]
    return ll, rows, lam, np.all([o[3] for o in out], axis=0)


# ------------------------------------------------------------------------------------------------ original-scale held-out error of the head (second gate)
def _heldout_losses(fold: _Fold, y: np.ndarray, name: str, lam: float, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row (squared error of the smeared mean prediction, absolute error of the median prediction) on the evaluation rows of ``fold``, original scale."""
    if name in _PARAMETRIC:
        z = _param_forward(name, y, np.array([lam]))[:, 0]
    else:
        z = _fixed_forward(name, y, scale)
    zbar, beta = fold.fit(z[:, None])
    zh = zbar[0] + fold.Dev @ beta[:, 0]
    res = z[fold.tr] - (zbar[0] + fold.Dtr @ beta[:, 0])
    q = np.quantile(res, (np.arange(_N_SMEAR) + 0.5) / _N_SMEAR)
    inv = (lambda v: _param_inverse(name, v, lam)) if name in _PARAMETRIC else (lambda v: _fixed_inverse(name, v, scale))
    yt = y[fold.ev]
    with np.errstate(all="ignore"):
        for m in (_MARGIN, 0.0):
            lo, hi = _safe_bounds(name, z[fold.tr], m, lam if name in _PARAMETRIC else None)
            med = inv(np.clip(zh, lo, hi))
            smear = inv(np.clip(zh[:, None] + q[None, :], lo, hi)).mean(1)
            if np.all(np.isfinite(med)) and np.all(np.isfinite(smear)):
                break
    sq = (smear - yt) ** 2
    ab = np.abs(med - yt)
    return np.where(np.isfinite(sq), sq, np.inf), np.where(np.isfinite(ab), ab, np.inf)


# ------------------------------------------------------------------------------------------------ results
@dataclasses.dataclass
class TransformAnalysis:
    """Everything computed about one (X, y): per-candidate gains, the null, the held-out errors of the candidates that beat the null. ``decide`` turns it into a choice."""

    names: List[str]
    gain: Dict[str, float]  # LL_f - LL_identity (nats per row), -inf when the candidate is unusable
    gain_rows: Dict[str, np.ndarray]  # per-row paired differences vs identity (for the tie rule)
    lam_search: Dict[str, float]  # lambda fitted on the training half(s)
    lam_full: Dict[str, float]  # lambda refitted on all rows (the recipe)
    best: Optional[str]
    best_gain: float
    tau: float
    p_value: float
    null_max: np.ndarray
    passing: List[str]
    err: Dict[str, Dict[str, float]]  # name -> {"rmse": ..., "mae": ...} held-out original-scale error (identity and passing candidates)
    err_rows: Dict[str, Tuple[np.ndarray, np.ndarray]]
    y_ref: np.ndarray
    n_rows: int
    head: str
    notes: List[str]
    abstain_reason: Optional[str]
    seconds: float
    fold_list: Optional[list] = None  # the fitted closed-form folds (used by extensions that re-score candidates on the held-out rows)
    scale: float = 1.0  # asinh scale


@dataclasses.dataclass(frozen=True)
class TargetTransformChoice:
    """Verdict. ``name == "identity"`` with ``abstained=True`` means the search found no transform that is both significant and not worse on the original scale."""

    name: str
    params: Dict[str, float]
    ll_gain: float  # gain of the chosen transform; of the best candidate when abstaining on the null
    null_threshold: float
    p_value: float
    err_identity: float  # held-out original-scale error of the identity head (RMSE or MAE, see ``metric``)
    err_chosen: float  # the same for the chosen transform (equal to ``err_identity`` when abstaining)
    metric: str
    abstained: bool
    reason: str
    best_candidate: Optional[str]
    gains: Dict[str, float]
    transform: Optional[FittedTransform]
    seconds: float


def analyse_target_transforms(
    X: np.ndarray, y: np.ndarray, *, candidates: Optional[Sequence[str]] = None, n_null: int = 40, alpha: float = 0.05, seed: int = 0, head: str = "linear",
    folds: int = 1, ridge: Optional[float] = None, max_rows: int = 20000, min_distinct: int = 20,
) -> TransformAnalysis:
    """Steps 1-5 plus the held-out errors of the significant candidates (see the module docstring)."""
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        sel = np.sort(rng.choice(len(y), max_rows, replace=False))
        X, y = X[sel], y[sel]
    n = len(y)

    def empty(reason: str, notes: Sequence[str] = ()) -> TransformAnalysis:
        return TransformAnalysis(["identity"], {"identity": 0.0}, {}, {}, {}, None, 0.0, float("inf"), 1.0, np.zeros(0), [], {}, {}, y, n, head, list(notes), reason, time.perf_counter() - t0)

    if n < 100:
        return empty(f"too few rows ({n} < 100)")
    if np.unique(y).size < min_distinct:
        return empty(f"target has {np.unique(y).size} distinct values (< {min_distinct}): not continuous")
    if float(np.ptp(y)) <= 0.0:
        return empty("constant target")
    wanted = list(candidates) if candidates is not None else list(DEFAULT_CANDIDATES)
    bad = [c for c in wanted if c not in _ALL_CANDIDATES]
    if bad:
        raise ValueError(f"unknown candidates {bad}; known {list(_ALL_CANDIDATES)}")
    names, notes = _valid_names(y, wanted)
    if len(names) == 1:
        return empty("no non-identity candidate is valid for the range of y", notes)
    spec = _Spec([m for m in names if m not in _PARAMETRIC], [m for m in names if m in _PARAMETRIC], float(np.std(y)) + 1e-12)
    names = spec.names

    D = _design(X, head)
    if D.shape[1] == 0:
        return empty("no usable feature column", notes)
    ridge = (1e-4 if head == "linear" else 1e-2) if ridge is None else ridge
    perm = rng.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    fold_list = [_Fold.make(D, A, B, ridge)] + ([_Fold.make(D, B, A, ridge)] if folds == 2 else [])

    ll, rows, lam_s, usable = _score_all(fold_list, spec, y, True)
    gain = {m: float(ll[i] - ll[0]) for i, m in enumerate(names)}
    gain_rows = {m: rows[:, i] - rows[:, 0] for i, m in enumerate(names) if usable[i]}
    lam_search = {m: float(lam_s[i]) for i, m in enumerate(names) if m in _PARAMETRIC}
    cand = [m for i, m in enumerate(names) if i > 0 and usable[i]]
    best = max(cand, key=lambda m: gain[m]) if cand else None
    best_gain = gain[best] if best else 0.0

    # null: sign-flip residual bootstrap around the identity head fitted on all rows
    full = _Fold.make(D, np.arange(n), np.arange(n), ridge)
    zbar, beta = full.fit(y[:, None])
    fit = zbar[0] + full.Dev @ beta[:, 0]
    r = y - fit
    lo_y, hi_y = float(y.min()), float(y.max())
    null_max = np.empty(n_null)
    for b in range(n_null):
        ys = np.clip(fit + (rng.integers(0, 2, n) * 2 - 1) * r, lo_y, hi_y)
        llb, _, _, ub = _score_all(fold_list, spec, ys, False)
        g = llb[1:] - llb[0]
        g = g[ub[1:]]
        null_max[b] = g.max() if g.size else -np.inf
    k = math.ceil((1.0 - alpha) * (n_null + 1))
    tau = float(np.sort(null_max)[k - 1]) if 1 <= k <= n_null else float("inf")
    p_value = float((1 + np.sum(null_max >= best_gain)) / (n_null + 1)) if best else 1.0
    passing = sorted([m for m in cand if gain[m] > tau], key=lambda m: -gain[m])

    # lambda of the recipe: refit on all rows; held-out errors on the evaluation rows of every fold
    lam_full: Dict[str, float] = {}
    for m in spec.param:
        lam_full[m] = _pick_lambda(full, m, _param_forward(m, y, _GRID[m]), _param_logjac(m, y, _GRID[m]))
    err_rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    err: Dict[str, Dict[str, float]] = {}
    for m in ["identity"] + passing + ([best] if best and best not in passing else []):
        lam = lam_search.get(m, 0.0)
        parts = [_heldout_losses(f, y, m, lam, spec.scale) for f in fold_list]
        sq, ab = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        err_rows[m] = (sq, ab)
        err[m] = {"rmse": float(np.sqrt(sq.mean())), "mae": float(ab.mean())}
    reason = None
    if best is None:
        reason = "no usable non-identity candidate (all collapsed to identity or non-finite)"
    return TransformAnalysis(names, gain, gain_rows, lam_search, lam_full, best, best_gain, tau, p_value, null_max, passing, err, err_rows, y, n, head, notes, reason, time.perf_counter() - t0, fold_list, spec.scale)


def decide(an: TransformAnalysis, *, metric: str = "rmse", tie_se: float = 1.0, fallthrough: bool = True, alpha: float = 0.05, select: str = "ll") -> TargetTransformChoice:
    """Apply the tie rule and the second gate of an analysis for ``metric`` ('rmse': smeared mean prediction; 'mae': median prediction).

    ``select="ll"`` (the wave-1 recipe) orders the significant candidates by LL gain; ``select="err"`` orders them by the held-out original-scale error instead (the LL test still decides significance)."""
    e_id = an.err.get("identity", {}).get(metric, float("nan"))

    def abstain(reason: str) -> TargetTransformChoice:
        return TargetTransformChoice("identity", {}, an.best_gain, an.tau, an.p_value, e_id, e_id, metric, True, reason, an.best, dict(an.gain), None, an.seconds)

    if an.abstain_reason is not None:
        return abstain(an.abstain_reason)
    if not an.passing:
        return abstain(f"best candidate {an.best} gain {an.best_gain:.5f} <= null threshold {an.tau:.5f} (p = {an.p_value:.3f}, alpha = {alpha})")
    order = list(an.passing)
    if select == "err":
        order.sort(key=lambda m: an.err[m][metric])
        tie_se = -1.0
    top = order[0]
    tied = []
    for m in order:
        if m == top:
            tied.append(m)
            continue
        d = an.gain_rows[top] - an.gain_rows[m]
        if tie_se >= 0.0 and d.mean() <= tie_se * d.std() / math.sqrt(len(d)):
            tied.append(m)
    first = min(tied, key=lambda m: (_COMPLEXITY[m], -an.gain[m])) if tie_se >= 0.0 else top
    order = [first] + [m for m in order if m != first]
    tried: List[str] = []
    for m in order:
        e_m = an.err[m][metric]
        if e_m <= e_id:
            lam = an.lam_full.get(m)
            tr = build_transform(m, an.y_ref, lam)
            return TargetTransformChoice(m, tr.params, an.gain[m], an.tau, an.p_value, e_id, e_m, metric, False, "", an.best, dict(an.gain), tr, an.seconds)
        tried.append(f"{m} ({e_m:.4g} > {e_id:.4g})")
        if not fallthrough:
            break
    return abstain(f"significant candidates failed the held-out {metric} gate vs identity: " + ", ".join(tried))


def recommend_target_transform(
    X: np.ndarray, y: np.ndarray, *, candidates: Optional[Sequence[str]] = None, n_null: int = 40, alpha: float = 0.05, seed: int = 0, head: str = "linear", metric: str = "rmse",
    folds: int = 1, tie_se: float = 1.0, fallthrough: bool = True, ridge: Optional[float] = None, max_rows: int = 20000,
) -> TargetTransformChoice:
    """Recommend a monotone target transform for a linear / additive-spline head, or abstain (identity). See the module docstring for the algorithm.

    ``candidates``: names from ``DEFAULT_CANDIDATES`` (+ ``box_cox``); None = all that the range of y allows. ``n_null`` sign-flip draws (>= 19 needed for alpha 0.05), ``seed`` fixes the split, the
    null and the subsample. ``head``: "linear" (closed-form ridge on [1, X]) or "spline" (additive natural cubic splines). ``metric``: "rmse" or "mae" for the second gate.
    ``folds=2`` cross-fits (A->B and B->A) instead of A->B only."""
    an = analyse_target_transforms(X, y, candidates=candidates, n_null=n_null, alpha=alpha, seed=seed, head=head, folds=folds, ridge=ridge, max_rows=max_rows)
    return decide(an, metric=metric, tie_se=tie_se, fallthrough=fallthrough, alpha=alpha)
