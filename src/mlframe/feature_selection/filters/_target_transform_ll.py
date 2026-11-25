"""Model-free recommender of a monotone TARGET transform: the Jacobian log-likelihood ranking, a sign-flip null and a held-out original-scale gate (PLAN_target_transform_in_mrmr, section 2.5).

Question answered without training the downstream model: which monotone unary transform ``g`` of the regression target ``y`` lets a LINEAR (or additive-spline) head predict ``y`` better, and is the identity good
enough? ``y = U(F(x))`` with ``U`` invertible is recognised by the pool member that is ``U^-1``: ``exp`` -> ``log``, a square -> ``signed_sqrt``, a cube -> ``cbrt``, a square root -> ``signed_square``, a reciprocal ->
``reciprocal``, a sigmoid -> ``logit``, ``sinh`` -> ``asinh``, ``tanh`` -> ``atanh``, ``log`` -> ``exp``. Numpy only, deterministic for a seed, every step vectorised over the candidates.

Algorithm (``recommend_target_transform``)
------------------------------------------
1. Candidates by the RANGE of y (``_valid_names``): identity, cbrt, cube, signed_sqrt, signed_square, asinh, sinh, Yeo-Johnson (lambda fitted) always; log, log1p and exp only for y > 0 / y >= 0 / y < 30; reciprocal
   only for single-signed y; logit only for y inside (0, 1); atanh only for y inside (-1, 1); Box-Cox only on request. cbrt and signed_sqrt have an infinite derivative at 0 and are dropped when more than 0.2 percent
   of y is exactly 0 (a tie, not a small value: the Jacobian would reward the atom).
2. Rows are split at random into halves A and B. The design is ``D = [1, X]`` (``head="linear"``) or ``[1, X, natural cubic spline terms of every continuous column]`` (``head="spline"``). The closed-form ridge
   solution is computed ONCE and applied to the whole matrix ``Z = [f_1(y), ..., f_K(y)]`` with a single matrix product.
3. Score on B, for all candidates at once, the Jacobian-corrected profile log-likelihood per row ``LL_f = -0.5 log MSE_B(f) + mean_B log|f'(y)|`` (the Box-Cox likelihood with covariates). A rank MI is the same for
   every monotone f and cannot select; an MI on equal-width bins is biased; the raw R2 on f(y) is not comparable across f. The gain is ``LL_f - LL_identity``. The Yeo-Johnson (Box-Cox) lambda is fitted on A by the
   profile likelihood on a 17-point grid refined by a parabola; a Yeo-Johnson with ``|lambda - 1| < 0.1`` is the identity and is not a candidate.
4. Null: ``n_null`` sign-flip residual bootstraps from the identity head fitted on all rows (``y* = clip(fit + s r, min y, max y)``: keeps heteroscedasticity and heavy tails, keeps y inside the domain of every
   candidate). Each draw reruns steps 2-3 including the lambda fits and records the MAXIMUM gain over the non-identity candidates. The threshold tau is the ceil((1 - alpha)(n_null + 1))-th smallest draw.
5. Every candidate whose own gain exceeds tau is significant (the max statistic controls the family-wise error). They are ordered by gain; a candidate within ``tie_se`` paired standard errors of the best is a
   tie and the SIMPLEST family wins (``select="ll"``).
6. Second gate and selection: the head fitted on f(y_A) is inverted (prediction clipped to the training range of f(y); the median ``inv(zhat)`` for ``metric="mae"``, the default, generalised Duan smearing over 100
   residual quantiles for ``metric="rmse"``) and its error on B on the ORIGINAL scale must be at least ``min_gain`` (2 percent) below that of the identity head. The significant candidates are ordered by that error (``select="err"``, the default: the LL can
   prefer a transform that fits the tails and predicts the mean worse), those within ``tie_se`` paired standard errors of the best are a tie and the simplest family wins; otherwise the function abstains and says why.

Valid for linear and additive heads; a tree head does not gain from a target transform (measured: neutral to slightly harmful), so the verdict is for linear / additive heads. double is used on purpose.
Not covered: classification, non-monotone transforms, targets with fewer than ``min_distinct`` distinct values (abstains), NaN rows (dropped)."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_CANDIDATES",
    "FittedTransform",
    "TargetTransformChoice",
    "TransformAnalysis",
    "analyse_target_transforms",
    "build_transform",
    "decide",
    "recommend_target_transform",
]

DEFAULT_CANDIDATES: Tuple[str, ...] = (
    "identity",
    "cbrt",
    "cube",
    "signed_sqrt",
    "signed_square",
    "asinh",
    "sinh",
    "log",
    "log1p",
    "exp",
    "reciprocal",
    "logit",
    "atanh",
    "yeo_johnson",
)
_ALL_CANDIDATES = (*DEFAULT_CANDIDATES, "box_cox")
_PARAMETRIC = ("yeo_johnson", "box_cox", "asinh", "sinh")
_SCALED = ("asinh", "sinh")
# ``asinh`` and ``sinh`` have a free scale c (``asinh(y / c)``): the parameter is log2(c / std(y)), so ``y = sinh(index)`` with the unit scale of the data is recognised, not only a scale tied to the spread of y
_GRID = {"yeo_johnson": np.linspace(-1.0, 3.0, 17), "box_cox": np.linspace(-2.0, 2.0, 17), "asinh": np.arange(-10.0, 2.01, 0.5), "sinh": np.arange(-10.0, 2.01, 0.5)}
#: Simplicity order of the families: among candidates the data cannot tell apart the lowest wins.
_COMPLEXITY = {"identity": 0, "log": 1, "cbrt": 2, "signed_sqrt": 2, "log1p": 2, "reciprocal": 2, "logit": 2, "atanh": 2, "exp": 2, "asinh": 3, "signed_square": 3, "cube": 3, "sinh": 4, "box_cox": 5, "yeo_johnson": 5}
_MARGIN = 0.05
_N_SMEAR = 100
_TINY = 1e-300
_YJ_COLLAPSE = 0.1
_ATOM_FRACTION = 0.002
#: ``exp`` is offered only when its forward value stays far from overflow; ``sinh`` grid points whose argument ``y / c`` exceeds ``_SINH_ARG_MAX`` are dropped.
_EXP_MAX = 30.0
_SINH_ARG_MAX = 30.0
_UNIT_EDGE = 1e-9
#: A transform is recommended only when its held-out error on the original scale is at least this share below the identity's (a skewed target always lets some transform win by a fraction of a percent).
MIN_GAIN = 0.02


# ------------------------------------------------------------------------------------------------ transforms: forward, log-Jacobian log|f'|, inverse
def _fixed_forward(name: str, y: np.ndarray, s: float) -> np.ndarray:
    """Forward transform ``name`` of ``y`` (``s`` is the scale of the scaled families)."""
    if name == "identity":
        return y
    if name == "cbrt":
        return np.asarray(np.cbrt(y))
    if name == "cube":
        return (y / s) ** 3
    if name == "signed_sqrt":
        return np.asarray(np.sign(y) * np.sqrt(np.abs(y)))
    if name == "signed_square":
        return np.asarray(np.sign(y) * y * y)
    if name == "log":
        return np.asarray(np.log(y))
    if name == "log1p":
        return np.asarray(np.log1p(y))
    if name == "exp":
        return np.asarray(np.exp(y))
    if name == "reciprocal":
        return -1.0 / y
    if name == "logit":
        c = np.clip(y, _UNIT_EDGE, 1.0 - _UNIT_EDGE)
        return np.asarray(np.log(c / (1.0 - c)))
    if name == "atanh":
        return np.asarray(np.arctanh(np.clip(y, -1.0 + _UNIT_EDGE, 1.0 - _UNIT_EDGE)))
    raise ValueError(name)


def _fixed_logjac(name: str, y: np.ndarray, s: float) -> np.ndarray:
    """``log |f'(y)|`` of the forward transform ``name`` (the Jacobian term of the profile likelihood)."""
    ay = np.maximum(np.abs(y), _TINY)
    if name == "identity":
        return np.zeros_like(y)
    if name == "cbrt":
        return np.asarray(-math.log(3.0) - (2.0 / 3.0) * np.log(ay))
    if name == "cube":
        return np.asarray(math.log(3.0) + 2.0 * np.log(ay) - 3.0 * math.log(s))
    if name == "signed_sqrt":
        return np.asarray(-math.log(2.0) - 0.5 * np.log(ay))
    if name == "signed_square":
        return np.asarray(math.log(2.0) + np.log(ay))
    if name == "log":
        return np.asarray(-np.log(y))
    if name == "log1p":
        return np.asarray(-np.log1p(y))
    if name == "exp":
        # Intentional float64: likelihood and Jacobian statistics of the link search
        return np.asarray(y, dtype=np.float64)
    if name == "reciprocal":
        return np.asarray(-2.0 * np.log(ay))
    if name == "logit":
        c = np.clip(y, _UNIT_EDGE, 1.0 - _UNIT_EDGE)
        return np.asarray(-np.log(c) - np.log1p(-c))
    if name == "atanh":
        c = np.clip(y, -1.0 + _UNIT_EDGE, 1.0 - _UNIT_EDGE)
        return np.asarray(-np.log1p(-c * c))
    raise ValueError(name)


def _fixed_inverse(name: str, z: np.ndarray, s: float) -> np.ndarray:
    """Inverse of the forward transform ``name``."""
    if name == "identity":
        return z
    if name == "cbrt":
        return z**3
    if name == "cube":
        return np.asarray(s * np.cbrt(z))
    if name == "signed_sqrt":
        return np.asarray(np.sign(z) * z * z)
    if name == "signed_square":
        return np.asarray(np.sign(z) * np.sqrt(np.abs(z)))
    if name == "log":
        return np.asarray(np.exp(z))
    if name == "log1p":
        return np.asarray(np.expm1(z))
    if name == "exp":
        return np.asarray(np.log(np.maximum(z, _TINY)))
    if name == "reciprocal":
        return -1.0 / z
    if name == "logit":
        return np.asarray(1.0 / (1.0 + np.exp(-z)))
    if name == "atanh":
        return np.asarray(np.tanh(z))
    raise ValueError(name)


def _scale_grid(lam: np.ndarray, s: float) -> np.ndarray:
    """The scales ``c = s * 2**lam`` of the grid points of ``asinh`` / ``sinh``."""
    return np.asarray(s * np.exp2(lam))


def _param_forward(name: str, y: np.ndarray, lam: np.ndarray, s: float = 1.0) -> np.ndarray:
    """(n,) x (L,) -> (n, L): the transform ``name`` of ``y`` for every parameter at once (Yeo-Johnson / Box-Cox: expm1 / log1p forms, exact limit at lambda -> 0; asinh / sinh: ``c = s 2^lam``)."""
    # Intentional float64: likelihood and Jacobian statistics of the link search
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    if name in _SCALED:
        c = _scale_grid(lam, s)[None, :]
        with np.errstate(all="ignore"):
            if name == "asinh":
                return np.asarray(np.arcsinh(y[:, None] / c))
            arg = y[:, None] / c
            return np.where(np.abs(arg) < _SINH_ARG_MAX, np.sinh(np.clip(arg, -_SINH_ARG_MAX, _SINH_ARG_MAX)), np.nan)
    small = np.abs(lam) < 1e-8
    if name == "box_cox":
        ly = np.log(y)[:, None]
        out = np.expm1(ly * np.where(small, 1.0, lam)[None, :]) / np.where(small, 1.0, lam)[None, :]
        out[:, small] = ly
        return np.asarray(out)
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


def _param_logjac(name: str, y: np.ndarray, lam: np.ndarray, s: float = 1.0) -> np.ndarray:
    """``log |f'(y)|`` of the parametric transform ``name`` for every parameter at once."""
    # Intentional float64: likelihood and Jacobian statistics of the link search
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    if name in _SCALED:
        c = _scale_grid(lam, s)[None, :]
        with np.errstate(all="ignore"):
            if name == "asinh":
                return np.asarray(-0.5 * np.log(c * c + (y * y)[:, None]))
            arg = np.clip(y[:, None] / c, -_SINH_ARG_MAX, _SINH_ARG_MAX)
            return np.asarray(np.log(np.cosh(arg)) - np.log(c))
    if name == "box_cox":
        return np.asarray(np.log(y)[:, None] * (lam - 1.0)[None, :])
    return np.asarray((np.sign(y) * np.log1p(np.abs(y)))[:, None] * (lam - 1.0)[None, :])


def _param_inverse(name: str, z: np.ndarray, lam: float, s: float = 1.0) -> np.ndarray:
    """Inverse of the parametric transform ``name`` at ``lam`` (the base of a power is floored so the inverse stays finite)."""
    eps = 1e-12
    if name in _SCALED:
        c = float(_scale_grid(np.array([lam]), s)[0])
        return np.asarray(c * np.sinh(z) if name == "asinh" else c * np.arcsinh(z))
    if name == "box_cox":
        if abs(lam) < 1e-8:
            return np.asarray(np.exp(z))
        return np.maximum(lam * z + 1.0, eps) ** (1.0 / lam)
    out = np.empty_like(z)
    pos = z >= 0
    zp, zn = z[pos], z[~pos]
    out[pos] = np.expm1(zp) if abs(lam) < 1e-8 else np.maximum(zp * lam + 1.0, eps) ** (1.0 / lam) - 1.0
    out[~pos] = -np.expm1(-zn) if abs(lam - 2.0) < 1e-8 else 1.0 - np.maximum(1.0 - (2.0 - lam) * zn, eps) ** (1.0 / (2.0 - lam))
    return out


def _safe_bounds(name: str, z_train: np.ndarray, margin: float, lam: Optional[float] = None) -> Tuple[float, float]:
    """Clip range for a prediction on the transformed scale before inverting: the training range of f(y) widened by ``margin``.

    No widening toward an asymptote: reciprocal never crosses zero, ``exp`` stays positive, and Yeo-Johnson / Box-Cox with a bounded image (lambda < 0 above, lambda > 2 / > 0 below) are not extended on that side
    (the inverse is unbounded at the image edge)."""
    lo, hi = float(z_train.min()), float(z_train.max())
    r = hi - lo
    zlo, zhi = lo, hi
    lo, hi = lo - margin * r, hi + margin * r
    if name == "reciprocal":
        if z_train.max() < 0:
            hi = min(hi, zhi / 2.0)
        else:
            lo = max(lo, zlo / 2.0)
    elif name == "exp":
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
        """The transformed target."""
        # Intentional float64: likelihood and Jacobian statistics of the link search
        y = np.asarray(y, dtype=np.float64)
        if self.name in _PARAMETRIC:
            return _param_forward(self.name, y.ravel(), np.array([self.params["lambda"]]), self.scale)[:, 0].reshape(y.shape)
        return _fixed_forward(self.name, y, self.scale)

    def _inv(self, z: np.ndarray) -> np.ndarray:
        """Inverse without clipping."""
        if self.name in _PARAMETRIC:
            return _param_inverse(self.name, z, self.params["lambda"], self.scale)
        return _fixed_inverse(self.name, z, self.scale)

    def inverse(self, z: np.ndarray, residuals: Optional[np.ndarray] = None) -> np.ndarray:
        """Back to the original scale; ``residuals`` (transformed-scale residuals of the model) switch on generalised Duan smearing (mean prediction), otherwise the median prediction."""
        # Intentional float64: likelihood and Jacobian statistics of the link search
        z = np.asarray(z, dtype=np.float64)
        with np.errstate(all="ignore"):
            if residuals is None:
                return self._inv(np.clip(z, self.z_lo, self.z_hi))
            q = np.quantile(residuals, (np.arange(_N_SMEAR) + 0.5) / _N_SMEAR)
            return np.asarray(self._inv(np.clip(z[..., None] + q, self.z_lo, self.z_hi)).mean(-1))


def build_transform(name: str, y_ref: np.ndarray, lam: Optional[float] = None) -> FittedTransform:
    """Fitted transform ``name`` with the clip range taken from ``y_ref`` (the training target); ``lam`` is required for Yeo-Johnson / Box-Cox."""
    # Intentional float64: likelihood and Jacobian statistics of the link search
    y_ref = np.asarray(y_ref, dtype=np.float64).ravel()
    s = max(float(np.std(y_ref)), 1e-12)
    if name in _PARAMETRIC:
        if lam is None:
            raise ValueError(f"build_transform: {name} needs its lambda")
        z = _param_forward(name, y_ref, np.array([lam]), s)[:, 0]
        params = {"lambda": float(lam)}
        if name in _SCALED:
            params["scale"] = float(_scale_grid(np.array([lam]), s)[0])
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

    def p3(v: np.ndarray) -> np.ndarray:
        """Truncated cube."""
        return np.asarray(np.maximum(v, 0.0) ** 3, dtype=np.float64)

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
    return np.asarray((D[:, sd > 1e-12] - D[:, sd > 1e-12].mean(0)) / sd[sd > 1e-12])


@dataclasses.dataclass
class _Fold:
    """Closed-form ridge fitted on rows ``tr`` and applied to rows ``ev`` (centred design blocks, projection matrix ``P = (G + ridge I)^-1 D_tr'``)."""

    tr: np.ndarray
    ev: np.ndarray
    P: np.ndarray
    Dtr: np.ndarray
    Dev: np.ndarray

    @staticmethod
    def make(D: np.ndarray, tr: np.ndarray, ev: np.ndarray, ridge: float) -> "_Fold":
        """The fold of the design ``D`` for training rows ``tr`` and evaluation rows ``ev``."""
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
    """The candidate names split into the fixed and the parametric families, and the scale of the scaled ones."""

    fixed: List[str]
    param: List[str]
    scale: float

    @property
    def names(self) -> List[str]:
        """All candidate names, fixed first."""
        return self.fixed + self.param


def _valid_names(y: np.ndarray, wanted: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(valid names, notes): candidates whose domain contains the range of ``y``; identity always first."""
    ymin, ymax = float(y.min()), float(y.max())
    ok = {
        "identity": True,
        "cbrt": True,
        "cube": True,
        "signed_sqrt": True,
        "signed_square": True,
        "asinh": True,
        "yeo_johnson": True,
        "sinh": True,
        "log": ymin > 0.0,
        "log1p": ymin >= 0.0,
        "exp": ymax < _EXP_MAX,
        "box_cox": ymin > 0.0,
        "reciprocal": ymin > 0.0 or ymax < 0.0,
        "logit": ymin > 0.0 and ymax < 1.0,
        "atanh": ymin > -1.0 and ymax < 1.0,
    }
    notes: List[str] = []
    if float(np.mean(y == 0.0)) > _ATOM_FRACTION:
        ok["cbrt"] = ok["signed_sqrt"] = False
        notes.append("atom at 0: cbrt and signed_sqrt dropped (infinite derivative at 0)")
    names = ["identity"] + [n for n in wanted if n != "identity" and ok.get(n, False)]
    return names, notes


def _build(y: np.ndarray, spec: _Spec) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Forward values and log-Jacobians of the fixed candidates, and the lambda grids of the parametric ones."""
    Zf = np.column_stack([_fixed_forward(n, y, spec.scale) for n in spec.fixed])
    Jf = np.column_stack([_fixed_logjac(n, y, spec.scale) for n in spec.fixed])
    grids = {n: (_param_forward(n, y, _GRID[n], spec.scale), _param_logjac(n, y, _GRID[n], spec.scale)) for n in spec.param}
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


def _score(fold: _Fold, spec: _Spec, y: np.ndarray, cols: Tuple, want_rows: bool):
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
        Zc.append(_param_forward(n, y, np.array([lv]), spec.scale))
        Jc.append(_param_logjac(n, y, np.array([lv]), spec.scale))
    Z, J = np.hstack(Zc), np.hstack(Jc)
    zbar, beta = fold.fit(Z)
    e = Z[fold.ev] - (zbar + fold.Dev @ beta)
    mse = np.maximum((e * e).mean(0), _TINY)
    rows = -0.5 * np.log(mse) - 0.5 * (e * e / mse - 1.0) + J[fold.ev]
    ll = rows.mean(0)
    use = np.array(use) & np.isfinite(ll)
    return np.where(use, ll, -np.inf), (rows if want_rows else None), np.array(lam), use


def _score_all(folds: Sequence[_Fold], spec: _Spec, y: np.ndarray, want_rows: bool):
    """:func:`_score` averaged over the folds."""
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
        z = _param_forward(name, y, np.array([lam]), scale)[:, 0]
    else:
        z = _fixed_forward(name, y, scale)
    zbar, beta = fold.fit(z[:, None])
    zh = zbar[0] + fold.Dev @ beta[:, 0]
    res = z[fold.tr] - (zbar[0] + fold.Dtr @ beta[:, 0])
    q = np.quantile(res, (np.arange(_N_SMEAR) + 0.5) / _N_SMEAR)
    inv = (lambda v: _param_inverse(name, v, lam, scale)) if name in _PARAMETRIC else (lambda v: _fixed_inverse(name, v, scale))
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
    fold_list: Optional[list] = None  # the fitted closed-form folds
    scale: float = 1.0  # scale of the scaled families (std of y)


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
    X: np.ndarray,
    y: np.ndarray,
    *,
    candidates: Optional[Sequence[str]] = None,
    n_null: int = 40,
    alpha: float = 0.05,
    seed: int = 0,
    head: str = "linear",
    folds: int = 1,
    ridge: Optional[float] = None,
    max_rows: int = 20000,
    min_distinct: int = 20,
) -> TransformAnalysis:
    """Steps 1-5 plus the held-out errors of the significant candidates (see the module docstring)."""
    t0 = time.perf_counter()
    # Intentional float64: likelihood and Jacobian statistics of the link search
    X = np.asarray(X, dtype=np.float64)
    # Intentional float64: likelihood and Jacobian statistics of the link search
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
        """The analysis of a target the search cannot work on."""
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
    spec = _Spec([m for m in names if m not in _PARAMETRIC], [m for m in names if m in _PARAMETRIC], max(float(np.std(y)), 1e-12))
    names = spec.names

    D = _design(X, head)
    if D.shape[1] == 0:
        return empty("no usable feature column", notes)
    ridge = (1e-4 if head == "linear" else 1e-2) if ridge is None else ridge
    perm = rng.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])
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
        lam_full[m] = _pick_lambda(full, m, _param_forward(m, y, _GRID[m], spec.scale), _param_logjac(m, y, _GRID[m], spec.scale))
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
    return TransformAnalysis(
        names, gain, gain_rows, lam_search, lam_full, best, best_gain, tau, p_value, null_max, passing, err, err_rows, y, n, head, notes, reason, time.perf_counter() - t0, fold_list, spec.scale
    )


def decide(an: TransformAnalysis, *, metric: str = "mae", tie_se: float = 1.0, fallthrough: bool = True, alpha: float = 0.05, select: str = "err", min_gain: float = MIN_GAIN) -> TargetTransformChoice:
    """Apply the tie rule and the second gate of an analysis for ``metric`` ('rmse': smeared mean prediction; 'mae': median prediction).

    ``select="ll"`` orders the significant candidates by LL gain; ``select="err"`` (the default: the error of the original-scale prediction is the quantity the user reads, MAE first) orders them by the held-out
    original-scale error instead (the LL test still decides significance)."""
    e_id = an.err.get("identity", {}).get(metric, float("nan"))

    def abstain(reason: str) -> TargetTransformChoice:
        """The identity verdict with the reason."""
        return TargetTransformChoice("identity", {}, an.best_gain, an.tau, an.p_value, e_id, e_id, metric, True, reason, an.best, dict(an.gain), None, an.seconds)

    if an.abstain_reason is not None:
        return abstain(an.abstain_reason)
    if not an.passing:
        return abstain(f"best candidate {an.best} gain {an.best_gain:.5f} <= null threshold {an.tau:.5f} (p = {an.p_value:.3f}, alpha = {alpha})")
    order = list(an.passing)
    if select == "err":
        # the held-out error of the original-scale prediction orders the significant candidates (the LL can prefer a transform that fits the tails and predicts the mean worse: a cube with a little noise
        # gets ``asinh`` by LL and half the error with ``cbrt``); a candidate whose paired per-row loss difference to the best one is within ``tie_se`` standard errors is a tie and the simplest family wins
        col = 0 if metric == "rmse" else 1
        order.sort(key=lambda m: an.err[m][metric])
        top = order[0]
        tied = []
        for m in order:
            d = an.err_rows[m][col] - an.err_rows[top][col]
            if m == top or d.mean() <= tie_se * d.std() / math.sqrt(len(d)):
                tied.append(m)
        first = min(tied, key=lambda m: (_COMPLEXITY[m], an.err[m][metric]))
    else:
        top = order[0]
        tied = []
        for m in order:
            if m == top:
                tied.append(m)
                continue
            d = an.gain_rows[top] - an.gain_rows[m]
            if d.mean() <= tie_se * d.std() / math.sqrt(len(d)):
                tied.append(m)
        first = min(tied, key=lambda m: (_COMPLEXITY[m], -an.gain[m]))
    order = [first] + [m for m in order if m != first]
    tried: List[str] = []
    for m in order:
        e_m = an.err[m][metric]
        if e_m <= (1.0 - min_gain) * e_id:
            lam = an.lam_full.get(m)
            tr = build_transform(m, an.y_ref, lam)
            return TargetTransformChoice(m, tr.params, an.gain[m], an.tau, an.p_value, e_id, e_m, metric, False, "", an.best, dict(an.gain), tr, an.seconds)
        tried.append(f"{m} ({e_m:.4g} against {e_id:.4g} for the identity, less than {min_gain:.0%} better)")
        if not fallthrough:
            break
    return abstain(f"significant candidates failed the held-out {metric} gate vs identity: " + ", ".join(tried))


def recommend_target_transform(
    X: np.ndarray,
    y: np.ndarray,
    *,
    candidates: Optional[Sequence[str]] = None,
    n_null: int = 40,
    alpha: float = 0.05,
    seed: int = 0,
    head: str = "linear",
    metric: str = "mae",
    folds: int = 1,
    tie_se: float = 1.0,
    fallthrough: bool = True,
    ridge: Optional[float] = None,
    max_rows: int = 20000,
    select: str = "err",
    min_gain: float = MIN_GAIN,
) -> TargetTransformChoice:
    """Recommend a monotone target transform for a linear / additive-spline head, or abstain (identity). See the module docstring for the algorithm.

    ``candidates``: names from ``DEFAULT_CANDIDATES`` (+ ``box_cox``); None = all that the range of y allows. ``n_null`` sign-flip draws (>= 19 needed for alpha 0.05), ``seed`` fixes the split, the
    null and the subsample. ``head``: "linear" (closed-form ridge on [1, X]) or "spline" (additive natural cubic splines). ``metric``: "mae" (median prediction, the default) or "rmse" (smeared mean prediction) for the second gate and the selection; ``select``: "err" (default) or "ll".
    ``folds=2`` cross-fits (A->B and B->A) instead of A->B only."""
    an = analyse_target_transforms(X, y, candidates=candidates, n_null=n_null, alpha=alpha, seed=seed, head=head, folds=folds, ridge=ridge, max_rows=max_rows)
    return decide(an, metric=metric, tie_se=tie_se, fallthrough=fallthrough, alpha=alpha, select=select, min_gain=min_gain)
