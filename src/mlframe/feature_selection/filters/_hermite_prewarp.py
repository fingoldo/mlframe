"""Per-operand pre-warp + rank-1 ALS helpers for the orthogonal-polynomial pair FE.

Closed-form supervised per-operand warps (``fit_operand_prewarp`` /
``fit_pair_prewarp_als`` / ``apply_operand_prewarp``), the rank-1 ALS warm
start (``warm_start_als_seed``), the canonical coefficient seeds and the
scale-saturating L2 penalty, plus the 1-D KSG MI objective. Independent of the
polyeval kernel-dispatch state in the parent (``hermite_fe``); the parent
helpers ``_POLY_BASES`` / ``build_basis_matrix`` are lazy-imported in-body to
avoid an import cycle.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

__all__ = [
    "warm_start_als_seed",
    "fit_operand_prewarp",
    "fit_pair_prewarp_als",
    "apply_operand_prewarp",
]


def _canonical_seeds(basis: str, degree: int) -> list:
    """Return canonical coefficient vectors (shape (degree+1,)) for warm-start, representing explicit low-degree polynomials."""
    seeds = []
    # Identity P_1: e_1 = [0, 1, 0, ..., 0]
    e1 = np.zeros(degree + 1, dtype=np.float64)
    if degree >= 1:
        e1[1] = 1.0
        seeds.append(e1)
    # Pure P_2 polynomial coefficient vector
    if degree >= 2:
        e2 = np.zeros(degree + 1, dtype=np.float64)
        e2[2] = 1.0
        seeds.append(e2)
        # Composite low-degree: P_0 + P_2 (captures mean + curvature)
        e02 = np.zeros(degree + 1, dtype=np.float64)
        e02[0] = -1.0
        e02[2] = 1.0
        seeds.append(e02)
    # Pure P_3
    if degree >= 3:
        e3 = np.zeros(degree + 1, dtype=np.float64)
        e3[3] = 1.0
        seeds.append(e3)
    return seeds




def _l2_normalize_pair(coef_a: np.ndarray, coef_b: np.ndarray,
                        target_norm: float = 1.0) -> tuple:
    """Project (c_a, c_b) jointly to the L2 sphere (or other target_norm). Used in direction_only search to remove
    the scaling ridge that confuses TPE/CMA on XOR-like targets where MI is invariant to overall scaling
    (bf=mul) or equivariant (bf=add/sub)."""
    norm = float(np.sqrt(np.sum(coef_a ** 2) + np.sum(coef_b ** 2)))
    if norm < 1e-12:
        return coef_a, coef_b
    scale = target_norm / norm
    return coef_a * scale, coef_b * scale


# Default saturation constant for the scale-invariant coefficient penalty (see
# ``_l2_penalty_value``). When ``l2_penalty_saturation > 0`` the penalty is
# ``lambda * ||c||^2 / (||c||^2 + saturation)`` -- it rises from 0 toward a
# CONSTANT ``lambda`` ceiling as ``||c||^2`` grows past ``saturation``, so a
# genuinely high-MI / high-coefficient solution is never crushed (the failure
# mode the raw ``lambda * ||c||^2`` penalty caused on the F-POLY pre-distortion
# fixture, where the true Chebyshev coefficients have ``||c||^2 ~ 86`` and the
# raw penalty ~4.3 dwarfed the MI peak ~1.5). ``saturation`` sets the coef-norm
# scale at which the penalty reaches half ``lambda``; 1.0 means small-coef noise
# solutions (||c||^2 << 1, e.g. an atan2 plateau artifact) still pay almost the
# full ``lambda``, preserving noise rejection.
_L2_PENALTY_SATURATION_DEFAULT = 1.0


def _l2_penalty_value(coef_a: np.ndarray, coef_b: np.ndarray,
                       l2_penalty: float,
                       l2_penalty_saturation: float = _L2_PENALTY_SATURATION_DEFAULT) -> float:
    """Coefficient-magnitude penalty subtracted from the raw MI objective.

    Two regimes, selected by ``l2_penalty_saturation``:

    * ``l2_penalty_saturation > 0`` (the default / recommended path): a
      SCALE-SATURATING penalty ``lambda * s / (s + sat)`` where ``s = ||c_a||^2
      + ||c_b||^2``. As ``s`` grows the penalty saturates toward the constant
      ``lambda`` instead of growing without bound, so it regularises pure noise
      (tiny ``s`` -> tiny penalty difference between candidates, plus the
      constant ceiling discourages adding magnitude for no MI gain) WITHOUT
      punishing genuinely-high-MI high-coefficient solutions. This is what lets
      the separable Chebyshev reconstruction of ``(a**3-2a)(b**2-b)`` (||c||^2
      ~ 86, MI ~ 1.5) win over the deceptive small-||c|| atan2/div plateau.

    * ``l2_penalty_saturation <= 0``: the legacy RAW penalty ``lambda *
      ||c||^2``. Kept for byte-compatibility / opt-out; this is the formula that
      crushed large-coefficient solutions.

    ``l2_penalty <= 0`` returns 0.0 in both regimes (penalty disabled).
    """
    if l2_penalty <= 0.0:
        return 0.0
    s = float(np.sum(coef_a ** 2) + np.sum(coef_b ** 2))
    if l2_penalty_saturation and l2_penalty_saturation > 0.0:
        return l2_penalty * (s / (s + l2_penalty_saturation))
    return l2_penalty * s


def warm_start_als_seed(B_a: np.ndarray, B_b: np.ndarray, y: np.ndarray,
                         *, iters: int = 3) -> tuple:
    """Per-operand warm-start coefficients for the multiplicative pair model
    ``y ~ f(x_a) * g(x_b)`` via a rank-1 alternating-least-squares (ALS) sweep
    in the orthogonal-polynomial basis.

    ``B_a`` / ``B_b`` are precomputed basis matrices ``B[i, k] = T_k(z[i])`` of
    shape ``(n, degree + 1)`` (see :func:`build_basis_matrix`). Returns
    ``(coef_a, coef_b)`` -- each length ``degree + 1`` -- such that ``B_a @
    coef_a`` and ``B_b @ coef_b`` are the rank-1 separable factors best fitting
    the centred target.

    Why ALS and not two independent 1-D fits: for a centred product target the
    marginal ``E[y | x_b]`` is ~ ``g(x_b) * E[f(x_a)] ~ 0``, so an independent
    1-D least-squares fit of ``y`` on ``B_b`` recovers almost nothing on the
    b-side (measured corr 0.49 vs Q on the F-POLY fixture). One ALS sweep -- fit
    ``f`` given the current ``g`` by regressing ``y`` on ``B_a`` scaled
    column-wise by ``g``, then symmetrically -- recovers BOTH factors exactly
    (corr 1.0 each on F-POLY) in three cheap ``lstsq`` solves. This is the
    highest-leverage, near-free warm start: it lands the joint optimiser
    directly in the true (large-coefficient) basin that CMA-ES otherwise never
    finds from the canonical unit-magnitude seeds.

    The returned coefficient SCALE is arbitrary for a ``mul`` combination (MI is
    scale-invariant under ``mul``); the magnitude is split between the two
    factors by the ALS normalisation and is intentionally NOT projected -- the
    saturating penalty (:func:`_l2_penalty_value`) makes that scale harmless.

    Returns ``(None, None)`` if the target has no variance or ``lstsq`` fails.
    """
    yc = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    yc = yc - yc.mean()
    if float(np.std(yc)) < 1e-12:
        return None, None
    try:
        # Initialise g(b) from a plain 1-D least-squares fit on the b-basis.
        cb, *_ = np.linalg.lstsq(B_b, yc, rcond=None)
        g = B_b @ cb
        ca = None
        for _ in range(max(1, int(iters))):
            g_norm = g / (float(np.std(g)) + 1e-12)
            ca, *_ = np.linalg.lstsq(B_a * g_norm[:, None], yc, rcond=None)
            f = B_a @ ca
            f_norm = f / (float(np.std(f)) + 1e-12)
            cb, *_ = np.linalg.lstsq(B_b * f_norm[:, None], yc, rcond=None)
            g = B_b @ cb
        if ca is None or not (np.all(np.isfinite(ca)) and np.all(np.isfinite(cb))):
            return None, None
        return np.ascontiguousarray(ca, dtype=np.float64), np.ascontiguousarray(cb, dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError):
        return None, None










def fit_operand_prewarp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    basis: str = "chebyshev",
    max_degree: int = 4,
) -> dict | None:
    """Fit a per-operand 1-D pre-warp ``f(x)`` that linearises the operand's
    relationship to the (possibly non-monotone) target ``y`` via a single
    orthogonal-polynomial least-squares solve.

    This is the lightest sufficient pre-warp for the *unary/binary* pair search:
    where a single library unary (``sqr``, ``log``, ...) cannot express a
    within-operand polynomial such as ``a**3 - 2a``, an orthogonal-series fit of
    ``y ~ poly(x)`` can. It is deliberately the SAME 1-D machinery the
    orthogonal-poly path warm-starts from (:func:`warm_start_als_seed` is its
    rank-1 ALS sibling); exposing it here lets BOTH paths share one
    implementation rather than duplicating the basis fit.

    The fit consumes ``y`` (it is supervised, like the MI scoring), but the
    returned spec is a CLOSED-FORM function of ``x`` alone -- the stored
    ``coef`` + basis ``preprocess`` params reproduce ``f(x)`` deterministically
    at transform() time with NO ``y`` reference (leak-safe replay).

    Returns a dict ``{basis, degree, coef, preprocess}`` consumable by
    :func:`apply_operand_prewarp`, or ``None`` if the target / operand has no
    usable variance or the solve fails.
    """
    from .hermite_fe import _POLY_BASES, build_basis_matrix
    bi = _POLY_BASES.get(basis)
    if bi is None or bi.get("kind") != "polynomial":
        # Pre-warp only defined for the orthogonal-polynomial families (closed-
        # form basis matrix + apply params); non-polynomial bases need per-call
        # eval closures that are not replay-portable here.
        return None
    xf = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    yf = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if xf.size == 0 or float(np.std(xf)) < 1e-12 or float(np.std(yf)) < 1e-12:
        return None
    deg = max(1, int(max_degree))
    z, params = bi["fit"](xf)
    z = np.ascontiguousarray(z, dtype=np.float64)
    try:
        B = build_basis_matrix(basis, z, deg)
        yc = yf - yf.mean()
        coef, *_ = np.linalg.lstsq(B, yc, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return {
        "basis": str(basis),
        "degree": int(deg),
        "coef": np.ascontiguousarray(coef, dtype=np.float64),
        "preprocess": dict(params),
    }


def fit_pair_prewarp_als(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    basis: str = "chebyshev",
    max_degree: int = 4,
) -> tuple:
    """Jointly fit a per-operand pre-warp for BOTH sides of a pair via the rank-1
    ALS sweep (:func:`warm_start_als_seed`), returning ``(spec_a, spec_b)`` each
    consumable by :func:`apply_operand_prewarp`.

    Why joint ALS and not two independent 1-D fits (:func:`fit_operand_prewarp`):
    for a centred product target ``y ~ P(a) * Q(b)`` the marginal ``E[y | b] ~
    Q(b) * E[P(a)] ~ 0``, so an INDEPENDENT 1-D fit of ``y`` on the b-basis
    recovers almost nothing on the b-side (measured corr ~0.1 on the F-POLY
    fixture). The ALS sweep alternates -- fit ``f`` given the current ``g``, then
    ``g`` given ``f`` -- and recovers BOTH factors (corr ~1.0 each). This is the
    SAME mechanism the orthogonal-poly path warm-starts the joint CMA optimiser
    with; reusing it here gives the elementary unary/binary search a genuine
    per-operand non-monotone pre-warp for product-structured pairs.

    Returns ``(None, None)`` on no-variance / solve failure or a non-polynomial
    basis (the closed-form replay needs the polynomial basis-matrix path).
    """
    from .hermite_fe import _POLY_BASES, build_basis_matrix
    bi = _POLY_BASES.get(basis)
    if bi is None or bi.get("kind") != "polynomial":
        return None, None
    xa = np.ascontiguousarray(np.asarray(x_a, dtype=np.float64))
    xb = np.ascontiguousarray(np.asarray(x_b, dtype=np.float64))
    yf = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if (xa.size == 0 or float(np.std(xa)) < 1e-12 or float(np.std(xb)) < 1e-12
            or float(np.std(yf)) < 1e-12):
        return None, None
    deg = max(1, int(max_degree))
    za, pa = bi["fit"](xa)
    zb, pb = bi["fit"](xb)
    za = np.ascontiguousarray(za, dtype=np.float64)
    zb = np.ascontiguousarray(zb, dtype=np.float64)
    try:
        Ba = build_basis_matrix(basis, za, deg)
        Bb = build_basis_matrix(basis, zb, deg)
        coef_a, coef_b = warm_start_als_seed(Ba, Bb, yf, iters=3)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
    if coef_a is None or coef_b is None:
        return None, None
    spec_a = {"basis": str(basis), "degree": int(deg),
              "coef": np.ascontiguousarray(coef_a, dtype=np.float64), "preprocess": dict(pa)}
    spec_b = {"basis": str(basis), "degree": int(deg),
              "coef": np.ascontiguousarray(coef_b, dtype=np.float64), "preprocess": dict(pb)}
    return spec_a, spec_b


def apply_operand_prewarp(x: np.ndarray, spec: dict) -> np.ndarray:
    """Replay a per-operand pre-warp ``f(x)`` from a spec produced by
    :func:`fit_operand_prewarp`. Closed-form in ``x`` (uses the stored basis
    ``preprocess`` params + ``coef``); no ``y`` reference, so transform()-time
    replay is bit-identical to fit time given the same ``x``."""
    from .hermite_fe import _POLY_BASES
    basis = str(spec["basis"])
    bi = _POLY_BASES[basis]
    xf = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    z = np.ascontiguousarray(bi["apply"](xf, dict(spec["preprocess"])), dtype=np.float64)
    coef = np.ascontiguousarray(spec["coef"], dtype=np.float64)
    out = bi["eval_dispatch"](z, coef)
    return np.asarray(out, dtype=np.float64).reshape(-1)


def _ksg_mi_1d(x: np.ndarray, y: np.ndarray, *, discrete_target: bool,
               n_neighbors: int = 3) -> float:
    """KSG MI of 1-D x with target -- used as the optimisation objective."""
    if discrete_target:
        return float(mutual_info_classif(x.reshape(-1, 1), y,
                                          n_neighbors=n_neighbors, random_state=42,
                                          discrete_features=False)[0])
    return float(mutual_info_regression(x.reshape(-1, 1), y,
                                         n_neighbors=n_neighbors, random_state=42,
                                         discrete_features=False)[0])
