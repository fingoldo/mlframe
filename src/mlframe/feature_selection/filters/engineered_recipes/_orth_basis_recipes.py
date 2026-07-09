"""Replay + builders for the orthogonal-basis engineered-recipe kinds.

Houses the orth-univariate / orth-pair-cross / orth-diff-basis / orth-cluster-basis
and the B-spline / Fourier recipe ``_apply_*`` + ``build_*`` functions, plus the
shared basis-eval + Cox-de Boor B-spline helpers. The ``apply_recipe`` dispatcher
in the parent ``engineered_recipes`` lazy-imports these; the ``EngineeredRecipe``
dataclass + ``_extract_column`` are lazy-imported in-body to avoid a cycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from . import EngineeredRecipe


# ---------------------------------------------------------------------------
# Layer 23 (2026-05-31): orthogonal-polynomial univariate / pair-cross recipes
# ---------------------------------------------------------------------------
#
# These replay the engineered columns produced by ``hybrid_orth_mi_fe`` /
# ``hybrid_orth_mi_pair_fe`` from ``_orthogonal_univariate_fe.py``. Both
# routes are CLOSED-FORM functions of the source column(s) alone -- the MI
# scoring that picked them at fit time consumed y, but the column value
# itself does not depend on y. Replay therefore reads only X.
#
# extra layout:
# * orth_univariate : {basis: str, degree: int}
# * orth_pair_cross : {basis_i: str, basis_j: str, deg_a: int, deg_b: int}


def _apply_orth_pre_transform(x: np.ndarray, pre_transform: str) -> np.ndarray:
    """Layer 58 (2026-05-31) per-column pre-transform: optionally reshape the
    raw column BEFORE the basis-domain preprocess kicks in. Supported values:

    * ``"raw"``  -- identity (no pre-transform; legacy Layer 21/57 path)
    * ``"log_abs"``  -- ``log(|x| + 1e-12)`` -- captures heavy-tail log-normal
                        targets where raw Hermite z-score collapses the signal.
    * ``"sqrt_abs"`` -- ``sign(x) * sqrt(|x|)`` -- mild non-linear stretch.
    * ``"tanh"``  -- ``tanh(x / max(std, 1e-12))`` -- bounded mapping; pairs
                        well with Chebyshev/Legendre on otherwise-unbounded
                        inputs.

    Replay invariant: stateless given x + pre_transform name. The std for
    ``tanh`` is computed on-the-fly from the SAME column so train/test parity
    holds (z-score in the basis preprocess already does the same). Unknown
    values fall back to identity with a logger warning rather than raising.
    """
    x = np.asarray(x, dtype=np.float64)
    if pre_transform in (None, "", "raw", "identity"):
        return np.asarray(x)
    if pre_transform == "log_abs":
        return np.asarray(np.log(np.abs(x) + 1e-12))
    if pre_transform == "sqrt_abs":
        return np.asarray(np.sign(x) * np.sqrt(np.abs(x)))
    if pre_transform == "tanh":
        sd = float(np.std(x))
        return np.tanh(x / sd) if sd > 1e-12 else np.tanh(x)
    # Unknown -- warn and fall back to identity. Avoids raising at replay
    # time on pickles produced by older clients that introduced bespoke
    # tags; the worst case is one orth column that's the wrong shape, which
    # downstream MRMR will deselect on the next refit.
    import logging as _lg_pretrans
    _lg_pretrans.getLogger(__name__).warning(
        "_apply_orth_pre_transform: unknown pre_transform %r; falling back " "to identity.",
        pre_transform,
    )
    return x


def _eval_orth_basis_column(
    x: np.ndarray,
    basis: str,
    degree: int,
    *,
    pre_transform: str = "raw",
    preprocess_params: Optional[dict] = None,
) -> np.ndarray:
    """Preprocess x to the basis domain (z-score for hermite, min-max for
    legendre/chebyshev, shift for laguerre), then evaluate the single basis
    function of the given degree via a one-hot coefficient vector.

    Mirrors ``_orthogonal_univariate_fe._evaluate_basis_column`` so that
    transform()-time replay produces the SAME value as fit-time generation.
    Lazy import of ``hermite_fe`` keeps the recipes module dependency-light.

    Layer 58 (2026-05-31): optional ``pre_transform`` applied to the column
    BEFORE the basis preprocess (log|x| for heavy-tail, tanh for bounded
    mapping, etc.). ``pre_transform='raw'`` (default) keeps Layer 21/57
    byte-identical -- existing recipes deserialized without the field
    behave unchanged.

    BUG2 FIX (2026-06-12): ``preprocess_params`` carries the FROZEN fit-time
    basis-preprocess statistics (z-score mean/std, min-max lo/hi, or shift lo,
    plus any robust clip). When present the basis axis is rebuilt with the
    basis ``apply`` function from these frozen params instead of REFITTING the
    axis from ``x`` -- so a row-slice transform replays BYTE-EXACTLY against the
    full-frame transform (a refit-from-slice recomputed mean/std drifting the
    z-score by ~1e-3, which after the downstream quantisation produced a
    |delta|=1 bin drift on a nested ``a__He2`` sub-operand). ``None`` (legacy /
    pre-fix pickles) falls back to the refit path, byte-identical to before.
    """
    from ..hermite_fe import _POLY_BASES, polyeval_dispatch
    basis_info = _POLY_BASES[basis]
    fit_fn = basis_info["fit"]
    # NaN-safe: mirror fit-time finite-mask behaviour. Fit-time uses the
    # COLUMN mean to fill NaN before z-score / min-max. Test-time must use
    # the SAME fill so the basis-evaluation parity holds row-by-row when the
    # train and test frames disagree on which rows are NaN. The hybrid FE
    # path uses ``np.nanmean`` on the column at fit time; we replicate.
    x = np.asarray(x, dtype=np.float64)
    finite_mask = np.isfinite(x)
    if not finite_mask.all():
        fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
        x = np.where(finite_mask, x, fill)
    # Layer 58: optional pre-transform before basis preprocess. Identity
    # when ``pre_transform='raw'`` (default) -- legacy bit-exact path.
    x = _apply_orth_pre_transform(x, pre_transform)
    # The pre-transform can produce non-finite values for pathological inputs
    # (e.g. log(0) when caller passed a value at the floor); guard with one
    # more NaN-safe fill so the basis preprocess (z-score / min-max) doesn't
    # propagate NaN through every downstream column.
    finite_mask2 = np.isfinite(x)
    if not finite_mask2.all():
        fill2 = float(np.nanmean(x[finite_mask2])) if finite_mask2.any() else 0.0
        x = np.where(finite_mask2, x, fill2)
    if preprocess_params is not None:
        # BUG2 FIX: replay the basis axis from the FROZEN fit-time params (no
        # data-dependent refit -> byte-exact on any row-slice).
        apply_fn = basis_info["apply"]
        z = apply_fn(np.asarray(x, dtype=np.float64), preprocess_params)
    else:
        z, _params = fit_fn(x)
    z = np.ascontiguousarray(z, dtype=np.float64)
    coef = np.zeros(int(degree) + 1, dtype=np.float64)
    coef[int(degree)] = 1.0
    return polyeval_dispatch(basis, z, coef)


def _apply_orth_univariate(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay an orthogonal-polynomial univariate column: extract the
    source column, evaluate basis_n(z) where z is the per-basis preprocessed
    value. Stateless given the stored basis + degree; no y reference.
    """
    from . import _extract_column
    if len(recipe.src_names) != 1:
        raise ValueError(f"orth_univariate recipe '{recipe.name}' must have exactly 1 " f"src_names; got {len(recipe.src_names)}")
    for key in ("basis", "degree"):
        if key not in recipe.extra:
            raise KeyError(f"orth_univariate recipe '{recipe.name}' missing '{key}' " f"in extra. Re-fit MRMR to regenerate.")
    src_name = recipe.src_names[0]
    basis = str(recipe.extra["basis"])
    degree = int(recipe.extra["degree"])
    # Layer 58 (2026-05-31): optional pre-transform applied to the raw column
    # before the basis preprocess. Default ``"raw"`` (identity) keeps recipes
    # produced by Layer 21/57 byte-identical -- existing pickles missing the
    # ``pre_transform`` key replay unchanged.
    pre_transform = str(recipe.extra.get("pre_transform", "raw"))
    # BUG2 FIX (2026-06-12): frozen fit-time basis-preprocess params (if the
    # recipe was built with them) make replay byte-exact on any row-slice.
    preprocess_params = recipe.extra.get("preprocess_params")
    vals = _extract_column(X, src_name)
    return _eval_orth_basis_column(
        vals, basis, degree, pre_transform=pre_transform,
        preprocess_params=preprocess_params,
    )


def _apply_orth_pair_cross(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay a pair-cross-basis column: extract both source columns,
    evaluate basis_a^{deg_a}(z_i) * basis_b^{deg_b}(z_j). Stateless given
    the stored bases + degrees; no y reference.
    """
    from . import _extract_column
    if len(recipe.src_names) != 2:
        raise ValueError(f"orth_pair_cross recipe '{recipe.name}' must have exactly 2 " f"src_names; got {len(recipe.src_names)}")
    for key in ("basis_i", "basis_j", "deg_a", "deg_b"):
        if key not in recipe.extra:
            raise KeyError(f"orth_pair_cross recipe '{recipe.name}' missing '{key}' " f"in extra. Re-fit MRMR to regenerate.")
    name_i, name_j = recipe.src_names
    basis_i = str(recipe.extra["basis_i"])
    basis_j = str(recipe.extra["basis_j"])
    deg_a = int(recipe.extra["deg_a"])
    deg_b = int(recipe.extra["deg_b"])
    # BUG2 FIX (2026-06-12): frozen per-operand fit-time preprocess params.
    pp_i = recipe.extra.get("preprocess_params_i")
    pp_j = recipe.extra.get("preprocess_params_j")
    vals_i = _extract_column(X, name_i)
    vals_j = _extract_column(X, name_j)
    h_a = _eval_orth_basis_column(vals_i, basis_i, deg_a, preprocess_params=pp_i)
    h_b = _eval_orth_basis_column(vals_j, basis_j, deg_b, preprocess_params=pp_j)
    return np.asarray(h_a * h_b)


def _freeze_preprocess_params(params: Optional[dict]) -> Optional[dict]:
    """Coerce fit-time basis-preprocess params to a JSON-light, pickle-stable
    dict of plain floats so the frozen recipe compares equal byte-for-byte and
    survives pickle round-trips. ``None`` -> ``None`` (legacy refit path)."""
    if not params:
        return None
    out: dict = {}
    for k, v in params.items():
        if v is None:
            continue
        out[str(k)] = float(v)
    return out or None


def build_orth_univariate_recipe(
    *,
    name: str,
    src_name: str,
    basis: str,
    degree: int,
    pre_transform: str = "raw",
    preprocess_params: Optional[dict] = None,
) -> EngineeredRecipe:
    """Frozen recipe for one orthogonal-polynomial univariate column
    ``basis_n(preprocess(pre_transform(X[src_name])))``. Replay is closed-form
    and deterministic; no y reference is captured.

    ``pre_transform`` defaults to ``"raw"`` (identity) so Layer 21/57
    recipes remain byte-identical. Layer 58 routing FE picks one of
    ``"raw" | "log_abs" | "sqrt_abs" | "tanh"`` per surviving column.
    """
    from . import EngineeredRecipe
    extra = {"basis": str(basis), "degree": int(degree)}
    # Only write the field when the caller picked a non-default value so
    # legacy pickles continue to compare equal byte-for-byte (the recipe's
    # ``__eq__`` walks ``extra`` content).
    if pre_transform and pre_transform != "raw":
        extra["pre_transform"] = str(pre_transform)
    # BUG2 FIX (2026-06-12): freeze the fit-time basis-preprocess params into the
    # recipe so transform() replays the axis byte-exactly (no slice-vs-full mean/std
    # refit drift). Omitted when None so legacy recipes stay byte-equal.
    _pp = _freeze_preprocess_params(preprocess_params)
    if _pp is not None:
        extra["preprocess_params"] = _pp
    return EngineeredRecipe(
        name=name,
        kind="orth_univariate",
        src_names=(src_name,),
        extra=extra,
    )


def build_orth_pair_cross_recipe(
    *, name: str, src_a_name: str, src_b_name: str,
    basis_i: str, basis_j: str, deg_a: int, deg_b: int,
    preprocess_params_i: Optional[dict] = None,
    preprocess_params_j: Optional[dict] = None,
) -> EngineeredRecipe:
    """Frozen recipe for one cross-basis pair column
    ``basis_i^{deg_a}(preprocess(X[a])) * basis_j^{deg_b}(preprocess(X[b]))``.
    """
    from . import EngineeredRecipe
    extra = {
        "basis_i": str(basis_i),
        "basis_j": str(basis_j),
        "deg_a": int(deg_a),
        "deg_b": int(deg_b),
    }
    # BUG2 FIX (2026-06-12): freeze each operand's fit-time preprocess params.
    _ppi = _freeze_preprocess_params(preprocess_params_i)
    _ppj = _freeze_preprocess_params(preprocess_params_j)
    if _ppi is not None:
        extra["preprocess_params_i"] = _ppi
    if _ppj is not None:
        extra["preprocess_params_j"] = _ppj
    return EngineeredRecipe(
        name=name,
        kind="orth_pair_cross",
        src_names=(src_a_name, src_b_name),
        extra=extra,
    )


def build_orth_diff_basis_recipe(
    *, name: str, col_a: str, col_b: str,
    basis: str, degree: int, pre_transform: str = "raw",
    preprocess_params: Optional[dict] = None,
) -> EngineeredRecipe:
    """Layer 59 (2026-05-31): frozen recipe for one diff-basis column
    ``basis_degree(preprocess(pre_transform(X[col_a] - X[col_b])))``.

    The diff orientation is FIXED as ``col_a - col_b`` so the recipe replays
    deterministically; reversing the column order yields a sign-flipped
    column which the MI scorer treats as a distinct candidate. Replay is a
    pure function of X (no y reference). ``pre_transform`` defaults to
    ``"raw"`` so the field is omitted from ``extra`` on the legacy path,
    keeping recipe byte-equality with earlier diff-basis pickles that pre-
    date the pre-transform feature.
    """
    from . import EngineeredRecipe
    extra = {"basis": str(basis), "degree": int(degree)}
    if pre_transform and pre_transform != "raw":
        extra["pre_transform"] = str(pre_transform)
    # REPLAY-FIDELITY FIX (2026-06-13): freeze the fit-time basis-preprocess params of the diff so
    # replay reproduces the axis byte-exactly (no slice-vs-full refit drift). Omitted when None so
    # legacy diff-basis pickles stay byte-equal.
    _pp = _freeze_preprocess_params(preprocess_params)
    if _pp is not None:
        extra["preprocess_params"] = _pp
    return EngineeredRecipe(
        name=name,
        kind="orth_diff_basis",
        src_names=(str(col_a), str(col_b)),
        extra=extra,
    )


def build_orth_cluster_basis_recipe(
    *, name: str, members: tuple[str, ...],
    basis: str, degree: int, aggregator: str = "mean_z",
    agg_stats: dict | None = None, basis_params: dict | None = None,
) -> EngineeredRecipe:
    """Layer 61 (2026-05-31): frozen recipe for one per-cluster shared-
    basis column ``basis_degree(preprocess(aggregator(members)))``.

    The member tuple is stored in deterministic (sorted) order so the
    aggregate orientation matches fit time exactly. ``aggregator`` is
    one of ``mean_z`` / ``median_z`` / ``pc1`` -- see
    :func:`compute_cluster_aggregate` in the cluster-basis FE module.
    Replay is a pure function of X (no y reference).

    2026-06-03 (audit cluster-aggregate-6/7): persist the fit-time aggregate
    stats (``agg_stats`` = per-member mean/std/signs + combiner weights) and the
    basis preprocess params (``basis_params`` = e.g. z-score mean/std) so replay
    APPLIES them rather than refitting on the test distribution -> byte parity
    under train/test drift. Both are plain float lists/dicts (recipe-__eq__- and
    pickle-safe). Omitted when absent so legacy recipes (which fall back to the
    refit path with a warning) are unaffected.
    """
    from . import EngineeredRecipe
    if len(members) < 2:
        raise ValueError(f"build_orth_cluster_basis_recipe: ``members`` must have >=2 " f"entries (cluster, not singleton); got {len(members)}.")
    extra = {
        "basis": str(basis),
        "degree": int(degree),
        "aggregator": str(aggregator),
    }
    if agg_stats is not None:
        extra["agg_stats"] = agg_stats
    if basis_params is not None:
        extra["basis_params"] = dict(basis_params)
    return EngineeredRecipe(
        name=name,
        kind="orth_cluster_basis",
        src_names=tuple(str(m) for m in members),
        extra=extra,
    )


def _bspline_basis_values(z: np.ndarray, knots: np.ndarray, idx: int, degree: int = 3) -> np.ndarray:
    """Evaluate the ``idx``-th cubic B-spline basis function at points ``z``.

    Uses the Cox-de Boor recursion. ``knots`` is the full augmented knot
    vector (with degree+1 repeated boundary knots). Returns shape (n,).
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    # Degree-0 indicator [t_k, t_{k+1})
    # Build up the Cox-de Boor recursion for the single basis index `idx`.
    # We do it in O(degree+1) per point by computing the full row of B-splines
    # via the standard algorithm, then picking the column we need.
    nk = len(knots)
    for i in range(n):
        zi = z[i]
        # Find span: largest k with knots[k] <= zi < knots[k+1]
        # Handle boundary: clip zi to [knots[degree], knots[-degree-1]]
        if zi >= knots[nk - degree - 1]:
            zi_eff = knots[nk - degree - 1] - 1e-12
        elif zi <= knots[degree]:
            zi_eff = knots[degree] + 1e-12
        else:
            zi_eff = zi
        # Compute non-zero B-splines of given degree at zi_eff using standard
        # de Boor algorithm (returns degree+1 non-zero values starting at span k-degree)
        # Locate k such that knots[k] <= zi_eff < knots[k+1]
        k = degree
        for kk in range(degree, nk - degree - 1):
            if knots[kk] <= zi_eff < knots[kk + 1]:
                k = kk
                break
        else:
            k = nk - degree - 2
        # B-splines of degree d at zi_eff for span k.
        # left[j] = zi - knots[k+1-j]; right[j] = knots[k+j] - zi
        # N[0] = 1, then for d = 1..degree update.
        N = np.zeros(degree + 1, dtype=np.float64)
        N[0] = 1.0
        for d in range(1, degree + 1):
            saved = 0.0
            for r in range(d):
                t_left = knots[k + 1 + r - d]
                t_right = knots[k + 1 + r]
                denom = t_right - t_left
                if denom <= 1e-12:
                    temp = 0.0
                else:
                    temp = N[r] / denom
                N[r] = saved + (t_right - zi_eff) * temp
                saved = (zi_eff - t_left) * temp
            N[d] = saved
        # Non-zero basis indices for this span are k-degree..k.
        # The column we want is `idx`. If idx in [k-degree, k] return N[idx - (k-degree)], else 0.
        rel = idx - (k - degree)
        if 0 <= rel <= degree:
            out[i] = N[rel]
        # else: 0 (already initialized)
    return out


def _fit_spline_knots(x: np.ndarray, n_inner_knots: int, degree: int = 3) -> tuple[np.ndarray, float, float]:
    """Fit a cubic B-spline knot vector at quantiles ``(1..K)/(K+1)`` of x.

    Returns the FULL augmented knot vector (with degree+1 boundary
    repetitions at 0 and 1) plus the (lo, hi) normalisation range.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        lo, hi = 0.0, 1.0
        inner = np.linspace(1.0 / (n_inner_knots + 1), n_inner_knots / (n_inner_knots + 1), n_inner_knots)
    else:
        xf = x[finite]
        lo, hi = float(xf.min()), float(xf.max())
        if hi - lo <= 1e-12:
            hi = lo + 1.0
        z = (xf - lo) / (hi - lo)
        qs = np.linspace(1.0 / (n_inner_knots + 1), n_inner_knots / (n_inner_knots + 1), n_inner_knots)
        inner = np.quantile(z, qs)
        # Ensure strictly increasing knots
        inner = np.unique(inner)
        if inner.size < n_inner_knots:
            # Top up with uniform fill where quantiles collapsed (ties).
            extra_n = n_inner_knots - inner.size
            uni = np.linspace(0.0, 1.0, extra_n + 2)[1:-1]
            inner = np.unique(np.concatenate([inner, uni]))
        inner = np.clip(inner, 1e-6, 1.0 - 1e-6)
    # Augment with degree+1 boundary repetitions on each end (clamped knot vector).
    boundary_lo = np.zeros(degree + 1, dtype=np.float64)
    boundary_hi = np.ones(degree + 1, dtype=np.float64)
    knots = np.concatenate([boundary_lo, inner.astype(np.float64), boundary_hi])
    return knots, float(lo), float(hi)


def _apply_orth_spline(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay one cubic B-spline basis column: B_{idx}(z) where z is min-max
    normalised x with knots fixed at fit time. Stateless given the stored
    knots + idx + (lo, hi); no y reference.
    """
    from . import _extract_column
    if len(recipe.src_names) != 1:
        raise ValueError(f"orth_spline recipe '{recipe.name}' must have exactly 1 " f"src_names; got {len(recipe.src_names)}")
    for key in ("knots", "idx", "lo", "hi"):
        if key not in recipe.extra:
            raise KeyError(f"orth_spline recipe '{recipe.name}' missing '{key}' " f"in extra. Re-fit MRMR to regenerate.")
    name = recipe.src_names[0]
    knots = np.asarray(recipe.extra["knots"], dtype=np.float64)
    idx = int(recipe.extra["idx"])
    lo = float(recipe.extra["lo"])
    hi = float(recipe.extra["hi"])
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    span = max(hi - lo, 1e-12)
    z = np.clip((vals - lo) / span, 0.0, 1.0)
    return _bspline_basis_values(z, knots, idx, degree=3)


def _apply_orth_fourier(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay one Fourier basis column: sin(2*pi*freq*z) or cos(2*pi*freq*z)
    where z = (x - lo) / span, with (lo, span) fixed at fit time.
    """
    from . import _extract_column
    if len(recipe.src_names) != 1:
        raise ValueError(f"orth_fourier recipe '{recipe.name}' must have exactly 1 " f"src_names; got {len(recipe.src_names)}")
    for key in ("kind", "freq", "lo", "span"):
        if key not in recipe.extra:
            raise KeyError(f"orth_fourier recipe '{recipe.name}' missing '{key}' " f"in extra. Re-fit MRMR to regenerate.")
    name = recipe.src_names[0]
    kind = str(recipe.extra["kind"])
    freq = float(recipe.extra["freq"])
    lo = float(recipe.extra["lo"])
    span = float(recipe.extra["span"])
    span = max(span, 1e-12)
    # power defaults to 1 (legacy recipes pre-dating power-argument Fourier).
    power = int(recipe.extra.get("power", 1))
    # arg defaults to "linear" (legacy recipes pre-dating the chirp warp).
    arg = str(recipe.extra.get("arg", "linear"))
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    if arg == "quadratic":
        # ADAPTIVE-CHIRP axis: z = (u - lo) / span where u = sign(zs)*zs**2,
        # zs = (x - mean) / std. mean/std/lo/span are the train-fit warp params,
        # so this reproduces the fit-time axis byte-for-byte from X alone (no y).
        mean = recipe.extra.get("mean")
        std = recipe.extra.get("std")
        if mean is None or std is None:
            raise KeyError(f"orth_fourier recipe '{recipe.name}' arg='quadratic' missing " f"'mean'/'std'. Re-fit MRMR to regenerate.")
        zs = (vals - float(mean)) / max(float(std), 1e-12)
        u = np.sign(zs) * (zs * zs)
        z = (u - lo) / span
    else:
        if power != 1:
            vals = np.power(vals, power)
        z = (vals - lo) / span
    ang = 2.0 * np.pi * freq * z
    if kind == "sin":
        return np.sin(ang)
    if kind == "cos":
        return np.cos(ang)
    raise ValueError(f"orth_fourier recipe '{recipe.name}': unknown kind {kind!r}")


def build_orth_spline_recipe(
    *, name: str, src_name: str, knots: np.ndarray, idx: int, lo: float, hi: float,
) -> EngineeredRecipe:
    """Frozen recipe for one cubic B-spline basis column ``B_{idx}(z)`` where
    ``z = clip((X[src_name] - lo) / (hi - lo), 0, 1)`` with quantile-placed
    knots fixed at fit time."""
    from . import EngineeredRecipe
    return EngineeredRecipe(
        name=name,
        kind="orth_spline",
        src_names=(src_name,),
        extra={
            "knots": np.asarray(knots, dtype=np.float64).copy(),
            "idx": int(idx),
            "lo": float(lo),
            "hi": float(hi),
        },
    )


def build_orth_fourier_recipe(
    *, name: str, src_name: str, kind: str, freq: float, lo: float, span: float,
    power: int = 1, adaptive: bool = False,
    arg: str = "linear", mean: Optional[float] = None, std: Optional[float] = None,
) -> EngineeredRecipe:
    """Frozen recipe for one Fourier basis column ``sin(2*pi*freq*z)`` or
    ``cos(2*pi*freq*z)`` where ``z = (X[src_name]**power - lo) / span`` with
    (power, lo, span) fixed at fit time. ``power`` > 1 builds the Fourier on the
    POWER-transformed argument (e.g. power=2 -> Fourier on x**2, recovering chirps
    like ``sin(a**2)``); the recipe is self-contained (raw src -> power -> Fourier),
    1-deep, replayable. ``power`` defaults to 1 (legacy linear-argument Fourier).

    ``arg`` selects the argument WARP applied before the Fourier (2026-06-03):
    * ``"linear"`` (default) -- ``z = (x**power - lo) / span`` (the legacy axis).
    * ``"quadratic"`` -- the ADAPTIVE-CHIRP axis ``z = (u - lo) / span`` where
      ``u = sign(zs)*zs**2`` and ``zs = (x - mean) / std``. Squaring the
      STANDARDISED, SIGNED z turns a growing-frequency chirp ``sin(2*pi*f*zs**2)``
      into a stationary-frequency sinusoid that the detector can lock; ``mean`` /
      ``std`` (the train-fit standardisation) are stored alongside (lo, span) so
      the warp replays leak-free. ``power`` is ignored for the quadratic arg.

    ``adaptive`` (default False) is a pure TAG stored in ``extra`` -- it marks a
    column emitted at an ADAPTIVELY-DETECTED z-space frequency (held-out
    validated) rather than a fixed-grid one. Replay reads ``arg``/``mean``/``std``
    to rebuild the warp but never reads ``adaptive``; the tag lets MRMR protect
    these columns past screening (a single sin/cos has low marginal MI -- phase --
    so the screen would otherwise drop the held-out-validated pair)."""
    from . import EngineeredRecipe
    if kind not in ("sin", "cos"):
        raise ValueError(f"orth_fourier kind must be 'sin' or 'cos'; got {kind!r}")
    if arg not in ("linear", "quadratic"):
        raise ValueError(f"orth_fourier arg must be 'linear' or 'quadratic'; got {arg!r}")
    if arg == "quadratic" and (mean is None or std is None):
        raise ValueError("orth_fourier arg='quadratic' requires both 'mean' and 'std' " "(the train-fit standardisation of the source column).")
    return EngineeredRecipe(
        name=name,
        kind="orth_fourier",
        src_names=(src_name,),
        extra={
            "kind": str(kind),
            "freq": float(freq),
            "lo": float(lo),
            "span": float(span),
            "power": int(power),
            "adaptive": bool(adaptive),
            "arg": str(arg),
            "mean": (None if mean is None else float(mean)),
            "std": (None if std is None else float(std)),
        },
    )
