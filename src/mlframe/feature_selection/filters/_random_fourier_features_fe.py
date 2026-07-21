"""Random Fourier Features (random kitchen sinks) multi-column kernel-approximation block
(mrmr_audit_2026-07-20 fe_expansion.md "Random Fourier Features (random kitchen sinks) multi-column
kernel-approximation block").

Rahimi & Recht (2007, "Random Features for Large-Scale Kernel Machines"): draw a random Gaussian
projection matrix ``W`` (p x m) and phases ``b ~ Uniform(0, 2*pi)``, emit::

    phi(x) = sqrt(2/m) * cos(X @ W / bandwidth + b)

as ``m`` new columns; the inner product ``phi(x_i).phi(x_j)`` approximates an RBF kernel
``k(x_i, x_j) = exp(-||x_i-x_j||^2 / (2*bandwidth^2))`` in expectation.

Why this catches a shape the catalog misses: every existing basis (Hermite/Legendre/Chebyshev/
Laguerre/wavelet/hinge/Fourier/spline) is a PER-COLUMN expansion, and every cross-basis family
(pair/triplet/quadruplet/adaptive-arity) is a PRODUCT of per-leg bases -- none of them build a
feature that is jointly a smooth function of MANY (5+) raw columns simultaneously without
combinatorial blow-up. Concrete scenario: ``y = exp(-||x||^2 / 2)`` on p=10 jointly-informative
numeric columns (a radial/Gaussian-bump target in 10-D) -- no pairwise product term or even a
quadruplet arity-4 cell captures a genuinely 10-way radial structure, while a handful of random
Fourier features linearly recovers it because the RBF kernel IS the radial-Gaussian target class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = [
    "random_fourier_features",
    "_draw_rff_params",
    "_apply_rff_params",
    "engineered_name_random_fourier",
    "generate_random_fourier_features_block",
    "apply_random_fourier_block",
    "build_random_fourier_recipe",
    "hybrid_random_fourier_fe",
]


def _draw_rff_params(p: int, m: int, random_state: int) -> "tuple[np.ndarray, np.ndarray]":
    """Draw the projection matrix ``W`` (p x m) and phase offsets ``b`` (m,) for a fixed
    ``random_state`` -- split out so a caller (the MRMR hybrid wiring) can freeze these arrays
    into a recipe at fit time and replay with the literal arrays rather than depend on RNG
    behavioral stability across numpy versions for transform()."""
    rng = np.random.default_rng(random_state)
    W = rng.standard_normal((p, m))
    b = rng.uniform(0.0, 2.0 * np.pi, m)
    return W, b


def _apply_rff_params(X: np.ndarray, W: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    """Closed-form RFF expansion given already-drawn (frozen) ``W``/``b``/``bandwidth`` -- the
    replay half of the fit/transform contract; no RNG involved."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    m = W.shape[1]
    if X.shape[0] < 1:
        return np.empty((0, m), dtype=np.float64)
    Z = (X @ W) / bandwidth + b
    return np.asarray(np.sqrt(2.0 / m) * np.cos(Z))


def _median_pairwise_distance(X: np.ndarray, *, max_sample: int = 500, random_state: int = 0) -> float:
    """Median pairwise Euclidean distance on a (deterministic) subsample of ``X`` -- the standard
    RBF bandwidth heuristic (Gretton 2005), shared with the module's own HSIC sibling. Returns 1.0
    on a degenerate (n<2 or all-identical-rows) subsample so the caller never divides by zero."""
    n = X.shape[0]
    if n < 2:
        return 1.0
    if n > max_sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_sample, replace=False)
        X = X[idx]
        n = max_sample
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(d2[iu])
    med = float(np.median(dist)) if dist.size else 0.0
    return med if med > 1e-12 else 1.0


def random_fourier_features(
    X: np.ndarray,
    *,
    m: int = 64,
    bandwidth: Optional[float] = None,
    random_state: int = 0,
) -> np.ndarray:
    """Random Fourier Feature expansion of a ``(n, p)`` block into ``(n, m)`` new columns
    approximating an RBF kernel over the FULL joint column set.

    Parameters
    ----------
    X : (n, p) array
        The candidate raw/engineered columns to jointly expand (p can be large -- this is exactly
        the family's point: a joint smooth function of many columns without combinatorial blow-up).
    m : int
        Number of random features to emit. Larger ``m`` approximates the RBF kernel more tightly
        (variance of the approximation shrinks as ``O(1/m)``) at the cost of ``m`` new columns.
    bandwidth : float, optional
        RBF bandwidth. ``None`` (default) uses the median-pairwise-distance heuristic on a
        deterministic subsample of ``X`` (the standard Gretton 2005 choice, shared with the
        module's HSIC sibling).
    random_state : int
        Seed for the projection matrix ``W``, the phase offsets ``b``, and the bandwidth
        subsample -- deterministic and replay-safe (the SAME seed reproduces the SAME features,
        which is load-bearing for a fit/transform contract: ``W``/``b`` must be frozen at fit time
        and reused verbatim at transform time, not re-drawn).

    Returns
    -------
    (n, m) float64 array. Degenerate input (n<1, p<1, or m<1) returns an ``(n, 0)`` array rather
    than raising -- callers treat zero emitted columns as "nothing to add", the same convention
    the audit's own sketch documents for a top_k=0-style empty family output.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n < 1 or p < 1 or m < 1:
        return np.empty((n, 0), dtype=np.float64)
    if not np.isfinite(X).all():
        return np.empty((n, 0), dtype=np.float64)

    bw = float(bandwidth) if bandwidth is not None else _median_pairwise_distance(X, random_state=random_state)
    bw = bw if bw > 1e-12 else 1.0

    W, b = _draw_rff_params(p, m, random_state)
    return _apply_rff_params(X, W, b, bw)


def engineered_name_random_fourier(cols: Sequence[str], idx: int) -> str:
    """Deterministic engineered-column name for the ``idx``-th RFF component of a joint block."""
    return "rff__" + "_".join(str(c) for c in cols) + f"__{idx}"


def generate_random_fourier_features_block(
    X: "pd.DataFrame", cols: Sequence[str], *, m: int = 64, bandwidth: Optional[float] = None, random_state: int = 0
) -> "tuple[pd.DataFrame, dict]":
    """Fit one joint RFF block over ``cols``: draw+freeze ``W``/``b``/bandwidth, emit ``m`` named
    columns. Returns ``(enc_df, payload)`` where ``payload`` carries the frozen arrays needed for
    leak-safe replay (no ``y`` reference)."""
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 1:
        return pd.DataFrame(index=X.index), {}
    X_cols = X[cols].to_numpy(dtype=np.float64)
    n, p = X_cols.shape
    if n < 1 or p < 1 or m < 1 or not np.isfinite(X_cols).all():
        return pd.DataFrame(index=X.index), {}

    bw = float(bandwidth) if bandwidth is not None else _median_pairwise_distance(X_cols, random_state=random_state)
    bw = bw if bw > 1e-12 else 1.0
    W, b = _draw_rff_params(p, m, random_state)
    Z = _apply_rff_params(X_cols, W, b, bw)
    names = [engineered_name_random_fourier(cols, i) for i in range(Z.shape[1])]
    enc = pd.DataFrame(Z, columns=names, index=X.index)
    payload = {"cols": tuple(cols), "W": W.copy(), "b": b.copy(), "bandwidth": bw}
    return enc, payload


def apply_random_fourier_block(X_test: "pd.DataFrame", payload: dict) -> "pd.DataFrame":
    """Replay a fitted RFF block on new rows: recompute the closed-form expansion from the frozen
    ``W``/``b``/``bandwidth`` and the raw source columns only."""
    cols = list(payload["cols"])
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise KeyError(f"apply_random_fourier_block: missing column(s) {missing} from X_test")
    X_cols = X_test[cols].to_numpy(dtype=np.float64)
    Z = _apply_rff_params(X_cols, payload["W"], payload["b"], payload["bandwidth"])
    names = [engineered_name_random_fourier(cols, i) for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=names, index=X_test.index)


def build_random_fourier_recipe(*, name: str, idx: int, cols: Sequence[str], W: np.ndarray, b: np.ndarray, bandwidth: float) -> "EngineeredRecipe":
    """Frozen recipe for one RFF component column. Stores the FULL projection matrix ``W`` (every
    column needs every row of ``W`` since each component is a joint linear combination of all
    source columns) + phase ``b[idx]`` + bandwidth; replay reads only X, so transform() is
    leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="random_fourier",
        src_names=tuple(str(c) for c in cols),
        extra={
            "cols": tuple(str(c) for c in cols),
            "idx": int(idx),
            "W": np.asarray(W, dtype=np.float64).copy(),
            "m_total": int(W.shape[1]),
            "b": float(b[idx]),
            "bandwidth": float(bandwidth),
        },
    )


def _apply_random_fourier_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe`` for a single RFF component. The
    ``sqrt(2/m)`` normalization constant must use the ORIGINAL full block size ``m_total`` (stored
    at fit time), not the width of the single-column ``W`` slice used here -- a per-component
    normalization mismatch would silently rescale every replayed value relative to the fitted
    training column."""
    cols = list(recipe.extra["cols"])
    if isinstance(X, pd.DataFrame):
        X_cols = X[cols].to_numpy(dtype=np.float64)
    else:
        try:
            import polars as _pl

            if isinstance(X, _pl.DataFrame):
                X_cols = X.select(cols).to_numpy().astype(np.float64)
            else:
                X_cols = np.column_stack([np.asarray(X[c], dtype=np.float64) for c in cols])
        except ImportError:
            X_cols = np.column_stack([np.asarray(X[c], dtype=np.float64) for c in cols])
    W = recipe.extra["W"]
    idx = int(recipe.extra["idx"])
    m_total = int(recipe.extra["m_total"])
    Z = (X_cols @ W[:, idx : idx + 1]) / recipe.extra["bandwidth"] + recipe.extra["b"]
    return np.asarray(np.sqrt(2.0 / m_total) * np.cos(Z))[:, 0]


def hybrid_random_fourier_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    m: int = 64,
    bandwidth: Optional[float] = None,
    max_cols_for_block: int = 8,
    top_k: int = 8,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    random_state: int = 0,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """End-to-end joint Random Fourier Feature block: bound the column pool by top raw-MI (the
    family's whole point is a JOINT smooth function of many columns, so the pool is capped, not
    combinatorially enumerated the way pair/triplet families are), draw ``m`` random features over
    the FULL bounded column set, MI-gate the ``m`` emitted components against the raw baseline,
    keep the top ``top_k``.

    Returns ``(X_aug, appended, recipes, enc_df)``. ``y`` is consumed only by the column-pool
    ranking + MI gate; recipes carry frozen ``W``/``b``/``bandwidth``, never ``y``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_random_fourier_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [c for c in num_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) < 1:
        return X.copy(), [], [], pd.DataFrame()

    if y is not None:
        from ._extra_fe_families import _top_mi_num_cols  # lazy: break parent<->sibling cycle

        num_cols = _top_mi_num_cols(X, num_cols, y, max_cols_for_block)
    else:
        num_cols = list(num_cols)[: int(max_cols_for_block)]
    if len(num_cols) < 1:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, payload = generate_random_fourier_features_block(X, num_cols, m=m, bandwidth=bandwidth, random_state=random_state)
    if enc_df.empty or not payload:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        _gate_top_k = int(mi_gate_top_k) if mi_gate_top_k else int(top_k)
        winners = local_mi_gate(enc_df, y, raw_X=X, top_k=_gate_top_k, reject_sink=reject_sink)
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_random_fourier_recipe(name=name, idx=int(name.rsplit("__", 1)[-1]), cols=payload["cols"], W=payload["W"], b=payload["b"], bandwidth=payload["bandwidth"])
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df
