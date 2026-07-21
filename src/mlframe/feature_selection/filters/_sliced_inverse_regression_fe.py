"""Sliced Inverse Regression (SIR) oblique-direction projection feature (mrmr_audit_2026-07-20
fe_expansion.md "Sliced Inverse Regression (SIR) oblique-direction projection feature").

Li (1991, "Sliced Inverse Regression for Dimension Reduction"): slice ``y`` into ``H`` bins,
compute the per-slice mean of ``X``, form the between-slice-mean covariance matrix
``M = Cov(E[X | slice])``, then solve the generalized eigenproblem ``Sigma^{-1} M v = lambda v``
(``Sigma`` = overall covariance of ``X``); the top eigenvector(s) ``v`` give the LINEAR COMBINATION
direction(s) ``w.x`` along which ``y`` varies most -- an effective dimension-reduction direction,
not restricted to any 2 or 3 named columns.

Why this catches a shape the catalog misses: ``y = 1{0.6*x1 + 0.5*x2 + 0.4*x3 + 0.3*x4 + 0.4*x5 >
c}`` -- a genuinely OBLIQUE (rotated) threshold spread thinly across 5 correlated columns, where
EVERY individual weight is too small for that column's marginal MI to clear the screening floor,
and no pairwise/triplet/quadruplet product of any 2-4 of the 5 columns reconstructs the linear
combination (axis-aligned bases multiplied together cannot represent a rotated hyperplane
economically). SIR recovers the direction ``w = (0.6, 0.5, 0.4, 0.3, 0.4)`` directly as a single
new feature ``w.x``, after which the existing argmax/gate/threshold machinery (or a plain MI
screen) picks it up trivially.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from ._mi_greedy_cmi_fe import _quantile_bin

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = [
    "sir_direction_features",
    "engineered_name_sir_direction",
    "generate_sir_direction_features",
    "apply_sir_direction",
    "build_sir_direction_recipe",
    "hybrid_sir_direction_fe",
]


def sir_direction_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_slices: int = 10,
    n_directions: int = 2,
) -> np.ndarray:
    """Sliced Inverse Regression projection features: the top ``n_directions`` SIR eigenvectors'
    projections of ``X``, as new ``(n, n_directions)`` columns.

    Parameters
    ----------
    X : (n, p) array
        Candidate numeric columns to jointly project (correlated columns are exactly where SIR's
        oblique direction beats any per-column or product-of-per-column basis).
    y : (n,) array
        Continuous or discrete target; sliced into ``n_slices`` equi-frequency bins (reusing the
        existing ``_quantile_bin`` helper) regardless of whether it is a classification or
        regression target -- SIR's own construction only needs a slicing, not a native
        classification/regression distinction.
    n_slices : int
        Number of equi-frequency slices of ``y``. Li (1991)'s own guidance: more slices resolve
        finer structure but each slice needs enough rows for a stable per-slice mean; the default
        10 is the standard textbook choice.
    n_directions : int
        Number of top eigenvector directions to project onto and return as columns.

    Returns
    -------
    (n, n_directions) float64 array of projections ``X @ v_1, ..., X @ v_{n_directions}``.
    Degenerate input (n < 2, p < 1, fewer than 2 realized slices, or a singular/near-singular
    ``Sigma``) returns an ``(n, 0)`` array rather than raising -- callers treat zero emitted
    directions as "nothing to add".
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64).ravel()
    if n < 2 or p < 1 or y.size != n or n_directions < 1:
        return np.empty((n, 0), dtype=np.float64)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return np.empty((n, 0), dtype=np.float64)

    slice_ids = _quantile_bin(y, nbins=n_slices)
    uniq_slices = np.unique(slice_ids)
    if uniq_slices.size < 2:
        return np.empty((n, 0), dtype=np.float64)

    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    Sigma = (Xc.T @ Xc) / n

    # Between-slice-mean covariance M = sum_h (n_h/n) * (mean_h - x_mean) @ (mean_h - x_mean)^T.
    M = np.zeros((p, p), dtype=np.float64)
    for s in uniq_slices:
        mask = slice_ids == s
        n_h = int(mask.sum())
        if n_h < 1:
            continue
        slice_mean_dev = X[mask].mean(axis=0) - x_mean
        M += (n_h / n) * np.outer(slice_mean_dev, slice_mean_dev)

    # Ridge-stabilize Sigma so the generalized eigenproblem stays solvable even when X's columns
    # are exactly collinear (a genuinely singular Sigma) -- a tiny trace-scaled shift that does not
    # move a well-conditioned Sigma's eigenvectors materially, mirroring the same trace-scaled-ridge
    # convention used elsewhere in this codebase (e.g. _fe_pure_form_retention_gpu_resident.py).
    trace = float(np.trace(Sigma))
    if trace <= 1e-12:
        return np.empty((n, 0), dtype=np.float64)
    Sigma_ridge = Sigma + (1e-8 * trace / p) * np.eye(p)

    try:
        eigvals, eigvecs = scipy.linalg.eigh(M, Sigma_ridge)
    except (scipy.linalg.LinAlgError, ValueError):
        return np.empty((n, 0), dtype=np.float64)

    # scipy.linalg.eigh returns ascending eigenvalues; SIR wants the LARGEST (most y-variation).
    order = np.argsort(eigvals)[::-1]
    k = min(n_directions, p)
    top_dirs = eigvecs[:, order[:k]]
    return np.asarray(Xc @ top_dirs)


def engineered_name_sir_direction(cols: Sequence[str], idx: int) -> str:
    """Deterministic engineered-column name for the ``idx``-th SIR projection direction."""
    return "sir__" + "_".join(str(c) for c in cols) + f"__dir{idx}"


def generate_sir_direction_features(
    X: "pd.DataFrame", cols: Sequence[str], y: np.ndarray, *, n_slices: int = 10, n_directions: int = 2
) -> "tuple[pd.DataFrame, dict]":
    """Fit SIR over ``cols``: compute+freeze the between-slice-mean eigendirections ``v`` and the
    centering ``x_mean`` (both pure functions of X/y, frozen once at fit time), emit
    ``n_directions`` named projection columns. Returns ``(enc_df, payload)`` where ``payload``
    carries the frozen ``x_mean``/``v`` arrays needed for leak-safe replay (``y`` itself is NOT
    stored -- only its already-baked-in effect on the frozen direction vectors)."""
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 1:
        return pd.DataFrame(index=X.index), {}
    X_cols = X[cols].to_numpy(dtype=np.float64)
    n, p = X_cols.shape
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if n < 2 or p < 1 or y_arr.size != n or not np.isfinite(X_cols).all() or not np.isfinite(y_arr).all():
        return pd.DataFrame(index=X.index), {}

    slice_ids = _quantile_bin(y_arr, nbins=n_slices)
    uniq_slices = np.unique(slice_ids)
    if uniq_slices.size < 2:
        return pd.DataFrame(index=X.index), {}

    x_mean = X_cols.mean(axis=0)
    Xc = X_cols - x_mean
    Sigma = (Xc.T @ Xc) / n

    M = np.zeros((p, p), dtype=np.float64)
    for s in uniq_slices:
        mask = slice_ids == s
        n_h = int(mask.sum())
        if n_h < 1:
            continue
        slice_mean_dev = X_cols[mask].mean(axis=0) - x_mean
        M += (n_h / n) * np.outer(slice_mean_dev, slice_mean_dev)

    trace = float(np.trace(Sigma))
    if trace <= 1e-12:
        return pd.DataFrame(index=X.index), {}
    Sigma_ridge = Sigma + (1e-8 * trace / p) * np.eye(p)

    try:
        eigvals, eigvecs = scipy.linalg.eigh(M, Sigma_ridge)
    except (scipy.linalg.LinAlgError, ValueError):
        return pd.DataFrame(index=X.index), {}

    order = np.argsort(eigvals)[::-1]
    k = min(int(n_directions), p)
    top_dirs = eigvecs[:, order[:k]]
    Z = Xc @ top_dirs
    names = [engineered_name_sir_direction(cols, i) for i in range(k)]
    enc = pd.DataFrame(Z, columns=names, index=X.index)
    payload = {"cols": tuple(cols), "x_mean": x_mean.copy(), "v": top_dirs.copy()}
    return enc, payload


def apply_sir_direction(X_test: "pd.DataFrame", payload: dict) -> "pd.DataFrame":
    """Replay a fitted SIR block on new rows: project the raw source columns onto the frozen
    directions after centering with the frozen ``x_mean`` -- reads only X, no ``y``."""
    cols = list(payload["cols"])
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise KeyError(f"apply_sir_direction: missing column(s) {missing} from X_test")
    X_cols = X_test[cols].to_numpy(dtype=np.float64)
    Xc = X_cols - payload["x_mean"]
    Z = Xc @ payload["v"]
    names = [engineered_name_sir_direction(cols, i) for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=names, index=X_test.index)


def build_sir_direction_recipe(*, name: str, idx: int, cols: Sequence[str], x_mean: np.ndarray, v: np.ndarray) -> "EngineeredRecipe":
    """Frozen recipe for one SIR projection-direction column. Stores the centering ``x_mean`` +
    the single direction vector ``v[:, idx]``; replay reads only X, so transform() is
    leakage-free (the recipe never stores ``y``, only its already-baked-in effect on ``v``)."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="sir_direction",
        src_names=tuple(str(c) for c in cols),
        extra={"cols": tuple(str(c) for c in cols), "x_mean": np.asarray(x_mean, dtype=np.float64).copy(), "v": np.asarray(v[:, idx], dtype=np.float64).copy()},
    )


def _apply_sir_direction_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe`` for a single SIR direction column."""
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
    Xc = X_cols - recipe.extra["x_mean"]
    return np.asarray(Xc @ recipe.extra["v"])


def hybrid_sir_direction_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    n_slices: int = 10,
    n_directions: int = 2,
    max_cols_for_block: int = 8,
    top_k: int = 2,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    random_state: int = 0,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """End-to-end SIR oblique-direction block: bound the column pool by top raw-MI (the family's
    whole point is a joint OBLIQUE linear combination over many correlated columns, so the pool is
    capped, not combinatorially enumerated), fit the top ``n_directions`` SIR eigendirections over
    the bounded column set, MI-gate the emitted direction columns against the raw baseline, keep
    the top ``top_k``.

    Returns ``(X_aug, appended, recipes, enc_df)``. ``y`` is consumed only by the column-pool
    ranking + the SIR fit itself + the MI gate; recipes carry the frozen ``x_mean``/direction
    vector, never ``y``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_sir_direction_fe: X must be a pandas DataFrame; got {type(X).__name__}")
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
    if len(num_cols) < 1 or y is None:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, payload = generate_sir_direction_features(X, num_cols, y, n_slices=n_slices, n_directions=n_directions)
    if enc_df.empty or not payload:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate:
        from ._unified_fe_gate import local_mi_gate

        _gate_top_k = int(mi_gate_top_k) if mi_gate_top_k else int(top_k)
        winners = local_mi_gate(enc_df, y, raw_X=X, top_k=_gate_top_k, reject_sink=reject_sink)
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_sir_direction_recipe(name=name, idx=int(name.rsplit("dir", 1)[-1]), cols=payload["cols"], x_mean=payload["x_mean"], v=payload["v"]) for name in winners
    ]
    return X_aug, winners, recipes, enc_df
