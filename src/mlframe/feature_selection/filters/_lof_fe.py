"""Local Outlier Factor / k-NN local density-ratio feature (mrmr_audit_2026-07-20
fe_expansion.md "Local Outlier Factor / k-NN local density-ratio feature").

Breunig et al. (2000, "LOF: Identifying Density-Based Local Outliers"): for each row, compute the
ratio of its local k-NN density to the AVERAGE local density of its k neighbors (reachability-
distance based); rows in locally sparse regions relative to their neighborhood score high even
when the neighborhood itself is not globally extreme.

Why this catches a shape the catalog misses: distinct from a global elliptical/Gaussian anomaly
score (Mahalanobis distance, which assumes a single global covariance shape), LOF is LOCAL and
non-parametric - it catches anomalies in a MULTI-MODAL joint distribution (e.g. several
well-separated Gaussian clusters of normal behavior, where a row is anomalous for sitting in a
locally-sparse gap BETWEEN clusters even though its raw distance to the GLOBAL mean/covariance is
unremarkable, since the global Mahalanobis ellipsoid straddles all clusters and a between-cluster
point can have a perfectly ordinary global Mahalanobis distance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from numba import njit, prange
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
import pandas as pd
from mlframe.feature_selection.filters._fe_family_timing import fe_timed

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = [
    "lof_scores",
    "engineered_name_lof",
    "generate_lof_features",
    "apply_lof_block",
    "build_lof_recipe",
    "hybrid_lof_fe",
]


def _pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """(n, n) squared Euclidean distance matrix via the matmul trick ``||a-b||^2 = ||a||^2 +
    ||b||^2 - 2*a.b``; the diagonal is forced to +inf so a row never counts itself as a neighbor."""
    # Centred and formed in double: the expansion cancels catastrophically in float32 once the columns sit far from the origin; the result keeps ``X``'s dtype.
    Xc = np.asarray(X, dtype=np.float64) - np.asarray(X, dtype=np.float64).mean(axis=0)  # Intentional float64: cancellation-prone distance expansion
    sq = np.sum(Xc * Xc, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Xc @ Xc.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    return np.asarray(d2.astype(X.dtype, copy=False))


def lof_scores(X: np.ndarray, *, k: int = 20) -> np.ndarray:
    """Local Outlier Factor score per row of ``X``.

    Parameters
    ----------
    X : (n, p) array
        Numeric columns to jointly compute the local density ratio over.
    k : int
        Neighborhood size. Clamped to ``n - 1`` when ``n`` is small (fewer than ``k + 1`` rows) so
        the function degrades gracefully rather than requesting more neighbors than exist.

    Returns
    -------
    (n,) float array of LOF scores. ``LOF ~= 1`` means a row's local density matches its
    neighbors' (ordinary); ``LOF >> 1`` means a row sits in a locally sparse region relative to its
    neighborhood (a local outlier). Degenerate input (n < 3, non-finite X) returns an all-NaN
    ``(n,)`` array rather than raising.

    Cost: brute-force O(n^2) pairwise distances (the matmul trick) - fine for moderate n; very
    large n (>~200k) would need chunking or an approximate k-NN library, per the audit's own note.
    """
    X = np.asarray(X, dtype=_crit_np_dtype())
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n < 3:
        return np.full(n, np.nan, dtype=_crit_np_dtype())
    if not np.isfinite(X).all():
        return np.full(n, np.nan, dtype=_crit_np_dtype())

    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.full(n, np.nan, dtype=_crit_np_dtype())

    d2 = _pairwise_sq_dist(X)
    dist = np.sqrt(d2)

    # k nearest neighbor indices per row (unordered within the k-set; LOF's own formula only needs
    # the SET of k-NN and each pairwise distance within it, not a strict rank ordering).
    nn_idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
    nn_dist = np.take_along_axis(dist, nn_idx, axis=1)
    k_distance = nn_dist.max(axis=1)  # (n,) distance to the k-th nearest neighbor of each row

    # reach-dist_k(p, o) = max(k-distance(o), dist(p, o)) for each of p's k neighbors o.
    reach_dist = np.maximum(k_distance[nn_idx], nn_dist)  # (n, k_eff)
    lrd = 1.0 / np.maximum(reach_dist.mean(axis=1), 1e-12)  # local reachability density per row

    # LOF(p) = mean over p's k neighbors o of lrd(o) / lrd(p).
    lof = lrd[nn_idx].mean(axis=1) / np.maximum(lrd, 1e-12)
    return np.asarray(lof)


def _lof_fit_reference(X_ref: np.ndarray, k: int) -> "tuple[np.ndarray, np.ndarray, int]":
    """Fit the reference-set internals LOF needs to score OUT-OF-SAMPLE rows later: each
    reference point's own local reachability density ``lrd_ref`` and ``k-distance_ref``. Returns
    ``(lrd_ref, k_distance_ref, k_eff)``."""
    n = X_ref.shape[0]
    k_eff = min(k, n - 1)
    d2 = _pairwise_sq_dist(X_ref)
    dist = np.sqrt(d2)
    nn_idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
    nn_dist = np.take_along_axis(dist, nn_idx, axis=1)
    k_distance = nn_dist.max(axis=1)
    reach_dist = np.maximum(k_distance[nn_idx], nn_dist)
    lrd = 1.0 / np.maximum(reach_dist.mean(axis=1), 1e-12)
    return lrd, k_distance, k_eff


def _lof_transform(X_new: np.ndarray, X_ref: np.ndarray, lrd_ref: np.ndarray, k_distance_ref: np.ndarray, k_eff: int) -> np.ndarray:
    """LOF scores of ``X_new`` against a frozen reference: the fused device kernel when the tuner says so and the shape fits it (the new rows go to the device in row blocks of ``rows_per_launch``,
    so the device memory does not grow with the number of rows), else :func:`_lof_transform_host`."""
    from ._lof_gpu import lof_transform_gpu, use_device_lof

    scored = None
    if use_device_lof(int(X_new.shape[0])):
        scored = lof_transform_gpu(X_new, X_ref, lrd_ref, k_distance_ref, k_eff)
    if scored is None:
        scored = _lof_transform_host(X_new, X_ref, lrd_ref, k_distance_ref, k_eff)
    bad = ~np.isfinite(np.asarray(X_new)).all(axis=1) if np.ndim(X_new) == 2 else np.zeros(0, dtype=bool)
    if bad.any():
        scored = np.array(scored, copy=True)
        scored[bad] = np.nan  # a non-finite row has no neighbours: NaN, as ``lof_scores`` returns, not a garbage density ratio
    return scored


@njit(cache=True, parallel=True, nogil=True)
def _lof_transform_njit(X_new: np.ndarray, X_ref: np.ndarray, lrd_ref: np.ndarray, k_distance_ref: np.ndarray, k_eff: int) -> np.ndarray:
    """The LOF scores of :func:`_lof_transform_host` by a brute-force k-nearest-neighbour scan, one new row per thread: the squared distance to every reference point is formed in double
    precision and the ``k_eff`` smallest are kept by insertion into a sorted buffer of ``k_eff`` entries (no ``(rows, n_ref)`` matrix, so no row blocks and memory of ``k_eff`` per thread). The
    neighbour set equals the host's except at exact distance ties on the k-th neighbour."""
    n = X_new.shape[0]
    m = X_ref.shape[0]
    p = X_new.shape[1]
    out = np.empty(n, dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    for i in prange(n):
        bd = np.full(k_eff, np.inf)
        bj = np.zeros(k_eff, dtype=np.int64)
        worst = np.inf
        for j in range(m):
            d2 = 0.0
            for t in range(p):
                dd = X_new[i, t] - X_ref[j, t]
                d2 += dd * dd
            if d2 < worst:
                s = k_eff - 1
                while s > 0 and bd[s - 1] > d2:
                    bd[s] = bd[s - 1]
                    bj[s] = bj[s - 1]
                    s -= 1
                bd[s] = d2
                bj[s] = j
                worst = bd[k_eff - 1]
        sum_reach = 0.0
        sum_lrd = 0.0
        for s in range(k_eff):
            d = np.sqrt(bd[s])
            kd = k_distance_ref[bj[s]]
            sum_reach += kd if kd > d else d
            sum_lrd += lrd_ref[bj[s]]
        mean_reach = sum_reach / k_eff
        lrd_new = 1.0 / (mean_reach if mean_reach > 1e-12 else 1e-12)
        out[i] = (sum_lrd / k_eff) / (lrd_new if lrd_new > 1e-12 else 1e-12)
    return out


def _lof_transform_host(X_new: np.ndarray, X_ref: np.ndarray, lrd_ref: np.ndarray, k_distance_ref: np.ndarray, k_eff: int) -> np.ndarray:
    """LOF scores of ``X_new`` against the frozen reference by the njit brute-force scan (:func:`_lof_transform_njit`, one new row per thread, memory of ``k_eff`` values per thread: no row
    blocks are needed on the host). 8 to 12 times faster than the matrix form it replaced (matmul, ``argpartition`` over every distance, row blocks sized from the free RAM; the tests keep it as their
    reference): 100000 rows x 2000 reference points x 3 coordinates 3699 ms against 302 ms."""
    out_dtype = np.asarray(lrd_ref).dtype  # the scores come back in the dtype of the frozen densities, as the matrix form returned them (a replay through apply_recipe is compared bit for bit with the fit)
    if X_new.shape[0] == 0:
        return np.empty(0, dtype=out_dtype)
    scores = _lof_transform_njit(
        np.ascontiguousarray(X_new),
        np.ascontiguousarray(X_ref),
        np.ascontiguousarray(lrd_ref, dtype=np.float64),  # Intentional float64: the frozen reference densities are float64 statistics
        np.ascontiguousarray(k_distance_ref, dtype=np.float64),  # Intentional float64: the frozen k-distances are float64 statistics
        int(k_eff),
    )
    return np.asarray(scores, dtype=out_dtype)


def engineered_name_lof(cols: Sequence[str]) -> str:
    """Deterministic engineered-column name for the joint LOF score of a column set."""
    return "lof__" + "_".join(str(c) for c in cols)


@fe_timed("lof")
def generate_lof_features(X: "pd.DataFrame", cols: Sequence[str], *, k: int = 20, max_ref: int = 2000, random_state: int = 0) -> "tuple[pd.DataFrame, dict]":
    """Fit one joint LOF score over ``cols``: subsample up to ``max_ref`` rows as the frozen
    reference set (RAM discipline - a bounded reference, never the whole fit frame), compute the
    LOF score of every fit row against that reference, and freeze the reference internals for
    leak-safe replay. Returns ``(enc_df, payload)``."""
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 1:
        return pd.DataFrame(index=X.index), {}
    X_cols = X[cols].to_numpy(dtype=_crit_np_dtype())
    n, p = X_cols.shape
    if n < 3 or p < 1 or not np.isfinite(X_cols).all():
        return pd.DataFrame(index=X.index), {}

    rng = np.random.default_rng(random_state)
    ref_size = min(int(max_ref), n)
    ref_idx = rng.choice(n, size=ref_size, replace=False) if ref_size < n else np.arange(n)
    X_ref = X_cols[ref_idx]

    k_eff_check = min(k, ref_size - 1)
    if k_eff_check < 1:
        return pd.DataFrame(index=X.index), {}

    lrd_ref, k_distance_ref, k_eff = _lof_fit_reference(X_ref, k)
    lof_all = _lof_transform(X_cols, X_ref, lrd_ref, k_distance_ref, k_eff)

    name = engineered_name_lof(cols)
    enc = pd.DataFrame({name: lof_all}, index=X.index)
    payload = {"cols": tuple(cols), "X_ref": X_ref.copy(), "lrd_ref": lrd_ref.copy(), "k_distance_ref": k_distance_ref.copy(), "k_eff": int(k_eff)}
    return enc, payload


def apply_lof_block(X_test: "pd.DataFrame", payload: dict) -> "pd.DataFrame":
    """Replay a fitted LOF score on new rows: score against the frozen reference set only -
    reads the raw source columns, no ``y``."""
    cols = list(payload["cols"])
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise KeyError(f"apply_lof_block: missing column(s) {missing} from X_test")
    X_cols = X_test[cols].to_numpy(dtype=_crit_np_dtype())
    lof_new = _lof_transform(X_cols, payload["X_ref"], payload["lrd_ref"], payload["k_distance_ref"], payload["k_eff"])
    name = engineered_name_lof(cols)
    return pd.DataFrame({name: lof_new}, index=X_test.index)


def build_lof_recipe(*, name: str, cols: Sequence[str], X_ref: np.ndarray, lrd_ref: np.ndarray, k_distance_ref: np.ndarray, k_eff: int) -> "EngineeredRecipe":
    """Frozen recipe for the LOF score column. Stores the bounded reference set + its precomputed
    density internals; replay reads only X, so transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="lof_score",
        src_names=tuple(str(c) for c in cols),
        extra={
            "cols": tuple(str(c) for c in cols),
            "X_ref": np.asarray(X_ref, dtype=_crit_np_dtype()).copy(),
            "lrd_ref": np.asarray(lrd_ref, dtype=_crit_np_dtype()).copy(),
            "k_distance_ref": np.asarray(k_distance_ref, dtype=_crit_np_dtype()).copy(),
            "k_eff": int(k_eff),
        },
    )


def _apply_lof_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    cols = list(recipe.extra["cols"])
    if isinstance(X, pd.DataFrame):
        X_cols = X[cols].to_numpy(dtype=_crit_np_dtype())
    else:
        try:
            import polars as _pl

            if isinstance(X, _pl.DataFrame):
                X_cols = X.select(cols).to_numpy().astype(_crit_np_dtype())
            else:
                X_cols = np.column_stack([np.asarray(X[c], dtype=_crit_np_dtype()) for c in cols])
        except ImportError:
            X_cols = np.column_stack([np.asarray(X[c], dtype=_crit_np_dtype()) for c in cols])
    return _lof_transform(X_cols, recipe.extra["X_ref"], recipe.extra["lrd_ref"], recipe.extra["k_distance_ref"], recipe.extra["k_eff"])


def hybrid_lof_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    k: int = 20,
    max_ref: int = 2000,
    max_cols_for_block: int = 8,
    top_k: int = 1,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    random_state: int = 0,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """End-to-end joint LOF anomaly-score feature: bound the column pool by top raw-MI (LOF's
    O(n^2)-in-the-reference cost and its own multi-modal-density point are both about a JOINT
    score over a bounded set of columns, not a per-column or combinatorial-pair enumeration),
    compute one LOF score column, MI-gate it against the raw baseline.

    Returns ``(X_aug, appended, recipes, enc_df)``. ``y`` is consumed only by the column-pool
    ranking + the MI gate; recipes carry the frozen bounded reference set, never ``y``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_lof_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [c for c in num_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) < 1:
        return X, [], [], pd.DataFrame()

    if y is not None:
        from ._extra_fe_families import _top_mi_num_cols  # lazy: break parent<->sibling cycle

        num_cols = _top_mi_num_cols(X, num_cols, y, max_cols_for_block)
    else:
        num_cols = list(num_cols)[: int(max_cols_for_block)]
    if len(num_cols) < 1:
        return X, [], [], pd.DataFrame()

    enc_df, payload = generate_lof_features(X, num_cols, k=k, max_ref=max_ref, random_state=random_state)
    if enc_df.empty or not payload:
        return X, [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        _gate_top_k = int(mi_gate_top_k) if mi_gate_top_k is not None and mi_gate_top_k > 0 else int(top_k)
        winners = local_mi_gate(enc_df, y, raw_X=X, top_k=_gate_top_k, reject_sink=reject_sink)
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X, [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_lof_recipe(name=name, cols=payload["cols"], X_ref=payload["X_ref"], lrd_ref=payload["lrd_ref"], k_distance_ref=payload["k_distance_ref"], k_eff=payload["k_eff"])
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df
