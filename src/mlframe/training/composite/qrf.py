"""Quantile-regression-forest distributional composite (single-fit full UQ).

``CompositeQRFEstimator`` is a NON-PARAMETRIC alternative to the dense-quantile
:class:`CompositeDistributionEstimator`. Where the dense-quantile estimator refits
ONE pinball-trained inner per requested level (K boosting fits for K quantiles), the
QRF composite fits a SINGLE quantile-regression FOREST on the transform
``T = f(y, base)`` and serves the CONDITIONAL DISTRIBUTION of ``T`` for ANY query
quantile from that one fitted model -- no per-level refit.

How it works.

- A forest (sklearn ``RandomForestRegressor`` / ``ExtraTreesRegressor``) is fit to
  predict ``T``. The Meinshausen (2006) quantile-regression-forest insight: a tree's
  leaf is a conditioning set, and the TRAINING ``T`` values that fell into the same
  leaf as a query row form a (weighted) sample of the conditional distribution
  ``T | X``. Averaging that membership over all trees gives a smooth conditional CDF.
- At fit we cache, per tree, the leaf index of every training row plus the training
  ``T`` vector. At predict we apply each tree, gather the per-leaf training-``T``
  weights for the query row, aggregate across trees, and read off any quantile by
  weighted inversion of the resulting empirical CDF -- ALL quantiles from ONE model.
- The forest's quantile of ``T`` is then inverted to the y-scale by the SAME shared
  transform machinery the rest of the composite subsystem uses: the QRF backend is
  the inner of a :class:`CompositeTargetEstimator`, so the transform fit, base
  extraction, domain filtering, T-clip, y-clip and fallback routing are reused
  verbatim (the backend only has to expose ``predict`` + ``predict_quantile(X, a)``).

Backend.

- If the optional ``quantile-forest`` package is installed it is preferred
  (``RandomForestQuantileRegressor``); it stores the leaf sets in compiled code and
  is faster on large forests.
- Otherwise a from-scratch :class:`_LeafResidualForest` (pure sklearn + numpy)
  caches the per-leaf training-``T`` arrays and does the weighted-quantile inversion
  itself. This is the default workhorse and needs no extra dependency.

Surface (mirrors :class:`CompositeDistributionEstimator`).

- :meth:`predict_quantile` -- y-scale quantiles ``(n, n_q)`` for ANY levels from the
  single fit; non-crossing by construction (a weighted-quantile of one sample is
  monotone in the level) and re-asserted with a per-row sort.
- :meth:`predict_cdf` -- the step-CDF implied by a dense internal quantile grid.
- :meth:`crps` -- the CRPS via the pinball / quantile-decomposition identity, exactly
  as the dense-quantile estimator computes it, so the two are directly comparable.

Memory. No frame copy. The backend pulls only the narrow feature matrix sklearn
needs; the wrapper operates on the small ``(n, K)`` quantile / leaf-weight matrices,
never on a copy of the feature frame.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from .estimator import CompositeTargetEstimator
from .transforms import get_transform

try:
    import numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a project dep but guard anyway
    numba = None  # type: ignore
    _HAS_NUMBA = False


def _leaf_weights_kernel(
    q_leaves: np.ndarray, train_leaves: np.ndarray, leaf_inv: np.ndarray, n_train: int,
) -> np.ndarray:
    """Meinshausen forest weights ``(n_query, n_train)`` without a per-tree bool matrix.

    For each (query i, tree t) the query falls in leaf ``q_leaves[i, t]`` whose inverse
    size is ``leaf_inv[t, leaf]``; every training row sharing that leaf gets that weight
    added. Looping (tree, query, train) in machine code avoids allocating the dense
    ``(n_query, n_train)`` boolean per tree that the numpy broadcast builds -- it walks
    the integer leaf ids directly. Parallel over query rows. Falls back to the same
    numpy broadcast when numba is unavailable.
    """
    n_query = q_leaves.shape[0]
    n_trees = q_leaves.shape[1]
    w = np.zeros((n_query, n_train), dtype=np.float64)
    for i in numba.prange(n_query):
        for t in range(n_trees):
            leaf = q_leaves[i, t]
            inv = leaf_inv[t, leaf]
            if inv <= 0.0:
                continue
            for j in range(n_train):
                if train_leaves[j, t] == leaf:
                    w[i, j] += inv
    if n_trees > 0:
        w /= n_trees
    return w


if _HAS_NUMBA:  # pragma: no branch
    _leaf_weights_kernel = numba.njit(parallel=True, cache=True)(_leaf_weights_kernel)


def _batch_weighted_quantiles_kernel(
    w: np.ndarray, y_train: np.ndarray, levels: np.ndarray, out: np.ndarray, start: int,
) -> None:
    """Per-row weighted-ECDF quantile inversion for a whole query batch in one prange pass.

    ``w`` is the dense ``(batch, n_train)`` membership matrix; per query row we gather the nonzero ``(y_train, weight)``
    pairs into a compact buffer, insertion-sort by value (stable, matching numpy's ``argsort(kind='mergesort')`` tie
    order), form the centered cumulative-weight plotting positions ``(cum - 0.5*w)/total`` and binary-search-interp the
    requested ``levels`` onto that monotone curve. This replaces the Python per-row loop that masked-then-argsort-then-
    ``np.interp``'d one row at a time; the kernel walks the row once, sorts only the nonzero members, and parallelises
    over rows. Flat zero-weight rows write NaN (the transform fallback routes them), exactly as the Python path did.
    """
    nq = w.shape[0]
    nt = w.shape[1]
    nl = levels.shape[0]
    for r in numba.prange(nq):
        m = 0
        for j in range(nt):
            if w[r, j] > 0.0:
                m += 1
        if m == 0:
            for li in range(nl):
                out[start + r, li] = np.nan
            continue
        vbuf = np.empty(m, dtype=np.float64)
        wbuf = np.empty(m, dtype=np.float64)
        p = 0
        total = 0.0
        for j in range(nt):
            wj = w[r, j]
            if wj > 0.0:
                vbuf[p] = y_train[j]
                wbuf[p] = wj
                total += wj
                p += 1
        for a in range(1, m):
            kv = vbuf[a]
            kw = wbuf[a]
            b = a - 1
            while b >= 0 and vbuf[b] > kv:
                vbuf[b + 1] = vbuf[b]
                wbuf[b + 1] = wbuf[b]
                b -= 1
            vbuf[b + 1] = kv
            wbuf[b + 1] = kw
        posbuf = np.empty(m, dtype=np.float64)
        cum = 0.0
        for j in range(m):
            cum += wbuf[j]
            posbuf[j] = (cum - 0.5 * wbuf[j]) / total
        for li in range(nl):
            x = levels[li]
            if x <= posbuf[0]:
                out[start + r, li] = vbuf[0]
                continue
            if x >= posbuf[m - 1]:
                out[start + r, li] = vbuf[m - 1]
                continue
            lo = 0
            hi = m - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if posbuf[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            p0 = posbuf[lo]
            p1 = posbuf[hi]
            if p1 == p0:
                out[start + r, li] = vbuf[lo]
            else:
                out[start + r, li] = vbuf[lo] + (vbuf[hi] - vbuf[lo]) * (x - p0) / (p1 - p0)


if _HAS_NUMBA:  # pragma: no branch
    _batch_weighted_quantiles_kernel = numba.njit(parallel=True, cache=True, fastmath=True)(
        _batch_weighted_quantiles_kernel
    )

logger = logging.getLogger(__name__)

# Internal dense grid used by predict_cdf / crps: 0.05 .. 0.95 step 0.05 (19 levels).
# A single QRF fit serves any level, so the grid only sets the CRPS-integral mesh.
_DEFAULT_DENSE_QUANTILES: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))

# Query-row batch size for the leaf-weight membership matrix. Bounds peak predict-time
# RAM to ``_PREDICT_BATCH * n_train`` floats so an arbitrarily large query frame cannot
# materialise an n_query x n_train weight matrix in one shot.
_PREDICT_BATCH: int = 512


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Weighted empirical quantiles of ``values`` at the requested ``levels``.

    ``values`` / ``weights`` are 1-D and aligned (the conditional sample and its
    membership weights for one query row); ``levels`` is the 1-D query grid in (0, 1).
    Uses the standard right-continuous weighted-ECDF inversion: sort by value, form the
    cumulative-weight midpoints ``(cum - 0.5*w)/total`` (Hyndman-Fan-style centering so
    the median of equal weights is the middle value), then ``np.interp`` the levels onto
    that monotone curve. Monotone in ``levels`` by construction, so the returned vector
    is already non-crossing. Empty / zero-weight input returns NaN (caller routes it
    through the transform fallback).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    lv = np.asarray(levels, dtype=np.float64).reshape(-1)
    total = w.sum()
    if v.size == 0 or total <= 0.0:
        return np.full(lv.shape[0], np.nan, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    v_s = v[order]
    w_s = w[order]
    cum = np.cumsum(w_s)
    # Centered plotting positions in [0, 1]; flat endpoints below/above the extremes.
    pos = (cum - 0.5 * w_s) / total
    return np.interp(lv, pos, v_s, left=v_s[0], right=v_s[-1])


class _LeafResidualForest:
    """From-scratch quantile-regression forest (pure sklearn + numpy).

    Fits a ``RandomForestRegressor`` / ``ExtraTreesRegressor`` to ``T`` and caches,
    per tree, the leaf index of every training row plus the training ``T`` vector. At
    predict it reconstructs the conditional distribution of ``T`` from the matching
    leaf sets (Meinshausen 2006) and exposes the sklearn-inner contract the
    :class:`CompositeTargetEstimator` needs: ``predict`` (conditional mean) and
    ``predict_quantile(X, alpha)`` (conditional quantiles, scalar or vector ``alpha``).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        forest_kind: str = "rf",
        min_samples_leaf: int = 5,
        max_features: Any = 1.0,
        n_jobs: int | None = -1,
        random_state: int | None = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.forest_kind = forest_kind
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "forest_kind": self.forest_kind,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "_LeafResidualForest":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None, **kw: Any) -> "_LeafResidualForest":
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        cls = ExtraTreesRegressor if self.forest_kind == "et" else RandomForestRegressor
        self.forest_ = cls(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.forest_.fit(Xa, ya, sample_weight=sample_weight)
        self.y_train_ = ya
        # Per-tree leaf id of every training row: (n_train, n_trees).
        self.train_leaves_ = self.forest_.apply(Xa)
        # Per-tree, per-leaf bincount of training rows (membership normaliser): the QRF
        # weight of a training row for a query is 1 / (#train rows sharing its leaf),
        # summed over trees. Precompute the per-tree leaf-size lookup once.
        n_trees = self.train_leaves_.shape[1]
        max_leaf = int(self.train_leaves_.max()) + 1 if self.train_leaves_.size else 1
        # Packed (n_trees, max_leaf) inverse-leaf-size table: leaf_inv[t, leaf] =
        # 1/|leaf| (0 for empty leaf ids in the padded tail). One contiguous array the
        # njit kernel can index without a python list of ragged bincounts.
        self._leaf_inv_ = np.zeros((n_trees, max_leaf), dtype=np.float64)
        for t in range(n_trees):
            counts = np.bincount(self.train_leaves_[:, t], minlength=max_leaf)
            nz = counts > 0
            self._leaf_inv_[t, nz] = 1.0 / counts[nz]
        # Train leaves transposed to (n_train, n_trees) is already the apply() layout;
        # the kernel reads train_leaves[j, t], so keep it as-is.
        return self

    def _leaf_weights(self, X: Any) -> np.ndarray:
        """QRF membership weights of every training row for every query row.

        Returns ``(n_query, n_train)``: weight ``w_ij`` is the Meinshausen forest
        weight of training row ``j`` for query ``i`` -- averaged over trees of
        ``1{leaf_t(i) == leaf_t(j)} / |leaf_t(i)|``. Each query row's weights sum to 1.
        """
        Xa = np.asarray(X, dtype=np.float64)
        q_leaves = np.ascontiguousarray(self.forest_.apply(Xa))  # (n_query, n_trees)
        n_train = self.train_leaves_.shape[0]
        if _HAS_NUMBA:
            # njit(parallel) kernel: walks leaf ids directly, no per-tree bool matrix.
            return _leaf_weights_kernel(q_leaves, self.train_leaves_, self._leaf_inv_, n_train)
        # numpy fallback: one vectorised (n_query x n_train) boolean broadcast per tree.
        n_query = q_leaves.shape[0]
        n_trees = q_leaves.shape[1]
        w = np.zeros((n_query, n_train), dtype=np.float64)
        for t in range(n_trees):
            tl = self.train_leaves_[:, t]
            ql = q_leaves[:, t]
            match = ql[:, None] == tl[None, :]
            inv = self._leaf_inv_[t, ql]
            w += match * inv[:, None]
        if n_trees > 0:
            w /= n_trees
        return w

    def predict(self, X: Any) -> np.ndarray:
        """Conditional-mean prediction of ``T`` (forest mean, matches the leaf mean)."""
        return np.asarray(self.forest_.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64)

    def predict_quantile(self, X: Any, alpha: float | Sequence[float] = 0.5) -> np.ndarray:
        """Conditional quantiles of ``T`` at ``alpha`` from the single fitted forest.

        ``alpha`` scalar -> ``(n_query,)``; ``alpha`` vector -> ``(n_query, K)``. Every
        column comes from the SAME fit (no refit per level). Per query row we form the
        weighted conditional sample (training ``T`` weighted by leaf membership) and
        invert its weighted ECDF at the requested level(s).

        Memory: the per-tree membership matrix is ``(batch, n_train)``; we process
        queries in row batches of ``_PREDICT_BATCH`` so peak RAM is bounded by
        ``batch * n_train`` regardless of how many query rows are passed (a 100 GB query
        frame never materialises a 100 GB weight matrix).
        """
        scalar = np.isscalar(alpha)
        levels = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        Xa = np.asarray(X, dtype=np.float64)
        n_query = Xa.shape[0]
        out = np.empty((n_query, levels.shape[0]), dtype=np.float64)
        for start in range(0, n_query, _PREDICT_BATCH):
            stop = min(start + _PREDICT_BATCH, n_query)
            w = self._leaf_weights(Xa[start:stop])  # (b, n_train)
            if _HAS_NUMBA:
                _batch_weighted_quantiles_kernel(w, self.y_train_, levels, out, start)
            else:
                for r in range(stop - start):
                    wi = w[r]
                    nz = wi > 0.0
                    out[start + r, :] = _weighted_quantiles(self.y_train_[nz], wi[nz], levels)
        return out.reshape(-1) if scalar else out


def _make_backend(
    prefer_quantile_forest: bool,
    n_estimators: int,
    forest_kind: str,
    min_samples_leaf: int,
    max_features: Any,
    n_jobs: int | None,
    random_state: int | None,
) -> Any:
    """Build the QRF backend, preferring the ``quantile-forest`` package if present.

    ``quantile-forest`` (``RandomForestQuantileRegressor``) stores leaf sets in
    compiled code and is faster on large forests; when unavailable (or disabled) we use
    the pure-sklearn :class:`_LeafResidualForest`. Both expose ``predict`` +
    ``predict_quantile(X, alpha)``, so the wrapper is backend-agnostic.
    """
    if prefer_quantile_forest:
        try:  # pragma: no cover - optional dep absent in CI
            from quantile_forest import RandomForestQuantileRegressor

            return _QuantileForestAdapter(
                RandomForestQuantileRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
            )
        except ImportError:
            logger.debug("quantile-forest not installed; using pure-sklearn leaf-residual forest.")
    return _LeafResidualForest(
        n_estimators=n_estimators,
        forest_kind=forest_kind,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )


class _QuantileForestAdapter(BaseEstimator, RegressorMixin):  # pragma: no cover - exercised only when quantile-forest installed
    """Adapt ``quantile_forest.RandomForestQuantileRegressor`` to the inner contract.

    Maps ``predict_quantile(X, alpha)`` onto the package's ``predict(X, quantiles=...)``
    and exposes a plain ``predict`` (median) so the wrapper treats it like any inner.

    ``model`` is the single ``__init__`` arg and is stored verbatim so the default
    BaseEstimator get_params/set_params round-trip and sklearn.clone (which recurses
    into the nested ``model``) both work -- delegating get_params to ``self.model``
    leaked the forest's params and made clone reconstruct the adapter with forest
    kwargs it does not accept (TypeError when an inner pipeline cloned this).
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None, **kw: Any) -> "_QuantileForestAdapter":
        self.model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1), sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        out = self.model.predict(np.asarray(X, dtype=np.float64), quantiles=[0.5])
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def predict_quantile(self, X: Any, alpha: float | Sequence[float] = 0.5) -> np.ndarray:
        scalar = np.isscalar(alpha)
        levels = list(np.atleast_1d(np.asarray(alpha, dtype=np.float64)))
        out = np.asarray(self.model.predict(np.asarray(X, dtype=np.float64), quantiles=levels), dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out.reshape(-1) if scalar else out


class CompositeQRFEstimator(BaseEstimator, RegressorMixin):
    """Quantile-regression-forest distributional composite: full UQ from ONE fit.

    Parameters
    ----------
    transform_name, base_column, base_columns, group_column,
    fallback_predict, drop_invalid_rows
        Forwarded to the inner :class:`CompositeTargetEstimator` (shared transform
        fit / inverse / domain / clip machinery). ``linear_residual`` is canonical.
    n_estimators, forest_kind, min_samples_leaf, max_features, n_jobs, random_state
        Forest hyper-parameters. ``forest_kind`` is ``"rf"`` (RandomForest, default) or
        ``"et"`` (ExtraTrees). ``min_samples_leaf`` controls the conditional-sample
        size per leaf -- larger smooths the distribution, smaller sharpens it.
    prefer_quantile_forest
        Use the optional ``quantile-forest`` package when installed (default True);
        falls back to the pure-sklearn leaf-residual forest otherwise.
    enforce_non_crossing
        Re-assert per-row ascending quantiles via a sort (default True). Weighted
        quantiles of one sample are already monotone, so this is a cheap safety net.

    Attributes set at fit
    ---------------------
    estimator_ : the fitted :class:`CompositeTargetEstimator` wrapping the QRF backend.
    feature_names_in_ : inherited from the inner (best effort).
    """

    def __init__(
        self,
        transform_name: str = "linear_residual",
        base_column: str = "",
        n_estimators: int = 200,
        forest_kind: str = "rf",
        min_samples_leaf: int = 5,
        max_features: Any = 1.0,
        n_jobs: int | None = -1,
        random_state: int | None = 0,
        prefer_quantile_forest: bool = True,
        fallback_predict: str = "y_train_median",
        drop_invalid_rows: bool = True,
        base_columns: Sequence[str] | None = None,
        group_column: str | None = None,
        enforce_non_crossing: bool = True,
    ) -> None:
        self.transform_name = transform_name
        self.base_column = base_column
        self.n_estimators = n_estimators
        self.forest_kind = forest_kind
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.prefer_quantile_forest = prefer_quantile_forest
        self.fallback_predict = fallback_predict
        self.drop_invalid_rows = drop_invalid_rows
        self.base_columns = base_columns
        self.group_column = group_column
        self.enforce_non_crossing = enforce_non_crossing

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None, **fit_kwargs: Any) -> "CompositeQRFEstimator":
        """Fit ONE quantile-regression forest on the transform ``T``.

        Validates the transform up front, builds the QRF backend, wraps it in a
        :class:`CompositeTargetEstimator` (so all transform / domain / clip / fallback
        state is shared), and fits it once. ``sample_weight`` / ``**fit_kwargs`` flow to
        the inner. Returns ``self``.
        """
        get_transform(self.transform_name)  # surface a typo'd name at fit
        backend = _make_backend(
            self.prefer_quantile_forest,
            self.n_estimators,
            self.forest_kind,
            self.min_samples_leaf,
            self.max_features,
            self.n_jobs,
            self.random_state,
        )
        inner = CompositeTargetEstimator(
            base_estimator=backend,
            transform_name=self.transform_name,
            base_column=self.base_column,
            base_columns=self.base_columns,
            group_column=self.group_column,
            fallback_predict=self.fallback_predict,
            drop_invalid_rows=self.drop_invalid_rows,
        )
        inner.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        self.estimator_ = inner
        ref_names = getattr(inner, "feature_names_in_", None)
        if ref_names is not None:
            self.feature_names_in_ = list(ref_names)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("CompositeQRFEstimator: call fit before this method.")

    # ------------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        """Median (0.5-quantile) y-scale point prediction from the single forest."""
        self._check_fitted()
        return self.predict_quantile(X, quantiles=[0.5]).reshape(-1)

    def predict_quantile(self, X: Any, quantiles: Sequence[float] | None = None) -> np.ndarray:
        """y-scale quantiles ``(n, n_q)`` for ANY ``quantiles`` from the ONE fit.

        Reads the forest's conditional ``T``-quantiles at the requested levels and
        inverts each to the y-scale via the wrapper's transform inverse. No per-level
        refit -- the same fitted forest serves every level. Rows are sorted ascending
        when ``enforce_non_crossing`` so ``q_low <= ... <= q_high``.
        """
        self._check_fitted()
        levels = (
            np.asarray(quantiles, dtype=np.float64).reshape(-1)
            if quantiles is not None
            else np.asarray(_DEFAULT_DENSE_QUANTILES, dtype=np.float64)
        )
        if levels.size == 0:
            raise ValueError("CompositeQRFEstimator.predict_quantile: quantiles is empty.")
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("CompositeQRFEstimator.predict_quantile: levels must be strictly in (0, 1).")
        levels = np.sort(levels)
        out = np.asarray(self.estimator_.predict_quantile(X, alpha=levels), dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if self.enforce_non_crossing and out.shape[1] > 1:
            out = np.sort(out, axis=1)
        return out

    def _dense_matrix(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """The dense ``(n, K)`` y-quantile matrix + its level grid (for cdf / crps)."""
        levels = np.asarray(_DEFAULT_DENSE_QUANTILES, dtype=np.float64)
        return self.predict_quantile(X, quantiles=levels), levels

    # ------------------------------------------------------------------
    def predict_cdf(self, X: Any, y_grid: Sequence[float]) -> np.ndarray:
        """Step-CDF implied by the QRF quantiles, evaluated at ``y_grid``.

        ``(n_samples, len(y_grid))``: entry ``[i, j]`` is the predicted
        ``F_i(y_grid[j]) = `` the highest fitted quantile LEVEL whose y-VALUE is
        ``<= y_grid[j]`` (0 below the lowest). Monotone non-decreasing in the grid by
        construction (the per-row quantile values are ascending), so each row is a valid
        CDF. Identical step-CDF definition to :class:`CompositeDistributionEstimator`.
        """
        self._check_fitted()
        qmat, levels = self._dense_matrix(X)  # (n, K), (K,)
        t = np.asarray(y_grid, dtype=np.float64).reshape(-1)
        leq = qmat[:, :, None] <= t[None, None, :]  # (n, K, G)
        level_if = np.where(leq, levels[None, :, None], 0.0)
        return level_if.max(axis=1)

    def crps(self, X: Any, y_true: Any, reduce: str = "mean") -> Any:
        """CRPS from the QRF quantile representation (quantile decomposition).

        ``CRPS(F_i, y_i) = (2 / K) * sum_k rho_{q_k}(y_i - Q_i(q_k))`` with the pinball
        loss ``rho_q(u) = u * (q - 1{u < 0})`` over the dense level grid (Gneiting &
        Raftery 2007). Lower is better; strictly proper, so a calibrated SHARP
        distribution scores below a wider one. Identical formula to
        :class:`CompositeDistributionEstimator`, so the two estimators' CRPS values are
        directly comparable on the same data.

        ``reduce`` : ``"mean"`` (scalar, default) or ``"none"`` (per-row ``(n,)``).
        """
        self._check_fitted()
        qmat, levels = self._dense_matrix(X)  # (n, K)
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        if y.shape[0] != qmat.shape[0]:
            raise ValueError(
                f"CompositeQRFEstimator.crps: y_true length {y.shape[0]} != n_samples {qmat.shape[0]}."
            )
        u = y[:, None] - qmat  # (n, K)
        rho = u * (levels[None, :] - (u < 0.0).astype(np.float64))
        per_row = (2.0 / levels.shape[0]) * rho.sum(axis=1)
        if reduce == "none":
            return per_row
        if reduce == "mean":
            return float(np.mean(per_row))
        raise ValueError(f"CompositeQRFEstimator.crps: reduce must be 'mean' or 'none', got {reduce!r}.")
