"""Group-aware mRMR via correlation-based pre-clustering.

Vanilla mRMR with high-correlation feature sets (one-hot expansion, repeated lags, calibration variants of the same sensor) selects ONE
representative and discards the rest. Often the operator wants the **group**, not just one member -- either to display them together in
a downstream report or to feed them into an ensemble that benefits from the redundancy.

Two-step approach:

1. ``cluster_features_by_correlation(X, threshold=0.9, ...)`` -- greedy clustering: every pair with ``|corr| > threshold`` ends in the
   same cluster (single-linkage on the correlation graph). Returns a ``cluster_id`` per feature.
2. ``GroupAwareMRMR(estimator, ...).fit(X, y)`` -- runs mRMR on the per-cluster medoids (the feature with highest mean abs-corr to its
   cluster mates), then expands the support to all members of any selected cluster. ``cluster_assignments_`` and ``selected_clusters_``
   are exposed for inspection.

For users with **explicit** group structure (one-hot expansions where the operator knows ``group_name -> column_list``), prefer
``RFECV(feature_groups=...)`` -- it has a more thorough all-or-nothing voting protocol. This module is for **discovered** groups via
correlation when the operator does not know the structure upfront.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from mlframe.utils.misc import rng_hygienic_fit

logger = logging.getLogger(__name__)

try:
    import numba

    _HAVE_NUMBA = True
except ImportError:  # numba is an optional dep; SU falls back to the pure-Python pair loop.
    _HAVE_NUMBA = False


def _su_pairs_python(codes: np.ndarray, ncats: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Pure-Python reference for the SU pair matrix (used when numba is absent). O(p^2 * n)."""
    p = codes.shape[1]
    su = np.zeros((p, p), dtype=np.float64)
    for a in range(p):
        for b in range(a + 1, p):
            joint = np.bincount(codes[:, a] * int(ncats[b]) + codes[:, b], minlength=int(ncats[a] * ncats[b]))
            tot = joint.sum()
            pr = joint[joint > 0] / tot if tot > 0 else np.empty(0)
            h_ab = float(-(pr * np.log(pr)).sum()) if pr.size else 0.0
            denom = h[a] + h[b]
            val = 0.0 if denom <= 1e-12 else max(0.0, 2.0 * (h[a] + h[b] - h_ab) / denom)
            su[a, b] = su[b, a] = val
    return su


if _HAVE_NUMBA:

    @numba.njit(cache=True, parallel=True)
    def _su_pairs_njit(codes, ncats, h):
        """Fused joint-histogram + entropy SU matrix. prange over row ``a`` writes ONLY the strict upper
        triangle (row a, cols > a) so no two threads touch the same cell; the caller symmetrises. Removes
        the per-pair ``np.bincount`` allocation of the Python path and scales across cores."""
        n, p = codes.shape
        su = np.zeros((p, p), dtype=np.float64)
        for a in numba.prange(p):
            for b in range(a + 1, p):
                nb = ncats[b]
                joint = np.zeros(ncats[a] * nb, dtype=np.int64)
                for i in range(n):
                    joint[codes[i, a] * nb + codes[i, b]] += 1
                tot = 0.0
                for k in range(joint.shape[0]):
                    tot += joint[k]
                h_ab = 0.0
                if tot > 0.0:
                    for k in range(joint.shape[0]):
                        c = joint[k]
                        if c > 0:
                            pr = c / tot
                            h_ab -= pr * np.log(pr)
                denom = h[a] + h[b]
                if denom > 1e-12:
                    val = 2.0 * (h[a] + h[b] - h_ab) / denom
                    su[a, b] = val if val > 0.0 else 0.0
        return su


def _numeric_codes_frame(X: "pd.DataFrame") -> "pd.DataFrame":
    """Return a float DataFrame where non-numeric columns are replaced by their
    integer factor codes, so correlation / SU clustering never float-coerces a
    raw categorical / object / string column (e.g. a ``"NA"`` level) and crashes.

    Redundancy clustering is an unsupervised same-row operation; per-column
    ``factorize`` (NaN -> code -1) is a valid integer encoding that preserves the
    original column count and order, so cluster indices still map back to the
    caller's feature positions. Numeric columns pass through unchanged.
    """
    # Iterate POSITIONALLY (``.iloc[:, j]``), not by label: duplicate column names (common after FE expansion -- repeated
    # lags, one-hot level collisions) make ``X[label]`` return a DataFrame, whose ``.dtype`` access raises. Positional
    # access always yields a Series, and we rebuild from a list so duplicate labels are preserved in original order.
    series_list = []
    for j in range(X.shape[1]):
        s = X.iloc[:, j]
        if pd.api.types.is_numeric_dtype(s.dtype):
            series_list.append(np.asarray(s.to_numpy()))
        else:
            try:
                codes, _ = pd.factorize(s, use_na_sentinel=True)
            except TypeError:
                # An object column whose values are themselves unhashable (e.g. an embedding column
                # storing one ndarray per row) cannot be factor-coded. Emit an all-zero (single-code)
                # column instead of raising -- it carries no redundancy signal, but preserves the
                # column-count/order invariant this function promises callers.
                codes = np.zeros(len(s), dtype=np.intp)
            series_list.append(np.asarray(codes))
    out = pd.DataFrame(np.column_stack(series_list) if series_list else np.empty((len(X), 0)), index=X.index)
    out.columns = list(X.columns)
    return out


def _su_redundancy_matrix(X, nbins: int = 10) -> np.ndarray:
    """Pairwise Symmetric-Uncertainty redundancy matrix in [0, 1] (diagonal 0).

    Unlike Pearson/Spearman (which only see monotone dependence), SU captures arbitrary -- including
    non-linear / non-monotone -- redundancy between two columns: two features that are deterministic but
    non-monotone functions of each other score ~1 here but ~0 under Pearson. Each column is quantile-binned
    to ``nbins`` codes (continuous) or used as-is when already low-cardinality, then SU = 2*I/(H_a+H_b) is
    computed per pair from the joint bincount. O(p^2) over the (already cluster-bounded) feature set.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    arr = _numeric_codes_frame(X).to_numpy()
    n, p = arr.shape
    codes = np.empty((n, p), dtype=np.int64)
    ncats = np.empty(p, dtype=np.int64)
    for j in range(p):
        col = arr[:, j]
        col = np.asarray(col, dtype=np.float64)
        finite = np.isfinite(col)
        uniq = np.unique(col[finite]) if finite.any() else np.array([0.0])
        if uniq.shape[0] <= nbins:
            # Low-cardinality / already-discrete: map values to dense codes. ``uniq`` is sorted and every
            # FINITE value is in it, so ``np.searchsorted`` returns each value's exact dense code; non-finite
            # values sort last -> ``len(uniq)`` sentinel, identical to the prior dict ``.get(v, len(lookup))``.
            # Vectorised: replaces an O(n) per-column Python dict-lookup list comprehension (7x at 300k).
            c = np.searchsorted(uniq, col).astype(np.int64)
            ncats[j] = uniq.shape[0] + (0 if finite.all() else 1)
        else:
            qs = np.quantile(col[finite], np.linspace(0, 1, nbins + 1)[1:-1])
            c = np.digitize(col, qs).astype(np.int64)
            c[~finite] = nbins  # NaN sentinel bin
            ncats[j] = nbins + (0 if finite.all() else 1)
        codes[:, j] = c

    def _entropy(counts):
        """Shannon entropy (nats) of a bincount vector; skips zero-count bins to avoid log(0)."""
        tot = counts.sum()
        if tot <= 0:
            return 0.0
        pr = counts[counts > 0] / tot
        return float(-(pr * np.log(pr)).sum())

    h = np.array([_entropy(np.bincount(codes[:, j], minlength=int(ncats[j]))) for j in range(p)])
    # The O(p^2 * n) pair loop is the SU bottleneck: numba fuses the joint-histogram + entropy and parallelises across
    # cores (upper triangle only, then symmetrised), falling back to the pure-Python reference when numba is absent.
    if _HAVE_NUMBA and p > 1:
        su = np.ascontiguousarray(_su_pairs_njit(np.ascontiguousarray(codes), ncats, h))
        su = su + su.T
    else:
        su = _su_pairs_python(codes, ncats, h)
    return su


def _redundancy_matrix(X, method: str) -> np.ndarray:
    """Abs redundancy matrix (diagonal 0) for the given method. ``'su'`` uses Symmetric Uncertainty (captures non-linear redundancy); everything else delegates to ``pandas.DataFrame.corr``."""
    if method == "su":
        return _su_redundancy_matrix(X)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # Factor-code non-numeric columns first: ``DataFrame.corr`` float-coerces every
    # column and would crash on a raw categorical / string level (e.g. ``"NA"``).
    corr = np.array(_numeric_codes_frame(X).corr(method=method).abs().to_numpy(), copy=True)
    np.fill_diagonal(corr, 0.0)
    return corr


def cluster_features_by_correlation(
    X,
    threshold: float = 0.9,
    method: str = "spearman",
    precomputed_corr: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy single-linkage clustering on the correlation graph.

    Two features land in the same cluster iff there's a chain of pairwise ``|corr| > threshold`` connections between them.

    Parameters
    ----------
    X : array or DataFrame, shape (n_samples, n_features)
    threshold : float, default 0.9
    method : {"pearson", "spearman", "kendall", "su"}
        Redundancy measure. The corr methods go through ``pandas.DataFrame.corr``; ``"su"`` uses Symmetric
        Uncertainty (captures non-linear / non-monotone redundancy the corr methods miss).

    Returns
    -------
    cluster_id : ndarray of shape (n_features,)
        Integer cluster labels in [0, n_clusters - 1].
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # The p x p redundancy matrix is a function of (X, method) alone; the caller (fit) computes it ONCE and threads it
    # through both this clustering pass and _cluster_medoids to avoid a second O(p^2) build (p = features, bounded).
    corr = precomputed_corr if precomputed_corr is not None else _redundancy_matrix(X, method)
    n = corr.shape[0]

    cluster_id = np.arange(n)  # union-find roots

    def _find(i: int) -> int:
        """Union-find root lookup with path halving (each visited node re-points to its grandparent)."""
        while cluster_id[i] != i:
            cluster_id[i] = cluster_id[cluster_id[i]]
            i = cluster_id[i]
        return i

    def _union(i: int, j: int) -> None:
        """Union-find merge: attach the higher-index root under the lower-index one so the smaller label wins as cluster id."""
        ri, rj = _find(i), _find(j)
        if ri != rj:
            cluster_id[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > threshold:
                _union(i, j)

    # Compress + relabel.
    roots = np.array([_find(i) for i in range(n)])
    _, compact = np.unique(roots, return_inverse=True)
    return compact


def _cluster_medoids(
    X,
    cluster_id: np.ndarray,
    method: str = "spearman",
    precomputed_corr: np.ndarray | None = None,
) -> list[int]:
    """For each cluster, pick the column with highest mean abs-corr to its cluster mates (the medoid). Singletons return their only member."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    corr = precomputed_corr if precomputed_corr is not None else _redundancy_matrix(X, method)
    n_clusters = int(cluster_id.max()) + 1

    medoids = []
    for c in range(n_clusters):
        members = np.where(cluster_id == c)[0]
        if len(members) == 1:
            medoids.append(int(members[0]))
            continue
        sub = corr[np.ix_(members, members)]
        # mean abs-corr to siblings (exclude self via zero diagonal).
        np.fill_diagonal(sub, 0.0)
        scores = sub.mean(axis=1)
        medoids.append(int(members[np.argmax(scores)]))
    return medoids


# TransformerMixin (not SelectorMixin): the wrapped mRMR estimator's transform can add engineered features,
# so it is not a pure mask-based selector and SelectorMixin's mask-only contract would be wrong here.
class GroupAwareMRMR(BaseEstimator, TransformerMixin):
    """Wraps an mRMR-family estimator with correlation pre-clustering.

    .fit fits the inner estimator on cluster medoids; .transform / .support_ expand to all cluster members of any selected medoid.

    Attributes after fit:
    * ``cluster_assignments_`` -- per-original-feature cluster id.
    * ``cluster_medoid_indices_`` -- per-cluster representative index (in original-feature space).
    * ``selected_clusters_`` -- ids of clusters whose medoid mRMR kept.
    * ``support_`` -- expanded set of all original-feature indices belonging to any selected cluster.
    """
    def __init__(
        self,
        estimator,
        corr_threshold: float = 0.9,
        corr_method: str = "spearman",
        expand: bool = True,
        min_reduction: float = 0.05,
    ):
        self.estimator = estimator
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.expand = expand
        # 2026-06-03 (audit integration-defaults-3): SAFETY/USEFULNESS guard for
        # default-ON use. The cluster-medoid reduction is a measured win
        # (~1.4-1.9x, no OOS loss) ONLY when genuine correlated redundancy
        # exists; on near-uncorrelated data it would reduce nothing yet still run
        # the wrapper on a same-size medoid set. When the clustering eliminates
        # fewer than ``min_reduction`` of the features, we BYPASS the medoid path
        # and fit the inner selector on the full X -> identical selection to a
        # bare wrapper, no wasted wrapper work. Broad validation
        # (bench_cross_selector_diverse) showed AUC delta in [-0.0004, +0.0081]
        # across synthetic + real datasets -> safe to default ON.
        self.min_reduction = min_reduction

    @staticmethod
    def _inner_support_indices(inner, columns):
        """Integer column indices the inner selector kept, normalising across
        selector conventions so GroupAwareMRMR wraps any of them:
          * ``support_`` -- boolean mask OR index array (sklearn RFECV, mRMR);
          * ``get_support()`` -- sklearn SelectorMixin;
          * ``accepted`` -- list of kept column NAMES (BorutaShap).
        ``columns`` is the ordered column names the inner was fit on (X or the
        medoid subset), used to map ``accepted`` names back to positions.
        """
        sup = getattr(inner, "support_", None)
        if sup is not None:
            sup = np.asarray(sup)
            return np.where(sup)[0] if sup.dtype == bool else sup.astype(np.int64)
        if hasattr(inner, "get_support"):
            try:
                sup = np.asarray(inner.get_support())
                return np.where(sup)[0] if sup.dtype == bool else sup.astype(np.int64)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in group_aware.py:313: %s", e)
                pass
        accepted = getattr(inner, "accepted", None)  # BorutaShap: kept col names
        if accepted is not None:
            pos = {str(c): i for i, c in enumerate(columns)}
            return np.array([pos[str(c)] for c in accepted if str(c) in pos], dtype=np.int64)
        raise AttributeError(
            f"{type(inner).__name__} exposes no support_/get_support()/accepted; " f"GroupAwareMRMR cannot map its selection back to clusters."
        )

    @staticmethod
    def _prune_rank_deficient(X, support_idx):
        """Drop columns of ``support_idx`` that are EXACT linear combinations of others already kept, so the wrapper never emits a singular design matrix.

        The pairwise near-dup / correlation cluster guards catch scaled/shifted replicas and ~0.95 collinear clusters, but NOT an exact multi-feature
        identity like ``x3 = 2*x1 - x2`` (x3's pairwise corr to either parent is moderate, so it survives every pairwise test yet makes the selected Gram
        SINGULAR). We QR-pivot the standardised selected block and keep only the columns that raise the numerical rank; later columns that add no rank are
        redundant linear combinations of earlier-kept ones and are dropped. Column order is preserved so the first member of any dependency survives.
        """
        idx = np.asarray(sorted(int(i) for i in support_idx), dtype=np.int64)
        if idx.size < 2:
            return idx
        # A categorical/string/object column (e.g. raw cat features still un-encoded at this wrapper stage,
        # or a weird-content sentinel like the literal string "null") can't be "a linear combination" of
        # other columns, so the rank test doesn't apply to it -- restrict to the numeric-dtype subset and
        # always keep the rest untouched, mirroring the existing non-finite early-exit below.
        numeric_mask = np.asarray([pd.api.types.is_numeric_dtype(t) for t in X.dtypes.iloc[idx]])
        if not numeric_mask.all():
            non_numeric_idx = idx[~numeric_mask]
            numeric_idx = idx[numeric_mask]
            kept_numeric = GroupAwareMRMR._prune_rank_deficient(X, numeric_idx)
            return np.sort(np.concatenate([kept_numeric, non_numeric_idx]))
        block = X.iloc[:, idx].to_numpy(dtype=float)
        finite = np.all(np.isfinite(block), axis=0)
        if not finite.all():
            return idx  # NaN/inf columns -- leave selection untouched (rank test undefined).
        block = block - block.mean(axis=0)
        sd = block.std(axis=0)
        nz = sd > 1e-12
        block[:, nz] = block[:, nz] / sd[nz]
        # Greedy rank-revealing pass in original column order: a column is kept only if it lifts the numerical rank of the kept set.
        kept_cols: list[int] = []
        kept_mat = None
        tol = max(block.shape) * np.finfo(float).eps
        for j in range(idx.size):
            cand = block[:, j]
            if not nz[j]:
                continue  # constant column carries no signal; drop from a multicollinearity prune.
            trial = cand[:, None] if kept_mat is None else np.column_stack([kept_mat, cand])
            if np.linalg.matrix_rank(trial, tol=tol) > len(kept_cols):
                kept_cols.append(j)
                kept_mat = trial
        if len(kept_cols) == idx.size:
            return idx
        return idx[np.asarray(kept_cols, dtype=np.int64)]

    @rng_hygienic_fit
    def fit(self, X, y, **fit_params):
        """Cluster near-duplicate features, run the wrapped selector on cluster medoids (or the full set if the reduction is too small), and expand support back to the original columns.

        Skips the medoid bypass entirely when clustering reduces the feature count by less than ``min_reduction``,
        so the wrapper degrades to a plain pass-through of the inner selector rather than wasting a clustering pass.
        """
        # **fit_params (e.g. ``groups`` for a GroupKFold cv, ``sample_weight``)
        # are row-aligned, so they pass straight through to the inner selector
        # whether it fits on the medoid subset or the full X (same rows).
        # A polars frame carries its REAL column names in ``X.columns`` (a list); bridge it to a pandas frame preserving those names so the medoid /
        # corr pass (positional ``.iloc``) runs unchanged AND ``feature_names_in_`` records the true names -- not the ``f_{i}`` placeholders the
        # bare-ndarray branch below synthesizes. Without this a polars fit silently divergeed its selected NAMES from the pandas fit (same positions,
        # wrong names).
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X = pd.DataFrame(X.to_numpy(), columns=list(X.columns))
        except ImportError:
            pass
        is_df = isinstance(X, pd.DataFrame)
        # GroupAwareMRMR itself tolerates duplicate column names (FE-expansion lag/one-hot collisions -- positional ``.iloc`` throughout the corr / medoid pass), so it does NOT blanket-reject. But when the inner selector itself rejects duplicate names (RFECV's _fit_init guard), the wrapper must surface that rejection at its OWN fit entry: the inner only sees the cluster-MEDOID subset (deduped), so without this propagation a duplicate-named X would silently slip past the inner's guard. Mirrors the inner contract for the wrapped-RFECV path while leaving the graceful MRMR-inner path untouched.
        if is_df and X.columns.has_duplicates and getattr(self.estimator, "rejects_duplicate_feature_names", False):
            dup_names = X.columns[X.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"GroupAwareMRMR.fit: the wrapped {type(self.estimator).__name__} rejects duplicate column names: {dup_names[:10]}. "
                f"De-duplicate (e.g. ``X.loc[:, ~X.columns.duplicated()]`` or rename) before fitting."
            )
        if not is_df:
            X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        # Build the p x p redundancy matrix ONCE and reuse it for both the clustering pass and the medoid pick --
        # they are both functions of (X, corr_method) and previously each rebuilt it (a redundant O(p^2) SU/corr pass).
        _corr = _redundancy_matrix(X, self.corr_method)
        self.cluster_assignments_ = cluster_features_by_correlation(
            X, threshold=self.corr_threshold, method=self.corr_method, precomputed_corr=_corr,
        )
        self.cluster_medoid_indices_ = _cluster_medoids(
            X, self.cluster_assignments_, method=self.corr_method, precomputed_corr=_corr,
        )
        n_clusters = len(self.cluster_medoid_indices_)
        n_feat = X.shape[1]
        reduction = (n_feat - n_clusters) / max(n_feat, 1)
        self.reduction_ = float(reduction)
        self.reduced_ = bool(reduction >= float(self.min_reduction))
        logger.info(
            "GroupAwareMRMR: %d original features -> %d cluster medoids "
            "(corr_threshold=%.2f, reduction=%.1f%%, applied=%s).",
            n_feat, n_clusters, self.corr_threshold, 100.0 * reduction,
            self.reduced_,
        )

        # Guard: when the reduction is below the threshold, run the inner
        # selector on the FULL feature set (no medoid bypass) so the result is
        # identical to a bare wrapper and no wrapper work is wasted clustering.
        if not self.reduced_:
            inner = clone(self.estimator)
            inner.fit(X, y, **fit_params)
            self.estimator_ = inner
            self.support_ = self._prune_rank_deficient(X, self._inner_support_indices(inner, list(X.columns)))
            self.selected_clusters_ = sorted(set(int(self.cluster_assignments_[i]) for i in self.support_))
            self.n_features_ = len(self.support_)
            self.n_features_in_ = n_feat
            if is_df:
                self.feature_names_in_ = list(X.columns)
            return self

        # Fit inner estimator on the medoid subset only.
        X_medoids = X.iloc[:, self.cluster_medoid_indices_]
        inner = clone(self.estimator)
        inner.fit(X_medoids, y, **fit_params)
        self.estimator_ = inner

        # Map the inner selector's kept medoids back to clusters via the
        # convention-agnostic helper (support_ index/mask, get_support(), or
        # BorutaShap's ``accepted`` names). 2026-06-03 (audit integration-
        # defaults-3): lets GroupAwareMRMR wrap ANY wrapper selector
        # (RFECV / BorutaShap) -- a MEASURED ~1.4-3x wall-clock win with no OOS
        # loss -- by running the wrapper on the cluster medoids instead of every
        # redundant column.
        sel_idx = self._inner_support_indices(inner, list(X_medoids.columns))
        medoid_cluster_ids = self.cluster_assignments_[self.cluster_medoid_indices_]
        self.selected_clusters_ = sorted(set(int(medoid_cluster_ids[int(i)]) for i in sel_idx))

        if self.expand:
            _sup = np.array(sorted([idx for idx in range(X.shape[1]) if self.cluster_assignments_[idx] in self.selected_clusters_]), dtype=np.int64)
        else:
            # Just the medoids of selected clusters.
            _sup = np.array(sorted([self.cluster_medoid_indices_[c] for c in self.selected_clusters_]), dtype=np.int64)
        # Prune any exact linear-combination column from the final selection so the wrapper never emits a singular design matrix (e.g. x3=2*x1-x2).
        self.support_ = self._prune_rank_deficient(X, _sup)

        self.n_features_ = len(self.support_)
        self.n_features_in_ = X.shape[1]
        if is_df:
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X, y=None):
        """Subset ``X`` to the columns retained in ``support_`` (expanded cluster members or medoids-only, per ``expand``)."""
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]

    def get_support(self, indices: bool = False):
        """sklearn SelectorMixin contract over the EXPANDED wrapper selection.

        Defined on the wrapper itself so it does NOT fall through __getattr__ to the inner selector, whose support_/get_support() are sized to the
        cluster-MEDOID subset (n_clusters), not the original feature space (n_features_in_). Returning the inner medoid mask would misalign with
        feature_names_in_ and silently mislabel kept features (e.g. expose cluster members the wrapper never selected). support_ is the expanded
        integer index array into feature_names_in_; convert to the requested form.
        """
        sup = np.asarray(self.support_)
        if indices:
            return sup.astype(np.int64)
        mask = np.zeros(int(self.n_features_in_), dtype=bool)
        if sup.size:
            mask[sup.astype(np.int64)] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """Selected feature names (sklearn transformer contract) so this is a
        faithful drop-in inside the training-suite pre-pipeline. ``support_`` is
        an integer index array into ``feature_names_in_``."""
        names = getattr(self, "feature_names_in_", None)
        # sklearn ``_check_feature_names_in`` contract: a passed input_features MUST match n_features_in_ (column-drift detection) and, when correct-length, OVERRIDES the stored feature_names_in_ -- so a caller can re-inject real names after an ndarray fit (which synthesized f_0..f_N placeholders).
        if input_features is not None:
            input_features = list(input_features)
            n_in = int(getattr(self, "n_features_in_", len(input_features)))
            if len(input_features) != n_in:
                raise ValueError(
                    f"input_features has {len(input_features)} elements, expected {n_in} "
                    f"(n_features_in_); names passed to get_feature_names_out must match the "
                    f"feature set this selector was fit on (sklearn column-drift contract)."
                )
            return np.asarray([input_features[int(i)] for i in self.support_], dtype=object)
        if names is not None:
            return np.asarray([names[int(i)] for i in self.support_], dtype=object)
        return np.asarray([f"f_{int(i)}" for i in self.support_], dtype=object)

    @property
    def accepted(self):
        """BorutaShap report contract: the training suite reads ``accepted``
        (kept column NAMES) for BorutaShap diagnostics. Return the EXPANDED
        selection (cluster members of accepted medoids) -- i.e. the same set
        ``support_`` / ``transform`` / ``get_feature_names_out`` expose -- so the
        report agrees with what actually feeds training, NOT the inner's
        medoid-only accepted list. Falls through (AttributeError -> __getattr__ ->
        inner) when the wrapper isn't fitted or the inner has no ``accepted``.
        """
        inner = self.__dict__.get("estimator_")
        if inner is not None and hasattr(inner, "accepted") and "support_" in self.__dict__:
            return list(self.get_feature_names_out())
        raise AttributeError("accepted")

    def __getattr__(self, name):
        """Transparently expose the fitted inner selector's attributes (e.g. the
        training-suite's ``_selector_kind`` / ``_mlframe_use_sample_weights_in_fs_``
        markers, RFECV's ``cv_results_`` etc.) so this wrapper is a faithful
        drop-in. Only consulted for attributes GroupAwareMRMR does NOT define
        itself (support_, transform, get_feature_names_out, etc. stay the
        wrapper's expanded versions). Guarded against pre-fit recursion.
        """
        if name in ("estimator_", "estimator", "__setstate__", "__getstate__"):
            raise AttributeError(name)
        inner = self.__dict__.get("estimator_")
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(f"{type(self).__name__!r} object (and its inner estimator) has no " f"attribute {name!r}")
