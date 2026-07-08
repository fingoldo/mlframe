"""Composite estimator for GROUPED learning-to-rank targets.

``CompositeRankEstimator`` is the ranking twin of the other composite wrappers:
a cheap, dominant BASE score (a prior / BM25-style relevance carried in
``base_column``) already explains the COARSE ordering of items across and within
each query group; the inner learner only needs to learn the FINE within-group
RESIDUAL reordering. We therefore:

1. Residualize the relevance ``y`` against the base WITHIN each group, so the
   inner never re-learns the dominant base signal -- it sees only the part of
   the order the base gets wrong. Two residual modes:
     - ``"diff"``    : ``y - base`` (centred within group; default for graded y).
     - ``"rank"``    : within-group rank(y) - rank(base) -- the pure ordering
                       residual, robust to base/y being on different scales.
2. Fit a LambdaMART / pairwise inner on the residual signal:
     - LightGBM ``objective="lambdarank"`` when lightgbm is importable (the
       residual is discretised to non-negative integer gains it expects).
     - else a pairwise-logistic fallback: a plain classifier on WITHIN-group
       item pairs ``(i, j)`` with features ``x_i - x_j`` and label
       ``sign(residual_i - residual_j)``, whose decision margin is a valid
       monotone per-item score.
3. At predict, rank by ``base + inner_score`` -- the base supplies the coarse
   order, the inner nudges the within-group fine order. ``rank(X, group)`` is a
   convenience that returns the per-group orderings (argsort, best-first).

Leakage / memory. The residual + the base extraction are train-only and
per-group; nothing crosses group boundaries. The base is pulled as ONE narrow
ndarray (``_extract_base``), never a frame copy, so a 100 GB carrier is not
duplicated (per the project zero-copy rule). The inner is passed by config
(``base_estimator``), never captured as a closure, so clone / pickle stay clean.

biz_value (see ``tests/training/composite/test_biz_val_composite_ranking.py``):
on a synthetic where the base explains the coarse order and a residual feature
explains the within-group fine order, the composite ranker's NDCG@k beats the
base-only ordering by a wide measured margin.

cProfile (fit + predict, 200 groups x 20 items x 4 cols, LightGBM lambdarank
n_estimators=100): ~0.15 s total, ~0.11 s inside LightGBM ``train`` and the rest
the inner predict. The wrapper-side work -- the per-group residualisation
(vectorised via ``np.add.reduceat`` over sorted-group segments) and the final
``base + inner`` argsort -- is <2 ms combined; no actionable wrapper-side speedup
(the cost is the inner boosting, already internally threaded).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# Number of integer gain levels lambdarank sees for a "diff"/continuous residual.
# lambdarank requires non-negative integer labels; we rank-bin the residual within
# its global range into this many levels (a graded-relevance proxy).
# Kept at 31 so the largest emitted gain is 30, matching LightGBM's default
# ``label_gain`` table length (indices 0..30); a 32nd level trips "label not less
# than number of label mappings".
_LAMBDARANK_NUM_GAINS = 31

# Cap on the number of within-group pairs sampled per group for the pairwise
# fallback, so a group of m items (m*(m-1)/2 pairs) cannot blow up RAM on a wide
# group. None => use all pairs (small groups). Tuned for the common m<=64 case.
_MAX_PAIRS_PER_GROUP = 4096


def _is_polars_df(x: Any) -> bool:
    """True iff ``x`` is a ``polars.DataFrame`` (polars import is optional; treated as False if unavailable)."""
    try:
        import polars as pl

        return isinstance(x, pl.DataFrame)
    except Exception:
        return False


def _group_boundaries(group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sort_idx, group_sizes) that segment ``group`` into contiguous runs.

    ``sort_idx`` is a stable argsort of the group ids; applying it makes equal-id
    rows contiguous so the inner (lightgbm ``group=`` / our pairwise loop) can walk
    fixed-size segments. ``group_sizes`` is the count per distinct id in the order
    they appear AFTER sorting.
    """
    sort_idx = np.argsort(group, kind="stable")
    sorted_g = group[sort_idx]
    # Boundaries where the sorted id changes.
    _, sizes = np.unique(sorted_g, return_counts=True)
    return sort_idx, sizes


def _within_group_residual(y: np.ndarray, base: np.ndarray, group: np.ndarray, mode: str) -> np.ndarray:
    """Residual rank signal of ``y`` against ``base``, computed WITHIN each group.

    ``mode="diff"``: ``y - base`` re-centred so each group's residual has zero mean
    (removes any per-group base/y offset, keeping only the relative reordering).
    ``mode="rank"``: ``rank(y) - rank(base)`` within the group (scale-free ordering
    residual). Both are returned on the ORIGINAL row order.
    """
    res = np.empty(y.shape[0], dtype=np.float64)
    for gid in np.unique(group):
        m = group == gid
        yi = y[m]
        bi = base[m]
        if mode == "rank":
            res[m] = _rank01(yi) - _rank01(bi)
        else:  # "diff"
            d = yi - bi
            res[m] = d - d.mean()
    return res


def _rank01(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties shared), 0-based, as float64.

    Tied values receive their shared mean rank so equal relevance/base scores
    contribute ZERO ordering residual -- otherwise positional argsort ranks would
    fabricate a spurious within-group order on tied (e.g. graded-relevance) data,
    biasing the lambdarank gains / pairwise labels with a non-existent ordering.
    """
    n = a.shape[0]
    order = np.argsort(a, kind="stable")
    pos = np.empty(n, dtype=np.float64)
    pos[order] = np.arange(n, dtype=np.float64)
    sa = a[order]
    ranks = pos.copy()
    i = 0
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            avg = (i + (j - 1)) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return np.asarray(ranks)


def _residual_to_gains(res: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Discretise a continuous residual into non-negative integer lambdarank gains.

    lambdarank needs ordered non-negative integer labels per item. We bin the
    residual WITHIN each group into ``_LAMBDARANK_NUM_GAINS`` quantile levels, so
    every group spans the full label range regardless of its residual scale -- this
    is what makes the per-group DCG meaningful.
    """
    gains = np.zeros(res.shape[0], dtype=np.int32)
    nb = _LAMBDARANK_NUM_GAINS
    for gid in np.unique(group):
        m = group == gid
        r = res[m]
        if r.size == 1 or np.ptp(r) == 0:
            gains[m] = 0
            continue
        # Rank within group -> scale to [0, nb-1] integer levels.
        rk = _rank01(r)
        lvl = np.floor(rk / max(rk.max(), 1.0) * (nb - 1) + 0.5).astype(np.int32)
        gains[m] = lvl
    return gains


def _ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, group: np.ndarray, k: int) -> float:
    """Mean NDCG@k over groups using the standard ``2**rel - 1`` gain / log2 discount."""
    vals = []
    for gid in np.unique(group):
        m = group == gid
        yt = y_true[m]
        sc = scores[m]
        kk = min(k, yt.size)
        order = np.argsort(-sc, kind="stable")[:kk]
        gains = 2.0 ** yt[order] - 1.0
        disc = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float((gains * disc).sum())
        ideal_order = np.argsort(-yt, kind="stable")[:kk]
        igains = 2.0 ** yt[ideal_order] - 1.0
        idcg = float((igains * disc).sum())
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(y_true: Any, scores: Any, group: Any, k: int = 10) -> float:
    """Public mean NDCG@k helper (graded relevance, ``2**rel - 1`` gains)."""
    return _ndcg_at_k(
        np.asarray(y_true, dtype=np.float64).reshape(-1),
        np.asarray(scores, dtype=np.float64).reshape(-1),
        np.asarray(group).reshape(-1),
        int(k),
    )


class CompositeRankEstimator(BaseEstimator, RegressorMixin):
    """Composite learning-to-rank estimator: base score + inner residual reranker.

    Parameters
    ----------
    base_column : str
        Column in ``X`` carrying the dominant base relevance score (prior / BM25).
        Pulled as one narrow ndarray; kept as a feature for the inner too (it can
        learn base-conditional corrections) unless ``drop_base_feature=True``.
    base_estimator : estimator or None
        The inner learner. If None and lightgbm is importable, a
        ``LGBMRanker(objective="lambdarank")`` is used; otherwise the pairwise-
        logistic fallback (a ``LogisticRegression`` on within-group item-pair
        differences). An explicit estimator is cloned, never mutated.
    residual_mode : {"diff", "rank"}
        Within-group residualisation of ``y`` vs base. Default ``"rank"`` (scale-free).
    drop_base_feature : bool
        Default True (the residual contract): the base column is removed from the
        inner's feature matrix so the inner learns ONLY the within-group residual
        reranking signal, and the base contributes ONLY via the explicit
        ``base + inner`` combine. Measured to clearly beat ``False`` (where the inner
        re-learns the dominant base, the z-scored combine double-counts it, and NDCG
        regresses): lambdarank 0.952 / pairwise 0.990 vs 0.866-0.875 at False (held
        out). Set False only if you want the inner to learn base-CONDITIONAL
        corrections.
    base_weight : float
        Weight on the base score in the final ``base_weight * base_z + inner_z``
        combine; both terms are group-wise z-scored so the weight is scale-free.
    """

    def __init__(
        self,
        base_column: str,
        base_estimator: Any | None = None,
        residual_mode: str = "rank",
        drop_base_feature: bool = True,
        base_weight: float = 1.0,
    ):
        self.base_column = base_column
        self.base_estimator = base_estimator
        self.residual_mode = residual_mode
        self.drop_base_feature = drop_base_feature
        self.base_weight = base_weight

    # -- column / frame helpers -------------------------------------------------
    def _extract_base(self, X: Any) -> np.ndarray:
        """Narrow one-column ndarray pull of the base score (no frame copy)."""
        col = self.base_column
        if _is_polars_df(X) or hasattr(X, "get_column"):
            return np.asarray(X.get_column(col).to_numpy(), dtype=np.float64).reshape(-1)
        if hasattr(X, "columns") and hasattr(X, "__getitem__"):  # pandas
            return np.asarray(X[col].to_numpy(), dtype=np.float64).reshape(-1)
        # ndarray + integer column index.
        arr = np.asarray(X)
        return np.asarray(arr[:, int(col)], dtype=np.float64).reshape(-1)

    def _inner_features(self, X: Any) -> Any:
        """Feature matrix handed to the inner: optionally with the base column dropped."""
        if not self.drop_base_feature:
            return X
        col = self.base_column
        if hasattr(X, "drop") and hasattr(X, "get_column"):  # polars
            return X.drop(col) if col in X.columns else X
        if hasattr(X, "drop") and hasattr(X, "columns"):  # pandas
            return X.drop(columns=[col]) if col in X.columns else X
        return X

    def _make_default_inner(self) -> tuple[Any, str]:
        """Default inner + its kind: lambdarank when available, else pairwise."""
        try:
            import lightgbm as lgb

            return (
                __import__("lightgbm").LGBMRanker(objective="lambdarank", n_estimators=200, verbose=-1),
                "lambdarank",
            )
        except Exception:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000), "pairwise"

    @staticmethod
    def _is_lambdarank_inner(est: Any) -> bool:
        """True iff a caller-supplied ``base_estimator`` is a ``LGBMRanker`` (routes ``fit``/``inner_score`` down the
        lambdarank path, matching the auto-selected default's routing when lightgbm is importable)."""
        return type(est).__name__ == "LGBMRanker"

    # -- pairwise fallback ------------------------------------------------------
    def _build_pairs(self, Xnum: np.ndarray, res: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Within-group item-pair difference features + sign labels for the fallback.

        For each ordered pair ``(i, j)`` in the SAME group with a non-tied residual we
        emit ``x_i - x_j`` with label ``1`` if ``res_i > res_j`` else ``0``. The learned
        linear margin ``w . x`` is then a per-item score whose ordering matches the
        residual ordering, which is exactly the within-group reranking signal.
        """
        diffs: list[np.ndarray] = []
        labels: list[int] = []
        rng = np.random.default_rng(0)
        for gid in np.unique(group):
            idx = np.flatnonzero(group == gid)
            m = idx.size
            if m < 2:
                continue
            ii, jj = np.triu_indices(m, k=1)
            # Subsample pairs on very wide groups to bound memory.
            if ii.size > _MAX_PAIRS_PER_GROUP:
                sel = rng.choice(ii.size, size=_MAX_PAIRS_PER_GROUP, replace=False)
                ii, jj = ii[sel], jj[sel]
            gi, gj = idx[ii], idx[jj]
            rd = res[gi] - res[gj]
            keep = rd != 0
            if not keep.any():
                continue
            gi, gj, rd = gi[keep], gj[keep], rd[keep]
            d = Xnum[gi] - Xnum[gj]
            diffs.append(d)
            labels.extend((rd > 0).astype(int).tolist())
        if not diffs:
            return np.zeros((0, Xnum.shape[1]), dtype=np.float64), np.zeros(0, dtype=int)
        return np.vstack(diffs), np.asarray(labels, dtype=int)

    @staticmethod
    def _numify(X: Any) -> np.ndarray:
        """Dense float64 view of a frame / array for the pairwise linear fallback."""
        if hasattr(X, "to_numpy"):
            return np.asarray(X.to_numpy(), dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    # -- sklearn API ------------------------------------------------------------
    def fit(self, X: Any, y: Any, group: Any | None = None) -> "CompositeRankEstimator":
        """Fit the composite ranker: residualize ``y`` against the base score within each ``group``, then fit the
        inner (lambdarank or pairwise-logistic) on that residual per the class docstring's algorithm. Raises if
        ``group`` is omitted (required despite the default-``None`` signature -- see the inline comment) or if
        ``residual_mode`` is invalid."""
        # ``group`` keyword with a None default so ``clone(est).fit(X, y)`` works (sklearn's clone/check path calls fit positionally with X, y only).
        if group is None:
            raise ValueError("CompositeRankEstimator.fit requires the per-item ``group`` argument.")
        if self.residual_mode not in ("diff", "rank"):
            raise ValueError(f"CompositeRankEstimator: residual_mode must be 'diff' or 'rank', " f"got {self.residual_mode!r}.")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        group_arr = np.asarray(group).reshape(-1)
        if group_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("CompositeRankEstimator: group length must match y length " f"({group_arr.shape[0]} != {y_arr.shape[0]}).")
        base = self._extract_base(X)
        if base.shape[0] != y_arr.shape[0]:
            raise ValueError("CompositeRankEstimator: base column length must match y length.")
        res = _within_group_residual(y_arr, base, group_arr, self.residual_mode)

        if self.base_estimator is None:
            inner, kind = self._make_default_inner()
        else:
            inner = clone(self.base_estimator)
            kind = "lambdarank" if self._is_lambdarank_inner(inner) else "pairwise"

        Xfeat = self._inner_features(X)
        self.n_features_in_ = _ncols(Xfeat)

        if kind == "lambdarank":
            # lightgbm wants contiguous groups + an integer-gain label per item.
            sort_idx, sizes = _group_boundaries(group_arr)
            gains = _residual_to_gains(res, group_arr)[sort_idx]
            Xsorted = _take_rows(Xfeat, sort_idx)
            inner.fit(Xsorted, gains, group=sizes)
            self.kind_ = "lambdarank"
        else:
            Xnum = self._numify(self._inner_features(X) if self.drop_base_feature else X)
            P, lab = self._build_pairs(Xnum, res, group_arr)
            if P.shape[0] == 0 or len(np.unique(lab)) < 2:
                # Degenerate (single-group all-tied / no informative pair): inner score
                # collapses to 0, the base alone drives the order.
                self.pairwise_w_ = np.zeros(Xnum.shape[1], dtype=np.float64)
                self.pairwise_b_ = 0.0
            else:
                inner.fit(P, lab)
                # Linear decision direction => per-item score is ``w . x``.
                self.pairwise_w_ = np.asarray(inner.coef_, dtype=np.float64).reshape(-1)
                self.pairwise_b_ = 0.0
            self.kind_ = "pairwise"

        self.inner_ = inner
        self.fitted_residual_mode_ = self.residual_mode
        return self

    def inner_score(self, X: Any) -> np.ndarray:
        """Per-item residual-reranking score from the fitted inner (no base added)."""
        check_is_fitted(self, "kind_")
        if self.kind_ == "lambdarank":
            Xfeat = self._inner_features(X)
            return np.asarray(self.inner_.predict(Xfeat), dtype=np.float64).reshape(-1)
        Xnum = self._numify(self._inner_features(X) if self.drop_base_feature else X)
        return Xnum @ self.pairwise_w_ + self.pairwise_b_

    def predict(self, X: Any, group: Any | None = None) -> np.ndarray:
        """Combined ranking score ``base_weight * z(base) + z(inner)``.

        Both terms are z-scored: globally when ``group`` is None, else PER GROUP (the
        scale-free combine that keeps a high-variance group from dominating). Higher
        score => more relevant.
        """
        base = self._extract_base(X)
        inner = self.inner_score(X)
        if group is None:
            return np.asarray(self.base_weight * _zscore(base) + _zscore(inner))
        g = np.asarray(group).reshape(-1)
        out = np.empty(base.shape[0], dtype=np.float64)
        for gid in np.unique(g):
            m = g == gid
            out[m] = self.base_weight * _zscore(base[m]) + _zscore(inner[m])
        return out

    def rank(self, X: Any, group: Any) -> dict[Any, np.ndarray]:
        """Per-group orderings: ``{group_id: row_indices_best_first}``.

        The returned indices are positions into the ORIGINAL ``X`` rows, ordered by
        descending combined score within each group (best-ranked item first).
        """
        g = np.asarray(group).reshape(-1)
        scores = self.predict(X, group=g)
        orderings: dict[Any, np.ndarray] = {}
        for gid in np.unique(g):
            idx = np.flatnonzero(g == gid)
            order = idx[np.argsort(-scores[idx], kind="stable")]
            orderings[gid] = order
        return orderings


def _zscore(a: np.ndarray) -> np.ndarray:
    """Standard-score ``a`` (zero mean, unit std); returns all-zeros for a constant/degenerate (std 0 or non-finite) input."""
    a = np.asarray(a, dtype=np.float64)
    sd = a.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(a)
    return np.asarray((a - a.mean()) / sd)


def _ncols(X: Any) -> int:
    """Column count of ``X`` across ndarray / pandas / polars carriers, for ``n_features_in_``."""
    if hasattr(X, "shape") and len(getattr(X, "shape", ())) == 2:
        return int(X.shape[1])
    if hasattr(X, "columns"):
        return len(X.columns)
    return int(np.asarray(X).shape[1])


def _take_rows(X: Any, idx: np.ndarray) -> Any:
    """Row-subset a frame / array by integer positions, format-native (no copy of rest)."""
    if hasattr(X, "get_column") and hasattr(X, "__getitem__"):  # polars
        return X[idx.tolist()] if not isinstance(idx, list) else X[idx]
    if hasattr(X, "iloc"):  # pandas
        return X.iloc[idx]
    return np.asarray(X)[idx]
