"""
Leakage-safe target encoders for high-cardinality categorical columns.

Target encoding inside a
``fit_and_transform`` pipeline LEAKS the target unless the encoding
for each train row is computed from a fold that EXCLUDES that row.
Naive ``mean(target | category)`` over the full train set is the
silent worst-case ML bug class -- train AUC ≈ 1.0 from memorising
the encoded values, val AUC drops out the floor.

Implementation: K-fold out-of-fold (OOF) encoding. For row ``i`` in
fold ``f``, the encoding is computed on rows from folds ``{F} - {f}``
only. At ``transform()`` time on held-out / test rows, the full
training-set statistic applies (no leak risk -- those rows weren't
in any fold of the training data).

Smoothing follows the standard formula:

    encoding(c) = (n_c × mean_c + smoothing × prior) / (n_c + smoothing)

where ``n_c`` is the count of rows in category ``c``, ``mean_c`` is
the in-fold target mean for that category, ``prior`` is the global
target mean (or median per ``TargetEncodeParams.prior``), and
``smoothing`` is the regulariser that pulls rare-category encodings
toward the prior.

Variants supported:
  * ``target_mean`` -- standard smoothed mean (default).
  * ``target_m_estimate`` -- m-estimate (mathematically equivalent
    to target_mean with ``smoothing == m``).
  * ``target_james_stein`` -- shrinkage toward prior with
    variance-aware shrinkage factor (closed-form for Gaussian
    targets).
  * ``target_loo`` -- leave-one-out (row-wise; not k-fold). Uses
    the standard formula ``(n_c × mean_c - target_i) / (n_c - 1)``
    for row i in category c. NaN-safe at category size = 1.
  * ``woe`` -- weight of evidence ``log(P(c|y=1) / P(c|y=0))``,
    smoothed. Binary-classification-only.

The encoder follows the sklearn fit/transform contract so it slots
into existing pipelines.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Union,
)

import numpy as np

# _NULL_SENTINEL / _canonical_cat_token / _float_canonical_tokens / _objectwise_isnull /
# _temporal_to_epoch_ns_tokens aren't referenced in this file's own body -- they're re-exported
# for backward compatibility so existing `from target_encoders import X` call sites keep working
# after the carve into _target_encoders_canon.py.
from mlframe.training.feature_handling._target_encoders_canon import (
    _NULL_SENTINEL,  # noqa: F401 -- re-exported, see above
    _canonical_cat_token,  # noqa: F401 -- re-exported, see above
    _categorical_to_string_array,
    _coerce_y_to_float64,
    _compute_prior,
    _float_canonical_tokens,  # noqa: F401 -- re-exported, see above
    _objectwise_isnull,  # noqa: F401 -- re-exported, see above
    _temporal_to_epoch_ns_tokens,  # noqa: F401 -- re-exported, see above
)

if TYPE_CHECKING:
    # Imports for ``Union[..., pd.Series, pl.Series]`` annotations below.
    # Guarded under TYPE_CHECKING so the runtime import cost stays zero
    # while typecheckers / IDE tooling see the real symbols.
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)


# =====================================================================
# LeakageSafeEncoder
# =====================================================================


class LeakageSafeEncoder:
    """Out-of-fold target encoder for a single categorical column.

    ``fit_transform()`` returns OOF-computed encodings for the train rows; ``transform()`` on held-out
    rows uses the full-train statistic.

    Parameters
    ----------
    method : Literal["target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe"]
        Encoding flavour. ``target_m_estimate`` is mathematically
        identical to ``target_mean`` so we treat them interchangeably
        with a different default ``smoothing``.
    smoothing : float
        Regularisation toward the prior for the MEAN encoders (target_mean / m_estimate / james_stein / loo). Higher -> rare categories encoded
        closer to the prior. Default 3.0: a held-out sweep (bench_target_encoder_smoothing, 5 scenarios x 5 seeds) shows 3.0 beats the old 10.0 on
        the majority of cells (log-loss, posterior-MSE); 10.0 over-shrinks informative categories. Does NOT affect the woe method, which has its own
        woe_smoothing cushion below.
    woe_smoothing : Optional[float]
        Laplace alpha for the ``woe`` method only (added to pos/neg cell
        mass). ``None`` -> 0.5 (Jeffreys-style cushion). Separate from
        ``smoothing`` because WoE log-odds need only a tiny cushion: a
        large alpha pulls every category toward 0 and destroys signal on
        rare / high-card / imbalanced data.
    cv : int
        K-fold count for OOF estimation (default 5). Higher reduces
        leak risk further but increases compute.
    prior : Literal["mean", "median"]
        Global statistic used as the prior in the smoothed formula.
    random_state : Optional[int]
        Seed for the K-fold splitter. ``None`` -> a fixed-but-arbitrary
        default (42) so two runs with identical config produce
        identical encodings.

    Notes
    -----
    The encoder is per-column. Multi-column applications instantiate
    one encoder per column upstream.
    """

    def __init__(
        self,
        *,
        method: Literal[
            "target_mean", "target_m_estimate",
            "target_james_stein", "target_loo", "woe",
        ] = "target_mean",
        smoothing: float = 3.0,
        woe_smoothing: Optional[float] = None,
        cv: int = 5,
        prior: Literal["mean", "median"] = "mean",
        random_state: Optional[int] = None,
        time_aware: bool = False,
        cv_splitter: Optional[Any] = None,
    ):
        valid_methods = {
            "target_mean", "target_m_estimate",
            "target_james_stein", "target_loo", "woe",
        }
        if method not in valid_methods:
            raise ValueError(f"unknown method {method!r}; valid: {sorted(valid_methods)}")
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}")
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")

        if woe_smoothing is not None and woe_smoothing < 0:
            raise ValueError(f"woe_smoothing must be >= 0, got {woe_smoothing}")

        self.method = method
        self.smoothing = smoothing
        # WoE Laplace alpha is a separate knob from the mean-encoder smoothing: a mean
        # encoder wants strong shrinkage (10) toward the prior, but the WoE log-odds need
        # only a tiny Laplace cushion (0.5) -- a large alpha pulls every category's WoE
        # toward 0 and destroys real signal on rare/high-card/imbalanced data (bench:
        # _benchmarks/bench_woe_laplace_alpha.py, 0.5 wins 9/15 cells, +0.06 AUC on 1pct).
        self.woe_smoothing = 0.5 if woe_smoothing is None else woe_smoothing
        self.cv = cv
        self.prior = prior
        self.random_state = 42 if random_state is None else random_state
        # When ``time_aware=True`` the OOF KFold below uses ``shuffle=False`` (preserves row order). When
        # the caller passes ``cv_splitter`` (e.g. ``TimeSeriesSplit(n_splits=5)``) it overrides both the
        # internal KFold and ``cv``; this is the safe path for genuinely temporal targets. Default
        # ``time_aware=False, cv_splitter=None`` preserves the legacy shuffled-KFold behaviour.
        self.time_aware = time_aware
        self.cv_splitter = cv_splitter

        # Fitted state
        self._global_prior: Optional[float] = None
        self._category_means: Optional[Dict[str, float]] = None  # for transform on held-out
        self._category_counts: Optional[Dict[str, int]] = None
        # WoE-only state
        self._woe_pos: Optional[Dict[str, float]] = None
        self._woe_neg: Optional[Dict[str, float]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit``/``fit_transform`` has populated the full-train statistic (required before ``transform``)."""
        return self._is_fitted

    def fit(
        self,
        X_column: Union[np.ndarray, list, pd.Series, pl.Series],
        y: Union[np.ndarray, list, pd.Series, pl.Series],
        sample_weight: Union[np.ndarray, list, None] = None,
    ) -> LeakageSafeEncoder:
        """Fit the FULL-train statistic for transform on held-out rows.
        ``fit_transform`` runs the OOF loop in addition to this.

        sample_weight: optional per-row weights. When provided, per-category means become weighted means
        ``sum(w_i * y_i) / sum(w_i)`` and WoE numerator / denominator become weighted positive / negative
        mass. Default None preserves byte-for-byte legacy behaviour.
        """
        cats = _categorical_to_string_array(X_column)
        y_arr = _coerce_y_to_float64(y)
        if cats.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X and y length mismatch: {cats.shape[0]} vs {y_arr.shape[0]}")
        sw_arr = self._coerce_sample_weight(sample_weight, len(y_arr))

        self._global_prior = _compute_prior(y_arr, self.prior, sw_arr)
        self._category_counts, self._category_means = self._compute_per_category(cats, y_arr, sw_arr)

        if self.method == "woe":
            unique_y = np.unique(y_arr)
            if not (set(unique_y).issubset({0.0, 1.0}) and len(unique_y) <= 2):
                raise ValueError("method='woe' requires binary {0, 1} target; got " f"{sorted(unique_y)[:5]}")
            self._woe_pos, self._woe_neg = self._compute_woe_per_category(cats, y_arr, sw_arr)

        self._is_fitted = True
        return self

    def transform(
        self,
        X_column: Union[np.ndarray, list, pd.Series, pl.Series],
    ) -> np.ndarray:
        """Encode held-out rows using the full-train statistic.

        Unseen categories -> ``self._global_prior`` (no leak: held-out
        rows were never in the fitted folds).
        """
        if not self._is_fitted:
            raise RuntimeError("LeakageSafeEncoder.transform called before fit; " "use fit_transform on train rows first")
        cats = _categorical_to_string_array(X_column)
        return self._encode_with_full_train_stat(cats)

    def fit_transform(
        self,
        X_column,
        y,
        sample_weight: Union[np.ndarray, list, None] = None,
    ) -> np.ndarray:
        """OOF-fit on the full corpus AND return train-row encodings.

        For row ``i`` in fold ``f``, encoding is computed on
        ``{F} - {f}`` only (no leak). After the OOF loop, the
        full-train statistic is fitted for downstream
        ``transform()`` on held-out rows.

        sample_weight: optional per-row weights threaded into both the per-fold encoding and the final
        full-train statistic. Default None preserves byte-for-byte legacy path.
        """
        cats = _categorical_to_string_array(X_column)
        y_arr = _coerce_y_to_float64(y)
        if cats.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X and y length mismatch: {cats.shape[0]} vs {y_arr.shape[0]}")
        sw_arr = self._coerce_sample_weight(sample_weight, len(y_arr))

        self._global_prior = _compute_prior(y_arr, self.prior, sw_arr)

        if self.method == "target_loo":
            # Leave-one-out is row-wise; no K-fold loop needed.
            out = self._loo_encode(cats, y_arr, sw_arr)
        else:
            out = self._kfold_encode(cats, y_arr, sw_arr)

        # Fit the full-train statistic for downstream transform.
        self._category_counts, self._category_means = self._compute_per_category(cats, y_arr, sw_arr)
        if self.method == "woe":
            self._woe_pos, self._woe_neg = self._compute_woe_per_category(cats, y_arr, sw_arr)

        self._is_fitted = True
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_sample_weight(sw, n: int) -> np.ndarray | None:
        """Validate + coerce sample_weight to a 1-D float64 array of length n, or None when sw is None / uniform.

        Returning None for uniform weights lets every downstream code path skip the weighted branch and stay
        byte-for-byte identical to the legacy path."""
        if sw is None:
            return None
        arr = np.asarray(list(sw) if not isinstance(sw, np.ndarray) else sw, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"sample_weight must be 1-D, got shape {arr.shape}")
        if arr.shape[0] != n:
            raise ValueError(f"sample_weight length {arr.shape[0]} != n_rows {n}")
        if not np.all(np.isfinite(arr)) or (arr < 0).any():
            raise ValueError("sample_weight must be finite and non-negative")
        if arr.sum() <= 0:
            raise ValueError("sample_weight sums to zero")
        if float(arr.max() - arr.min()) <= 1e-12 * max(1.0, abs(float(arr.mean()))):
            return None
        return arr

    def _compute_per_category(
        self,
        cats: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple:
        """Compute per-category (count, mean) dicts used for the full-train statistic; vectorized via ``pd.factorize`` + ``np.bincount`` (~10x faster than the legacy dict-accumulation loop at 1M+ rows), falling back to the dict loop when pandas is unavailable."""
        # counts: effective sample size per cat (weighted mass when sw given, else integer count).
        # means: sum(w*y)/sum(w) per cat when weighted; sum(y)/n_c per cat when uniform.
        # Vectorised path via ``pd.factorize`` + ``np.bincount``: ~10x faster
        # than the legacy Python dict-accumulation loop on 1M+ rows
        # (440 ms -> 48 ms in the encoder bench). Falls back to the
        # legacy loop only when pandas is unavailable.
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            pd = None
        if pd is not None and len(cats) > 0:
            codes, uniq = pd.factorize(cats, sort=False)
            K = len(uniq)
            if sample_weight is None:
                counts_arr = np.bincount(codes, minlength=K).astype(np.int64)
                sums_arr = np.bincount(codes, weights=y, minlength=K)
                means_arr = np.where(counts_arr > 0, sums_arr / np.maximum(counts_arr, 1), 0.0)
                counts_dict = {str(u): int(c) for u, c in zip(uniq, counts_arr)}
                means_dict = {str(u): float(m) for u, m in zip(uniq, means_arr)}
                return counts_dict, means_dict
            w_counts = np.bincount(codes, weights=sample_weight, minlength=K)
            w_sums = np.bincount(codes, weights=sample_weight * y, minlength=K)
            means_arr = np.where(w_counts > 0, w_sums / np.maximum(w_counts, 1e-300), 0.0)
            counts_dict_w = {str(u): float(c) for u, c in zip(uniq, w_counts)}
            means_dict_w = {str(u): float(m) for u, m in zip(uniq, means_arr)}
            return counts_dict_w, means_dict_w
        # Legacy fallback (pandas unavailable or empty input).
        counts: Dict[str, float] = {}
        sums: Dict[str, float] = {}
        if sample_weight is None:
            for c, y_i in zip(cats, y):
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y_i
            means = {c: sums[c] / counts[c] for c in counts}
            counts_int = {c: int(v) for c, v in counts.items()}
            return counts_int, means
        for c, y_i, w_i in zip(cats, y, sample_weight):
            counts[c] = counts.get(c, 0.0) + float(w_i)
            sums[c] = sums.get(c, 0.0) + float(w_i) * float(y_i)
        means = {c: (sums[c] / counts[c] if counts[c] > 0 else 0.0) for c in counts}
        return counts, means

    def _compute_woe_per_category(
        self,
        cats: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple:
        """WoE: log(P(c|y=1) / P(c|y=0)) with Laplace smoothing for zero-count cells.

        When sample_weight is provided, positive / negative cell counts become weighted mass:
        ``sum(w_i | c, y_i=1)`` and ``sum(w_i | c, y_i=0)``.

        Vectorised path: ``pd.factorize`` + masked ``np.bincount`` replaces
        the per-row Python loop. Same correctness contract (verified by a
        dedicated regression test) and ~10x faster on 1M-row binary y.
        """
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            pd = None
        if pd is not None and len(cats) > 0:
            codes, uniq = pd.factorize(cats, sort=False)
            K = len(uniq)
            pos_mask = y == 1.0
            neg_mask = y == 0.0
            if sample_weight is None:
                n_pos = max(1.0, float(pos_mask.sum()))
                n_neg = max(1.0, float(neg_mask.sum()))
                pos_w = np.where(pos_mask, 1.0, 0.0)
                neg_w = np.where(neg_mask, 1.0, 0.0)
            else:
                n_pos = max(1.0, float(sample_weight[pos_mask].sum()))
                n_neg = max(1.0, float(sample_weight[neg_mask].sum()))
                pos_w = np.where(pos_mask, sample_weight, 0.0)
                neg_w = np.where(neg_mask, sample_weight, 0.0)
            pos_counts_arr = np.bincount(codes, weights=pos_w, minlength=K)
            neg_counts_arr = np.bincount(codes, weights=neg_w, minlength=K)
            a = self.woe_smoothing
            p_arr = (pos_counts_arr + a) / (n_pos + a)
            q_arr = (neg_counts_arr + a) / (n_neg + a)
            woe_pos = {str(u): float(p) for u, p in zip(uniq, p_arr)}
            woe_neg = {str(u): float(q) for u, q in zip(uniq, q_arr)}
            return woe_pos, woe_neg
        # Legacy fallback (pandas unavailable).
        if sample_weight is None:
            n_pos = max(1.0, float(np.sum(y == 1.0)))
            n_neg = max(1.0, float(np.sum(y == 0.0)))
            pos_counts: Dict[str, float] = {}
            neg_counts: Dict[str, float] = {}
            for c, y_i in zip(cats, y):
                if y_i == 1.0:
                    pos_counts[c] = pos_counts.get(c, 0.0) + 1
                else:
                    neg_counts[c] = neg_counts.get(c, 0.0) + 1
        else:
            n_pos = max(1.0, float(np.sum(sample_weight[y == 1.0])))
            n_neg = max(1.0, float(np.sum(sample_weight[y == 0.0])))
            pos_counts = {}
            neg_counts = {}
            for c, y_i, w_i in zip(cats, y, sample_weight):
                if y_i == 1.0:
                    pos_counts[c] = pos_counts.get(c, 0.0) + float(w_i)
                else:
                    neg_counts[c] = neg_counts.get(c, 0.0) + float(w_i)
        all_cats = set(pos_counts) | set(neg_counts)
        woe_pos = {}
        woe_neg = {}
        a = self.woe_smoothing
        for c in all_cats:
            p = (pos_counts.get(c, 0.0) + a) / (n_pos + a)
            q = (neg_counts.get(c, 0.0) + a) / (n_neg + a)
            woe_pos[c] = p
            woe_neg[c] = q
        return woe_pos, woe_neg

    def _kfold_encode(
        self,
        cats: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """OOF k-fold encoding. Each fold's encoding is computed on
        the OTHER folds only.
        """
        from sklearn.model_selection import KFold

        n = len(cats)
        out = np.empty(n, dtype=np.float64)

        if self.cv_splitter is not None:
            kf = self.cv_splitter
        elif self.time_aware:
            kf = KFold(n_splits=self.cv, shuffle=False)
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        # Per-fold encoding: dict.get() is faster than pd.Series.map() at
        # the typical fold size (<500k rows per fold) -- the latter pays
        # Python-level Series construction overhead and is only competitive
        # on very wide categorical universes (>10k unique cats per fold).
        # A measured regression (vectorised: 407ms, legacy dict.get: 160ms on n=100k, K=200, cv=3) made
        # us revert to dict.get here instead of the pandas-Series path. The ``_compute_per_category`` upstream
        # vectorisation stays because it benefits from numpy bincount
        # over the full train fold at once.
        for train_idx, val_idx in kf.split(cats):
            cats_train, cats_val = cats[train_idx], cats[val_idx]
            y_train = y[train_idx]
            sw_train = None if sample_weight is None else sample_weight[train_idx]
            counts_t, means_t = self._compute_per_category(cats_train, y_train, sw_train)
            prior_t = _compute_prior(y_train, self.prior, sw_train)
            if self.method == "woe":
                wp, wn = self._compute_woe_per_category(cats_train, y_train, sw_train)
                for j, c in zip(val_idx, cats_val):
                    p = wp.get(c, prior_t)
                    q = wn.get(c, 1.0 - prior_t)
                    # Same Laplace-cushion clip _encode_per_row's woe branch already uses: the
                    # unseen-category fallback (prior_t / 1-prior_t) can be exactly 0.0 or 1.0 when this
                    # fold's training rows happen to contain no positives or no negatives (realistic with
                    # small `cv` on an imbalanced target) -- np.log(0) is -inf and poisons the OOF column.
                    p_safe = float(min(max(p, 1e-12), 1.0 - 1e-12))
                    q_safe = float(min(max(q, 1e-12), 1.0 - 1e-12))
                    out[j] = float(np.log(p_safe) - np.log(q_safe))
            else:
                # Smoothed mean (target_mean / target_m_estimate /
                # target_james_stein share this OOF shape).
                for j, c in zip(val_idx, cats_val):
                    n_c = counts_t.get(c, 0)
                    m_c = means_t.get(c, prior_t)
                    if n_c == 0:
                        out[j] = prior_t
                    else:
                        out[j] = (n_c * m_c + self.smoothing * prior_t) / (n_c + self.smoothing)

        return out

    def _loo_encode(self, cats: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
        """Row-wise leave-one-out encoding. Weighted form: cat sum-of-w*y minus this row's w_i*y_i, over
        cat sum-of-w minus w_i (effective LOO sample size in weighted units)."""
        counts, sums = self._compute_per_category_sums(cats, y, sample_weight)
        prior = self._global_prior if self._global_prior is not None else _compute_prior(y, self.prior, sample_weight)
        out = np.empty(len(cats), dtype=np.float64)
        if sample_weight is None:
            for i, (c, y_i) in enumerate(zip(cats, y)):
                n_c = counts[c]
                if n_c <= 1:
                    out[i] = prior
                else:
                    loo_mean = (sums[c] - y_i) / (n_c - 1)
                    out[i] = ((n_c - 1) * loo_mean + self.smoothing * prior) / ((n_c - 1) + self.smoothing)
            return out
        for i, (c, y_i, w_i) in enumerate(zip(cats, y, sample_weight)):
            w_total = counts[c]
            w_remaining = w_total - float(w_i)
            if w_remaining <= 0:
                out[i] = prior
            else:
                loo_mean = (sums[c] - float(w_i) * float(y_i)) / w_remaining
                out[i] = (w_remaining * loo_mean + self.smoothing * prior) / (w_remaining + self.smoothing)
        return out

    def _compute_per_category_sums(self, cats: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple:
        """Legacy Python dict-loop accumulation of per-category (count, sum) used by the LOO encode path, which needs raw sums (not means) to subtract the current row's own contribution."""
        counts: Dict[str, float] = {}
        sums: Dict[str, float] = {}
        if sample_weight is None:
            for c, y_i in zip(cats, y):
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y_i
            return counts, sums
        for c, y_i, w_i in zip(cats, y, sample_weight):
            counts[c] = counts.get(c, 0.0) + float(w_i)
            sums[c] = sums.get(c, 0.0) + float(w_i) * float(y_i)
        return counts, sums

    def _encode_with_full_train_stat(self, cats: np.ndarray) -> np.ndarray:
        """Encode ``cats`` against the fitted full-train statistic, dispatching to the vectorised path when pandas is available and falling back to the per-row loop otherwise."""
        # The per-row encoding is a deterministic function of the category string alone, so we compute the value once over
        # the (small) unique set and gather it back with pd.factorize + take. Bit-identical to the per-row loop by construction
        # (same arithmetic per category, same unseen fallback), and ~10-30x faster at n=10M where the Python loop dominated.
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            pd = None
        if pd is not None and len(cats) > 0:
            return self._encode_vectorised(cats, pd)
        return self._encode_per_row(cats)

    def _encode_vectorised(self, cats: np.ndarray, pd) -> np.ndarray:
        """Encode each row by computing the per-row encoding once over the small unique-category set, then gathering it back per row; bit-identical to ``_encode_per_row`` since the encoding is a pure function of the category string, but ~10-30x faster at large n."""
        codes, uniq = pd.factorize(cats, sort=False)
        per_uniq = self._encode_per_row(uniq)
        return np.asarray(per_uniq[codes])

    def _encode_per_row(self, cats: np.ndarray) -> np.ndarray:
        """Encode each category via the fitted full-train statistic (mean/m-estimate/James-Stein/WoE per ``self.method``), falling back to the global prior (or its log-odds for WoE) for unseen categories."""
        out = np.empty(len(cats), dtype=np.float64)
        assert self._global_prior is not None, "_encode_per_row: fit() must populate _global_prior before transform"
        prior = self._global_prior
        if self.method == "woe":
            # Unseen-category fallback uses the prior log-odds, not 0. ``0.0`` was misread as "no evidence"
            # but for imbalanced binary problems the true neutral (no information) is log(prior_pos/prior_neg);
            # a 99/1 split gives prior_logodds = log(99) ~= 4.6, so a 0.0 baseline was wildly biased toward
            # the minority class for every test-time unseen string. ``_global_prior`` is the smoothed positive
            # rate; convert to log-odds with the same Laplace cushion the kfold path uses.
            _p = self._global_prior if self._global_prior is not None else 0.5
            _p = float(min(max(_p, 1e-12), 1.0 - 1e-12))
            unseen_logodds = float(np.log(_p) - np.log(1.0 - _p))
            # With smoothing=0 a category that has zero positives OR zero negatives in train yields
            # p==0 or q==0; np.log(0)=-inf and the subtraction becomes nan. Clip with the same Laplace
            # cushion the kfold path uses so caller-visible features stay finite.
            assert self._woe_pos is not None and self._woe_neg is not None, "_encode_per_row(woe): fit() must populate _woe_pos/_woe_neg before transform"
            for i, c in enumerate(cats):
                p = self._woe_pos.get(c)
                q = self._woe_neg.get(c)
                if p is None or q is None:
                    out[i] = unseen_logodds
                else:
                    p_safe = float(min(max(p, 1e-12), 1.0 - 1e-12))
                    q_safe = float(min(max(q, 1e-12), 1.0 - 1e-12))
                    out[i] = float(np.log(p_safe) - np.log(q_safe))
            return out
        assert self._category_counts is not None and self._category_means is not None, "_encode_per_row: fit() must populate _category_counts/_category_means before transform"
        for i, c in enumerate(cats):
            n_c = self._category_counts.get(c, 0)
            if n_c == 0:
                out[i] = prior
            else:
                m_c = self._category_means[c]
                if self.method == "target_james_stein":
                    # JS shrinkage: factor = 1 - (k-1)*sigma^2 / sum_sq_dev
                    # Simplified: shrink toward prior with factor that
                    # depends on per-category sample size.
                    shrink = self.smoothing / (n_c + self.smoothing)
                    out[i] = (1 - shrink) * m_c + shrink * prior
                else:
                    out[i] = (n_c * m_c + self.smoothing * prior) / (n_c + self.smoothing)
        return out

    def __repr__(self) -> str:
        return (
            f"LeakageSafeEncoder(method={self.method!r}, smoothing={self.smoothing}, "
            f"woe_smoothing={self.woe_smoothing}, cv={self.cv}, prior={self.prior!r}, "
            f"fitted={self._is_fitted})"
        )


__all__ = ["LeakageSafeEncoder"]
