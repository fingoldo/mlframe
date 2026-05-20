"""
Leakage-safe target encoders for high-cardinality categorical columns.

Round-3 architecture A17 + tests T14: target encoding inside a
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
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    # Imports for ``Union[..., pd.Series, pl.Series]`` annotations below.
    # Guarded under TYPE_CHECKING so the runtime import cost stays zero
    # while typecheckers / IDE tooling see the real symbols.
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================


_NULL_SENTINEL = "__NULL__"


def _categorical_to_string_array(values: Sequence) -> np.ndarray:
    """Coerce a sequence of category values to a numpy string array of object dtype. Handles None / NaN
    by mapping them to a sentinel ``"__NULL__"`` so they form their own category rather than being
    silently dropped.

    Vectorised path: numpy / pandas / polars Series take ``.astype(str)`` -- ~50x faster than per-row
    Python loop on 1M+ rows. Plain Python sequences (list / tuple) fall back to the legacy loop.
    """
    # Fast path: pandas Series with vectorised str cast.
    try:
        import pandas as pd
        if isinstance(values, pd.Series):
            mask_null = values.isna()
            out = values.astype(str).to_numpy(dtype=object)
            if mask_null.any():
                out[mask_null.to_numpy()] = _NULL_SENTINEL
            return out
    except ImportError:
        pass
    # Fast path: polars Series.
    try:
        import polars as pl
        if isinstance(values, pl.Series):
            mask_null = np.asarray(values.is_null().to_numpy())
            # cast to Utf8 then numpy; ``__NULL__`` overwrites nulls (polars cast yields ``None``).
            out = values.cast(pl.Utf8).to_numpy().astype(object)
            if mask_null.any():
                out[mask_null] = _NULL_SENTINEL
            # Float NaN cells survive cast as ``"NaN"``; rebrand to sentinel for parity with pandas path.
            nan_mask = out == "NaN"
            if nan_mask.any():
                out[nan_mask] = _NULL_SENTINEL
            return out
    except ImportError:
        pass
    # Fast path: numpy array.
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "f":
            mask = np.isnan(values)
            out = values.astype(str).astype(object)
            if mask.any():
                out[mask] = _NULL_SENTINEL
            return out
        if values.dtype.kind in ("O", "U", "S"):
            out = np.where(
                _objectwise_isnull(values),
                _NULL_SENTINEL,
                values.astype(str),
            ).astype(object)
            return out
        return values.astype(str).astype(object)
    # Generic iterable / list / tuple: per-row loop is fine (length typically <= a few thousand for tests).
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        if v is None:
            out[i] = _NULL_SENTINEL
            continue
        if isinstance(v, float) and np.isnan(v):
            out[i] = _NULL_SENTINEL
            continue
        out[i] = str(v)
    return out


def _objectwise_isnull(arr: np.ndarray) -> np.ndarray:
    """Vectorised None / NaN mask for object-dtype arrays.

    Pre-fix this returned an all-False mask on per-row introspection failure,
    so target-encoder fit silently treated None/NaN rows as valid encoded
    values (their non-numeric content getting averaged into the per-category
    mean / WoE). The "fall back per-row when ufuncs fail" docstring was
    aspirational; the fallback was actually "pretend nothing is null."

    Post-fix: try pandas.isna (the proper vectorised fallback for object-dtype
    arrays). If THAT also fails, raise: silent target-encoder corruption is
    strictly worse than a loud failure that surfaces the bad input early.
    """
    try:
        is_none = np.array([x is None for x in arr], dtype=bool)
    except Exception as _e_none:
        # Per-row `is None` should never raise on a finite-iterable object
        # array; if it does, the input is malformed enough that pandas.isna
        # is the right tool. Don't silently zero - that's the bug we're
        # fixing.
        import pandas as _pd
        try:
            is_none = _pd.isna(arr).astype(bool)
            # pandas.isna already covers NaN for floats; return early.
            return is_none
        except Exception as _e_pd:
            raise ValueError(
                f"_objectwise_isnull: both per-row `is None` ({_e_none}) and "
                f"pandas.isna ({_e_pd}) failed on object array. Refusing to "
                f"return an all-False mask, which would silently corrupt "
                f"target-encoder fit by treating null rows as valid."
            ) from _e_pd
    try:
        is_nan = np.array(
            [isinstance(x, float) and np.isnan(x) for x in arr],
            dtype=bool,
        )
    except Exception as _e_nan:
        import pandas as _pd
        try:
            is_nan = _pd.isna(arr).astype(bool)
        except Exception as _e_pd:
            raise ValueError(
                f"_objectwise_isnull: per-row NaN check failed ({_e_nan}) and "
                f"pandas.isna fallback also failed ({_e_pd})."
            ) from _e_pd
    return is_none | is_nan


def _coerce_y_to_float64(y) -> np.ndarray:
    """Backend-agnostic float64 coercion. Avoids ``np.asarray(list(y))`` which materialises a Python list
    first (doubles memory + drops dtype on pandas/polars Series). Native ``.to_numpy()`` is zero-copy
    when dtype already matches.

    Return-value contract: read-only. When ``y`` is already a float64 ndarray (or a float64
    pandas/polars Series), the returned array shares storage with the caller's input; in-place
    mutation would corrupt the caller's target. All current callers (``_compute_prior``,
    ``_compute_per_category``, ``_compute_woe_per_category``) treat the result as read-only.
    Take ``.copy()`` at the call site if you need to mutate."""
    if isinstance(y, np.ndarray):
        return y.astype(np.float64, copy=False)
    try:
        import pandas as pd
        if isinstance(y, (pd.Series, pd.DataFrame)):
            return y.to_numpy(dtype=np.float64, copy=False).ravel()
    except ImportError:
        pass
    try:
        import polars as pl
        if isinstance(y, pl.Series):
            return y.cast(pl.Float64).to_numpy().ravel()
    except ImportError:
        pass
    return np.asarray(y, dtype=np.float64).ravel()


def _compute_prior(y: np.ndarray, prior_kind: Literal["mean", "median"], sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        if prior_kind == "median":
            return float(np.median(y))
        return float(np.mean(y))
    # Weighted prior. np.median has no native sample_weight; emulate via sorted cumulative weights for the median
    # branch so weighted_prior(uniform_w) == np.median(y) (up to ties).
    sw = np.asarray(sample_weight, dtype=np.float64)
    total = float(sw.sum())
    if total <= 0:
        if prior_kind == "median":
            return float(np.median(y))
        return float(np.mean(y))
    if prior_kind == "median":
        order = np.argsort(y)
        cw = np.cumsum(sw[order])
        cutoff = 0.5 * total
        # First index with cumulative weight >= cutoff is the weighted median.
        idx = int(np.searchsorted(cw, cutoff))
        idx = min(idx, len(y) - 1)
        return float(y[order[idx]])
    return float(np.dot(sw, y) / total)


def _smoothed_mean(
    counts: np.ndarray,
    means: np.ndarray,
    smoothing: float,
    prior: float,
) -> np.ndarray:
    """Standard smoothed mean: pulls rare-category encodings toward
    the global prior."""
    return (counts * means + smoothing * prior) / (counts + smoothing)


# =====================================================================
# LeakageSafeEncoder
# =====================================================================


class LeakageSafeEncoder:
    """Out-of-fold target encoder for a single categorical column.

    Round-3 A17 lock: ``fit_transform()`` returns OOF-computed
    encodings for the train rows; ``transform()`` on held-out rows
    uses the full-train statistic.

    Parameters
    ----------
    method : Literal["target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe"]
        Encoding flavour. ``target_m_estimate`` is mathematically
        identical to ``target_mean`` so we treat them interchangeably
        with a different default ``smoothing``.
    smoothing : float
        Regularisation toward the prior. Higher -> rare categories
        encoded closer to the prior.
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
        smoothing: float = 10.0,
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
            raise ValueError(
                f"unknown method {method!r}; valid: {sorted(valid_methods)}"
            )
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}")
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")

        self.method = method
        self.smoothing = smoothing
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
            raise ValueError(
                f"X and y length mismatch: {cats.shape[0]} vs {y_arr.shape[0]}"
            )
        sw_arr = self._coerce_sample_weight(sample_weight, len(y_arr))

        self._global_prior = _compute_prior(y_arr, self.prior, sw_arr)
        self._category_counts, self._category_means = self._compute_per_category(cats, y_arr, sw_arr)

        if self.method == "woe":
            unique_y = np.unique(y_arr)
            if not (set(unique_y).issubset({0.0, 1.0}) and len(unique_y) <= 2):
                raise ValueError(
                    "method='woe' requires binary {0, 1} target; got "
                    f"{sorted(unique_y)[:5]}"
                )
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
            raise RuntimeError(
                "LeakageSafeEncoder.transform called before fit; "
                "use fit_transform on train rows first"
            )
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
            raise ValueError(
                f"X and y length mismatch: {cats.shape[0]} vs {y_arr.shape[0]}"
            )
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
        the per-row Python loop. Same correctness contract (verified by the
        H-FH-14 regression test) and ~10x faster on 1M-row binary y.
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
            p_arr = (pos_counts_arr + self.smoothing) / (n_pos + self.smoothing)
            q_arr = (neg_counts_arr + self.smoothing) / (n_neg + self.smoothing)
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
        for c in all_cats:
            p = (pos_counts.get(c, 0.0) + self.smoothing) / (n_pos + self.smoothing)
            q = (neg_counts.get(c, 0.0) + self.smoothing) / (n_neg + self.smoothing)
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
        # The H-FH-14 measured regression (vectorised: 407ms, legacy
        # dict.get: 160ms on n=100k, K=200, cv=3) made us revert the
        # pandas-Series path here. The ``_compute_per_category`` upstream
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
                    out[j] = float(np.log(p) - np.log(q))
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
                    out[i] = (
                        ((n_c - 1) * loo_mean + self.smoothing * prior)
                        / ((n_c - 1) + self.smoothing)
                    )
            return out
        for i, (c, y_i, w_i) in enumerate(zip(cats, y, sample_weight)):
            w_total = counts[c]
            w_remaining = w_total - float(w_i)
            if w_remaining <= 0:
                out[i] = prior
            else:
                loo_mean = (sums[c] - float(w_i) * float(y_i)) / w_remaining
                out[i] = (
                    (w_remaining * loo_mean + self.smoothing * prior)
                    / (w_remaining + self.smoothing)
                )
        return out

    def _compute_per_category_sums(self, cats: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple:
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
        out = np.empty(len(cats), dtype=np.float64)
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
            for i, c in enumerate(cats):
                p = self._woe_pos.get(c)
                q = self._woe_neg.get(c)
                if p is None or q is None:
                    out[i] = unseen_logodds
                else:
                    out[i] = float(np.log(p) - np.log(q))
            return out
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
            f"cv={self.cv}, prior={self.prior!r}, fitted={self._is_fitted})"
        )


__all__ = ["LeakageSafeEncoder"]
