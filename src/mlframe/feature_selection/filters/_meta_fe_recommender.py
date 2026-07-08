"""Layer 99 — META FE-RECOMMENDER: turn ~50 opt-in ``fe_*`` flags into 1 auto mode.

MRMR ships ~50 opt-in feature-engineering generators (Layers 33-95), each one a
master ``fe_*_enable`` flag that helps ONLY on a specific data shape and adds
compute everywhere else. Users who don't read the docstrings don't know which to
turn on, so in practice the entire FE zoo stays off and the wins go unrealised.

This module closes that gap with TWO complementary layers, both built ON the
Layer-98 Param-Oracle (:mod:`mlframe.utils._param_oracle`):

A. **Rule-based cold-start recommender** (:func:`recommend_fe_flags_by_rules`)
   Cheaply fingerprints ``(X, y)`` -- reusing Param-Oracle's
   :func:`default_fingerprint` plus a handful of FE-relevant structural
   detectors (int-as-cat group columns, object/category cats, a time+entity
   pair, NaN rate, continuous structure) -- and maps each matched precondition
   to the master flag(s) of the generator(s) that pay off on that shape. Needs
   NO history; it is the cold-start prior.

B. **Param-Oracle-backed learned recommender** (:class:`MetaFERecommender`)
   Wraps (A) as its cold-start fallback and layers a :class:`ParamOracle` keyed
   on the dataset fingerprint over it. ``fit_observe(X, y, flags_used,
   cv_score)`` records which flag-set scored what on which fingerprint;
   ``recommend(X, y)`` returns the learned best flag-set once enough confident
   history exists for that fingerprint, else falls back to the rules. The
   learned layer therefore IMPROVES on the static rules over time as more
   datasets are seen, without ever doing worse than the cold-start prior.

KEY CONSTRAINT (inherited from Param-Oracle): the learned store is STAT-ONLY --
only scalar fingerprint stats, the flag-set, and the CV score touch disk; raw
arrays never do.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from mlframe.utils import (
    ParamOracle,
    bucketize_fingerprint,
    default_fingerprint,
)

logger = logging.getLogger(__name__)

# The master ``fe_*_enable`` flags this recommender reasons about. Sub-knobs
# (cols / top_k / periods) are left at their auto-detect defaults; the master
# switch is the only thing a 1-knob user needs flipped. Names verified against
# the MRMR ctor signature (mrmr.py).
FE_GROUP_FLAGS = (
    "fe_grouped_agg_enable",
    "fe_composite_group_agg_enable",
    "fe_grouped_quantile_enable",
)
FE_CAT_FLAGS = (
    "fe_count_encoding_enable",
    "fe_frequency_encoding_enable",
    "fe_cat_pair_enable",
)
FE_CAT_TRIPLE_FLAG = "fe_cat_triple_enable"
FE_TEMPORAL_FLAG = "fe_temporal_agg_enable"
FE_MISSINGNESS_FLAG = "fe_missingness_indicator_enable"
FE_HYBRID_ORTH_FLAG = "fe_hybrid_orth_enable"
FE_NUMERIC_DECOMPOSE_FLAG = "fe_numeric_decompose_enable"
FE_MODULAR_FLAG = "fe_modular_enable"
FE_CAT_NUM_INTERACTION_FLAG = "fe_cat_num_interaction_enable"

# Full universe of master flags this recommender can toggle. The rule
# recommender always returns a dict over THIS exact key-set (all False unless a
# precondition fired) so callers / the Param-Oracle param_space see a stable,
# complete flag vector.
ALL_FE_MASTER_FLAGS = (
    FE_GROUP_FLAGS
    + FE_CAT_FLAGS
    + (
        FE_CAT_TRIPLE_FLAG,
        FE_TEMPORAL_FLAG,
        FE_MISSINGNESS_FLAG,
        FE_HYBRID_ORTH_FLAG,
        FE_NUMERIC_DECOMPOSE_FLAG,
        FE_MODULAR_FLAG,
        FE_CAT_NUM_INTERACTION_FLAG,
    )
)

# Detector thresholds. Int-as-cat group cardinality band mirrors the existing
# ``detect_group_column_candidates`` heuristic (min_unique=3, max_unique=500).
_GROUP_MIN_CARD = 3
_GROUP_MAX_CARD = 500
_MISSINGNESS_MIN_RATE = 0.01
_HEAVY_KURTOSIS = 3.0  # excess kurtosis -> heavy tails / non-Gaussian structure
_HEAVY_SKEW = 1.0


# ---------------------------------------------------------------------------
# Cheap FE-relevant structural detectors (stat-only, no raw data retained)
# ---------------------------------------------------------------------------

def _fe_structure(X, y=None) -> dict:
    """Detect FE-relevant structure of ``X`` cheaply. Returns a dict of bools /
    small scalars used by the rule recommender. Never raises; degrades to an
    all-False structure on un-introspectable inputs.

    Detected:
      * ``int_as_cat_group``: at least one numeric column whose distinct-count
        sits in ``[3, 500]`` (a plausible group key).
      * ``object_cat``: at least one object/category/string column.
      * ``n_cat``: count of categorical-ish columns (object + int-as-cat).
      * ``time_col``: at least one monotone-ish int or datetime column.
      * ``entity_col``: at least one int-as-cat column (a plausible entity key).
      * ``max_nan_rate``: highest per-column NaN fraction.
      * ``has_continuous``: at least one high-cardinality numeric column.
      * ``heavy_structure``: a continuous column with heavy tails / skew
        (non-Gaussian -> non-linear interactions worth orthogonal-basis FE).
      * ``anchored_numeric``: a numeric column with many repeated discrete-ish
        values (rounding anchors / encoded ids -> decompose / modular FE).
    """
    import numpy as np

    out = {
        "int_as_cat_group": False,
        "object_cat": False,
        "n_cat": 0,
        "time_col": False,
        "entity_col": False,
        "max_nan_rate": 0.0,
        "has_continuous": False,
        "heavy_structure": False,
        "anchored_numeric": False,
    }

    cols = _iter_columns(X)
    if not cols:
        return out

    n_cat = 0
    for _name, arr in cols:
        n = arr.shape[0]
        if n == 0:
            continue
        kind = arr.dtype.kind

        # datetime column -> time signal.
        if kind in ("M", "m"):
            out["time_col"] = True
            continue

        if kind in ("O", "U", "S") or kind == "b":
            # object / string / bool -> categorical.
            out["object_cat"] = True
            n_cat += 1
            continue

        if kind in ("i", "u", "f"):
            finite = np.isfinite(arr) if kind == "f" else np.ones(n, dtype=bool)
            n_finite = int(finite.sum())
            nan_rate = float((n - n_finite) / n) if n else 0.0
            if nan_rate > out["max_nan_rate"]:
                out["max_nan_rate"] = nan_rate
            if n_finite < 3:
                continue
            vals = arr[finite]
            try:
                card = int(np.unique(vals).size)
            except Exception:
                card = n_finite

            is_intish = kind in ("i", "u") or _is_integral(vals)
            if is_intish and _GROUP_MIN_CARD <= card <= _GROUP_MAX_CARD:
                out["int_as_cat_group"] = True
                out["entity_col"] = True
                n_cat += 1
                # A strictly increasing integer column is a plausible time axis.
                if _is_monotone(vals):
                    out["time_col"] = True

            # High-cardinality numeric -> a continuous feature.
            if card >= max(20, int(0.2 * n_finite)):
                out["has_continuous"] = True
                skew, exkurt = _skew_kurt(vals.astype(np.float64, copy=False))
                if abs(skew) > _HEAVY_SKEW or exkurt > _HEAVY_KURTOSIS:
                    out["heavy_structure"] = True

            # Anchored / discrete-pattern numeric: cardinality far below row
            # count (many repeats) yet not a low-card group -> rounding anchors
            # or encoded ids that decompose / modular FE can crack.
            if is_intish and card > _GROUP_MAX_CARD and card < 0.5 * n_finite:
                out["anchored_numeric"] = True

    out["n_cat"] = n_cat
    return out


def _iter_columns(X):
    """Yield ``(name, 1d-ndarray)`` per column of a pandas / polars DataFrame or
    a 2D numpy array. Returns [] on un-introspectable input."""
    import numpy as np

    # pandas
    if hasattr(X, "columns") and hasattr(X, "iloc"):
        out = []
        for c in X.columns:
            try:
                out.append((str(c), np.asarray(X[c].to_numpy())))
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _meta_fe_recommender.py:204: %s", e)
                continue
        return out
    # polars
    if hasattr(X, "columns") and hasattr(X, "get_column"):
        out = []
        for c in X.columns:
            try:
                out.append((str(c), np.asarray(X.get_column(c).to_numpy())))
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _meta_fe_recommender.py:213: %s", e)
                continue
        return out
    # numpy / array-like
    try:
        arr = np.asarray(X)
    except Exception:
        return []
    if arr.ndim == 1:
        return [("0", arr)]
    if arr.ndim == 2:
        return [(str(j), arr[:, j]) for j in range(arr.shape[1])]
    return []


def _is_integral(vals) -> bool:
    import numpy as np
    try:
        return bool(np.all(np.equal(np.mod(vals, 1.0), 0.0)))
    except Exception:
        return False


def _is_monotone(vals) -> bool:
    import numpy as np
    try:
        d = np.diff(vals)
        return bool(np.all(d >= 0)) or bool(np.all(d <= 0))
    except Exception:
        return False


def _skew_kurt(x):
    n = x.size
    if n < 3:
        return 0.0, 0.0
    m = x.mean()
    d = x - m
    var = float((d**2).mean())
    if var <= 1e-12:
        return 0.0, 0.0
    sd = var**0.5
    z = d / sd
    return float((z**3).mean()), float((z**4).mean() - 3.0)


# ---------------------------------------------------------------------------
# A. Rule-based cold-start recommender
# ---------------------------------------------------------------------------

def recommend_fe_flags_by_rules(X, y=None) -> dict:
    """Cold-start FE-flag recommendation from data shape alone (no history).

    Returns a dict over :data:`ALL_FE_MASTER_FLAGS`; every key is present, value
    ``True`` iff the generator's data-shape precondition is met. A clean
    continuous frame with no cats / groups / time / NaNs returns all-False (no
    spurious enables -- the key no-false-positive property the L99 tests pin).

    Rule table
    ----------
    * int-as-cat group column (card 3..500)  -> grouped_agg, composite_group_agg,
      grouped_quantile
    * object / category columns               -> count_encoding, frequency_encoding,
      cat_pair ; (>= 3 cats) -> cat_triple ; (+ numeric) -> cat_num_interaction
    * time column + entity column             -> temporal_agg
    * any column NaN rate >= 1%               -> missingness_indicator
    * heavy-tailed / skewed continuous        -> hybrid_orth
    * anchored / discrete-pattern numeric     -> numeric_decompose, modular
    """
    s = _fe_structure(X, y)
    flags = {f: False for f in ALL_FE_MASTER_FLAGS}

    if s["int_as_cat_group"]:
        for f in FE_GROUP_FLAGS:
            flags[f] = True

    if s["object_cat"]:
        for f in FE_CAT_FLAGS:
            flags[f] = True
        if s["n_cat"] >= 3:
            flags[FE_CAT_TRIPLE_FLAG] = True
        if s["has_continuous"]:
            flags[FE_CAT_NUM_INTERACTION_FLAG] = True

    if s["time_col"] and s["entity_col"]:
        flags[FE_TEMPORAL_FLAG] = True

    if s["max_nan_rate"] >= _MISSINGNESS_MIN_RATE:
        flags[FE_MISSINGNESS_FLAG] = True

    if s["heavy_structure"]:
        flags[FE_HYBRID_ORTH_FLAG] = True

    if s["anchored_numeric"]:
        flags[FE_NUMERIC_DECOMPOSE_FLAG] = True
        flags[FE_MODULAR_FLAG] = True

    return flags


# ---------------------------------------------------------------------------
# B. Param-Oracle-backed learned recommender
# ---------------------------------------------------------------------------

def _flags_to_combo_key(flags: Mapping[str, bool]) -> tuple:
    """Stable hashable key for a flag-set: sorted (name, bool) tuple over the
    full master-flag universe (missing keys default False)."""
    return tuple((f, bool(flags.get(f, False))) for f in ALL_FE_MASTER_FLAGS)


class MetaFERecommender:
    """Learned FE-flag recommender, built on :class:`ParamOracle`.

    The rule recommender is the cold-start fallback; the oracle learns, per
    dataset fingerprint, which flag-set earns the best downstream CV score and
    overrides the rules once it has confident history.

    Parameters
    ----------
    store_path:
        Param-Oracle parquet store path. Defaults to a bare filename under the
        shared Param-Oracle store dir.
    min_observations:
        Confidence gate forwarded to :class:`ParamOracle`. A learned flag-set is
        only trusted once it has at least this many recorded observations on the
        matching fingerprint bucket.
    fn_name:
        Logical key under which observations are stored / looked up.
    """

    FN_NAME = "mrmr_fe_flags"

    def __init__(
        self,
        store_path: str = "meta_fe_recommender.parquet",
        *,
        min_observations: int = 2,
        fn_name: Optional[str] = None,
    ):
        self.fn_name = fn_name or self.FN_NAME
        # param_space declares the toggled flags; combos are the flag-sets we
        # actually observe (recorded via record(), not swept), so the grid is
        # only used for the cold caller-default. maximize the CV score.
        self.oracle = ParamOracle(
            store_path,
            param_space={f: [False, True] for f in ALL_FE_MASTER_FLAGS},
            mode="inference",
            maximize="cv_score",
            min_observations=int(min_observations),
        )

    # ----- fingerprint -----

    @staticmethod
    def _fingerprint(X, y=None) -> dict:
        return default_fingerprint((X,), {})

    # ----- observe / learn -----

    def fit_observe(self, X, y, flags_used: Mapping[str, bool], cv_score: float, ts: Optional[str] = None) -> None:
        """Record that ``flags_used`` scored ``cv_score`` (higher = better) on the
        fingerprint of ``(X, y)``. Future :meth:`recommend` calls on a
        similar fingerprint learn from this."""
        fp = self._fingerprint(X, y)
        combo = {f: bool(flags_used.get(f, False)) for f in ALL_FE_MASTER_FLAGS}
        self.oracle.record(
            fp, combo, {"cv_score": float(cv_score)},
            ts=ts, fn_name=self.fn_name,
        )

    # ----- recommend -----

    def recommend(self, X, y=None) -> dict:
        """Best FE flag-set for ``(X, y)``.

        Param-Oracle lookup first: if a confident learned best exists for this
        fingerprint (exact bucket or near-neighbour), return it. Otherwise fall
        back to the rule-based cold-start prior. Always returns a dict over
        :data:`ALL_FE_MASTER_FLAGS`."""
        rules = recommend_fe_flags_by_rules(X, y)
        fp = self._fingerprint(X, y)
        if not self._has_confident_history(fp):
            return rules
        learned = self.oracle.recommend(fp, fn_name=self.fn_name)
        # Normalise to the full flag universe (recommend can return the bare
        # caller-default first-combo when history is thin).
        return {f: bool(learned.get(f, rules.get(f, False))) for f in ALL_FE_MASTER_FLAGS}

    def _has_confident_history(self, fp: Mapping[str, Any]) -> bool:
        """True iff the store holds a confident (>= min_observations) row whose
        fingerprint bucket EXACTLY matches ``fp``'s bucket for this fn. Distinct
        from the oracle's own k-NN fallback: we only override the rules when we
        have direct evidence for THIS shape, never a transplanted neighbour."""
        target_key = _stable_bucket_key(fp)
        for r in self.oracle.store.read_rows():
            if r.get("fn_name") != self.fn_name or r.get("host") != self.oracle.host:
                continue
            if r.get("fp_bucket_json") != target_key:
                continue
            if int(r.get("n_obs", 0) or 0) >= self.oracle.min_observations:
                return True
        return False


def _stable_bucket_key(fp: Mapping[str, Any]) -> str:
    import orjson
    # orjson emits compact output (no spaces), matching the old
    # ``separators=(",", ":")``; decode to keep the ``str`` return contract.
    return orjson.dumps(bucketize_fingerprint(fp), default=str, option=orjson.OPT_SORT_KEYS).decode("utf-8")


__all__ = [
    "recommend_fe_flags_by_rules",
    "MetaFERecommender",
    "ALL_FE_MASTER_FLAGS",
    "FE_GROUP_FLAGS",
    "FE_CAT_FLAGS",
]
