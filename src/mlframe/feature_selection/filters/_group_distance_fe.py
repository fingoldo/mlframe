"""Layer 95 PART B (2026-06-01): PER-GROUP DISTRIBUTION-DISTANCE FE.

Extends Layer 88 (``_grouped_quantile_fe``: percentile-rank-within-group +
per-group spread). Where Layer 88 captures where a row sits WITHIN its own
group's distribution, this layer captures how far the row's GROUP is from the
GLOBAL (pooled) distribution -- a group-anomaly detector broadcast to rows.

For each ``(group_col, num_col)`` three distance features:

* ``gzdist``  -- group-level z: ``(group_mean - global_mean) / global_std``.
  How many global-scale standard deviations the group's centre sits from the
  pooled centre. Recovers a target that depends on whether a row belongs to a
  group whose typical value is unusual relative to everyone else.
* ``gkldist`` -- per-group KL divergence ``KL(P_group || P_global)`` over a
  fixed global histogram binning, broadcast to every row in the group. The
  information-theoretic distance between the group's value distribution and the
  global one -- catches shape anomalies (a group whose values cluster in a
  region the global distribution rarely visits) that a mean-shift z misses.
* ``gwdist``  -- per-group Wasserstein-1 (earth-mover) distance between the
  group's empirical distribution and the global one, broadcast to rows. A
  metric-aware distance (sensitive to HOW FAR the mass moved, not just bin
  overlap), complementary to KL.

Captures GROUP-ANOMALY signals: rows that sit in atypical groups get flagged,
independent of the row's own within-group position (orthogonal to Layer 88).

The IT enhancement (MI gate)
----------------------------
``hybrid_group_distance_fe`` reuses Layer 88's plug-in MI-uplift gate
(``score_grouped_quantile_by_mi_uplift``): each engineered column is ranked by
``MI(col; y)`` and kept only when it clears a minimum MI and beats the source
num_col's own marginal MI. A group-distance feature on a column whose group
structure carries no y-signal lands at near-zero MI and is dropped.

Leakage safety (CRITICAL)
-------------------------
Recipes store ONLY per-group distance scalars + the pooled-global fallback,
all computed on TRAIN. Replay reads only X (maps each row's group key through
the stored lookup; unseen groups fall back to the global value). No ``y``
reference is captured -- transform() is leakage-free by construction.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_group_distance_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._internals import group_key_strings

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_group_zdist",
    "engineered_name_group_kl",
    "engineered_name_group_wasserstein",
    "generate_group_distance_features",
    "apply_group_distance",
    "hybrid_group_distance_fe",
]

# Minimum per-group sample size to trust a per-group distribution distance;
# smaller groups fall back to 0 (no measurable distance from global).
_MIN_GROUP_SIZE = 8
# Fixed number of global histogram bins for the KL / Wasserstein distances.
_DIST_NBINS = 20
# Laplace smoothing for the KL histogram so an empty group bin doesn't make
# KL blow up to +inf.
_KL_EPS = 1e-9

_VALID_DISTANCE_TYPES = ("zdist", "kl", "wasserstein")


def engineered_name_group_zdist(num_col: str, group_col: str) -> str:
    return f"gzdist({num_col}|{group_col})"


def engineered_name_group_kl(num_col: str, group_col: str) -> str:
    return f"gkldist({num_col}|{group_col})"


def engineered_name_group_wasserstein(num_col: str, group_col: str) -> str:
    return f"gwdist({num_col}|{group_col})"


def _broadcast_lookup(g_keys: np.ndarray, lookup: dict, glob: float) -> np.ndarray:
    """Map each row's group key through ``lookup`` (str-keyed), unseen -> glob.

    Mirrors the Layer-88 ``_broadcast_lookup`` hot-path: resolve once per UNIQUE
    key via ``np.unique(return_inverse)`` and broadcast back, rather than once
    per row. Falls back to a per-row mapping on the TypeError np.unique raises
    for unorderable mixed-type objects.
    """
    g_keys = np.asarray(g_keys)
    try:
        uniq, inverse = np.unique(g_keys, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        uniq_vals = np.array(
            [lookup.get(str(_k), glob) for _k in uniq], dtype=np.float64
        )
        out = uniq_vals[inverse]
    except (TypeError, ValueError):
        out = np.array(
            [lookup.get(str(_k), glob) for _k in g_keys], dtype=np.float64
        )
    return np.nan_to_num(out, nan=glob, posinf=glob, neginf=glob)


def _kl_divergence(group_vals: np.ndarray, global_edges: np.ndarray,
                   global_hist: np.ndarray) -> float:
    """``KL(P_group || P_global)`` over the fixed global bin edges. Both
    histograms Laplace-smoothed + renormalised so an empty bin never produces
    a division-by-zero / +inf term."""
    if group_vals.size == 0 or global_edges.size < 2:
        return 0.0
    g_counts, _ = np.histogram(group_vals, bins=global_edges)
    p = g_counts.astype(np.float64) + _KL_EPS
    p /= p.sum()
    q = global_hist.astype(np.float64) + _KL_EPS
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _wasserstein1(group_vals: np.ndarray, global_sorted: np.ndarray) -> float:
    """Wasserstein-1 (earth-mover) distance between the group's empirical
    distribution and the global one, via the integral of |CDF_group - CDF_global|.
    Reuses ``scipy.stats.wasserstein_distance`` when available; otherwise a
    pure-numpy quantile-grid approximation."""
    if group_vals.size == 0 or global_sorted.size == 0:
        return 0.0
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(group_vals, global_sorted))
    except Exception:
        # Pure-numpy fallback: mean |Q_group(u) - Q_global(u)| over a fixed
        # quantile grid (Wasserstein-1 for 1-D == integral of |inverse-CDF diff|).
        u = np.linspace(0.0, 1.0, 101)
        qg = np.quantile(group_vals, u)
        qglob = np.quantile(global_sorted, u)
        return float(np.mean(np.abs(qg - qglob)))


def generate_group_distance_features(
    X: pd.DataFrame,
    group_cols: Sequence[str],
    num_cols: Sequence[str],
):
    """Compute per-(group_col, num_col) group-level z / KL / Wasserstein-1
    distance from the global distribution, broadcast to rows.

    Returns ``(enc_df, raw_recipes)`` where ``raw_recipes[name]`` is the kwargs
    payload for :func:`apply_group_distance` (and the recipe builder). Each
    payload stores the per-group distance scalar lookup + the global fallback;
    replay reads only X (no ``y`` reference), so transform() is leakage-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"generate_group_distance_features: X must be a pandas DataFrame; "
            f"got {type(X).__name__}"
        )
    group_cols = [c for c in group_cols if c in X.columns]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not group_cols:
        return pd.DataFrame(index=X.index), raw_recipes

    for group_col in group_cols:
        g_keys = group_key_strings(X[group_col])
        cur_num_cols = [
            c for c in num_cols
            if c in X.columns and c != group_col
            and pd.api.types.is_numeric_dtype(X[c])
        ]
        for num_col in cur_num_cols:
            x = np.asarray(X[num_col].to_numpy(), dtype=np.float64)
            finite_all = x[np.isfinite(x)]
            if finite_all.size == 0:
                global_mean = 0.0
                global_std = 1.0
                global_edges = np.array([0.0, 1.0], dtype=np.float64)
                global_hist = np.array([1.0], dtype=np.float64)
                global_sorted = np.array([], dtype=np.float64)
            else:
                global_mean = float(np.mean(finite_all))
                _std = float(np.std(finite_all))
                global_std = _std if _std > 0.0 else 1.0
                lo, hi = float(finite_all.min()), float(finite_all.max())
                if hi <= lo:
                    hi = lo + 1.0
                global_edges = np.linspace(lo, hi, _DIST_NBINS + 1)
                global_hist, _ = np.histogram(finite_all, bins=global_edges)
                global_hist = global_hist.astype(np.float64)
                global_sorted = np.sort(finite_all)

            z_lookup: dict[str, float] = {}
            kl_lookup: dict[str, float] = {}
            w_lookup: dict[str, float] = {}
            for gv, idx in pd.Series(np.arange(len(x))).groupby(g_keys, sort=False):
                rows = idx.to_numpy()
                vals = x[rows]
                fin = vals[np.isfinite(vals)]
                key = str(gv)
                if fin.size >= _MIN_GROUP_SIZE:
                    z_lookup[key] = float(
                        (np.mean(fin) - global_mean) / global_std
                    )
                    kl_lookup[key] = _kl_divergence(fin, global_edges, global_hist)
                    w_lookup[key] = _wasserstein1(fin, global_sorted)
                else:
                    z_lookup[key] = 0.0
                    kl_lookup[key] = 0.0
                    w_lookup[key] = 0.0

            common = {
                "group_col": group_col,
                "num_col": num_col,
                "z_lookup": z_lookup,
                "kl_lookup": kl_lookup,
                "w_lookup": w_lookup,
                "global_mean": global_mean,
                "global_std": global_std,
            }

            z_name = engineered_name_group_zdist(num_col, group_col)
            encoded[z_name] = _broadcast_lookup(g_keys, z_lookup, 0.0)
            raw_recipes[z_name] = {**common, "distance_type": "zdist"}

            kl_name = engineered_name_group_kl(num_col, group_col)
            encoded[kl_name] = _broadcast_lookup(g_keys, kl_lookup, 0.0)
            raw_recipes[kl_name] = {**common, "distance_type": "kl"}

            w_name = engineered_name_group_wasserstein(num_col, group_col)
            encoded[w_name] = _broadcast_lookup(g_keys, w_lookup, 0.0)
            raw_recipes[w_name] = {**common, "distance_type": "wasserstein"}

    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


def apply_group_distance(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay one group-distance column from the stored recipe, reading only
    ``X_test``. ``distance_type`` selects the per-group scalar lookup
    (``zdist`` / ``kl`` / ``wasserstein``); each row maps its group key through
    the lookup. Unseen groups fall back to 0 (no measurable distance)."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_group_distance: X_test must be a DataFrame; got "
            f"{type(X_test).__name__}"
        )
    group_col = recipe["group_col"]
    distance_type = recipe.get("distance_type", "zdist")
    if group_col not in X_test.columns:
        raise KeyError(
            f"apply_group_distance: missing group column {group_col!r} from "
            f"X_test"
        )
    g_keys = group_key_strings(X_test[group_col])
    if distance_type == "zdist":
        lookup = dict(recipe["z_lookup"])
    elif distance_type == "kl":
        lookup = dict(recipe["kl_lookup"])
    elif distance_type == "wasserstein":
        lookup = dict(recipe["w_lookup"])
    else:
        raise ValueError(
            f"apply_group_distance: unknown distance_type {distance_type!r}"
        )
    return _broadcast_lookup(g_keys, lookup, 0.0)


# ---------------------------------------------------------------------------
# Recipe adapter consumed by engineered_recipes.apply_recipe
# ---------------------------------------------------------------------------


def _coerce_X(X, group_col: str, recipe_name: str) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return pd.DataFrame({group_col: X[group_col].to_numpy()})
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({group_col: X[group_col]})
    raise TypeError(
        f"recipe '{recipe_name}': cannot extract {group_col!r} from X of type "
        f"{type(X).__name__}"
    )


def _apply_group_distance_recipe(recipe, X) -> np.ndarray:
    """Adapter: pull the stored payload from ``recipe.extra`` and replay via
    :func:`apply_group_distance`."""
    group_col = str(recipe.extra["group_col"])
    X_view = _coerce_X(X, group_col, recipe.name)
    return apply_group_distance(
        X_view,
        {
            "group_col": group_col,
            "distance_type": str(recipe.extra.get("distance_type", "zdist")),
            "z_lookup": dict(recipe.extra.get("z_lookup", {})),
            "kl_lookup": dict(recipe.extra.get("kl_lookup", {})),
            "w_lookup": dict(recipe.extra.get("w_lookup", {})),
        },
    )


def build_group_distance_recipe(
    *, name: str, group_col: str, num_col: str, distance_type: str,
    z_lookup: dict, kl_lookup: dict, w_lookup: dict,
    global_mean: float = 0.0, global_std: float = 1.0,
):
    """Layer 95 PART B: frozen recipe for one per-group distribution-distance
    feature. Stores the per-group distance scalar lookups + the pooled-global
    stats; replay maps a row's group key through the lookup for
    ``distance_type``. Unseen groups fall back to 0. Replay reads only X (no y),
    so transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    if distance_type not in _VALID_DISTANCE_TYPES:
        raise ValueError(
            f"group_distance distance_type must be one of "
            f"{_VALID_DISTANCE_TYPES}; got {distance_type!r}"
        )
    return EngineeredRecipe(
        name=name,
        kind="group_distance",
        src_names=(str(group_col), str(num_col)),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "distance_type": str(distance_type),
            "z_lookup": {str(k): float(v) for k, v in z_lookup.items()},
            "kl_lookup": {str(k): float(v) for k, v in kl_lookup.items()},
            "w_lookup": {str(k): float(v) for k, v in w_lookup.items()},
            "global_mean": float(global_mean),
            "global_std": float(global_std),
        },
    )


# ---------------------------------------------------------------------------
# Auto-detection helpers (shared shape with Layer 88)
# ---------------------------------------------------------------------------


def _auto_detect_group_cols(X: pd.DataFrame, max_cols: int = 4) -> list[str]:
    try:
        from ...training.composite import detect_group_column_candidates
        cands = detect_group_column_candidates(X)
        return [name for name, _info in cands[:max_cols]]
    except Exception as _e:
        logger.debug(
            "group_distance auto-detect: detector import failed (%s); using "
            "fallback cardinality scan.", _e,
        )
        out: list[str] = []
        n = len(X)
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_float_dtype(col):
                continue
            nun = int(col.nunique(dropna=True))
            if 3 <= nun <= min(500, max(3, n // 2)):
                out.append(str(c))
        return out[:max_cols]


def _auto_detect_num_cols(
    X: pd.DataFrame, group_cols: Sequence[str], max_cols: int = 8,
) -> list[str]:
    group_set = set(group_cols)
    out: list[str] = []
    for c in X.columns:
        if c in group_set:
            continue
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        if pd.api.types.is_float_dtype(col):
            out.append(str(c))
            continue
        if int(col.nunique(dropna=True)) > 500:
            out.append(str(c))
    return out[:max_cols]


def hybrid_group_distance_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    group_cols: Optional[Sequence[str]] = None,
    num_cols: Optional[Sequence[str]] = None,
    top_k: int = 6,
    n_bins_mi: int = 10,
    min_mi: float = 1e-4,
    min_uplift: float = 0.0,
):
    """End-to-end per-group distribution-distance FE pipeline.

    1. Auto-detect ``group_cols`` / ``num_cols`` when not supplied.
    2. Materialise per-group z / KL / Wasserstein-1 distance from the global
       distribution, broadcast to rows.
    3. Score by ``MI(col; y)`` and gate by uplift over the source num_col
       marginal MI (Layer 88's ``score_grouped_quantile_by_mi_uplift``); keep
       the top ``top_k`` survivors.
    4. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed only by the MI gate; the persisted recipes carry no ``y``
    reference, so transform() replay is leakage-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"hybrid_group_distance_fe: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if group_cols is None or len(group_cols) == 0:
        group_cols = _auto_detect_group_cols(X)
    else:
        group_cols = [c for c in group_cols if c in X.columns]
    if not group_cols:
        return X.copy(), [], [], pd.DataFrame()
    if num_cols is None or len(num_cols) == 0:
        num_cols = _auto_detect_num_cols(X, group_cols)
    else:
        num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_group_distance_features(
        X, group_cols, num_cols,
    )
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    from ._grouped_quantile_fe import score_grouped_quantile_by_mi_uplift

    eng_to_source = {name: raw_recipes[name]["num_col"] for name in enc_df.columns}
    scores = score_grouped_quantile_by_mi_uplift(
        X, enc_df, y, n_bins=n_bins_mi, eng_to_source=eng_to_source,
    )
    keep = scores[
        (scores["mi"] >= float(min_mi)) & (scores["uplift"] >= float(min_uplift))
    ]
    winners = list(keep["engineered_col"].head(int(top_k)))
    if not winners:
        return X.copy(), [], [], scores

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = []
    for name in winners:
        payload = raw_recipes[name]
        recipes.append(build_group_distance_recipe(name=name, **payload))
    return X_aug, winners, recipes, scores
