"""Auto-parameter tuning skeleton for RFECV (TODO A, Wave 6 prelim 2026-05-28).

The full plan (see TODO.md section A) is to run a synthetic benchmark sweep
across data distributions (normal / heavy-tailed / categorical / mixed),
target types (binary / multiclass / regression), MI strengths, redundancy
patterns -- and fit a small classifier that maps a ``DataFingerprint`` -> the
winning ``SearchConfig + FIConfig + RobustnessConfig`` combo from a calibrated
table.

This module ships a STARTING POINT:

1. ``DataFingerprint.from_xy(X, y)`` extracts ~12 cheap signals (n_rows, p,
   target type, target imbalance, p:n ratio, mean col-MI to y, fraction of
   numeric / categorical / high-card columns, fraction of zero-variance,
   max abs Pearson corr to y, presence of nan).

2. ``suggest_configs(fp)`` returns 3 Config objects based on a small
   rule-based table (no ML classifier yet -- that's the Wave 7+ work). The
   rules encode the bench learnings from Wave 1-5:
      - p >> n -> prescreen='univariate_ht', init_design_size higher
      - flat/plateau curves (low max-MI) -> n_features_selection_rule='one_se_max'
      - balanced classification with strong signal -> argmax + parsimony bias
      - regression high-noise -> convergence_tol kick in
      - tiny p (<10) -> init_design_size='auto' fallback to 2

3. ``RFECV(auto_tune=True)`` wires the three suggested configs at fit entry
   when the user didn't pass their own.

When the ML-driven table replaces the rule table, the public API stays
identical -- only the inner ``suggest_configs`` body changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .rfecv._configs import SearchConfig, FIConfig, RobustnessConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataFingerprint:
    """Cheap (X, y) signals consumed by the auto-tuner. All computed in O(n*p)."""

    n_rows: int
    n_features: int
    target_type: str  # 'binary' | 'multiclass' | 'regression'
    target_imbalance: float  # min-class fraction for classification; std/mean for regression
    p_n_ratio: float
    frac_numeric: float
    frac_high_card: float
    frac_constant: float
    max_abs_corr_to_y: float
    mean_abs_corr_to_y: float
    has_nan: bool

    @classmethod
    def from_xy(cls, X, y) -> "DataFingerprint":
        """Compute a ``DataFingerprint`` (shape, target type, numeric/high-card/constant column fractions, target correlation stats, NaN presence) from raw ``(X, y)``, used by ``suggest_configs`` to pick sensible defaults."""
        if hasattr(X, "shape"):
            n, p = int(X.shape[0]), int(X.shape[1])
        else:
            arr = np.asarray(X)
            n, p = int(arr.shape[0]), int(arr.shape[1])
        y_arr = np.asarray(y)
        # Target type
        # Absolute label cap (mirrors _univariate_ht._MULTICLASS_MAX_LABELS): a sqrt(n)-growing threshold mis-labels a
        # high-cardinality integer regression target (counts / ages / integer-coded ordinals) as 'multiclass' on large n.
        _MAX_LABELS = 50
        if y_arr.dtype.kind in "iub":
            uniq = np.unique(y_arr)
            if uniq.size == 2:
                target_type = "binary"
            elif uniq.size <= _MAX_LABELS and uniq.size <= 0.05 * n:
                target_type = "multiclass"
            else:
                target_type = "regression"
        else:
            uniq = np.unique(y_arr[~np.isnan(y_arr)]) if y_arr.dtype.kind == "f" else np.unique(y_arr)
            if uniq.size == 2:
                target_type = "binary"
            elif uniq.size <= _MAX_LABELS and uniq.size <= 0.05 * n:
                target_type = "multiclass"
            else:
                target_type = "regression"
        # Target imbalance
        if target_type in ("binary", "multiclass"):
            # np.unique handles negative / non-contiguous integer labels; np.bincount(astype(int)) would raise
            # on negative labels and silently allocate up-to-max_label bins on sparse high integer codes.
            counts = np.unique(y_arr, return_counts=True)[1]
            counts = counts[counts > 0]
            target_imbalance = float(counts.min() / counts.sum()) if counts.size else 0.0
        else:
            ystd = float(np.nanstd(y_arr))
            ymean = float(np.nanmean(y_arr))
            target_imbalance = float(ystd / (abs(ymean) + 1e-12))
        # Column-level signals (cheap pass).
        frac_numeric = 1.0
        frac_high_card = 0.0
        frac_constant = 0.0
        max_corr = 0.0
        mean_corr = 0.0
        has_nan = False
        if isinstance(X, pd.DataFrame):
            from pandas.api.types import is_numeric_dtype
            numeric_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
            frac_numeric = len(numeric_cols) / max(1, p)
            hi_card = 0
            const = 0
            corrs = []
            yfloat = y_arr.astype(float)
            for c in numeric_cols:
                col = X[c].to_numpy()
                try:
                    nu = int(X[c].nunique(dropna=True))
                except (TypeError, ValueError):
                    nu = -1
                if nu <= 1:
                    const += 1
                    continue
                # 'high-card' is meaningful only for INTEGER-typed columns where
                # nunique > 0.5*n suggests a hash / ID-like encoding. Continuous
                # floats trivially exceed this threshold and are NOT high-card.
                if X[c].dtype.kind in "iub" and nu > 0.5 * n:
                    hi_card += 1
                col_f = col.astype(float, copy=False)
                mask = np.isfinite(col_f) & np.isfinite(yfloat)
                if mask.sum() < 10:
                    continue
                if np.isnan(col_f).any():
                    has_nan = True
                xstd = np.nanstd(col_f[mask])
                if xstd < 1e-12:
                    continue
                rho = float(np.corrcoef(col_f[mask], yfloat[mask])[0, 1])
                if np.isfinite(rho):
                    corrs.append(abs(rho))
            frac_high_card = hi_card / max(1, p)
            frac_constant = const / max(1, p)
            if corrs:
                max_corr = float(np.max(corrs))
                mean_corr = float(np.mean(corrs))
        else:
            # ndarray path -- assume fully numeric.
            X_arr = np.asarray(X, dtype=float)
            has_nan = bool(np.isnan(X_arr).any())
            yfloat = y_arr.astype(float)
            corrs = []
            for j in range(p):
                col = X_arr[:, j]
                mask = np.isfinite(col) & np.isfinite(yfloat)
                if mask.sum() < 10:
                    continue
                xstd = np.nanstd(col[mask])
                if xstd < 1e-12:
                    continue
                rho = float(np.corrcoef(col[mask], yfloat[mask])[0, 1])
                if np.isfinite(rho):
                    corrs.append(abs(rho))
            if corrs:
                max_corr = float(np.max(corrs))
                mean_corr = float(np.mean(corrs))
        return cls(
            n_rows=n, n_features=p, target_type=target_type,
            target_imbalance=target_imbalance,
            p_n_ratio=p / max(1, n),
            frac_numeric=frac_numeric,
            frac_high_card=frac_high_card,
            frac_constant=frac_constant,
            max_abs_corr_to_y=max_corr,
            mean_abs_corr_to_y=mean_corr,
            has_nan=has_nan,
        )


def suggest_configs(fp: DataFingerprint) -> Tuple[SearchConfig, FIConfig, RobustnessConfig]:
    """Rule-based suggestion (Wave 6 prelim). Returns (search, fi, robustness).

    Rules encode the bench learnings from Wave 1-5:
      - p>>n (p_n_ratio > 1)        : prescreen='univariate_ht', larger init_design
      - p>=1000                     : larger init_design_size, prescreen top_k cap
      - flat curve (max corr < 0.3) : one_se_max rule; convergence_tol active
      - peak curve (max corr > 0.6) : argmax rule
      - severe imbalance (<5%)      : submit_dummy_to_optimizer=False
      - high-card present           : prefer permutation-based importance
      - tiny p (<10)                : init_design_size fixed to 2

    Future: ML classifier on (fp -> winning combo) replaces the if-chain.
    """
    search_kwargs: Dict[str, Any] = {}
    fi_kwargs: Dict[str, Any] = {}
    robust_kwargs: Dict[str, Any] = {}

    # init_design_size
    if fp.n_features < 10:
        search_kwargs["init_design_size"] = 2
    elif fp.n_features >= 100:
        search_kwargs["init_design_size"] = 7
    # else: leave as 'auto'

    # prescreen for high-p
    if fp.p_n_ratio > 1.0 or fp.n_features >= 1000:
        robust_kwargs["prescreen"] = "univariate_ht"
        if fp.n_features >= 5000:
            robust_kwargs["prescreen_top_k"] = max(500, fp.n_features // 10)

    # selection rule based on signal strength (proxy via max |corr|)
    if fp.max_abs_corr_to_y < 0.3:
        # Flat curve - one_se_max default; tol convergence active
        fi_kwargs["n_features_selection_rule"] = "one_se_max"
        search_kwargs["convergence_tol"] = 1e-3
    elif fp.max_abs_corr_to_y > 0.6:
        # Clear signal peak - use argmax for parsimony
        fi_kwargs["n_features_selection_rule"] = "argmax"

    # imbalanced binary on accuracy-style: do not submit dummy
    if fp.target_type == "binary" and fp.target_imbalance < 0.05:
        search_kwargs["submit_dummy_to_optimizer"] = False

    # FI decay for long runs (p>=50 implies user likely runs many iters)
    if fp.n_features >= 50:
        fi_kwargs["fi_decay_rate"] = 0.05

    # NaN-in-X warning suggests permutation rather than coef path
    if fp.has_nan:
        fi_kwargs["importance_getter"] = "permutation"

    # High-cardinality features biases tree importance -> permutation
    if fp.frac_high_card > 0.1:
        fi_kwargs["importance_getter"] = "permutation"

    return (
        SearchConfig(**search_kwargs),
        FIConfig(**fi_kwargs),
        RobustnessConfig(**robust_kwargs),
    )


def explain_suggestion(fp: DataFingerprint) -> str:
    """Human-readable trace of why each knob was picked. For ``auto_tune_decision_``."""
    lines = [
        f"DataFingerprint: n={fp.n_rows}, p={fp.n_features}, p/n={fp.p_n_ratio:.3f}, "
        f"target={fp.target_type}, imbalance={fp.target_imbalance:.3f}, "
        f"max|corr|={fp.max_abs_corr_to_y:.3f}, mean|corr|={fp.mean_abs_corr_to_y:.3f}, "
        f"frac_numeric={fp.frac_numeric:.2f}, frac_high_card={fp.frac_high_card:.2f}, "
        f"frac_constant={fp.frac_constant:.2f}, has_nan={fp.has_nan}",
    ]
    rules: list = []
    if fp.n_features < 10:
        rules.append("p<10 -> init_design_size=2")
    elif fp.n_features >= 100:
        rules.append("p>=100 -> init_design_size=7")
    if fp.p_n_ratio > 1.0 or fp.n_features >= 1000:
        rules.append("p_n>1 or p>=1000 -> prescreen='univariate_ht'")
    if fp.max_abs_corr_to_y < 0.3:
        rules.append("flat curve (max|corr|<0.3) -> rule=one_se_max + convergence_tol=1e-3")
    elif fp.max_abs_corr_to_y > 0.6:
        rules.append("peak curve (max|corr|>0.6) -> rule=argmax")
    if fp.target_type == "binary" and fp.target_imbalance < 0.05:
        rules.append("imbalanced (<5%) -> submit_dummy_to_optimizer=False")
    if fp.n_features >= 50:
        rules.append("p>=50 -> fi_decay_rate=0.05")
    if fp.has_nan or fp.frac_high_card > 0.1:
        rules.append("nan/high-card -> importance_getter='permutation'")
    if not rules:
        rules.append("(no rules fired; using defaults)")
    return "\n".join(lines + [f"  - {r}" for r in rules])
