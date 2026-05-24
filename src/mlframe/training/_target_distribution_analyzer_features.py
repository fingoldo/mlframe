"""Feature-side distribution analyzer carved out of
``mlframe.training._target_distribution_analyzer``: ``FeatureDistributionReport``
dataclass, ``_pairwise_redundant_features`` helper, ``_normalise_X``
input-shape dispatcher, and the public ``analyze_feature_distribution``
function.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training._target_distribution_analyzer import analyze_feature_distribution``
resolves transparently.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ._target_distribution_analyzer import (
    _HIGH_CARDINALITY_MAX,
    _LEAKAGE_CORR_THRESHOLD,
    _LOW_VAR_REL_STD,
    _NAN_FRACTION_THRESHOLD,
    _REDUNDANCY_MAX_NUMERIC_FEATURES,
    _REDUNDANT_CORR_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureDistributionReport:
    """Report from :func:`analyze_feature_distribution`."""

    n_samples: int
    n_features: int
    pathologies: list[str] = field(default_factory=list)
    # Per-feature warning detail: ``{feature_name: [pathology_strings]}``.
    feature_warnings: dict[str, list[str]] = field(default_factory=dict)
    # Suggested drop candidates (near-constant / NaN-heavy / one side of a
    # redundant pair). Operator should review before applying.
    drop_candidates: list[str] = field(default_factory=list)
    # Suspected target-leakage feature names (not auto-actioned).
    leakage_candidates: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    knob_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def _pairwise_redundant_features(
    X_numeric: np.ndarray,
    feature_names: list[str],
    threshold: float = _REDUNDANT_CORR_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Return list of (feature_a, feature_b, |corr|) for pairs above the threshold.

    Uses np.corrcoef on the FULL matrix (O(n_features^2) memory). The caller
    is responsible for capping feature count via _REDUNDANCY_MAX_NUMERIC_FEATURES.
    """
    if X_numeric.shape[1] < 2:
        return []
    # corrcoef along rows -> transpose so each row is a feature.
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X_numeric, rowvar=False)
    pairs: list[tuple[str, str, float]] = []
    n_feats = X_numeric.shape[1]
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            corr = float(C[i, j])
            if not math.isfinite(corr):
                continue
            if abs(corr) >= threshold:
                pairs.append((feature_names[i], feature_names[j], abs(corr)))
    pairs.sort(key=lambda t: -t[2])
    return pairs


def _normalise_X(
    X,
    feature_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (df, numeric_cols, categorical_cols).

    Numpy input gets generic ``f0...fN`` names; pandas keeps its columns.
    Polars input is zero-copy-converted to pandas via ``.to_pandas()`` so
    column dtypes (Float32 / Int64 / Boolean / String) are preserved -- a
    naive ``np.asarray(polars_df)`` collapses everything to object dtype
    and downstream ``is_numeric_dtype`` checks misclassify every column as
    categorical. TVT 2026-05-21 reproduction: the suite called the analyzer
    after _phase_train_val_test_split but BEFORE _phase_fit_pipeline (train_df
    still polars), and the analyzer logged
    ``high_cardinality_categorical(n=25)`` for 25 Float32 numeric features.
    Categorical detection: object / category / pandas string dtype goes to
    ``categorical_cols``; numeric int/float/bool goes to ``numeric_cols``.
    """
    if isinstance(X, pd.DataFrame):
        df = X
    else:
        # Polars-family input handling (2026-05-21 P0 #3 + follow-up). Three
        # shapes need explicit dispatch; the first version of this branch
        # only handled DataFrame and the other two fell through to the
        # numpy path with the same misclassification bug the original fix
        # targeted:
        #   - polars.DataFrame: to_pandas() returns a pd.DataFrame; canonical.
        #   - polars.LazyFrame: NO to_pandas method; must collect() first then
        #     to_pandas(). Without explicit handling np.asarray hits a 0-d
        #     object array and every column reads as categorical.
        #   - polars.Series: to_pandas() returns a pd.Series (NOT DataFrame),
        #     downstream df.columns AttributeErrors. Convert to a 1-column
        #     DataFrame.
        _module = type(X).__module__ if X is not None else ""
        _is_polars = isinstance(_module, str) and _module.startswith("polars")
        df = None
        if _is_polars:
            _typename = type(X).__name__
            if _typename == "LazyFrame":
                # Materialise to DataFrame then convert. LazyFrame has neither
                # to_pandas nor columns directly, so caller must accept the
                # collect() cost; a LazyFrame walked through analyze_feature_
                # distribution is going to be materialised anyway downstream.
                try:
                    df = X.collect().to_pandas()
                except Exception:
                    df = None
            elif _typename == "Series":
                # 1-D series -> single-column frame. Use the series name if
                # caller didn't supply feature_names.
                _name = (feature_names[0] if feature_names else (getattr(X, "name", None) or "f0"))
                try:
                    df = pd.DataFrame({_name: X.to_pandas()})
                except Exception:
                    df = None
            else:
                _to_pandas = getattr(X, "to_pandas", None)
                if callable(_to_pandas):
                    df = _to_pandas()
        if df is None:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=feature_names)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for c in df.columns:
        s = df[c]
        # pandas <2 has ``object`` for strings; >=2 has ArrowExtension or StringDtype.
        # Treat anything non-numeric as categorical for analysis purposes.
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric_cols.append(str(c))
        elif pd.api.types.is_bool_dtype(s):
            # Bools coalesce into numeric for std-based checks (cast to int8).
            numeric_cols.append(str(c))
        else:
            categorical_cols.append(str(c))
    # E3.2 (2026-05-21): post-conversion sanity check. If the dataframe has any
    # numeric columns but ALL of them were classified as categorical (or vice
    # versa), the input shape almost certainly broke the duck-typing dispatch
    # above (e.g. a future input form we forgot to handle landed in
    # np.asarray with object dtype). Loud-fail rather than silent
    # misclassification (the TVT-2026-05-21 P0 #3 prod symptom was exactly
    # this -- 25 Float32 columns reported as ``high_cardinality_categorical``;
    # a defensive assertion would have caught it at the analyzer call site
    # before the rest of the suite ran).
    _n_cols_total = len(df.columns)
    if _n_cols_total > 0 and not numeric_cols and not categorical_cols:
        # Pure-empty classification under non-empty frame: definitely a bug.
        raise ValueError(
            f"_normalise_X: produced 0 numeric AND 0 categorical from {_n_cols_total} columns. "
            f"Input type was {type(X).__name__}; check the dispatch above."
        )
    # If EVERY column ended up categorical AND the input was numpy-shaped, that's the
    # original misclassification path. Fire the WARN only when the SOURCE input actually
    # carried numeric polars / numpy dtypes that we lost during dispatch -- if the polars
    # frame was genuinely all-String/all-Categorical (e.g. hypothesis-generated text
    # columns), all-cat classification is the CORRECT outcome and a WARN is pure noise.
    if (
        _n_cols_total >= 2
        and not numeric_cols
        and len(categorical_cols) == _n_cols_total
        and not isinstance(X, pd.DataFrame)
    ):
        # Probe the source for at least one numeric polars dtype; if none, the
        # all-cat classification is correct and we should stay silent. For
        # numpy ndarray inputs we can't probe per-column dtype reliably (the
        # array has a single dtype which may legitimately be object for a
        # broken-dispatch shape the test layer wants to surface), so default
        # to warning -- preferring false positive over silent miss.
        _source_had_numeric = False
        try:
            if "polars" in type(X).__module__:
                import polars as _pl
                _numeric_pl_dtypes = (
                    _pl.Float32, _pl.Float64,
                    _pl.Int8, _pl.Int16, _pl.Int32, _pl.Int64,
                    _pl.UInt8, _pl.UInt16, _pl.UInt32, _pl.UInt64,
                    _pl.Boolean,
                )
                _schema = getattr(X, "schema", None) or {}
                for _src_dt in _schema.values():
                    if isinstance(_src_dt, _numeric_pl_dtypes) or _src_dt in _numeric_pl_dtypes:
                        _source_had_numeric = True
                        break
            else:
                # Non-polars, non-pandas -- e.g. raw numpy ndarray. We can't
                # tell whether the user expected numeric or not; warn so
                # broken-dispatch shapes (test layer's intent) still surface.
                _source_had_numeric = True
        except Exception:
            # If the probe itself fails, assume there could be numeric data
            # and surface the WARN -- prefer false positive over silent miss.
            _source_had_numeric = True
        if _source_had_numeric:
            logger.warning(
                "_normalise_X: classified ALL %d columns as categorical from %s input "
                "despite source having numeric dtypes. This was the P0 #3 misclassification "
                "symptom on polars frames; verify the dispatch handled your input shape correctly.",
                _n_cols_total, type(X).__name__,
            )
    return df, numeric_cols, categorical_cols


def analyze_feature_distribution(
    X,
    y: Optional[np.ndarray] = None,
    *,
    feature_names: Optional[list[str]] = None,
    target_type: Literal["regression", "classification", "auto"] = "auto",
    low_variance_rel_std: float = _LOW_VAR_REL_STD,
    redundant_corr_threshold: float = _REDUNDANT_CORR_THRESHOLD,
    high_cardinality_max: int = _HIGH_CARDINALITY_MAX,
    nan_fraction_threshold: float = _NAN_FRACTION_THRESHOLD,
    leakage_corr_threshold: float = _LEAKAGE_CORR_THRESHOLD,
    redundancy_max_numeric_features: int = _REDUNDANCY_MAX_NUMERIC_FEATURES,
) -> FeatureDistributionReport:
    """Inspect the FEATURE matrix and (optionally) target; surface pathologies.

    Each detector is documented in the module docstring. The function never
    mutates X / y. Recommendations are observational; the suite caller decides
    whether to action them.
    """
    df, numeric_cols, categorical_cols = _normalise_X(X, feature_names=feature_names)
    n_samples = int(df.shape[0])
    n_features = int(df.shape[1])
    diagnostics: dict[str, Any] = {
        "n_samples": n_samples, "n_features": n_features,
        "n_numeric": len(numeric_cols), "n_categorical": len(categorical_cols),
    }
    pathologies: list[str] = []
    feature_warnings: dict[str, list[str]] = {}
    drop_candidates: list[str] = []
    leakage_candidates: list[str] = []
    knob_overrides: dict[str, dict[str, Any]] = {}

    if n_samples < 30:
        # Too small for any reliable detection; return early.
        return FeatureDistributionReport(
            n_samples=n_samples, n_features=n_features,
            pathologies=["insufficient_samples_n<30"],
            diagnostics=diagnostics,
        )

    def _add_warning(col: str, msg: str) -> None:
        feature_warnings.setdefault(col, []).append(msg)

    # --- numeric: low-variance + nan fraction ---
    # bench-attempt-rejected (2026-05-21, 200k rows / 15 cols): tried
    # materialising df[numeric_cols] -> (N, F) block once and reducing via
    # np.nanmean / np.nanstd / np.isfinite axis=0 instead of the per-column
    # loop below. Full-pass time unchanged (~270 ms), y-skipping path got
    # ~15 ms SLOWER. Root cause: np.nanmean and np.nanstd each do their own
    # internal NaN scan, so vectorising on top of an isfinite mask scans the
    # block 3x instead of the loop's 2x; pandas BlockManager extraction also
    # has overhead vs per-column .to_numpy. Keep the per-column path.
    low_var_features: list[str] = []
    nan_heavy_features: list[str] = []
    for c in numeric_cols:
        col = df[c].to_numpy()
        col_f = np.asarray(col, dtype=np.float64)
        nan_mask = ~np.isfinite(col_f)
        nan_frac = float(nan_mask.mean()) if col_f.size > 0 else 0.0
        if nan_frac >= nan_fraction_threshold:
            nan_heavy_features.append(c)
            _add_warning(c, f"nan_fraction={nan_frac:.2f} >= {nan_fraction_threshold}")
            drop_candidates.append(c)
            continue  # Skip further stats on a NaN-dominated feature.
        finite = col_f[~nan_mask]
        if finite.size < 2:
            _add_warning(c, "insufficient_finite_values")
            drop_candidates.append(c)
            continue
        mu = float(np.mean(finite))
        sd = float(np.std(finite))
        rel = abs(sd) / (abs(mu) + 1e-9) if abs(mu) > 1e-9 else (sd if sd > 0 else 0.0)
        if sd <= 0.0 or rel < low_variance_rel_std:
            low_var_features.append(c)
            _add_warning(c, f"low_variance(rel_std={rel:.2e})")
            drop_candidates.append(c)
    if low_var_features:
        pathologies.append(f"low_variance_features(n={len(low_var_features)})")
        diagnostics["low_variance_features"] = list(low_var_features)
    if nan_heavy_features:
        pathologies.append(f"nan_heavy_features(n={len(nan_heavy_features)})")
        diagnostics["nan_heavy_features"] = list(nan_heavy_features)
        # Recommend explicit NaN strategy at the preprocessing layer.
        knob_overrides.setdefault("preprocessing_config", {})["review_nan_strategy"] = True

    # --- categorical: high cardinality ---
    high_card_features: list[str] = []
    for c in categorical_cols:
        n_unique = int(df[c].nunique(dropna=False))
        if n_unique > high_cardinality_max:
            high_card_features.append(c)
            _add_warning(c, f"high_cardinality(n_unique={n_unique} > {high_cardinality_max})")
    if high_card_features:
        pathologies.append(f"high_cardinality_categorical(n={len(high_card_features)})")
        diagnostics["high_cardinality_features"] = list(high_card_features)
        # Recommend a target / hashing encoder over one-hot (preprocessing layer hint).
        knob_overrides.setdefault("preprocessing_config", {})["prefer_high_cardinality_encoder"] = True

    # --- redundant pairs (numeric only; skip the dropped low-var/nan-heavy set) ---
    candidate_numeric = [c for c in numeric_cols if c not in low_var_features and c not in nan_heavy_features]
    if len(candidate_numeric) > redundancy_max_numeric_features:
        diagnostics["redundancy_skipped"] = (
            f"n_numeric={len(candidate_numeric)} > cap={redundancy_max_numeric_features}; "
            "pairwise correlation O(n^2) would be too costly. Lower the threshold via "
            "redundancy_max_numeric_features or pre-filter."
        )
    elif len(candidate_numeric) >= 2:
        # Fill NaNs with column means for the corrcoef pass so a sparse NaN row
        # doesn't poison every correlation in its row/column. We're not
        # imputing the data downstream, just computing redundancy.
        sub = df[candidate_numeric].to_numpy(dtype=np.float64, na_value=np.nan)
        # pandas / polars `.to_numpy()` may return a zero-copy view into the
        # underlying ArrowExtensionArray buffer, which is read-only on
        # pyarrow >=18. The in-place NaN fill below would crash with
        # ``ValueError: assignment destination is read-only`` on those
        # versions; force a writable copy when the view is not writable.
        if not sub.flags.writeable:
            sub = np.array(sub, copy=True)
        col_means = np.nanmean(sub, axis=0)
        # Vectorised mean fill -- np.take_along_axis would be overkill here.
        for j in range(sub.shape[1]):
            mask = ~np.isfinite(sub[:, j])
            if mask.any():
                sub[mask, j] = col_means[j] if math.isfinite(col_means[j]) else 0.0
        pairs = _pairwise_redundant_features(sub, candidate_numeric, threshold=redundant_corr_threshold)
        if pairs:
            pathologies.append(f"redundant_feature_pairs(n={len(pairs)})")
            diagnostics["redundant_feature_pairs"] = [
                {"a": a, "b": b, "corr": c} for a, b, c in pairs[:50]  # cap log to top 50
            ]
            for a, b, c in pairs:
                _add_warning(a, f"redundant_with({b}, corr={c:.3f})")
                _add_warning(b, f"redundant_with({a}, corr={c:.3f})")

    # --- target leakage (only if y supplied) ---
    if y is not None and len(candidate_numeric) > 0:
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.size == n_samples and y_arr.dtype.kind in ("f", "i", "u", "b"):
            # pandas.DataFrame.corrwith vectorises pairwise-complete-obs
            # correlation across all columns in one C-level call: ~65 ms vs
            # the prior per-column np.corrcoef loop's ~145 ms at 200k rows /
            # 15 cols, and bit-exact -- it builds the per-column finite mask
            # internally instead of imputing NaN cells to the column mean
            # (which would dilute the correlation enough to drop legitimate
            # leakage below the 0.99 threshold on combos with sparse NaN).
            y_series = pd.Series(y_arr, index=df.index)
            try:
                corrs = df[candidate_numeric].corrwith(y_series, drop=False)
            except Exception:
                # Object-dtype mix or other corrwith refusal — fall through
                # to nothing rather than crash the analyzer.
                corrs = pd.Series(dtype=np.float64)
            for c, corr_val_raw in corrs.items():
                if not pd.notna(corr_val_raw):
                    continue
                corr_val = float(corr_val_raw)
                if abs(corr_val) >= leakage_corr_threshold:
                    leakage_candidates.append(c)
                    _add_warning(c, f"suspected_target_leakage(corr_with_y={corr_val:.4f})")
        if leakage_candidates:
            pathologies.append(f"suspected_target_leakage(n={len(leakage_candidates)})")
            diagnostics["leakage_candidates"] = list(leakage_candidates)

    return FeatureDistributionReport(
        n_samples=n_samples, n_features=n_features,
        pathologies=pathologies,
        feature_warnings=feature_warnings,
        drop_candidates=drop_candidates,
        leakage_candidates=leakage_candidates,
        diagnostics=diagnostics,
        knob_overrides=knob_overrides,
    )
