"""Data-driven config suggestion for composite-target discovery (UX helper).

``suggest_discovery_config(df, target_col, feature_cols)`` inspects a frame
CHEAPLY -- on a SAMPLE of rows, never a whole-frame ``to_pandas`` /
materialisation -- and returns a populated ``CompositeTargetDiscoveryConfig``
with sensible, data-derived choices plus a short human-readable ``rationale``
mapping each choice to WHY it was made.

What it inspects and how it steers the config:

* **n rows** -> ``mi_sample_n``.  Tiny frames keep the full train (``None``);
  mid frames keep ``100_000``; very large frames cap to keep the MI screen
  under a minute.
* **monotone / timestamp column** -> ``time_column`` +
  ``time_series_transforms_enabled``.  Reuses ``detect_time_column_candidates``
  (datetime dtype OR strictly-monotone numeric); when one is found the three
  chronological-order transforms become valid (the config validator appends
  them) and the MI screen sorts by time.
* **target right-skew** -> ``signed_power_y`` / ``log_y`` transforms.  A
  strongly right-skewed target (skew above a threshold) benefits from a tail-
  compressing unary y-transform; ``log_y`` is added only when the target is
  strictly positive (its domain), ``signed_power_y`` always (signed-safe).
* **heavy tail** -> ``mi_sample_strategy="stratified_quantile"`` + a boosted
  ``mi_n_strata``.  Heavy-tail targets carry the signal in the rare tail rows;
  per-stratum quotas keep those rows in the MI screen.
* **structural base hints** -> ``dominant_features_hint``.  Reuses
  ``structural_affinity_scores`` (near-affine predictor / low-card integer
  grouping / monotone time index) on the sample to surface OBVIOUS base columns
  so the auto-base ranking starts from a strong prior.

The function only READS the frame and returns a fresh config object; it never
mutates the caller's frame and never copies it.  Everything that needs row
data runs on a bounded sample (``_AUTOCONFIG_SAMPLE_N`` rows).

Cost / cProfile note: the only non-O(1) work is the bounded-sample column pull
(``_extract_column_array(df, col, rows=sample_idx)`` per inspected column) plus
the whole-sample-matrix ``structural_affinity_scores`` pass, both already
vectorised + sample-bounded.  On a 4M-row x 200-col frame the suggestion runs
on a <=20k-row sample, so it is dominated by the per-column gather, not by any
full-frame scan.  No actionable speedup at the call frequency (once per
training run); the detectors it reuses are themselves at the numpy floor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .discovery._structural_hints import structural_affinity_scores
from .discovery.auto_detect import detect_time_column_candidates
from .discovery.screening import _extract_column_array, _is_numeric_column

logger = logging.getLogger(__name__)


# Rows the inspection runs on. Bounded so a 100+ GB frame is never materialised
# -- the structural-hint pass + skew estimate are sample statistics, so a
# representative slice is sufficient and keeps the suggestion sub-second.
_AUTOCONFIG_SAMPLE_N: int = 20_000

# Row-count bands for ``mi_sample_n``. Below the small band the full train is
# cheap enough to screen exactly (None); within the mid band the shipped 100k
# default is adequate; above the large band the screen must cap to stay under a
# minute on 4M+ rows.
_MI_SAMPLE_SMALL_N: int = 50_000  # below this -> full train (mi_sample_n=None)
_MI_SAMPLE_MID_N: int = 500_000  # below this -> 100k sample
_MI_SAMPLE_LARGE: int = 100_000  # the cap used above the mid band

# Target-skew thresholds. ``skew`` here is the standardised third moment of the
# target sample; ``> _SKEW_TRANSFORM_MIN`` is "strongly right-skewed" (a
# tail-compressing unary y-transform helps a linear / neural downstream model).
_SKEW_TRANSFORM_MIN: float = 1.0
# Heavy-tail trigger (mirrors the discovery in-fit auto-boost: skew>2 or
# kurt>5) -> stratified sampling + boosted strata.
_HEAVY_TAIL_SKEW: float = 2.0
_HEAVY_TAIL_KURT: float = 5.0
# Boosted strata count for heavy-tail targets (matches the config's
# ``mi_n_strata_heavy_tail`` shipped default so the suggestion is consistent
# with what discovery would auto-boost to anyway).
_HEAVY_TAIL_N_STRATA: int = 30


def _sample_indices(n_rows: int, sample_n: int, seed: int) -> np.ndarray:
    """Sorted row indices for a bounded inspection sample (full range when the
    frame already fits under ``sample_n``). Sorted so a monotone time column
    keeps its order for the ``detect_time_column_candidates`` monotonicity
    check."""
    if n_rows <= sample_n:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample_n, replace=False))


def _target_skew_kurt(y: np.ndarray) -> Tuple[float, float]:
    """Standardised skew + excess kurtosis of the finite target sample.

    Returns ``(0.0, 0.0)`` for a degenerate (constant / too-small) sample so no
    skew / heavy-tail branch fires on uninformative data. Mirrors the chained-
    multiply moment estimate used in the discovery in-fit heavy-tail boost.
    """
    yf = y[np.isfinite(y)]
    if yf.size < 20:
        return 0.0, 0.0
    std = float(yf.std())
    if std <= 1e-12:
        return 0.0, 0.0
    z = (yf - yf.mean()) / std
    z2 = z * z
    skew = float(np.mean(z2 * z))
    kurt = float(np.mean(z2 * z2) - 3.0)
    return skew, kurt


def suggest_discovery_config(
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    sample_n: int = _AUTOCONFIG_SAMPLE_N,
    seed: int = 0,
    preset: Optional[str] = None,
    **config_overrides: Any,
) -> Tuple[Any, Dict[str, str]]:
    """Inspect ``df`` cheaply and return a populated discovery config + rationale.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Source frame. Read-only; never copied or materialised whole -- only a
        bounded ``sample_n``-row slice of the target + inspected feature columns
        is pulled.
    target_col : str
        Name of the regression target column.
    feature_cols : sequence of str
        Candidate feature columns (the pool discovery would search for bases).
    sample_n : int, default 20_000
        Inspection sample size. The structural / skew detectors are sample
        statistics, so this bounds cost on 100+ GB frames.
    seed : int, default 0
        Seed for the inspection-sample draw (reproducible suggestion).
    preset : {"fast", "thorough"}, optional
        Compose with ``CompositeTargetDiscoveryConfig.preset``: the preset's field bundle overlays
        the data-derived suggestions (an explicit speed/quality intent beats a heuristic), and
        ``**config_overrides`` still win over both. Data-derived fields the preset does not pin
        (time_column / heavy-tail strata / transform additions / dominant_features_hint) survive.
    **config_overrides
        Forwarded to the ``CompositeTargetDiscoveryConfig`` constructor, taking
        precedence over every suggested field (caller wins).

    Returns
    -------
    (config, rationale)
        ``config`` is a ``CompositeTargetDiscoveryConfig`` with ``enabled=True``
        and the data-derived fields populated. ``rationale`` maps each steered
        field name to a one-line human-readable reason.
    """
    # Imported here (not at module top) to avoid a hard import cycle: the config
    # module lives under ``training`` and pulls in the broader configs package.
    from ..configs import CompositeTargetDiscoveryConfig

    feature_cols = list(feature_cols)
    rationale: Dict[str, str] = {}
    suggested: Dict[str, Any] = {"enabled": True}
    # Preset composition: the preset's deliberately-pinned bundle overlays the data-derived
    # suggestions at construction time (see the two construction sites below); overrides win last.
    _preset_fields: Dict[str, Any] = {}
    if preset is not None:
        _preset_fields = CompositeTargetDiscoveryConfig.preset_fields(preset)
        rationale["preset"] = f"'{preset}' preset overlays its pinned fields on the data-derived suggestions: {sorted(_preset_fields)}."

    n_rows = len(df)

    # ---- mi_sample_n: scale to frame size ----------------------------------
    if n_rows < _MI_SAMPLE_SMALL_N:
        suggested["mi_sample_n"] = None
        rationale["mi_sample_n"] = f"n={n_rows} < {_MI_SAMPLE_SMALL_N}: small frame, screen the full " "train exactly (no sampling)."
    elif n_rows < _MI_SAMPLE_MID_N:
        suggested["mi_sample_n"] = _MI_SAMPLE_LARGE
        rationale["mi_sample_n"] = f"n={n_rows}: mid frame, keep the {_MI_SAMPLE_LARGE} default MI " "sample (adequate, sub-minute)."
    else:
        suggested["mi_sample_n"] = _MI_SAMPLE_LARGE
        rationale["mi_sample_n"] = (
            f"n={n_rows} >= {_MI_SAMPLE_MID_N}: large frame, cap the MI sample " f"at {_MI_SAMPLE_LARGE} to keep the screen under a minute."
        )

    # Degenerate frame: no rows / no target -> conservative defaults only.
    if n_rows == 0 or target_col not in _frame_columns(df):
        rationale["enabled"] = "Empty frame or target absent: returning conservative defaults."
        cfg = CompositeTargetDiscoveryConfig(**{**suggested, **_preset_fields, **config_overrides})
        return cfg, rationale

    sample_idx = _sample_indices(n_rows, max(2, int(sample_n)), seed)

    # ---- target skew / heavy tail ------------------------------------------
    y_sample = _extract_column_array(df, target_col, rows=sample_idx)
    skew, kurt = _target_skew_kurt(y_sample)

    base_transforms: List[str] = list(CompositeTargetDiscoveryConfig().transforms)
    added_transforms: List[str] = []

    if skew > _SKEW_TRANSFORM_MIN:
        # signed_power_y is signed-safe (works on any-sign target); log_y needs a
        # strictly-positive domain. Add both only where their domain holds.
        if "signed_power_y" not in base_transforms:
            added_transforms.append("signed_power_y")
        yf = y_sample[np.isfinite(y_sample)]
        strictly_positive = bool(yf.size and float(yf.min()) > 0.0)
        if strictly_positive and "log_y" not in base_transforms:
            added_transforms.append("log_y")
        rationale["transforms"] = (
            f"target skew={skew:.2f} > {_SKEW_TRANSFORM_MIN}: strongly right-"
            "skewed, added tail-compressing y-transform(s) "
            f"{added_transforms} (log_y only when target>0)."
        )

    heavy_tail = abs(skew) > _HEAVY_TAIL_SKEW or kurt > _HEAVY_TAIL_KURT
    if heavy_tail:
        suggested["mi_sample_strategy"] = "stratified_quantile"
        suggested["mi_n_strata"] = _HEAVY_TAIL_N_STRATA
        rationale["mi_sample_strategy"] = (
            f"heavy tail (skew={skew:.2f}, kurt={kurt:.2f}): stratified-quantile "
            f"MI sampling with {_HEAVY_TAIL_N_STRATA} strata so rare tail rows "
            "stay in the screen."
        )

    if added_transforms:
        suggested["transforms"] = base_transforms + added_transforms

    # ---- monotone / timestamp time column ----------------------------------
    # Only inspect columns that exist; restrict candidate scan to the supplied
    # feature pool to avoid flagging an unrelated bookkeeping column. We build a
    # tiny sample-frame view via the detector, which is polars/pandas aware.
    time_candidates = _detect_time_on_sample(df, feature_cols, sample_idx)
    if time_candidates:
        time_col, info = time_candidates[0]
        suggested["time_column"] = time_col
        suggested["time_series_transforms_enabled"] = True
        rationale["time_column"] = (
            f"'{time_col}' looks chronological (" f"{'datetime' if info.get('is_datetime') else 'monotone numeric'}): " "set as time_column."
        )
        rationale["time_series_transforms_enabled"] = (
            "Time column found: enabled the chronological-order transforms "
            "(ewma_residual / rolling_quantile_ratio / frac_diff) and time-"
            "ordered MI screen."
        )

    # ---- structural base hints ---------------------------------------------
    hint_cols, kinds = _structural_base_hints(
        df, target_col, feature_cols, sample_idx, time_col=suggested.get("time_column"),
    )
    if hint_cols:
        suggested["dominant_features_hint"] = hint_cols
        kind_desc = ", ".join(f"{c}={kinds.get(c, '?')}" for c in hint_cols)
        rationale["dominant_features_hint"] = (
            f"structural detectors surfaced obvious base(s): {kind_desc} -- " "seeded as dominant_features_hint so auto-base starts from them."
        )

    cfg = CompositeTargetDiscoveryConfig(**{**suggested, **_preset_fields, **config_overrides})
    return cfg, rationale


def _frame_columns(df: Any) -> List[str]:
    """Column names for a polars or pandas frame."""
    cols = getattr(df, "columns", None)
    if cols is None:
        return []
    return list(cols)


def _detect_time_on_sample(
    df: Any, feature_cols: Sequence[str], sample_idx: np.ndarray,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Run ``detect_time_column_candidates`` over the feature pool on the sample.

    Datetime-dtype columns are detected directly on ``df`` (dtype is frame-level,
    no row pull needed); monotone numeric detection needs ordered rows, so we
    pass the sample-ordered slice. Reuses the shared detector -- no duplication.
    """
    present = [c for c in feature_cols if c in _frame_columns(df)]
    if not present:
        return []
    # The detector reads dtype + a to_numpy on each candidate. To keep it bounded
    # on a 100+ GB frame we hand it a sample-sliced view built per-flavour. For
    # pandas this is an ``.iloc`` view (no copy of untouched columns); for polars
    # a ``.gather`` of the sample rows over the candidate columns only.
    try:
        sample_view = _sample_view(df, present, sample_idx)
    except Exception as _e:  # pragma: no cover - defensive; fall back to no time col
        logger.debug("autoconfig: time-column sample view failed: %s", _e)
        return []
    return detect_time_column_candidates(sample_view, candidate_columns=present)


def _sample_view(df: Any, cols: Sequence[str], sample_idx: np.ndarray) -> Any:
    """Bounded sample-row view over ``cols`` (polars gather / pandas iloc).

    Never copies columns outside ``cols`` and never materialises the full frame.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.select(list(cols))[sample_idx]
    except ImportError:  # pragma: no cover - polars optional
        pass
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        return df.iloc[sample_idx][list(cols)]
    raise TypeError(f"_sample_view: unsupported df type {type(df).__name__}")


def _structural_base_hints(
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    sample_idx: np.ndarray,
    *,
    time_col: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Reuse ``structural_affinity_scores`` to surface obvious base columns.

    Returns ``(ordered_hint_cols, kinds)`` where ``kinds`` maps each hinted
    column to its detected base kind (``linear_residual`` / ``grouped`` /
    ``time``). Only NUMERIC feature columns (excluding the target and any
    detected ``time_col``, which is already wired as ``time_column``) are
    scored, on the bounded sample.
    """
    numeric = [c for c in feature_cols if c != target_col and c != time_col and c in _frame_columns(df) and _is_numeric_column(df, c)]
    if not numeric:
        return [], {}
    try:
        y = _extract_column_array(df, target_col, rows=sample_idx).astype(np.float64, copy=False)
        cols_arrays = [_extract_column_array(df, c, rows=sample_idx).astype(np.float64, copy=False) for c in numeric]
        x_matrix = np.column_stack(cols_arrays)
    except Exception as _e:  # pragma: no cover - defensive
        logger.debug("autoconfig: structural-hint matrix build failed: %s", _e)
        return [], {}

    # ``structural_affinity_scores`` operates on the finite-masked matrix; mask
    # rows where the target or any candidate is non-finite (same contract the
    # discovery scorer callers use).
    finite = np.isfinite(y) & np.isfinite(x_matrix).all(axis=1)
    if finite.sum() < 3:
        return [], {}
    scores, kinds = structural_affinity_scores(
        x_matrix[finite], y[finite], numeric,
    )
    # Keep only columns that actually fired a detector, ordered by score desc.
    fired = [(numeric[j], float(scores[j])) for j in range(len(numeric)) if scores[j] > 0.0]
    fired.sort(key=lambda kv: kv[1], reverse=True)
    hint_cols = [c for c, _ in fired]
    kinds_out = {c: kinds[c] for c in hint_cols if c in kinds}
    return hint_cols, kinds_out
