"""Per-category-level discriminability screen for a binary target (Weight-of-Evidence magnitude).

Derived from Dyakonov's PZAD ``case_sdsj`` "gender coefficient" idea (rank category levels by ``|m - f| / (m + f)``, how
strongly a level tilts one class vs the other) and generalized to a proper supervised screen: for each level ``L`` of each
categorical feature, ``p = P(y=1 | level==L)`` against the overall ``base = P(y=1)``, and the discriminability is the
Weight-of-Evidence magnitude ``|WoE| = |ln( (p/(1-p)) / (base/(1-base)) )|`` with Laplace smoothing. Levels below a minimum
support floor are dropped (and the drop COUNT is logged, never silently), so a rare level's noisy rate cannot top the ranking.
The headline output is a horizontal bar of the top ``feature=level`` cells by ``|WoE|``, signed so a reader sees which levels
push toward the positive class (WoE > 0) vs the negative class (WoE < 0), with a reference line at WoE = 0.

RAM: never copies the frame. Only the individual categorical column views are pulled (as integer codes), one at a time; on a
huge frame the count pass runs on a bounded seeded row subsample (``_COUNT_SUBSAMPLE_CAP``) since a bincount over codes is cheap
and 200k rows already pin every level's rate tightly.

cProfile verdict (see ``_benchmarks/profile_category_discriminability.py``): the hot kernel is ``level_woe``'s per-row count
pass; the njit ``_level_counts_njit`` beats the two-``np.bincount`` numpy fallback 5.87x at n=100k (0.123 vs 0.721 ms) and
8.46x at n=1M (1.42 vs 12.04 ms) at 50 levels, so it is the default (numpy fallback kept for the no-numba env). The WoE itself
is a vectorized O(n_levels) close. Full-table cProfile at n=1M (subsampled to the 200k count cap) is ~43 ms end to end, split
across the count pass, the raw-rate bincount, and the top_k sort; no other actionable hotspot -- the per-feature
``astype("category")`` codes build is pandas C and is bounded by the high-cardinality skip (>1000 levels) + column-at-a-time
iteration.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.spec import BarPanelSpec, FigureSpec

logger = logging.getLogger(__name__)

# A bincount over codes is cheap and 200k rows pin every level's rate tightly, so cap the count pass here on huge frames.
_COUNT_SUBSAMPLE_CAP: int = 200_000
# Columns with more levels than this are treated as identifiers / free text, not categoricals worth a per-level screen.
_MAX_CARDINALITY: int = 1000
# Positive-class color (WoE > 0, level tilts toward y=1) vs negative-class color (WoE < 0), used to sign the bars.
_POS_COLOR: str = "#2ca02c"
_NEG_COLOR: str = "#d62728"


try:
    import numba

    @numba.njit(cache=True)
    def _level_counts_njit(codes: np.ndarray, y: np.ndarray, n_levels: int):
        # One row pass accumulating positive-count + total-count per level; codes < 0 (missing / NaN category) are skipped.
        pos = np.zeros(n_levels, dtype=np.float64)
        tot = np.zeros(n_levels, dtype=np.float64)
        for i in range(codes.shape[0]):
            c = codes[i]
            if c < 0:
                continue
            pos[c] += y[i]
            tot[c] += 1.0
        return pos, tot

    _HAS_NUMBA = True
except Exception:  # numba unavailable: two-bincount numpy fallback (bit-identical accumulation, just slower).
    _HAS_NUMBA = False

    def _level_counts_njit(codes: np.ndarray, y: np.ndarray, n_levels: int):  # type: ignore[misc]
        keep = codes >= 0
        kc = codes[keep]
        tot = np.bincount(kc, minlength=n_levels).astype(np.float64)
        pos = np.bincount(kc, weights=y[keep], minlength=n_levels)
        return pos, tot


def level_woe(
    level_codes: np.ndarray,
    y: np.ndarray,
    n_levels: int,
    base_rate: float,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weight-of-Evidence magnitude driver: return ``(woe, count)`` per level of one categorical feature.

    ``woe[L] = ln( (p_L/(1-p_L)) / (base/(1-base)) )`` with Laplace-smoothed ``p_L = (pos_L + alpha) / (tot_L + 2*alpha)``; the
    sign is kept (positive => level tilts toward ``y=1``). ``count`` is the raw per-level total. The per-row count pass is the
    hot kernel (njit; numpy-bincount fallback). Missing-category codes (< 0, e.g. pandas ``cat.codes`` of NaN) are skipped.
    """
    codes = np.ascontiguousarray(level_codes, dtype=np.int64)
    yf = np.ascontiguousarray(y, dtype=np.float64)
    nl = int(n_levels)
    if nl <= 0:
        return np.zeros(0), np.zeros(0)
    pos, tot = _level_counts_njit(codes, yf, nl)
    p = (pos + alpha) / (tot + 2.0 * alpha)
    base = min(max(float(base_rate), 1e-12), 1.0 - 1e-12)
    base_logit = np.log(base / (1.0 - base))
    woe = np.log(p / (1.0 - p)) - base_logit
    return woe, tot


def _iter_categorical_columns(X: Any, features: Optional[Sequence[str]]):
    """Yield ``(feature_name, codes, labels)`` for each categorical column: caller-provided ``features`` or auto-detected.

    Codes are the integer category codes (missing => -1); labels are the level names. High-cardinality (> ``_MAX_CARDINALITY``)
    and non-categorical columns are skipped so an identifier / continuous column never enters the per-level screen.
    """
    import pandas as pd

    if features is not None:
        cols = list(features)
    else:
        cols = [c for c in X.columns if X[c].dtype == object or isinstance(X[c].dtype, pd.CategoricalDtype)]
    for col in cols:
        s = X[col]
        cat = s if isinstance(s.dtype, pd.CategoricalDtype) else s.astype("category")
        labels = list(cat.cat.categories)
        if len(labels) < 1 or len(labels) > _MAX_CARDINALITY:
            if len(labels) > _MAX_CARDINALITY:
                logger.info("category_discriminability: skipped high-cardinality column %r (%d levels)", col, len(labels))
            continue
        codes = np.ascontiguousarray(cat.cat.codes.to_numpy(), dtype=np.int64)
        yield str(col), codes, labels


def category_discriminability_table(
    X: Any,
    y: np.ndarray,
    features: Optional[Sequence[str]] = None,
    *,
    top_k: int = 15,
    min_support: int = 30,
    alpha: float = 0.5,
) -> List[Tuple[str, str, float, int, float]]:
    """Rank ``(feature, level)`` cells by ``|WoE|``: return the top_k as ``(feature, level_label, woe, support, p_rate)``.

    Iterates the categorical columns (caller ``features`` or auto-detected object / category dtype), pulls each as codes, and
    scores every level with :func:`level_woe`. Levels below ``min_support`` are dropped and the total drop count is logged (not
    silently discarded). ``p_rate`` is the raw (unsmoothed) ``P(y=1 | level)``. On a huge frame the count pass runs on a bounded
    seeded row subsample so the pass stays RAM-safe on 100+ GB frames.
    """
    y = np.ascontiguousarray(np.asarray(y), dtype=np.float64)
    n = y.shape[0]
    row_idx = None
    if n > _COUNT_SUBSAMPLE_CAP:
        rng = np.random.default_rng(0)
        row_idx = rng.choice(n, size=_COUNT_SUBSAMPLE_CAP, replace=False)
        row_idx.sort()
        y_use = y[row_idx]
    else:
        y_use = y
    base_rate = float(y_use.mean()) if y_use.size else float("nan")

    rows: List[Tuple[str, str, float, int, float]] = []
    dropped = 0
    for name, codes, labels in _iter_categorical_columns(X, features):
        if row_idx is not None:
            codes = np.ascontiguousarray(codes[row_idx])
        n_levels = len(labels)
        woe, tot = level_woe(codes, y_use, n_levels, base_rate, alpha=alpha)
        keep = codes >= 0
        pos = np.bincount(codes[keep], weights=y_use[keep], minlength=n_levels)
        for lvl in range(n_levels):
            support = int(tot[lvl])
            if support < min_support:
                if support > 0:
                    dropped += 1
                continue
            p_rate = float(pos[lvl] / tot[lvl]) if tot[lvl] > 0 else base_rate
            rows.append((name, str(labels[lvl]), float(woe[lvl]), support, p_rate))

    if dropped:
        logger.info("category_discriminability: dropped %d levels below min_support=%d", dropped, min_support)

    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows[: max(1, int(top_k))]


def category_discriminability_panel(
    X: Any,
    y: np.ndarray,
    features: Optional[Sequence[str]] = None,
    *,
    top_k: int = 15,
    min_support: int = 30,
    alpha: float = 0.5,
) -> BarPanelSpec:
    """Horizontal signed-WoE bar of the top_k ``feature=level`` cells (green => tilts to y=1, red => to y=0), zero line at WoE=0."""
    rows = category_discriminability_table(X, y, features, top_k=top_k, min_support=min_support, alpha=alpha)
    if not rows:
        return BarPanelSpec(
            categories=("(no level above min_support)",),
            values=np.array([0.0]),
            title="Category discriminability (|WoE|)",
            orientation="horizontal",
        )
    cats = tuple(f"{feat}={lbl}  (n={support:_}, p={p_rate:.2f})" for feat, lbl, _woe, support, p_rate in rows)
    vals = np.array([r[2] for r in rows], dtype=np.float64)
    # Per-bar signed color: the bar direction already encodes the sign (positive extends right of the 0 line), the color
    # reinforces which class the level tilts toward. Kept as a per-bar tuple so a signed-aware renderer can color each bar.
    colors = tuple(_POS_COLOR if v >= 0.0 else _NEG_COLOR for v in vals)
    return BarPanelSpec(
        categories=cats,
        values=vals,
        title="Category discriminability (signed |WoE|; green=>y=1, red=>y=0; label = support n + P(y=1|level))",
        xlabel="Weight of Evidence  ln[ (p/(1-p)) / (base/(1-base)) ]",
        ylabel="feature=level",
        orientation="horizontal",
        colors=colors,
        hline=(0.0, "black", "WoE = 0 (base rate)"),
    )


def compose_category_discriminability_figure(
    X: Any,
    y: np.ndarray,
    features: Optional[Sequence[str]] = None,
    *,
    top_k: int = 15,
    min_support: int = 30,
    alpha: float = 0.5,
    suptitle: str = "Category discriminability (|WoE|)",
) -> FigureSpec:
    """One-panel FigureSpec wrapping :func:`category_discriminability_panel`."""
    panel = category_discriminability_panel(X, y, features, top_k=top_k, min_support=min_support, alpha=alpha)
    return FigureSpec(suptitle=suptitle, panels=((panel,),), figsize=(10.0, max(5.0, 0.5 * len(panel.categories) + 2.0)))


__all__ = [
    "level_woe",
    "category_discriminability_table",
    "category_discriminability_panel",
    "compose_category_discriminability_figure",
]
