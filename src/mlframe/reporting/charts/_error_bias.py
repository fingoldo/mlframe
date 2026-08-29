"""Per-feature signed-residual bias: where the model systematically over- or under-predicts.

Carved out of ``error_analysis.py``, which had grown past the house carve band. The other builders there rank
segments by error MAGNITUDE; this one keeps the SIGN, because "the model under-predicts for this segment" and
"the model is wrong for this segment" are different findings, and only the first tells you which way to correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, LinePanelSpec, PanelSpec

from ._error_analysis_shared import (
    DEFAULT_OVERLAY_BINS, DEFAULT_TAIL_FRACTION, _as_float_1d, _pull_columns_at_rows, _resolve_feature_names,
    _row_count,
)
from ._layout import figsize_for_grid, pack_panels

@dataclass(frozen=True)
class ErrorBiasResult:
    """Per-feature OVER/UNDER/MAJORITY overlay figure + the group-mean table.

    ``group_means`` is a small DataFrame indexed by feature, columns OVER/UNDER/MAJORITY (each the group's mean
    feature value). ``group_masks`` are the boolean row selectors so a caller can reuse the tagging.
    """

    figure: FigureSpec
    group_means: Any  # pandas.DataFrame
    group_masks: Dict[str, np.ndarray]


def _signed_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed residual y_true - y_pred: positive => the model UNDER-predicts here, negative => it OVER-predicts.

    ONE convention for the whole figure. Group tagging previously used ``y_pred - y_true`` while the per-segment
    bias readout in the same figure used ``y_true - y_pred``, so the two halves disagreed about which sign meant
    "over" and a comment existed solely to reconcile them. The suptitle already declares this one.
    """
    return np.asarray(_as_float_1d(y_true) - _as_float_1d(y_pred))


def _tag_error_groups(
    signed_err: np.ndarray,
    tail_fraction: float,
) -> Dict[str, np.ndarray]:
    """Split rows into OVER / UNDER / MAJORITY by signed-error quantile.

    The top ``tail_fraction`` of signed errors (most positive) are OVER-estimates, the bottom ``tail_fraction``
    (most negative) UNDER-estimates, the middle is MAJORITY. Quantile cut via ``np.quantile`` (k-way partition,
    O(n)); no full sort.
    """
    finite = np.isfinite(signed_err)
    hi_cut = np.quantile(signed_err[finite], 1.0 - tail_fraction) if finite.any() else np.inf
    lo_cut = np.quantile(signed_err[finite], tail_fraction) if finite.any() else -np.inf
    if hi_cut == lo_cut:
        # A constant signed error makes both tail tests true for every row, so OVER and UNDER each held ALL rows and
        # MAJORITY held none -- three identical densities, which reads as "no error bias" rather than "no variation".
        return {"OVER": np.zeros_like(finite), "UNDER": np.zeros_like(finite), "MAJORITY": finite}
    # `signed_err` is y_true - y_pred, so the MOST NEGATIVE tail is where the model over-predicts.
    over = finite & (signed_err <= lo_cut)
    under = finite & (signed_err >= hi_cut)
    majority = finite & ~over & ~under
    return {"OVER": over, "UNDER": under, "MAJORITY": majority}


def error_bias_per_feature(
    X: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    features: Optional[Sequence[str]] = None,
    max_features: int = 4,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    nbins: int = DEFAULT_OVERLAY_BINS,
    title: str = "Error bias per feature (OVER / UNDER / MAJORITY)",
    seed: int = 0,
) -> ErrorBiasResult:
    """Own Evidently error-bias reimplementation: which feature values drive extreme errors.

    Rows are tagged into the top-``tail_fraction`` signed-error OVER-estimates, the bottom-``tail_fraction``
    UNDER-estimates, and the MAJORITY middle. For each (up to ``max_features``) feature the three groups' value
    distributions are overlaid as density histograms (one LinePanelSpec per feature), and a group-mean table is
    returned. Per-feature binning is via ``np.histogram`` (O(n)); group means via masked sums.
    """
    import pandas as pd

    names = _resolve_feature_names(X, feature_names)
    signed = _signed_error(y_true, y_pred)
    masks = _tag_error_groups(signed, tail_fraction)
    resid_signed = signed

    missing_features: List[str] = []
    if features is not None:
        missing_features = [f for f in features if f not in names]
        sel = [names.index(f) for f in features if f in names]
    else:
        sel = list(range(min(max_features, len(names))))
    missing_note = f"\nrequested features not found, skipped: {', '.join(missing_features)}" if missing_features else ""

    # Densify ONLY the selected columns over all rows -- the overlay touches at most ``max_features`` (default 4) of
    # what can be a several-hundred-column frame, so building the whole dense matrix here just to discard all but
    # ``sel`` is wasted O(n*cols) work. Column pull is bit-identical to ``_resolve_feature_matrix(X)[:, j]``.
    all_rows = np.arange(_row_count(X), dtype=np.int64)
    col_vals = _pull_columns_at_rows(X, sel, all_rows)

    group_colors = {"OVER": "#d62728", "UNDER": "#1f77b4", "MAJORITY": "#7f7f7f"}
    panels: List[PanelSpec] = []
    rows: Dict[str, List[float]] = {g: [] for g in ("OVER", "UNDER", "MAJORITY")}
    feat_index: List[str] = []
    # Track the single worst-bias segment across every feature for the figure title: (|mean signed resid|, signed text).
    global_worst_abs = -1.0
    global_worst_text = ""

    for j in sel:
        col = col_vals[j]
        finite = np.isfinite(col)
        cf = col[finite]
        if cf.size == 0:
            continue
        try:
            edges = np.histogram_bin_edges(cf, bins=nbins)
        except ValueError:
            # A near-constant finite column (e.g. an MLP-engineered feature with ~zero variance) has a range so
            # tiny that numpy can't carve it into ``nbins`` distinct finite-precision edges -- raises "Too many
            # bins for data range" (caught live via a fuzz combo with mlp in the model mix). This one feature's
            # bias panel isn't meaningful anyway (no spread to bin), so skip it rather than aborting the whole
            # diagnostic (best-effort, matches every other panel in this dispatcher).
            continue
        if edges.size < 2 or not np.all(np.diff(edges) > 0):
            # numpy doesn't always raise on a near-constant huge-magnitude column -- it can instead
            # return edges collapsed to fewer distinct floats than requested (zero-width bins), which
            # silently produces NaN/inf densities downstream via divide-by-zero in np.histogram's
            # normalization instead of raising. Skip this feature's panel the same way as above.
            continue
        centers = (edges[:-1] + edges[1:]) / 2.0
        series: List[np.ndarray] = []
        labels: List[str] = []
        cols: List[str] = []
        for g in ("OVER", "UNDER", "MAJORITY"):
            gvals = col[masks[g] & finite]
            dens, _ = np.histogram(gvals, bins=edges, density=True)
            series.append(dens)
            labels.append(g)
            cols.append(group_colors[g])
            rows[g].append(float(gvals.mean()) if gvals.size else float("nan"))
        feat_index.append(names[j])

        # Per-segment signed-residual bias: bin this feature's values, take the mean residual (y_true - y_pred) in each
        # bin. The segment with the largest |mean residual| is the model's worst-bias slice for this feature; its sign
        # tells direction (> 0 UNDER-predict, < 0 OVER-predict). Aggregated O(n) via two bincounts, no per-row Python.
        bin_idx = np.clip(np.digitize(col[finite], edges[1:-1]), 0, len(centers) - 1)
        nbin = len(centers)
        cnt = np.bincount(bin_idx, minlength=nbin).astype(np.float64)
        ssum = np.bincount(bin_idx, weights=resid_signed[finite], minlength=nbin)
        with np.errstate(invalid="ignore", divide="ignore"):
            seg_bias = np.where(cnt > 0, ssum / np.where(cnt > 0, cnt, 1.0), np.nan)
        worst_b = int(np.nanargmax(np.where(np.isfinite(seg_bias), np.abs(seg_bias), -np.inf))) if np.isfinite(seg_bias).any() else -1
        if worst_b >= 0:
            wb = float(seg_bias[worst_b])
            direction = "UNDER-predicts" if wb > 0 else "OVER-predicts"
            seg_lo, seg_hi = float(edges[worst_b]), float(edges[worst_b + 1])
            worst_note = f"worst-bias segment: {names[j]} in [{seg_lo:.3g}, {seg_hi:.3g}] -> model {direction} (mean resid {wb:+.3g})"
            if abs(wb) > global_worst_abs:
                global_worst_abs = abs(wb)
                global_worst_text = f"{names[j]} in [{seg_lo:.3g}, {seg_hi:.3g}] {direction} (resid {wb:+.3g})"
        else:
            worst_note = "worst-bias segment: n/a"

        panels.append(LinePanelSpec(
            x=centers,
            y=tuple(series),
            series_labels=tuple(labels),
            colors=tuple(cols),
            line_styles=("-", "-", "--"),
            title=f"{names[j]} value distribution by error group\n{worst_note}",
            xlabel=names[j],
            ylabel="Density",
        ))

    group_means = pd.DataFrame(
        {g: rows[g] for g in ("OVER", "UNDER", "MAJORITY")},
        index=feat_index,
    )
    if not panels:
        # No usable feature (all-NaN columns, zero features, or none selected); an empty grid would crash the renderer.
        ann = AnnotationPanelSpec(text=f"No usable feature column: every candidate is all-NaN, or none was selected.{missing_note}", title="")
        return ErrorBiasResult(FigureSpec(suptitle=title + missing_note, panels=((ann,),), figsize=(8.0, 3.0)), group_means, masks)
    grid = pack_panels(panels, max_cols=2)
    n_rows = len(grid)
    worst_line = f"Worst-bias segment overall: {global_worst_text}" if global_worst_text else "Worst-bias segment overall: n/a"
    suptitle = f"{title}\nSigned residual = y_true - y_pred; > 0 in a segment => model UNDER-predicts there, < 0 => OVER-predicts.\n" f"{worst_line}{missing_note}"
    fig = FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(max(n_rows, 1), 2, cell_width=6.0, cell_height=4.0),
        caption=(
            f"Rows are split by signed residual (y_true - y_pred) into the bottom {tail_fraction:.0%} where the model "
            f"OVER-predicts, the top {tail_fraction:.0%} where it UNDER-predicts, and the middle MAJORITY. Each panel "
            "overlays those three groups' value distributions for one feature: wherever the tail curves pull away "
            "from the majority, that feature's values are what drive the extreme errors. y is a density, so each "
            "curve integrates to 1 and the three groups' heights are NOT comparable as counts."
        ),
    )
    return ErrorBiasResult(fig, group_means, masks)


__all__ = ["ErrorBiasResult", "error_bias_per_feature"]
