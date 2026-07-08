"""Cross-split OVERFITTING panel: one model's headline metrics across train / val / test (+OOF), side by side.

``compose_split_comparison_figure`` answers the first question a data scientist asks of any model -- "how big is the
train-test gap?" -- which the per-split model card cannot show because it only ever describes ONE split in isolation.
This view puts the splits next to each other: a grouped-bar panel of the task-appropriate headline metrics (one bar
per split per metric), a delta table of the train->val, val->test and train->test change per metric, and a single
traffic-light OVERFIT verdict driven by the degradation on the headline discrimination metric.

Classification headline: ROC_AUC / PR_AUC / ECE / Brier / KS. Regression headline: RMSE / MAE / R2 / bias. Metrics are
computed once per split through the SAME kernels the model card uses (``_classification_metrics`` over one ``_ScoreSort``,
``_regression_metrics``); raw arrays are subsampled per split before the metric pass so a 100M-row split stays cheap. A
split with only one class / no finite pairs is annotated rather than faked, and missing splits simply do not appear.

The overfit rule is deliberately simple and stated: for classification a train->test ROC_AUC drop above a threshold
fires RED (large) / AMBER (moderate); for regression a test RMSE materially larger than train RMSE (ratio) fires the
same. The verdict object carries the gap + reason so triage code and tests can read it without rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid
from mlframe.reporting.charts.binary import _ScoreSort, _finite_binary
from mlframe.reporting.charts.model_card import _classification_metrics, _regression_metrics
from mlframe.reporting.charts.regression import _finite_pair
from mlframe.reporting.spec import AnnotationPanelSpec, BarPanelSpec, FigureSpec, PanelSpec

# Canonical split order so train always reads leftmost and the eye walks train -> val -> test -> oof. Any other split
# name is appended after these in first-seen order.
_SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test", "oof")

# Headline metrics per task: (display name, key in the metric dict, higher_is_better). The first entry is the
# discrimination metric the overfit verdict is built on.
_CLF_HEADLINE: Tuple[Tuple[str, str, bool], ...] = (
    ("ROC_AUC", "ROC_AUC", True), ("PR_AUC", "PR_AUC", True),
    ("KS", "KS", True), ("ECE", "ECE", False), ("Brier", "Brier", False),
)
_REG_HEADLINE: Tuple[Tuple[str, str, bool], ...] = (
    ("R2", "R2", True), ("RMSE", "RMSE", False), ("MAE", "MAE", False), ("bias", "bias", False),
)

# Overfit thresholds. Classification: a train->test ROC_AUC drop is the canonical overfit signature. >0.10 is a serious
# gap (RED); 0.03-0.10 is worth a flag (AMBER); below that the model generalizes (GREEN). Regression mirrors it on the
# test/train RMSE ratio (test error inflated relative to train): >1.50x RED, 1.15-1.50x AMBER.
AUC_GAP_RED: float = 0.10
AUC_GAP_AMBER: float = 0.03
RMSE_RATIO_RED: float = 1.50
RMSE_RATIO_AMBER: float = 1.15

# Per-split subsample cap before the metric pass (metrics are stable far below this; the cap bounds the O(n log n) sort).
_METRIC_SUBSAMPLE: int = 200_000

# One stable color per split so the same split is the same color across every metric group.
_SPLIT_COLORS: Dict[str, str] = {
    "train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c", "oof": "#9467bd",
}
_FALLBACK_COLORS: Tuple[str, ...] = ("#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


@dataclass(frozen=True)
class OverfitVerdict:
    """The cross-split overfit traffic-light + the gap that drove it (read by tests / triage, not just drawn)."""

    color: str  # "green" / "amber" / "red"
    label: str  # short headline, e.g. "GENERALIZES" / "MILD OVERFIT" / "OVERFIT"
    reason: str  # human-readable justification incl. the headline gap
    gap: float  # the headline degradation (train->test AUC drop, or test/train RMSE ratio)
    metric: str  # the metric the gap is measured on


def _order_splits(names: Sequence[str]) -> List[str]:
    """Canonical-then-first-seen split ordering so train/val/test/oof read left-to-right."""
    present = list(names)
    ordered = [s for s in _SPLIT_ORDER if s in present]
    ordered.extend(s for s in present if s not in _SPLIT_ORDER)
    return ordered


def _split_color(name: str, idx: int) -> str:
    """Look up a split's chart color by its canonical name (train/val/test), falling back to a cycling palette entry keyed on position for any unrecognized split name."""
    return _SPLIT_COLORS.get(name.lower(), _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _subsample(*arrays: np.ndarray, cap: int, seed: int) -> Tuple[np.ndarray, ...]:
    """Aligned per-split subsample to ``cap`` rows (shared index across the passed arrays); identity below the cap."""
    n = arrays[0].shape[0]
    if n <= cap:
        return arrays
    idx = np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))
    return tuple(a[idx] for a in arrays)


def _metrics_for_split(
    task: str, entry: Mapping[str, Any], threshold: float, subsample: int, seed: int,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Compute one split's headline metrics, reusing the model-card kernels. Returns (metrics, annotation).

    ``annotation`` is non-None only on a degenerate split (single class / no finite pairs); ``metrics`` is then None.
    Precomputed metrics passed via ``entry["metrics"]`` short-circuit the raw-array path (no recompute, no kernels).
    """
    pre = entry.get("metrics")
    if pre:
        return {str(k): float(v) for k, v in pre.items()}, None

    if task in ("classification", "binary"):
        score = entry.get("y_score")
        if score is None:
            score = entry.get("y_pred")
        if score is None or entry.get("y_true") is None:
            return None, "no y_true / y_score"
        yt, ys = _finite_binary(entry["y_true"], score)
        if yt.size == 0:
            return None, "no finite (label, score) pairs"
        if (yt == 1).sum() == 0 or (yt == 0).sum() == 0:
            return None, "single class -- discrimination undefined"
        yt, ys = _subsample(yt, ys, cap=subsample, seed=seed)
        sort = _ScoreSort(yt, ys)
        return _classification_metrics(sort, yt, ys, threshold), None

    if task == "regression":
        pred = entry.get("y_pred")
        if pred is None:
            pred = entry.get("y_score")
        if pred is None or entry.get("y_true") is None:
            return None, "no y_true / y_pred"
        yt, yp = _finite_pair(entry["y_true"], pred)
        if yt.size == 0:
            return None, "no finite (y_true, y_pred) pairs"
        yt, yp = _subsample(yt, yp, cap=subsample, seed=seed)
        m = _regression_metrics(yt, yp)
        m["y_std"] = float(np.std(yt))  # carried for the 0-1 quality normalization of RMSE/MAE/|bias| bars
        return m, None

    raise ValueError(f"unknown task {task!r}; expected 'classification'/'binary'/'regression'")


def _overfit_verdict(task: str, per_metrics: Mapping[str, Dict[str, float]]) -> OverfitVerdict:
    """Traffic-light overfit verdict from the train->test degradation on the headline discrimination metric.

    Needs both 'train' and 'test' present; if either is missing it falls back to the widest available pair
    (first vs last split in canonical order) so a train+oof or val+test pairing still gets a verdict.
    """
    ordered = _order_splits(list(per_metrics.keys()))
    lo = "train" if "train" in per_metrics else (ordered[0] if ordered else None)
    hi = "test" if "test" in per_metrics else (ordered[-1] if ordered else None)
    if lo is None or hi is None or lo == hi:
        return OverfitVerdict("green", "N/A", "need >= 2 splits to measure a train-test gap", 0.0, "")

    if task in ("classification", "binary"):
        a_train = per_metrics[lo].get("ROC_AUC", float("nan"))
        a_test = per_metrics[hi].get("ROC_AUC", float("nan"))
        gap = float(a_train - a_test)
        if not np.isfinite(gap):
            return OverfitVerdict("green", "N/A", "ROC_AUC missing on a split", 0.0, "ROC_AUC")
        if gap >= AUC_GAP_RED:
            color, label = "red", "OVERFIT"
        elif gap >= AUC_GAP_AMBER:
            color, label = "amber", "MILD OVERFIT"
        else:
            color, label = "green", "GENERALIZES"
        reason = f"{lo}->{hi} ROC_AUC drop {gap:+.3f} ({a_train:.3f} -> {a_test:.3f})"
        return OverfitVerdict(color, label, reason, gap, "ROC_AUC")

    r_train = per_metrics[lo].get("RMSE", float("nan"))
    r_test = per_metrics[hi].get("RMSE", float("nan"))
    ratio = float(r_test / r_train) if r_train > 0 else float("inf")
    if not np.isfinite(ratio):
        return OverfitVerdict("green", "N/A", "RMSE undefined on a split (zero train error?)", 0.0, "RMSE")
    if ratio >= RMSE_RATIO_RED:
        color, label = "red", "OVERFIT"
    elif ratio >= RMSE_RATIO_AMBER:
        color, label = "amber", "MILD OVERFIT"
    else:
        color, label = "green", "GENERALIZES"
    reason = f"{lo}->{hi} RMSE ratio {ratio:.2f}x ({r_train:.4g} -> {r_test:.4g})"
    return OverfitVerdict(color, label, reason, ratio, "RMSE")


def _grouped_bar_panel(
    task: str, splits: List[str], per_metrics: Mapping[str, Dict[str, float]], verdict_color: str,
) -> PanelSpec:
    """Grouped bars: one group per headline metric, one bar per split. Lower-is-better metrics shown as 0-1 quality.

    Every metric is mapped into a comparable [0,1] 'quality' so bars share one axis: higher-is-better metrics clip to
    [0,1] directly; lower-is-better (ECE/Brier/RMSE/MAE/|bias|) map to ``max(0, 1 - normalized)`` so a tall bar always
    reads 'good'. RMSE/MAE/|bias| are normalized by the train target spread (carried in the metric dict as 'y_std')
    when present, else left raw-clipped. The point of the panel is the cross-split COMPARISON within each group.
    """
    headline = _CLF_HEADLINE if task in ("classification", "binary") else _REG_HEADLINE
    cats = tuple(disp for disp, _, _ in headline)
    y_std = max(per_metrics.get("train", {}).get("y_std", 0.0), 1e-12)
    series: List[np.ndarray] = []
    colors: List[str] = []
    for i, s in enumerate(splits):
        m = per_metrics[s]
        row: List[float] = []
        for _, key, higher in headline:
            raw = m.get(key, float("nan"))
            if not np.isfinite(raw):
                row.append(0.0)
                continue
            if higher:
                q = raw
            elif key in ("RMSE", "MAE", "bias"):
                q = 1.0 - min(1.0, abs(raw) / y_std)
            else:
                q = 1.0 - abs(raw)
            row.append(float(np.clip(q, 0.0, 1.0)))
        series.append(np.asarray(row, dtype=np.float64))
        colors.append(_split_color(s, i))
    return BarPanelSpec(
        categories=cats,
        values=tuple(series),
        series_labels=tuple(splits),
        title="Headline metrics per split (taller = better, [0,1])",
        xlabel="metric",
        ylabel="quality",
        colors=tuple(colors),
    )


def _delta_table_panel(
    task: str, splits: List[str], per_metrics: Mapping[str, Dict[str, float]], verdict: OverfitVerdict,
) -> AnnotationPanelSpec:
    """Compact delta table: train->val, val->test and train->test RAW change per headline metric + the overfit verdict.

    Deltas are reported on the raw metric (not the 0-1 quality) so the numbers match what a DS reads off a metrics
    table. Only transitions whose both endpoints are present are shown.
    """
    headline = _CLF_HEADLINE if task in ("classification", "binary") else _REG_HEADLINE
    transitions: List[Tuple[str, str]] = []
    for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
        if a in per_metrics and b in per_metrics:
            transitions.append((a, b))
    # Fall back to consecutive present-split transitions when the canonical ones are absent (e.g. train+oof only).
    if not transitions and len(splits) >= 2:
        transitions = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]
        if len(splits) > 2:
            transitions.append((splits[0], splits[-1]))

    dot = {"green": "[GREEN]", "amber": "[AMBER]", "red": "[RED]"}.get(verdict.color, "[RED]")
    lines: List[str] = [f"{dot} {verdict.label}", verdict.reason, ""]
    header = f"{'metric':<9s}" + "".join(f"{a[:2]}->{b[:2]:<6s}" for a, b in transitions)
    lines.append(header)
    for disp, key, _ in headline:
        cells: List[str] = []
        for a, b in transitions:
            va, vb = per_metrics[a].get(key, float("nan")), per_metrics[b].get(key, float("nan"))
            cells.append(f"{(vb - va):+.3f}" if np.isfinite(va) and np.isfinite(vb) else "  n/a ")
        lines.append(f"{disp:<9s}" + "".join(f"{c:<8s}" for c in cells))
    return AnnotationPanelSpec(text="\n".join(lines), title="Cross-split deltas + overfit verdict", fontsize=10)


def overfit_verdict(
    *,
    task: str,
    per_split: Mapping[str, Mapping[str, Any]],
    threshold: float = 0.5,
    subsample: int = _METRIC_SUBSAMPLE,
    seed: int = 0,
) -> OverfitVerdict:
    """Compute the cross-split overfit verdict without building a figure (triage / tests).

    ``per_split`` maps split name -> ``{"y_true", "y_score"/"y_pred"}`` (raw arrays) or ``{"metrics": {...}}``
    (precomputed). Raises when fewer than two non-degenerate splits are available.
    """
    per_metrics: Dict[str, Dict[str, float]] = {}
    for name, entry in per_split.items():
        m, _ = _metrics_for_split(task, entry, threshold, subsample, seed)
        if m is not None:
            per_metrics[name] = m
    if len(per_metrics) < 2:
        raise ValueError("overfit verdict needs >= 2 non-degenerate splits")
    return _overfit_verdict(task, per_metrics)


def compose_split_comparison_figure(
    per_split: Mapping[str, Mapping[str, Any]],
    task: str,
    *,
    model_name: str = "model",
    threshold: float = 0.5,
    subsample: int = _METRIC_SUBSAMPLE,
    seed: int = 0,
    suptitle: Optional[str] = None,
    cell_width: float = 7.0,
    cell_height: float = 5.0,
) -> FigureSpec:
    """Build a cross-split OVERFITTING FigureSpec for ONE model: grouped headline-metric bars + delta table + verdict.

    Parameters
    ----------
    per_split : mapping ``split_name -> entry``. Each entry is EITHER raw arrays
        (``{"y_true": ..., "y_score": ...}`` for classification / ``{"y_true": ..., "y_pred": ...}`` for regression)
        OR precomputed metrics (``{"metrics": {"ROC_AUC": ..., ...}}``). Canonical split names train/val/test/oof
        order left-to-right; any other names append after. Missing splits simply do not appear.
    task : "classification"/"binary" -> ROC_AUC/PR_AUC/KS/ECE/Brier headline; "regression" -> R2/RMSE/MAE/bias.
    threshold : operating point passed through to the classification metric kernel (MCC confusion counts).
    subsample : per-split row cap before the metric pass (default 200k); metrics are stable well below it.

    Layout: row 0 = [grouped-bar metrics-per-split | delta table + traffic-light overfit verdict]. Degenerate splits
    (single class / no finite pairs) are listed in the delta panel and excluded from the bars + verdict. With fewer
    than two usable splits the figure degrades to an honest 1-panel note.

    The verdict (``OverfitVerdict``) is reachable standalone via ``overfit_verdict(...)`` for tests / triage.
    """
    t = task.strip().lower()
    figsize = figsize_for_grid(1, 2, cell_width=cell_width, cell_height=cell_height)
    if not per_split:
        ann = AnnotationPanelSpec(text="compose_split_comparison_figure: no splits supplied", title="Split comparison")
        return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)

    per_metrics: Dict[str, Dict[str, float]] = {}
    degenerate: List[Tuple[str, str]] = []
    for name in _order_splits(list(per_split.keys())):
        m, note = _metrics_for_split(t, per_split[name], threshold, subsample, seed)
        if m is not None:
            per_metrics[name] = m
        elif note is not None:
            degenerate.append((name, note))

    if len(per_metrics) < 2:
        present = ", ".join(f"{n} ({why})" for n, why in degenerate) or "fewer than 2 usable splits"
        ann = AnnotationPanelSpec(
            text=f"{model_name}: cross-split overfit view needs >= 2 usable splits.\nUnusable: {present}",
            title="Split comparison", fontsize=11,
        )
        return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)

    splits = _order_splits(list(per_metrics.keys()))
    verdict = _overfit_verdict(t, per_metrics)
    bar = _grouped_bar_panel(t, splits, per_metrics, verdict.color)
    table = _delta_table_panel(t, splits, per_metrics, verdict)
    if degenerate:
        # Surface annotated splits in the table panel so they are not silently dropped.
        skipped = "; ".join(f"{n}: {why}" for n, why in degenerate)
        table = AnnotationPanelSpec(text=table.text + f"\n\nskipped: {skipped}", title=table.title, fontsize=table.fontsize)

    title = suptitle if suptitle is not None else f"Cross-split overfit -- {model_name} -- {verdict.label}"
    return FigureSpec(suptitle=title, panels=((bar, table),), figsize=figsize)


__all__ = [
    "compose_split_comparison_figure",
    "overfit_verdict",
    "OverfitVerdict",
    "AUC_GAP_RED",
    "AUC_GAP_AMBER",
    "RMSE_RATIO_RED",
    "RMSE_RATIO_AMBER",
]
