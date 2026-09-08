"""One quality frame per run, and an honest way to compare runs across heterogeneous targets.

A suite trains many targets of different KINDS in one go -- an LtR target beside a binary classifier
beside a quantile regression -- and each kind reports its own metrics on its own scale. Comparing two runs
that differ only in their input features or hyperparameters then has no obvious answer: NDCG@10 = 0.81
against RMSE = 3.2 is not a comparison, and averaging them is not one either.

Two things are built here:

* :func:`targets_performance_frame` turns one run's ``(models, metadata)`` into a per-target frame, sorted
  by target name, with a trailing aggregate row.
* :func:`compare_targets_performance` puts two or more of those side by side and names a winner.

**Why the aggregate needs two scalings.** Metrics cannot be compared across targets in absolute terms, so
each metric is normalised ACROSS THE RUNS BEING COMPARED -- the only frame of reference that exists here.
That is the first scaling, and it is why a single run has no aggregate score: there is nothing to normalise
against, and this module returns NaN rather than inventing one. The second scaling averages a target's
metrics into one per-target score before targets are averaged together; without it a target type reporting
twelve metrics would outvote one reporting two, and a run with five binary targets would outvote a run
with one LtR target.

**What is deliberately excluded.** A metric whose direction is unknown to
:mod:`mlframe.training.metrics_registry` cannot be normalised -- higher might be better or worse -- so it
appears in the frame and is kept out of the aggregate, counted in ``excluded_metrics`` rather than dropped
silently. Comparison uses the INTERSECTION of targets and, within each target, the intersection of
metrics: a run that happens to report an extra metric must not thereby move its own score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core._misc_helpers import _entry_metric
from .metrics_registry import metric_name_higher_is_better

logger = logging.getLogger(__name__)

#: Split whose metrics the frame reports by default. ``test`` is untouched during fitting and is the
#: honest estimate; ``val`` drove early stopping and is optimistically biased, so it is never the default.
DEFAULT_SPLIT = "test"

#: Label of the trailing summary row. Not a target name, so it is filtered out before any per-target maths.
AGGREGATE_ROW = "<aggregate>"

#: How a metric is normalised across runs before averaging. ``rank`` is the default: it is immune to the
#: metric's scale and to a single wild run, which is what makes averaging NDCG next to RMSE defensible at
#: all. ``minmax`` keeps the size of the gap between runs and is the right choice when that gap is the
#: question, at the cost of one outlier compressing everything else.
NORMALISATIONS = ("rank", "minmax")


@dataclass(frozen=True)
class TargetsComparison:
    """Result of comparing runs: the long per-metric frame, the per-run scores, and the winner.

    ``winner`` is ``None`` when no run can be scored -- no shared targets, no shared metrics with a known
    direction, or a single run -- rather than an arbitrary pick. ``reason`` says which of those it was.
    """

    frame: pd.DataFrame
    scores: pd.DataFrame
    winner: Optional[str]
    reason: str
    common_targets: Tuple[Tuple[str, str], ...]
    excluded_metrics: Tuple[str, ...]


def _iter_target_entries(models: Mapping[str, Any]) -> List[Tuple[str, str, List[Any]]]:
    """``(target_type, target_name, entries)`` for every target the run trained, in target-name order."""
    out: List[Tuple[str, str, List[Any]]] = []
    for target_type, by_name in (models or {}).items():
        if not isinstance(by_name, Mapping):
            continue
        for target_name, entries in by_name.items():
            listed = list(entries) if isinstance(entries, (list, tuple)) else [entries]
            if listed:
                out.append((str(target_type), str(target_name), listed))
    return sorted(out, key=lambda row: (row[1], row[0]))


def _entry_metric_names(entry: Any, split: str) -> List[str]:
    """Metric names an entry reports for ``split``, flattening the class-indexed classification layout."""
    inner_entry = entry[0] if isinstance(entry, tuple) and entry else entry
    metrics = getattr(inner_entry, "metrics", None)
    if not isinstance(metrics, Mapping):
        return []
    inner = metrics.get(split)
    if not isinstance(inner, Mapping):
        return []
    names: List[str] = []
    for key, value in inner.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            names.append(str(key))
        elif isinstance(key, int) and isinstance(value, Mapping):
            names.extend(str(k) for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    return sorted(dict.fromkeys(names))


def _primary_metric(metadata: Mapping[str, Any], target_type: str, target_name: str) -> Optional[str]:
    """The target's primary metric, without the split prefix the dummy baselines record it with."""
    by_type = (metadata or {}).get("dummy_baselines", {})
    report = by_type.get(target_type, {}).get(target_name, {}) if isinstance(by_type, Mapping) else {}
    primary = report.get("primary_metric") if isinstance(report, Mapping) else None
    if not isinstance(primary, str):
        return None
    for prefix in ("val_", "test_", "train_"):
        if primary.startswith(prefix):
            return primary[len(prefix) :]
    return primary


def _best_entry(entries: Sequence[Any], split: str, metric: Optional[str]) -> Tuple[Any, float]:
    """The entry scoring best on ``metric``, and that score. ``(entries[0], nan)`` when it cannot be read."""
    if not metric:
        return entries[0], float("nan")
    higher = metric_name_higher_is_better(metric)
    if higher is None:
        return entries[0], float("nan")
    best, best_value = entries[0], float("nan")
    for entry in entries:
        value = _entry_metric(entry, split, metric)
        if not np.isfinite(value):
            continue
        if not np.isfinite(best_value) or (value > best_value if higher else value < best_value):
            best, best_value = entry, value
    return best, best_value


def _entry_name(entry: Any) -> str:
    """A human name for a model entry, matching what the suite's own verdict block prints."""
    inner = entry[0] if isinstance(entry, tuple) and entry else entry
    return str(getattr(inner, "model_name", None) or type(getattr(inner, "model", inner)).__name__)


LEAD_COLUMNS = ("target_name", "target_type", "best_model", "n_models", "primary_metric", "primary_value")


def targets_performance_frame(
    models: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    split: str = DEFAULT_SPLIT,
) -> pd.DataFrame:
    """One row per target, sorted by target name, with a trailing aggregate row.

    Each row carries the best model for that target (by its primary metric) and every metric that model
    reports on ``split``. Metric columns are the union across targets, so a cell is NaN where the metric
    does not apply to that target's kind -- an LtR target has no RMSE.

    The aggregate row summarises COVERAGE, not quality: how many targets and metrics the run reports, and
    how many of those metrics have a direction the registry knows. A quality aggregate needs other runs to
    normalise against; see :func:`compare_targets_performance`.
    """
    rows: List[Dict[str, Any]] = []
    for target_type, target_name, entries in _iter_target_entries(models):
        primary = _primary_metric(metadata or {}, target_type, target_name)
        best, primary_value = _best_entry(entries, split, primary)
        row: Dict[str, Any] = {
            "target_name": target_name,
            "target_type": target_type,
            "best_model": _entry_name(best),
            "n_models": len(entries),
            "primary_metric": primary or "",
            "primary_value": primary_value,
        }
        for metric in _entry_metric_names(best, split):
            value = _entry_metric(best, split, metric)
            if np.isfinite(value):
                row[metric] = float(value)
        rows.append(row)

    lead = list(LEAD_COLUMNS)
    if not rows:
        return pd.DataFrame(columns=[*lead, "split"])

    frame = pd.DataFrame(rows)
    metric_cols = sorted(c for c in frame.columns if c not in lead)
    frame = frame[[*lead, *metric_cols]]
    frame.insert(len(lead), "split", split)

    known = [c for c in metric_cols if metric_name_higher_is_better(c) is not None]
    aggregate = {
        "target_name": AGGREGATE_ROW,
        "target_type": f"{frame['target_type'].nunique()} kind(s)",
        "best_model": "",
        "n_models": int(frame["n_models"].sum()),
        "primary_metric": f"{len(known)}/{len(metric_cols)} metrics scoreable",
        "primary_value": float("nan"),
        "split": split,
    }
    return pd.concat([frame, pd.DataFrame([aggregate])], ignore_index=True)


def _normalise(values: np.ndarray, higher_is_better: bool, method: str) -> np.ndarray:
    """Map raw metric values across runs onto a 0..1 scale where 1 is best. NaN stays NaN."""
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=float)
    if int(finite.sum()) < 2:
        # One usable run cannot be ranked against anything; scoring it 1.0 would invent a comparison.
        return out
    usable = values[finite]
    if method == "rank":
        order = usable.argsort(kind="stable")
        ranks = np.empty(usable.size, dtype=float)
        ranks[order] = np.arange(usable.size, dtype=float)
        # Ties must not break "equal runs score equally", so tied values share their mean rank.
        for value in np.unique(usable):
            tied = usable == value
            if int(tied.sum()) > 1:
                ranks[tied] = ranks[tied].mean()
        scaled = ranks / (usable.size - 1)
    else:
        lo, hi = float(usable.min()), float(usable.max())
        scaled = np.full(usable.shape, 0.5) if hi <= lo else (usable - lo) / (hi - lo)
    out[finite] = scaled if higher_is_better else 1.0 - scaled
    return out


def load_targets_performance(path: str) -> pd.DataFrame:
    """Read a quality frame the suite wrote to disk, ready to hand straight to the comparison.

    The runs worth comparing are usually not both in memory -- yesterday's against today's is the normal
    case -- so the suite writes the frame as CSV and this reads it back. CSV, not a pickle, so a frame
    survives the classes that produced it being refactored away.
    """
    return pd.read_csv(path)


def _as_frame(run: Any, split: str) -> pd.DataFrame:
    """Accept either a live ``(models, metadata)`` pair, an already-built frame, or a path to a saved one."""
    if isinstance(run, pd.DataFrame):
        return run
    if isinstance(run, str):
        return load_targets_performance(run)
    models, metadata = run
    return targets_performance_frame(models, metadata, split=split)


def compare_targets_performance(
    runs: Mapping[str, Any],
    *,
    split: str = DEFAULT_SPLIT,
    normalisation: str = "rank",
) -> TargetsComparison:
    """Compare two or more runs over the targets they share and name a winner.

    ``runs`` maps a run label to any of: a live ``(models, metadata)`` pair, a frame from
    :func:`targets_performance_frame`, or the path of one the suite saved. Only targets present in EVERY
    run are scored, and within each target only the metrics every run reports for it -- a run must not be
    able to move its own score by reporting more.
    """
    if normalisation not in NORMALISATIONS:
        raise ValueError(f"normalisation must be one of {NORMALISATIONS}, got {normalisation!r}")

    labels = list(runs)
    frames = {label: _as_frame(run, split) for label, run in runs.items()}
    per_run_targets = {
        label: {(str(row.target_type), str(row.target_name)) for row in frame.itertuples() if str(row.target_name) != AGGREGATE_ROW}
        for label, frame in frames.items()
    }
    common = sorted(set.intersection(*per_run_targets.values())) if per_run_targets else []
    long_rows: List[Dict[str, Any]] = []
    excluded: set = set()

    for target_type, target_name in common:
        per_run_metrics: Dict[str, Dict[str, float]] = {}
        for label, frame in frames.items():
            match = frame[(frame["target_name"] == target_name) & (frame["target_type"] == target_type)]
            values = match.iloc[0].to_dict() if len(match) else {}
            per_run_metrics[label] = {
                str(k): float(v)
                for k, v in values.items()
                if k not in LEAD_COLUMNS and isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
            }
        shared = set.intersection(*(set(m) for m in per_run_metrics.values())) if per_run_metrics else set()
        for metric in sorted(shared):
            higher = metric_name_higher_is_better(metric)
            if higher is None:
                excluded.add(metric)
                continue
            raw = np.array([per_run_metrics[label].get(metric, np.nan) for label in labels], dtype=float)
            scaled = _normalise(raw, higher, normalisation)
            for label, raw_value, scaled_value in zip(labels, raw, scaled):
                long_rows.append(
                    {
                        "run": label,
                        "target_name": target_name,
                        "target_type": target_type,
                        "metric": metric,
                        "value": float(raw_value),
                        "higher_is_better": bool(higher),
                        "scaled": float(scaled_value),
                    }
                )

    columns = ["run", "target_name", "target_type", "metric", "value", "higher_is_better", "scaled"]
    frame = pd.DataFrame(long_rows, columns=columns)
    scoreable = frame.dropna(subset=["scaled"]) if not frame.empty else frame
    if scoreable.empty or len(labels) < 2:
        reason = (
            "a single run has nothing to normalise against"
            if len(labels) < 2
            else "the runs share no target" if not common else "no shared metric has a direction the registry knows"
        )
        return TargetsComparison(frame, pd.DataFrame(columns=["run", "score", "n_targets", "n_metrics"]), None, reason, tuple(common), tuple(sorted(excluded)))

    # Second scaling: metrics -> one score per target, then targets -> one score per run. Equal weight per
    # target, so neither a metric-rich target type nor a repeated target kind can outvote the rest.
    per_target = scoreable.groupby(["run", "target_name", "target_type"], as_index=False)["scaled"].mean()
    scores = per_target.groupby("run", as_index=False)["scaled"].mean().rename(columns={"scaled": "score"})
    counts = scoreable.groupby("run").agg(n_targets=("target_name", "nunique"), n_metrics=("metric", "nunique")).reset_index()
    scores = scores.merge(counts, on="run").sort_values("score", ascending=False, ignore_index=True)

    top = scores[scores["score"] == scores["score"].max()]
    if len(top) > 1:
        return TargetsComparison(frame, scores, None, f"tied at {float(top['score'].iloc[0]):.4f}", tuple(common), tuple(sorted(excluded)))
    return TargetsComparison(
        frame,
        scores,
        str(top["run"].iloc[0]),
        f"best mean scaled score over {int(scores['n_targets'].iloc[0])} shared target(s)",
        tuple(common),
        tuple(sorted(excluded)),
    )
