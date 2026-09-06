"""Chart specs for the benchmark, built through this repository's own renderer rather than beside it.

Three figures, one per question a reader actually asks, and the form follows the question rather than the
data's shape:

* **Did it beat doing nothing, and by how much?** -- a signed horizontal bar per arm, sorted, with the
  paired confidence interval on each bar and a zero line. Magnitude AND polarity, which is what a diverging
  pair of colours is for; the interval is not optional, because a bar chosen for being longest reads as a
  precise measurement without it.
* **What did that cost?** -- cost against advantage as a scatter, with the frontier arms labelled inline.
  Two measures on different scales never share an axis, so cost is the x axis and quality the y, never a
  second y on the bar chart.
* **How wide would the region of practical equivalence have to be?** -- the posterior CDF of the absolute
  pooled effect as a line per arm, so a reader applies their own threshold instead of arguing about ours.

Colour is assigned by job and validated, not chosen. The signed bars use one blue and one red
(``#1f77b4`` / ``#d62728``: adjacent CVD separation 21.1 under protanopia, 31.7 for normal vision). The
line chart draws at most four arms in a fixed order that passes the same check -- tab10's orange and green
are 0.7 apart under protanopia when adjacent, so the obvious "first four of the palette" order would have
been unreadable to a red-green colourblind reader. Every series is also direct-labelled and dash-cycled, so
identity never rests on colour alone.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.spec import BarPanelSpec, FigureSpec, LinePanelSpec

from ._leaderboard import NULL_ARM, extract_long_rows
from ._paired_stats import average_over_cv_seed, paired_differences, paired_t_test

logger = logging.getLogger(__name__)

__all__ = ["POSITIVE_COLOR", "NEGATIVE_COLOR", "SERIES_COLORS", "contrast_figure", "pareto_figure", "rope_curve_figure", "save_benchmark_figures"]

# Validated as an adjacent pair: protan dE 21.1, normal-vision dE 31.7, both above surface contrast 3:1.
POSITIVE_COLOR = "#1f77b4"
NEGATIVE_COLOR = "#d62728"
# Fixed order, validated as a four-slot categorical palette. NOT the palette's first four: orange and green
# sit 0.7 apart under protanopia when adjacent, which is invisible.
SERIES_COLORS: Tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c")
# Orange warns on surface contrast (2.47:1), which the direct labels and the accompanying text tables relieve.
SERIES_DASHES: Tuple[str, ...] = ("-", "--", "-.", ":")
MAX_SERIES = len(SERIES_COLORS)


def _contrasts(records: Sequence[Dict[str, Any]], scenario: str, model: str, k_label: str, metric: str) -> List[Tuple[str, Any]]:
    """Return `(arm, paired t result)` for every arm with paired seeds on one cell, best first."""
    rows = average_over_cv_seed(extract_long_rows(records, model=model, k_label=k_label, metric=metric))
    arms = sorted({str(row["arm"]) for row in rows if str(row["arm"]) != NULL_ARM and str(row["scenario"]) == scenario})
    out: List[Tuple[str, Any]] = []
    for arm in arms:
        deltas = paired_differences(rows, arm=arm, null_arm=NULL_ARM, scenario=scenario)
        if len(deltas) >= 2:
            out.append((arm, paired_t_test(deltas)))
    return sorted(out, key=lambda item: item[1].mean_delta, reverse=True)


def contrast_figure(
    records: Sequence[Dict[str, Any]],
    scenario: str,
    model: str = "lightgbm",
    k_label: str = "1k",
    metric: str = "roc_auc",
) -> Optional[FigureSpec]:
    """Signed bars: each arm's paired advantage over the null, with its confidence interval.

    Returns ``None`` when the cell has no arm with paired seeds, rather than an empty figure -- a blank chart
    and a chart of zeros look alike at a glance and mean opposite things.
    """
    contrasts = _contrasts(records, scenario, model, k_label, metric)
    if not contrasts:
        logger.info("no paired contrast to draw for %s/%s at %s", scenario, model, k_label)
        return None

    arms = tuple(arm for arm, _ in contrasts)
    deltas = np.asarray([stat.mean_delta for _, stat in contrasts], dtype=np.float64)
    # Asymmetric distances from the bar to each interval end, which is what the spec's value_err expects; a
    # degenerate interval (every seed moved identically) contributes zero width rather than a missing bar.
    lower = np.asarray([stat.mean_delta - (stat.ci_low if stat.ci_low is not None else stat.mean_delta) for _, stat in contrasts])
    upper = np.asarray([(stat.ci_high if stat.ci_high is not None else stat.mean_delta) - stat.mean_delta for _, stat in contrasts])
    colors = tuple(POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in deltas)
    hover = tuple(
        f"{arm}: delta={stat.mean_delta:+.4f}, m={stat.m} seeds, p={'n/a' if stat.p_value is None else f'{stat.p_value:.4g}'}" for arm, stat in contrasts
    )

    panel = BarPanelSpec(
        categories=arms,
        values=deltas,
        colors=colors,
        title=f"{model} @ {k_label}",
        xlabel=f"paired delta in {metric} vs {NULL_ARM}",
        ylabel="",
        orientation="horizontal",
        value_err=(np.abs(lower), np.abs(upper)),
        hline=(0.0, "#7f7f7f", "no selection"),
        hovertext=hover,
        grid=True,
    )
    return FigureSpec(
        suptitle=f"Does selection beat doing nothing? {scenario}",
        panels=((panel,),),
        figsize=(10.0, 0.42 * len(arms) + 2.2),
        caption=(
            "Bars are the mean per-dataset-seed difference against the all-features null; whiskers are the 95% "
            "paired interval. Blue is a gain, red a loss. A bar whose whisker crosses zero has not separated "
            "from doing nothing on this bed."
        ),
    )


def pareto_figure(
    records: Sequence[Dict[str, Any]],
    scenario: str,
    model: str = "lightgbm",
    k_label: str = "1k",
    metric: str = "roc_auc",
) -> Optional[FigureSpec]:
    """Cost against advantage, with the frontier arms labelled.

    Cost and quality are different measures on different scales, so they get two axes of one scatter rather
    than two y-scales on one bar chart.
    """
    from ._pareto import pareto_points

    points = [point for point in pareto_points(records, scenario=scenario, model=model, k_label=k_label, metric=metric) if point.cost is not None]
    if len(points) < 2:
        logger.info("no priced arms to draw a frontier for %s/%s at %s", scenario, model, k_label)
        return None

    costs = np.asarray([point.cost for point in points], dtype=np.float64)
    deltas = np.asarray([point.advantage for point in points], dtype=np.float64)
    frontier = np.asarray([index for index, point in enumerate(points) if point.on_frontier], dtype=int)
    # Only the frontier is drawn in the accent colour and named in the legend: a label on every point is how
    # a scatter becomes unreadable, and the dominated arms are named in the text table this figure accompanies.

    # Two marker series rather than the scatter's highlight channel: that channel's legend entry is named for
    # its own domain use (worst-K regression errors), so borrowing it would label the frontier "worst-K".
    on = np.zeros(len(points), dtype=bool)
    on[frontier] = True
    panel = LinePanelSpec(
        x=(costs[on], costs[~on]),
        y=(deltas[on], deltas[~on]),
        series_labels=("on the frontier", "dominated"),
        colors=(POSITIVE_COLOR, "#7f7f7f"),
        line_styles=("markers", "markers"),
        title=f"{model} @ {k_label}",
        xlabel="cost: mean model fits per cell",
        ylabel=f"paired delta in {metric} vs {NULL_ARM}",
        grid=True,
    )
    # The frontier arms are named in the caption rather than labelled on the points: at a dozen arms the
    # labels collide, and a short ordered list carries the same identity without the collision.
    named = ", ".join(points[i].arm for i in frontier) or "none"
    return FigureSpec(
        suptitle=f"What did the advantage cost? {scenario}",
        panels=((panel,),),
        figsize=(9.0, 5.5),
        caption=(
            "Each point is one arm; the coloured ones are on the cost-quality frontier, where nothing is both "
            f"cheaper and at least as good. On the frontier here: {named}. The null hypothesis sits at zero cost "
            "and zero advantage, so any arm below zero on the vertical axis is beaten by not selecting at all."
        ),
    )


def rope_curve_figure(
    records: Sequence[Dict[str, Any]],
    model: str = "lightgbm",
    k_label: str = "1k",
    arms: Sequence[str] = (),
    radii: Sequence[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
) -> Optional[FigureSpec]:
    """Posterior CDF of the absolute pooled effect, one line per arm.

    At most four arms are drawn. A fifth would need a colour outside the validated set, and the honest fix
    for "too many series" is fewer series, not a generated hue.
    """
    from ._bayes import extract_skill_rows, rope_curve, scenario_effects

    rows = average_over_cv_seed(extract_skill_rows(records, model=model, k_label=k_label))
    present = sorted({str(row["arm"]) for row in rows if str(row["arm"]) != NULL_ARM})
    if arms:
        chosen = [arm for arm in arms if arm in present][:MAX_SERIES]
    else:
        # Alphabetical would put `ace` and `boruta` on every chart forever. The four arms whose pooled effect
        # is largest in absolute value are the ones a reader is deciding between, in either direction.
        scored = []
        for arm in present:
            effects = scenario_effects(rows, arm=arm)
            if len(effects) >= 2:
                scored.append((abs(float(np.mean([effect.delta for effect in effects]))), arm))
        chosen = [arm for _, arm in sorted(scored, reverse=True)[:MAX_SERIES]]
    if not chosen:
        return None

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    labels: List[str] = []
    for arm in chosen:
        curve = rope_curve(scenario_effects(rows, arm=arm), radii=radii)
        masses = [mass for _, mass in curve]
        if not np.all(np.isfinite(masses)):
            logger.info("arm %s has too few scenarios for a pooled posterior; omitted from the ROPE curve", arm)
            continue
        xs.append(np.asarray([radius for radius, _ in curve], dtype=np.float64))
        ys.append(np.asarray(masses, dtype=np.float64))
        labels.append(arm)
    if not labels:
        return None

    panel = LinePanelSpec(
        x=tuple(xs),
        y=tuple(ys),
        series_labels=tuple(labels),
        title=f"{model} @ {k_label}",
        xlabel="r: half-width of the region of practical equivalence (normalized skill)",
        ylabel="P(|pooled effect| < r)",
        colors=SERIES_COLORS[: len(labels)],
        line_styles=SERIES_DASHES[: len(labels)],
        grid=True,
    )
    return FigureSpec(
        suptitle="How large would 'practically equivalent' have to be?",
        panels=((panel,),),
        figsize=(9.0, 5.0),
        caption=(
            "Each curve is the posterior probability that an arm's pooled advantage is smaller than r, on the "
            "normalized-skill scale. Read your own threshold off the x axis rather than accepting ours: a curve "
            "that reaches 1.0 early is an arm whose advantage is small however you define small."
        ),
    )


def save_benchmark_figures(
    records: Sequence[Dict[str, Any]],
    out_dir: str,
    model: str = "lightgbm",
    k_label: str = "1k",
    scenarios: Sequence[str] = (),
    backends: str = "matplotlib[png]",
) -> List[str]:
    """Render every figure for one (model, K) and return the base paths written.

    Saved through the repository's own renderer, so both backends and every configured format come for free
    and the charts match the rest of the reporting surface instead of being a second visual language.
    """
    from mlframe.reporting.output import parse_plot_output_dsl
    from mlframe.reporting.renderers.save import render_and_save

    os.makedirs(out_dir, exist_ok=True)
    beds = list(scenarios) or sorted({str(record["scenario"]) for record in records if record.get("status") == "ok"})
    output = parse_plot_output_dsl(backends)

    written: List[str] = []
    for scenario in beds:
        for name, spec in (
            ("contrast", contrast_figure(records, scenario, model=model, k_label=k_label)),
            ("pareto", pareto_figure(records, scenario, model=model, k_label=k_label)),
        ):
            if spec is None:
                continue
            base = os.path.join(out_dir, f"{scenario}_{model}_{k_label}_{name}")
            render_and_save(spec, output, base)
            written.append(base)

    rope = rope_curve_figure(records, model=model, k_label=k_label)
    if rope is not None:
        base = os.path.join(out_dir, f"rope_{model}_{k_label}")
        render_and_save(rope, output, base)
        written.append(base)
    return written
