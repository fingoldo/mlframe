"""Tests for the benchmark's chart specs.

Charts are checked as data, not as pixels: the spec is the contract both renderers consume, so asserting on
it catches the mistakes that matter (a bar coloured against its sign, an interval that vanished, a series
count that outgrew the validated palette) without pinning anything to a rendering backend.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from mlframe.feature_selection._benchmarks.fs_hybrid._figures import (
    MAX_SERIES,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    SERIES_COLORS,
    contrast_figure,
    pareto_figure,
    rope_curve_figure,
)


def _record(arm: str, scenario: str, seed: int, auc: float, fits: int = 10, skill: float = 0.3) -> Dict[str, Any]:
    """One ok cell carrying a lightgbm AUC and skill at k5."""
    return {
        "status": "ok",
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": seed,
        "cv_seed": 0,
        "n_model_fits": fits,
        "scores": {"k5": {"models": {"lightgbm": {"roc_auc": auc}}, "skill": {"lightgbm": skill}}},
    }


def _cell(spec: Dict[str, Any], scenario: str = "bed", seeds: int = 6) -> List[Dict[str, Any]]:
    """Records for one bed from ``{arm: (auc_offset, fits)}``, null at 0.70."""
    out: List[Dict[str, Any]] = []
    for seed in range(seeds):
        drift = 0.002 * seed
        out.append(_record("all-features", scenario, seed, 0.70 + drift, fits=10))
        for arm, (offset, fits) in spec.items():
            out.append(_record(arm, scenario, seed, 0.70 + drift + offset, fits=fits, skill=0.3 + offset))
    return out


class TestContrastFigure:
    """Signed bars against the null."""

    def test_absent_data_returns_no_figure(self) -> None:
        """A blank chart and a chart of zeros look alike at a glance and mean opposite things."""
        assert contrast_figure([], "bed", k_label="k5") is None

    def test_bar_colour_follows_the_sign(self) -> None:
        """Colour carries polarity here, so a gain drawn in the loss colour is a lie in the encoding."""
        figure = contrast_figure(_cell({"winner": (0.05, 10), "loser": (-0.05, 10)}), "bed", k_label="k5")
        assert figure is not None
        panel = figure.panels[0][0]
        by_arm = dict(zip(panel.categories, panel.colors or ()))
        assert by_arm["winner"] == POSITIVE_COLOR
        assert by_arm["loser"] == NEGATIVE_COLOR

    def test_bars_are_sorted_best_first(self) -> None:
        """A reader scans top-down; an unsorted bar chart makes them do the sorting."""
        figure = contrast_figure(_cell({"a": (0.01, 10), "b": (0.05, 10), "c": (-0.02, 10)}), "bed", k_label="k5")
        assert figure is not None
        values = np.asarray(figure.panels[0][0].values, dtype=float)
        assert list(values) == sorted(values, reverse=True)

    def test_every_bar_carries_its_interval(self) -> None:
        """A bar picked out for being longest reads as a precise measurement without its whisker."""
        figure = contrast_figure(_cell({"a": (0.01, 10), "b": (0.05, 10)}), "bed", k_label="k5")
        assert figure is not None
        lower, upper = figure.panels[0][0].value_err
        assert len(lower) == len(upper) == len(figure.panels[0][0].categories)
        assert np.all(lower >= 0) and np.all(upper >= 0)

    def test_the_zero_line_is_drawn(self) -> None:
        """Zero is the null hypothesis on this axis, not a gridline the reader has to infer."""
        figure = contrast_figure(_cell({"a": (0.01, 10)}), "bed", k_label="k5")
        assert figure is not None and figure.panels[0][0].hline is not None
        assert figure.panels[0][0].hline[0] == 0.0

    def test_the_bed_is_named_once(self) -> None:
        """The suptitle names the bed; repeating it in the panel title spends space on nothing."""
        figure = contrast_figure(_cell({"a": (0.01, 10)}), "bed", k_label="k5")
        assert figure is not None
        assert "bed" in figure.suptitle
        assert "bed" not in figure.panels[0][0].title


class TestParetoFigure:
    """Cost against advantage."""

    def test_frontier_and_dominated_are_separate_labelled_series(self) -> None:
        """Identity comes from the legend, not from a channel borrowed from another chart's semantics."""
        figure = pareto_figure(_cell({"cheap": (0.05, 12), "dear": (0.01, 400)}), "bed", k_label="k5")
        assert figure is not None
        assert figure.panels[0][0].series_labels == ("on the frontier", "dominated")

    def test_the_frontier_arms_are_named_in_the_caption(self) -> None:
        """A dozen labels on a scatter collide; a short ordered list carries the same identity."""
        figure = pareto_figure(_cell({"cheap": (0.05, 12), "dear": (0.01, 400)}), "bed", k_label="k5")
        assert figure is not None
        assert "cheap" in figure.caption

    def test_cost_and_quality_never_share_an_axis(self) -> None:
        """Two measures on different scales get two axes of one scatter, never two y-scales."""
        figure = pareto_figure(_cell({"cheap": (0.05, 12)}), "bed", k_label="k5")
        assert figure is not None
        panel = figure.panels[0][0]
        assert panel.secondary_y in (None, False)
        assert "cost" in panel.xlabel and "delta" in panel.ylabel

    def test_too_few_priced_arms_returns_no_figure(self) -> None:
        """A frontier over one point is a point, and drawing it would imply a comparison nobody made."""
        assert pareto_figure([], "bed", k_label="k5") is None


class TestRopeCurve:
    """The posterior CDF of the pooled effect."""

    def _multi_bed(self) -> List[Dict[str, Any]]:
        """Three beds, so the pooled fit has something to pool."""
        out: List[Dict[str, Any]] = []
        for index, bed in enumerate(("bed_a", "bed_b", "bed_c")):
            out += _cell({"big": (0.05 + 0.01 * index, 10), "small": (0.001, 10), "mid": (0.02, 10), "tiny": (0.0005, 10), "other": (0.003, 10)}, scenario=bed)
        return out

    def test_series_are_capped_at_the_validated_palette(self) -> None:
        """A fifth series would need a colour outside the set that passed the CVD check."""
        figure = rope_curve_figure(self._multi_bed(), k_label="k5")
        assert figure is not None
        assert len(figure.panels[0][0].series_labels) <= MAX_SERIES

    def test_the_default_selection_is_by_effect_not_alphabet(self) -> None:
        """Alphabetical would put the same two arms on every chart forever, whatever the data said."""
        figure = rope_curve_figure(self._multi_bed(), k_label="k5")
        assert figure is not None
        assert "big" in figure.panels[0][0].series_labels

    def test_identity_does_not_rest_on_colour_alone(self) -> None:
        """Colour plus a dash pattern plus a legend: any one of the three can fail a reader."""
        figure = rope_curve_figure(self._multi_bed(), k_label="k5")
        assert figure is not None
        panel = figure.panels[0][0]
        assert panel.colors and panel.line_styles and panel.series_labels
        assert len(set(panel.colors)) == len(panel.colors)
        assert len(set(panel.line_styles)) == len(panel.line_styles)


class TestPalette:
    """The colours are validated values, not preferences."""

    def test_the_signed_pair_is_the_validated_one(self) -> None:
        """Swapping in another red/blue pair silently drops the CVD separation that was measured."""
        assert (POSITIVE_COLOR, NEGATIVE_COLOR) == ("#1f77b4", "#d62728")

    def test_the_series_order_is_not_the_palette_default(self) -> None:
        """tab10's orange and green sit 0.7 apart under protanopia when adjacent -- invisible to a reader."""
        assert SERIES_COLORS == ("#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c")
        orange, green = SERIES_COLORS.index("#ff7f0e"), SERIES_COLORS.index("#2ca02c")
        assert abs(orange - green) > 1, "the two colours that fail CVD separation must not be adjacent"


def test_a_rendered_figure_survives_the_repository_renderer(tmp_path: Any) -> None:
    """The spec is only worth asserting on if the repository's own renderer accepts it."""
    from mlframe.feature_selection._benchmarks.fs_hybrid._figures import save_benchmark_figures

    written = save_benchmark_figures(_cell({"a": (0.03, 12), "b": (-0.02, 40)}), str(tmp_path), k_label="k5")
    assert written
    produced = sorted(path.name for path in tmp_path.iterdir())
    assert any(name.endswith(".png") for name in produced), produced
