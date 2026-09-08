"""A collapsed grid column must not take a panel down with it.

``col_width_ratios`` applies to the whole grid; there is no per-row spanning. The model card collapsed its
third column to ~0 to widen a header whose row leaves that cell empty -- and the row BELOW puts a mini panel
there, so "mini gain" rendered as a sliver with its legend wider than the panel. The invariant is stated
generally here because any builder reaching for the same trick lands in the same place.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.model_card import compose_model_card_figure

MIN_COLUMN_SHARE = 0.05


def _card(task: str = "classification"):
    """A model card on ordinary data, both task shapes."""
    rng = np.random.default_rng(0)
    n = 12_000
    if task == "classification":
        y = (rng.random(n) < 0.3).astype(int)
        score = 1.0 / (1.0 + np.exp(-(rng.standard_normal(n) + 1.2 * y)))
        return compose_model_card_figure(task="classification", y_true=y, y_score=score, model_name="lightgbm_dart_v2", split="holdout")
    y = rng.normal(1000, 300, n)
    return compose_model_card_figure(task="regression", y_true=y, y_pred=y + rng.normal(0, 120, n), model_name="xgboost_hist", split="oof")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_no_occupied_column_is_collapsed_to_nothing(task):
    """Every column holding a panel in ANY row must keep a usable share of the width."""
    spec = _card(task)
    ratios = spec.col_width_ratios
    if ratios is None:
        pytest.skip("this card does not set column ratios")

    occupied = {c for row in spec.panels for c, panel in enumerate(row) if panel is not None}
    total = float(sum(ratios))
    for c in sorted(occupied):
        share = float(ratios[c]) / total
        assert share >= MIN_COLUMN_SHARE, f"column {c} holds a panel but is given {share:.4%} of the width; it will render as a sliver " f"(ratios={ratios})"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_mini_panel_gets_comparable_width(task):
    """The three mini panels sit in one row and answer one question each; one of them being 100x narrower is a layout bug, not a choice."""
    spec = _card(task)
    ratios = spec.col_width_ratios
    if ratios is None or len(spec.panels) < 2:
        pytest.skip("this card has no mini row or no column ratios")

    mini_cols = [c for c, panel in enumerate(spec.panels[-1]) if panel is not None]
    assert len(mini_cols) >= 2, "fixture no longer produces a multi-panel mini row"
    widths = [float(ratios[c]) for c in mini_cols]
    assert max(widths) / min(widths) <= 2.0, f"mini panels differ in width by {max(widths) / min(widths):.0f}x (ratios={ratios})"


def test_the_card_header_carries_no_stray_markup():
    """The header read "name  #  --  split": a hash with nothing to its right, printed for the user to see."""
    spec = _card()
    header = spec.panels[0][0]
    text = getattr(header, "text", "")
    assert text, "the card has no header text"
    first_line = text.splitlines()[0]
    assert "#" not in first_line, f"stray markup in the card header: {first_line!r}"
    assert "lightgbm_dart_v2" in first_line and "holdout" in first_line, f"header lost its identity: {first_line!r}"
