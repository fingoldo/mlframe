"""The headline bar rescales unlike quantities onto one axis, so it must not imply they are commensurate.

Bar LENGTH is a rescaled quality in [0, 1] -- an error metric is shown as ``1 - metric`` so long always
reads as good. That makes the lengths comparable and the numbers unreadable, and it previously carried one
grey reference line at 0.5 across all of them: 0.5 is chance for ROC_AUC and for KS and means nothing at all
for a rescaled Brier, so a chance-level AUC bar sat level with a catastrophic ``1 - Brier`` and with the
reference itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.model_card import compose_model_card_figure
from mlframe.reporting.spec import BarPanelSpec

INVERTED = {"ECE", "Brier"}


def _headline(task: str = "classification") -> BarPanelSpec:
    """The headline bar panel of a model card."""
    rng = np.random.default_rng(0)
    n = 12_000
    if task == "classification":
        y = (rng.random(n) < 0.3).astype(int)
        score = 1.0 / (1.0 + np.exp(-(rng.standard_normal(n) + 1.2 * y)))
        spec = compose_model_card_figure(task="classification", y_true=y, y_score=score)
    else:
        y = rng.normal(1000, 300, n)
        spec = compose_model_card_figure(task="regression", y_true=y, y_pred=y + rng.normal(0, 150, n))
    return next(p for row in spec.panels for p in row if isinstance(p, BarPanelSpec) and p.orientation == "horizontal")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_there_is_no_shared_reference_line(task):
    """One line across metrics with different baselines asserts a comparison that is not defined."""
    panel = _headline(task)
    assert panel.hline is None, f"{task}: a shared reference line is back: {panel.hline!r}"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_bar_names_its_raw_value(task):
    """Length is a rescaled quality; without the raw number the reader cannot recover the measurement."""
    panel = _headline(task)
    for label in panel.categories:
        assert any(ch.isdigit() for ch in label), f"{task}: bar label carries no value: {label!r}"


def test_an_inverted_metric_names_itself_not_its_complement():
    """ "1-ECE 0.275" beside a bar of length 0.725 states something false: 0.275 is the ECE."""
    panel = _headline()
    labels = {lbl.split()[0]: float(lbl.split()[-1]) for lbl in panel.categories}
    lengths = dict(zip((lbl.split()[0] for lbl in panel.categories), (float(v) for v in panel.values)))
    inverted_present = INVERTED & set(labels)
    assert inverted_present, f"fixture has no inverted metric to check; labels were {sorted(labels)}"
    for name in inverted_present:
        assert not name.startswith("1-"), f"{name!r} still names the complement rather than the metric"
        # The label is printed to three decimals, so the comparison is to that precision -- not to float equality.
        assert lengths[name] == pytest.approx(1.0 - labels[name], abs=5e-4), (
            f"{name}: label says {labels[name]}, bar length is {lengths[name]} -- the label must name the RAW metric "
            "while the bar shows the rescaled quality"
        )


def test_the_bar_order_is_stable_across_cards():
    """A model card is read against other cards; sorting per card puts a metric on a different row each time."""
    first = _headline()
    rng = np.random.default_rng(7)
    n = 12_000
    y = (rng.random(n) < 0.45).astype(int)
    score = 1.0 / (1.0 + np.exp(-(0.3 * rng.standard_normal(n) + 2.5 * y)))
    other = next(
        p
        for row in compose_model_card_figure(task="classification", y_true=y, y_score=score).panels
        for p in row
        if isinstance(p, BarPanelSpec) and p.orientation == "horizontal"
    )
    names_a = [lbl.split()[0] for lbl in first.categories]
    names_b = [lbl.split()[0] for lbl in other.categories]
    assert names_a == names_b, f"two cards ordered their metrics differently: {names_a} vs {names_b}"
    assert list(first.values) != list(other.values), "fixture produced identical metrics, so this test proves nothing"
