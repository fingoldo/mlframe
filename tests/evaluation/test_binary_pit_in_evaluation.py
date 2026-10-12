"""evaluate_estimators draws the binary PIT histogram next to the calibration plot, only for labels coded 0/1."""
import numpy as np
import pytest

from mlframe.evaluation import reports


@pytest.fixture
def recorded_pit(monkeypatch):
    """Replace plot_pit_diagram with a recorder and return the list of recorded calls."""
    calls = []
    monkeypatch.setattr(reports, "plot_pit_diagram", lambda **kwargs: calls.append(kwargs))
    return calls


def test_binary_pit_rendered_for_zero_one_labels(recorded_pit):
    """Labels coded 0/1 are passed to plot_pit_diagram with the positive-class probabilities and the estimator name as title prefix."""
    probs = np.column_stack([np.full(40, 0.5), np.linspace(0.05, 0.95, 40)])
    labels = np.tile([0, 1], 20)

    reports._plot_binary_pit_for_evaluation(probs, labels, (8, 4), "lgb")

    assert len(recorded_pit) == 1
    call = recorded_pit[0]
    np.testing.assert_allclose(call["predicted_probs"], probs[:, 1])
    np.testing.assert_array_equal(call["true_labels"], labels)
    assert call["title_prefix"] == "lgb: "


def test_binary_pit_skipped_for_non_zero_one_labels(recorded_pit):
    """String labels leave the PIT histogram undrawn."""
    probs = np.column_stack([np.full(4, 0.5), np.full(4, 0.5)])

    reports._plot_binary_pit_for_evaluation(probs, np.array(["a", "b", "a", "b"]), (8, 4), "lgb")

    assert recorded_pit == []
