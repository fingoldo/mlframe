"""Regression: a failing confusion-matrix savefig must not leak the matplotlib figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.evaluation import reports


def test_confusion_matrix_savefig_failure_still_closes_figure(monkeypatch, tmp_path):
    """savefig raising must propagate and the confusion-matrix figure must be closed."""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 2)
    y = (X[:, 0] > 0).astype(int)
    est = LogisticRegression().fit(X, y)
    made = []

    def _fake_plot(*args, **kwargs):
        """Return a real figure whose savefig always fails."""
        fig, ax = plt.subplots()

        def _boom(*a, **k):
            """Simulate a disk failure."""
            raise OSError("disk full")

        fig.savefig = _boom
        made.append(fig)
        return fig, ax

    monkeypatch.setattr(reports, "plot_confusion_matrix", _fake_plot)
    plt.close("all")
    with pytest.raises(OSError):
        reports._evaluate_test_set(
            est, est, "m", X, y, len(y), 0.5, 1, None, None, {0: "a", 1: "b"}, ["a", "b"], False, True, str(tmp_path / "cm.png"),
            None, True, "Blues", None, 0, ".0f", False, 50, None, False, False, 10, (4, 4), [], None, None, None,
        )
    assert made and not plt.fignum_exists(made[0].number)
