"""Tests for ``mlframe.reporting.charts.plot_confusion_matrix`` -- our own-implementation replacement for
``sklearn.metrics.ConfusionMatrixDisplay``.

Coverage:
  * hand-computed small case (counts match our njit kernel exactly);
  * normalize modes {None, 'true', 'pred', 'all'};
  * custom labels (order + subset) and string labels;
  * headless figure build (Agg) + drawing onto a supplied ``ax``;
  * sklearn-equivalence: matrix values match ``sklearn.metrics.confusion_matrix`` (the one sanctioned sklearn import).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from mlframe.reporting.charts import plot_confusion_matrix, confusion_matrix_counts


def test_counts_hand_case():
    yt = np.array([0, 0, 1, 1, 2, 2])
    yp = np.array([0, 1, 1, 1, 2, 0])
    mat, labels = confusion_matrix_counts(yt, yp)
    expected = np.array([[1, 1, 0],
                         [0, 2, 0],
                         [1, 0, 1]], dtype=np.int64)
    assert np.array_equal(mat, expected)
    assert list(labels) == [0, 1, 2]


def test_string_labels_sorted():
    yt = np.array(["cat", "dog", "cat", "bird"])
    yp = np.array(["cat", "cat", "cat", "bird"])
    mat, labels = confusion_matrix_counts(yt, yp)
    assert list(labels) == ["bird", "cat", "dog"]
    # bird: 1 correct; cat: 2 correct; dog->cat: 1 misroute
    assert mat.sum() == 4


def test_labels_subset_and_order():
    yt = np.array([0, 1, 2, 2])
    yp = np.array([0, 1, 2, 1])
    mat, labels = confusion_matrix_counts(yt, yp, labels=[2, 1, 0])
    assert list(labels) == [2, 1, 0]
    # top-left is (true=2, pred=2)=1
    assert mat[0, 0] == 1


@pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
def test_normalize_modes_build_figure(normalize):
    yt = np.array([0, 0, 1, 1, 1])
    yp = np.array([0, 1, 1, 1, 0])
    fig, ax = plot_confusion_matrix(yt, yp, normalize=normalize)
    assert type(fig).__name__ == "Figure"
    assert ax.figure is fig
    if normalize == "true":
        # each row of the DISPLAYED matrix sums to 1 (true-normalised)
        im = ax.images[0]
        arr = im.get_array()
        np.testing.assert_allclose(np.asarray(arr).sum(axis=1), [1.0, 1.0])
    if normalize == "all":
        im = ax.images[0]
        np.testing.assert_allclose(float(np.asarray(im.get_array()).sum()), 1.0)


def test_draw_onto_supplied_ax():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    fig2, ax2 = plot_confusion_matrix(np.array([0, 1]), np.array([0, 1]), ax=ax)
    assert fig2 is fig and ax2 is ax


def test_invalid_normalize_raises():
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([0, 1]), np.array([0, 1]), normalize="rows")


def test_headless_savefig(tmp_path):
    fig, _ = plot_confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), title="cm")
    out = tmp_path / "cm.png"
    fig.savefig(str(out))
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# sklearn equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 5, 13])
def test_matrix_equivalence_vs_sklearn(seed):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    K = int(rng.integers(2, 8))
    n = int(rng.integers(20, 500))
    yt = rng.integers(0, K, n)
    yp = rng.integers(0, K, n)
    mat, labels = confusion_matrix_counts(yt, yp)
    sk_mat = sk.confusion_matrix(yt, yp)
    assert np.array_equal(mat, sk_mat)


def test_matrix_equivalence_string_labels_vs_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    yt = np.array(["a", "b", "c", "a", "b", "c", "a"])
    yp = np.array(["a", "a", "c", "b", "b", "c", "a"])
    mat, _ = confusion_matrix_counts(yt, yp)
    assert np.array_equal(mat, sk.confusion_matrix(yt, yp))
