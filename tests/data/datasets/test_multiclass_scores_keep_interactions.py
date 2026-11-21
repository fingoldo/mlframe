"""A multiclass link keeps its interaction terms in every class score (audit 2026-11-20, finding 04-F3)."""
import numpy as np

from mlframe.data.datasets.generator import multiclass_class_scores
from mlframe.data.datasets.spec import LinkSpec


def test_interaction_only_link_gives_a_nonzero_score_for_every_class():
    """A link with only an interaction used to give zero class scores (only the rotated additive weights were read)."""
    link = LinkSpec(kind="linear", interactions=(("a", "b"),), interaction_weights=(1.0,))
    columns = {"a": np.array([1.0, 2.0, -1.0]), "b": np.array([3.0, -1.0, 2.0])}
    scores = multiclass_class_scores(link, columns, n=3, n_classes=3, scale=1.0)
    assert scores.shape == (3, 3)
    assert np.all(np.abs(scores) > 0)
    np.testing.assert_allclose(scores[:, 0], columns["a"] * columns["b"])


def test_additive_weights_are_still_rotated_per_class():
    """The additive part keeps the per-class rotation: the classes differ when the weights do."""
    link = LinkSpec(kind="linear", coefficients={"a": 1.0, "b": 2.0})
    columns = {"a": np.array([1.0, 0.5]), "b": np.array([0.5, 1.0])}
    scores = multiclass_class_scores(link, columns, n=2, n_classes=2, scale=1.0)
    assert not np.allclose(scores[:, 0], scores[:, 1])
