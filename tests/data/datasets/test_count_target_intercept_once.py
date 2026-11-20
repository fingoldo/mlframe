"""The count-target link adds its intercept once, not twice (audit 2026-11-20, finding 04-F1)."""
import numpy as np

import mlframe.data.datasets._targets_nonbinary as nonbinary
from mlframe.data.datasets.spec import DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec
from mlframe.data.datasets.generator import generate


def _count_means(intercept, monkeypatch):
    """Return the Poisson means the generator passes to count_mean for a count target with the given link intercept."""
    seen = []
    original = nonbinary.count_mean

    def spy(score, *args, **kwargs):
        """Record the means and pass them through unchanged."""
        seen.append(np.asarray(original(score, *args, **kwargs), dtype=np.float64).copy())
        return seen[-1]

    monkeypatch.setattr(nonbinary, "count_mean", spy)
    spec = DatasetSpec(
        name="count_intercept",
        n_samples=300,
        root_seed=0,
        features=(FeatureSpec(name="x1"),),
        targets=(TargetSpec(name="y", kind="count", prevalence=0.3, link=LinkSpec(kind="linear", coefficients={"x1": 1.0}, intercept=intercept)),),
        edges=(EdgeSpec(source="x1", target="y"),),
    )
    generate(spec)
    assert seen, "count_mean was not called"
    return seen[0]


def test_intercept_enters_the_count_mean_once(monkeypatch):
    """With unit scale, raising the intercept by 1 raises the exponent by exactly 1; the old code added it twice (scale * intercept + intercept)."""
    base = _count_means(0.0, monkeypatch)
    shifted = _count_means(1.0, monkeypatch)
    np.testing.assert_allclose(shifted, base * np.e, rtol=1e-6)
