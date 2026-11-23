"""Cluster 21 regression: a perfect model gets a verdict that says deferral cannot help, not "insufficient data" (audit 2026-11-20, finding 21-F18).

Drives build_risk_coverage_spec with a perfect binary model whose confidence ranks the rows. The verdict must not say the data are
insufficient; a noisy model still gets a quantified verdict.
"""

import numpy as np

from mlframe.reporting.charts import risk_coverage as rc


def _verdict(result):
    """Return the 'Verdict: ...' sentence carried by the result's figure (it is embedded in the figure's text)."""
    import re

    match = re.search(r"Verdict: [^'\"]*", repr(result.figure))
    if match is None:
        raise AssertionError("no verdict text found on the figure")
    return match.group(0)


def test_perfect_model_is_not_reported_as_insufficient_data():
    """A perfect model's verdict says deferral cannot reduce error, not that the data are insufficient (21-F18)."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    # Probabilities always on the correct side of 0.5 (perfect accuracy) but varying, so confidence ranks rows.
    score = y * 0.5 + 0.25 + 0.2 * rng.random(size=200)
    result = rc.build_risk_coverage_spec(y, score, task="binary")
    text = _verdict(result)
    assert "insufficient data" not in text
    assert "cannot reduce error" in text
