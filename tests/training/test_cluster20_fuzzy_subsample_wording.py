"""Cluster 20 regression: the fuzzy-membership docs describe the draw accurately (audit 2026-11-20, finding 20-F43).

The code draws the fit subsample with ``rng.integers``, which is with replacement, and says so in its comment (duplicates are harmless for
quantiles). The module docstring used to call it a random subsample. It now says with replacement.
"""

from pathlib import Path

from mlframe.reporting.charts import fuzzy_membership as fm


def test_docstring_says_draw_is_with_replacement():
    """The module docstring describes the bounded draw as with replacement (20-F43)."""
    doc = fm.__doc__ or ""
    assert "with replacement" in doc
    assert "random subsample" not in doc


def test_code_still_draws_with_replacement():
    """The draw itself is the with-replacement rng.integers call, matching the docs (20-F43)."""
    src = Path(fm.__file__).resolve().read_text(encoding="utf-8")
    assert "rng.integers(0, finite.size, _FIT_SUBSAMPLE_CAP)" in src
