"""Cluster 39 regression: the CUDA-lost CPU fallback states its precision change (audit 2026-11-20, finding 39-F26).

The fallback reloads the model in float32 while the primary load is fp16. The warning says so, and the comment no longer claims the embeddings
are identical to the pre-failure output.
"""

from pathlib import Path

from mlframe.training.feature_handling import hf_provider


def test_fallback_warning_names_the_precision_change():
    """The CPU fallback warning states that the dtype changes and that earlier batches were fp16 (39-F26)."""
    src = Path(hf_provider.__file__).read_text(encoding="utf-8")
    assert "in float32 for the rest of this transform" in src
    assert "change precision mid-run" in src


def test_fallback_comment_no_longer_claims_identical_embeddings():
    """The comment does not claim the fallback output matches the primary output (39-F26)."""
    src = Path(hf_provider.__file__).read_text(encoding="utf-8")
    assert "matching the primary load path" not in src
    assert "NOT identical to the pre-failure fp16 output" in src
