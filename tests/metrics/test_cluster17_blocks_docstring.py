"""Cluster 17 regression: the block-kernel module documents its own functions, not the extras module's API (audit 2026-11-20, finding 17-F23).

Pinned: the module docstring names each block kernel it defines and points to _classification_extras.py for the public metrics; it no longer
lists the top_k_accuracy / ks_statistic public API as its own.
"""

from mlframe.metrics.classification import _classification_extras_blocks as blocks


def test_docstring_names_the_block_kernels_defined_here():
    """Each block kernel defined in the module is named in its docstring (17-F23)."""
    doc = blocks.__doc__ or ""
    for name in ("fast_binary_confusion_metrics_block", "fast_binary_probability_metrics_block", "fast_multiclass_confusion_metrics_block"):
        assert hasattr(blocks, name)
        assert name in doc


def test_docstring_points_to_the_public_metrics_module():
    """The public metrics are attributed to _classification_extras.py, not to this module (17-F23)."""
    doc = blocks.__doc__ or ""
    assert "_classification_extras.py" in doc
    assert "Public API (re-exported" not in doc
