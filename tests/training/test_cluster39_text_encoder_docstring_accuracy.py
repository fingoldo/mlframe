"""Cluster 39 regression: the column-iterator docstring states where the values are materialised (audit 2026-11-20, finding 39-F49).

The docstring used to claim the lazy generator avoids all double allocation, but the TF-IDF fit path materialises it with ``list(texts)``.
The docstring now names that exception.
"""

import inspect

from mlframe.training.feature_handling import text_encoder as te


def test_docstring_names_the_tfidf_materialisation():
    """The docstring says the TF-IDF path materialises the iterator (39-F49)."""
    doc = inspect.getdoc(te._column_to_string_iter)
    assert "list(texts)" in doc
    assert "TF-IDF" in doc


def test_docstring_no_longer_claims_full_avoidance():
    """The unconditional no-double-allocation claim is gone (39-F49)."""
    doc = inspect.getdoc(te._column_to_string_iter)
    assert "so 1M-row text columns don't double-allocate" not in doc
