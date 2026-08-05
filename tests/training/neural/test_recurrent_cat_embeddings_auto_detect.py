"""TRAINING_NEURAL-6 regression test: the recurrent wrapper's categorical-factorization boundary must
auto-detect un-named non-numeric columns, mirroring the flat-MLP sibling (``_FitPrepMixin._factorize_cats_fit``).

Pre-fix, ``_RecurrentCatEmbeddingMixin._factorize_cats_fit`` only factorized columns explicitly present in
the caller-supplied ``cat_features`` list; a categorical column absent from that list survived as raw
object/string dtype and would crash downstream numeric-only consumers instead of being handled gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.neural._recurrent_cat_embeddings import _RecurrentCatEmbeddingMixin


class _Host(_RecurrentCatEmbeddingMixin):
    """Minimal host exposing only the attribute the mixin reads (operates purely on self)."""

    use_learnable_cat_embeddings = True


def test_un_named_categorical_column_is_auto_detected_and_factorized():
    """A non-numeric column absent from cat_features must still be factorized, not left raw."""
    host = _Host()
    df = pd.DataFrame(
        {
            "named_cat": ["a", "b", "a", "c"],
            "un_named_cat": ["x", "y", "x", "z"],  # not in cat_features -- must be auto-detected
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = host._factorize_cats_fit(df, cat_features=["named_cat"])

    assert "un_named_cat" in host._cat_cols_, "un-named categorical column must be auto-detected"
    assert "named_cat" in host._cat_cols_
    assert host._n_cat_features_ == 2
    # Every column of the returned frame must be numeric -- no raw object dtype survives.
    for col in out.columns:
        assert pd.api.types.is_numeric_dtype(out[col]), f"column {col!r} is not numeric after factorization"


def test_all_named_no_auto_detection_needed_matches_prior_behaviour():
    """When every categorical column is already named, behaviour is unchanged (no spurious auto-detect)."""
    host = _Host()
    df = pd.DataFrame({"named_cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
    host._factorize_cats_fit(df, cat_features=["named_cat"])
    assert host._cat_cols_ == ["named_cat"]
    assert host._n_cat_features_ == 1


def test_no_categorical_columns_at_all_is_noop():
    """A pure-numeric frame with no cat_features and nothing to auto-detect stays a no-op."""
    host = _Host()
    df = pd.DataFrame({"num": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    out = host._factorize_cats_fit(df, cat_features=None)
    assert host._cat_cols_ is None
    assert host._n_cat_features_ == 0
    assert list(out.columns) == ["num"]
