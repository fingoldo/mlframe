"""Unit + regression tests for MRMR embedding / free-text passthrough.

The embedding-passthrough feature routes object-cells-are-list/ndarray (embedding-vector) columns and long free-text columns THROUGH the MI screen unchanged so a
downstream learnable-embedding network consumes them, instead of crashing the discretiser on the non-scalar cells. These tests pin: pass-through is byte-untouched,
scalar features are still selected, no crash, the detector classifies correctly, the sklearn n_features_in_ contract holds, and the opt-out restores legacy behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters._mrmr_passthrough import detect_passthrough_columns


def _make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": sig + rng.normal(scale=0.1, size=n),  # strong signal
            "b": rng.normal(size=n),  # noise
            "c": rng.normal(size=n),  # noise
            "emb": [rng.normal(size=4).astype(np.float32) for _ in range(n)],
        }
    )
    y = (sig > 0).astype(int)
    return df, y


def test_detect_passthrough_classifies_embedding_and_text():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=20),
            "short_cat": ["A", "B"] * 10,  # short strings -> NOT text passthrough
            "emb": [rng.normal(size=3) for _ in range(20)],
            "text": ["the quick brown fox jumped over the lazy dog repeatedly"] * 20,
        }
    )
    emb, txt = detect_passthrough_columns(df)
    assert emb == ["emb"]
    assert txt == ["text"]
    assert "short_cat" not in emb and "short_cat" not in txt
    assert "num" not in emb and "num" not in txt


def test_mrmr_embedding_passed_through_untouched_numerics_selected():
    df, y = _make_frame()
    m = MRMR(max_runtime_mins=0.5, fe_max_steps=0, random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
        out = m.transform(df)

    # Embedding column passed through and present in the output.
    assert m._passthrough_features_ == ["emb"]
    assert "emb" in out.columns
    # The strong scalar signal survives the screen; pure-noise columns can be dropped.
    assert "a" in out.columns
    # n_features_in_ contract covers the full input width including the embedding.
    assert m.n_features_in_ == 4
    assert list(m.feature_names_in_) == ["a", "b", "c", "emb"]

    # Cells are byte-identical to the input (passed through, never discretised / coerced).
    for i in (0, 5, len(df) - 1):
        np.testing.assert_array_equal(out["emb"].iloc[i], df["emb"].iloc[i])


def test_mrmr_embedding_passthrough_opt_out_is_legacy():
    """With passthrough disabled the non-scalar column is no longer routed through;
    MRMR either drops it or errors -- either way it is NOT in _passthrough_features_."""
    df, y = _make_frame(n=200)
    m = MRMR(max_runtime_mins=0.3, fe_max_steps=0, random_seed=0, embedding_passthrough=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m.fit(df, y)
        except Exception:
            # Legacy path can crash on the ndarray cells -- that is exactly the bug the
            # default-on passthrough fixes; the opt-out is allowed to reproduce it.
            return
        # If the legacy path did not crash, the embedding must NOT have been passthrough-routed.
        assert getattr(m, "_passthrough_features_", []) == []


def test_mrmr_no_embedding_columns_is_noop():
    """A purely scalar frame must leave _passthrough_features_ empty (detector finds nothing)."""
    rng = np.random.default_rng(2)
    n = 300
    sig = rng.normal(size=n)
    df = pd.DataFrame({"a": sig, "b": rng.normal(size=n), "cat": (["x", "y", "z"] * n)[:n]})
    y = (sig > 0).astype(int)
    m = MRMR(max_runtime_mins=0.3, fe_max_steps=0, random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
    assert m._passthrough_features_ == []


def test_mrmr_text_column_passed_through():
    rng = np.random.default_rng(3)
    n = 300
    sig = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": sig + rng.normal(scale=0.1, size=n),
            "b": rng.normal(size=n),
            "descr": ["a moderately long free-text description string here number %d" % i for i in range(n)],
        }
    )
    y = (sig > 0).astype(int)
    m = MRMR(max_runtime_mins=0.3, fe_max_steps=0, random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
        out = m.transform(df)
    assert "descr" in m._passthrough_features_
    assert "descr" in out.columns
    assert "a" in out.columns
