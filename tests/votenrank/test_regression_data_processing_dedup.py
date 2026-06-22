"""Regression: preprocess_glue/sglue de-dup of repeated model names was a chained assignment
(glue["Model"].loc[mask] += ...) that under Copy-on-Write writes to a temporary and is a silent
no-op, leaving duplicate index labels. Also np.NaN was used (removed in NumPy>=2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.votenrank import data_processing as dp


def _make_glue():
    # Two rows share the model name "DupModel" -> must be disambiguated to DupModel_1 / DupModel_2.
    return pd.DataFrame(
        {
            "Rank": [1, 2, 3],
            "Name": ["n1", "n2", "n3"],
            "URL": ["u1", "u2", "u3"],
            "Score": [10.0, 9.0, 8.0],
            "Model": ["DupModel", "DupModel", "Unique"],
            "MRPC": ["0.8/0.7", "0.6/0.5", "0.4/0.3"],
            "STS-B": ["0.9/0.8", "0.7/0.6", "0.5/0.4"],
            "QQP": ["0.7/0.6", "0.5/0.4", "0.3/0.2"],
            "AX": [0.5, 0.4, 0.3],
        }
    )


def test_preprocess_glue_actually_dedups_model_names():
    # Force Copy-on-Write so the pre-fix chained assignment (glue["Model"].loc[mask] += ...)
    # is a silent no-op (writes land on a throwaway temporary) and the duplicate survives.
    with pd.option_context("mode.copy_on_write", True):
        glue, weights = dp.preprocess_glue(_make_glue())
    # The index is the (now disambiguated) Model column; it must contain no duplicates.
    assert glue.index.is_unique, f"duplicate model labels survived: {list(glue.index)}"
    assert "DupModel_1" in glue.index
    assert "DupModel_2" in glue.index


def test_preprocess_sglue_uses_np_nan_and_dedups():
    sglue = pd.DataFrame(
        {
            "Rank": [1, 2, 3],
            "Name": ["n1", "n2", "n3"],
            "URL": ["u1", "u2", "u3"],
            "Score": [10.0, 9.0, 8.0],
            "Model": ["Dup", "Dup", "Solo"],
            "CB": ["0.8/0.7", "-", "0.4/0.3"],
            "MultiRC": ["0.9/0.8", "0.7/0.6", "0.5/0.4"],
            "ReCoRD": ["0.7/0.6", "0.5/0.4", "0.3/0.2"],
            "AX-g": ["0.6/0.5", "0.4/0.3", "0.2/0.1"],
            "AX-b": [0.5, 0.4, 0.3],
        }
    )
    with pd.option_context("mode.copy_on_write", True):
        out, weights = dp.preprocess_sglue(sglue)
    assert out.index.is_unique
    assert "Dup_1" in out.index and "Dup_2" in out.index
