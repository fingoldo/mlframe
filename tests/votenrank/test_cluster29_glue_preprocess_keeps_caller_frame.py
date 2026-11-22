"""Audit 2026-11-20 finding 29-F23: preprocess_glue does not rename models in the caller's frame.

Duplicate model names used to get suffixes written into the caller's frame, so a second call gave different names. The test calls it on
a frame with a duplicated name and checks the caller's Model column is unchanged.
"""

import pandas as pd

from mlframe.votenrank.data_processing import preprocess_glue


def test_duplicate_model_names_are_not_suffixed_in_the_callers_frame():
    """The caller's Model column keeps both duplicate names as they were."""
    glue = pd.DataFrame(
        {
            "Rank": [1, 2],
            "Name": ["x", "y"],
            "URL": ["u", "v"],
            "Score": [1, 2],
            "Model": ["bert", "bert"],
            "CoLA": [0.5, 0.6],
            "MRPC": ["0.8/0.7", "0.9/0.8"],
            "STS-B": ["0.5/0.4", "0.6/0.5"],
            "QQP": ["0.7/0.6", "0.8/0.7"],
            "AX": [0.1, 0.2],
        }
    )
    preprocess_glue(glue)
    assert glue["Model"].tolist() == ["bert", "bert"]
