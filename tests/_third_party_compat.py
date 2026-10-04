"""Probes for known third-party incompatibilities, so a test skips only when the incompatibility is reproduced on the spot.

A blanket ``except AttributeError`` around the code under test also swallows a genuine mlframe regression; these probes exercise the third-party call
in isolation and let any other failure propagate.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

_TAGS_MARKER = "__sklearn_tags__"


@functools.cache
def catboost_encoder_tags_broken() -> bool:
    """True when ``category_encoders.CatBoostEncoder.fit`` raises the ``__sklearn_tags__`` super() AttributeError on this sklearn / category_encoders pair."""
    from category_encoders import CatBoostEncoder

    frame = pd.DataFrame({"c": list("abababab")})
    try:
        CatBoostEncoder(cols=["c"]).fit(frame, np.array([0, 1, 0, 1, 1, 0, 1, 0]))
    except AttributeError as exc:
        if _TAGS_MARKER in str(exc):
            return True
        raise
    return False


@functools.cache
def binning_process_tags_broken() -> bool:
    """True when ``optbinning.BinningProcess.fit`` raises the ``__sklearn_tags__`` AttributeError on this sklearn / optbinning pair."""
    from optbinning import BinningProcess

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.standard_normal(200)})
    target = (frame["x"].to_numpy() > 0).astype(int)
    try:
        BinningProcess(variable_names=["x"]).fit(frame, target)
    except AttributeError as exc:
        if _TAGS_MARKER in str(exc):
            return True
        raise
    return False


def skip_if_catboost_encoder_tags_broken() -> None:
    """Skip the calling test only when the CatBoostEncoder / sklearn ``__sklearn_tags__`` incompatibility is reproduced in isolation."""
    if catboost_encoder_tags_broken():
        pytest.skip("category_encoders.CatBoostEncoder.fit is incompatible with the installed sklearn (__sklearn_tags__ super() chain)")


def skip_if_binning_process_tags_broken() -> None:
    """Skip the calling test only when the optbinning / sklearn ``__sklearn_tags__`` incompatibility is reproduced in isolation."""
    if binning_process_tags_broken():
        pytest.skip("optbinning.BinningProcess.fit is incompatible with the installed sklearn (__sklearn_tags__ super() chain)")
