"""Cluster 15 regression: _scaled_coef reports when std scaling is skipped for non-numeric data (audit 2026-11-20, finding 15-F14).

The failure used to be swallowed with pass. It now logs a WARNING; the coefficients are returned unscaled, as before, and a numeric frame
still scales.
"""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers import _importance_methods as im


def test_non_numeric_data_warns_and_stays_unscaled(caplog):
    """Object columns make the std computation fail; a WARNING is logged and the coefficients are unscaled (15-F14)."""
    coef = np.array([1.0, -2.0])
    data = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    with caplog.at_level(logging.WARNING, logger=im.logger.name):
        out = im._scaled_coef(coef, data, None, "max", "train")
    assert np.allclose(np.asarray(out, dtype=float), np.abs(coef))
    assert any(r.levelno == logging.WARNING and "left unscaled" in r.getMessage() for r in caplog.records)


def test_numeric_data_is_scaled_and_silent(caplog):
    """Control: numeric data scales the coefficients by the column stds with no warning (15-F14)."""
    coef = np.array([1.0, 1.0])
    data = np.column_stack([np.array([0.0, 2.0]), np.array([0.0, 4.0])])
    with caplog.at_level(logging.WARNING, logger=im.logger.name):
        out = im._scaled_coef(coef, data, None, "max", "train")
    assert np.allclose(np.asarray(out, dtype=float), np.abs(coef) * np.nanstd(data, axis=0))
    assert not any("left unscaled" in r.getMessage() for r in caplog.records)
