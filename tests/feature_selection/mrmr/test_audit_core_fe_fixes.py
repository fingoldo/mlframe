"""Regression tests for the core-class / FE-step / FE-pairs findings of the mrmr audit fix wave.

Pins the nested-config default parity (HybridOrthScorersConfig must not empty the ensemble roster), the
set_params config-expansion contract (a config passed through set_params must reach the flat attrs so
GridSearchCV-over-config works), the FE-step non-numeric introspection now logs on failure, and the numpy
feature-matrix round-trip restores nulls symmetrically with the pandas/polars branches.
"""

from __future__ import annotations

import logging

import numpy as np


def test_hybrid_orth_scorers_config_default_matches_flat_roster():
    """HybridOrthScorersConfig().ensemble_scorers must equal the flat MRMR ensemble-scorers default, so an
    all-defaults HybridOrthConfig() does not overwrite the roster with an empty tuple."""
    from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import HybridOrthConfig, HybridOrthScorersConfig

    assert HybridOrthScorersConfig().ensemble_scorers == ("plug_in", "ksg", "copula", "dcor", "hsic")
    # an all-defaults nested config must carry the same roster its scorers block does
    assert HybridOrthConfig().scorers.ensemble_scorers == ("plug_in", "ksg", "copula", "dcor", "hsic")


def test_set_params_expands_nested_config_onto_flats():
    """MRMR.set_params(dcd_config=...) must expand the config's fields onto the flat attrs (mirroring __init__),
    not silently drop them -- otherwise GridSearchCV over a config param no-ops every candidate."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import DCDConfig

    m = MRMR()
    flipped = not bool(m.dcd_enable)
    m.set_params(dcd_config=DCDConfig(dcd_enable=flipped))
    assert bool(m.dcd_enable) == flipped, "config passed through set_params must take effect on the flat attr"


def test_non_numeric_column_indices_logs_on_introspection_failure(caplog):
    """_non_numeric_column_indices must log (not silently swallow) when dtype introspection raises, since the
    empty-set fallback marks every column numeric and can crash the downstream numeric basis."""
    from mlframe.feature_selection.filters._mrmr_fe_step._helpers import _non_numeric_column_indices

    class _Boom:
        """Frame stand-in whose dtype introspection always raises, forcing the swallow path."""

        columns = ["a", "b"]

        @property
        def schema(self):
            """Raise, simulating a polars schema access that fails."""
            raise RuntimeError("boom")

        @property
        def dtypes(self):
            """Raise, simulating a pandas dtypes access that fails."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.filters._mrmr_fe_step._helpers"):
        out = _non_numeric_column_indices(_Boom(), ["a", "b"])
    assert out == set()
    assert any("introspection failed" in r.message for r in caplog.records)


def test_feature_matrix_numpy_roundtrip_restores_nulls():
    """from_feature_matrix must restore NaN positions on the numpy branch (symmetric with pandas/polars), not
    return the raw float32 plane that skipped the null-mask restore."""
    from mlframe.feature_selection.filters._fe_matrix_io import from_feature_matrix, to_feature_matrix

    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]], dtype=np.float64)
    out = from_feature_matrix(to_feature_matrix(X))
    assert out.shape == X.shape
    assert np.isnan(out[0, 1]) and np.isnan(out[2, 0]), "null positions must survive the round-trip"
    assert not np.isnan(out[1, 0]) and out[1, 1] == 4.0
