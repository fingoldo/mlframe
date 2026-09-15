"""Small FE-cascade contracts: gate-operand attributes are seeded with the other rosters, and the artifact validator checks what its schema asks.

* ``_gate_raw_operands_`` / ``_gate_col_src_vars_`` were seeded only inside one cascade stage, so a fit path that never ran it (the
  multioutput fan-out, which seeds rosters via ``seed_empty_fe_rosters``) left them undefined for their readers.
* ``validate_artifact_dict`` accepted a dict from a newer schema version and a ``bins`` without its ``nbins_per_feature``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_artifacts import ARTIFACT_SCHEMA_VERSION, validate_artifact_dict
from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import seed_empty_fe_rosters


def test_seed_empty_fe_rosters_also_seeds_gate_operand_attrs():
    """A multioutput-style estimator seeded with empty rosters must expose both gate-operand attributes, empty."""
    est = SimpleNamespace()
    seed_empty_fe_rosters(est)
    assert getattr(est, "_gate_raw_operands_", None) == set()
    assert getattr(est, "_gate_col_src_vars_", None) == {}


def _valid():
    """A minimal valid artifact dict."""
    return {"feature_names": ["a", "b"], "su_to_target": np.array([0.1, 0.2]), "schema_version": ARTIFACT_SCHEMA_VERSION, "n_samples_at_fit": 100}


def test_validate_artifact_dict_rejects_future_schema_version():
    """A dict from a newer schema than this reader understands is rejected."""
    art = _valid()
    art["schema_version"] = ARTIFACT_SCHEMA_VERSION + 1
    assert validate_artifact_dict(art) is False
    assert validate_artifact_dict(_valid()) is True


def test_validate_artifact_dict_rejects_half_present_bins():
    """``bins`` and ``nbins_per_feature`` are present together or not at all."""
    art = _valid()
    art["bins"] = {"a": np.zeros(100, dtype=np.int32)}
    assert validate_artifact_dict(art) is False
    art["nbins_per_feature"] = {"a": 1}
    assert validate_artifact_dict(art) is True


def test_validate_artifact_dict_checks_row_count_when_given():
    """A caller passing its own row count gets a mismatch rejected; without it the count is not checked."""
    assert validate_artifact_dict(_valid(), n_samples=100) is True
    assert validate_artifact_dict(_valid(), n_samples=99) is False
