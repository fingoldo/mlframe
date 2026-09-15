"""A prewarp whose operand length differs from the target's must be rejected, not accepted (IMPL-3, found in mrmr_audit_2026-09-14).

``_prewarp_generalises`` returned ``True`` ("accept the warp") when an operand's length differed from the validation target's, with the
comment "subsample edge -> don't block". There is no such legitimate edge: under subsampling the pair search narrows ``X`` itself to the
sample rows, operands are read from that ``X``, and the prewarp target is cut to the same rows, so on every correct path the lengths agree.
A mismatch can only mean the operand and target rows are misaligned, and accepting it both skipped the held-out check and let the full ALS
fit pair operand rows with the wrong target rows. With held-out validation switched off the closure accepted before ever comparing.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import hermite_fe
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import _fit_prewarp_and_gate_med

_N = 120  # >= 60 so the held-out path is active when enabled


def _run(monkeypatch, operand_rows: int, min_val_corr: float):
    """Prewarp setup for one prospective pair; operands have ``operand_rows`` rows against an ``_N``-row target."""

    def _fake_als(a, b, y, basis=None, max_degree=None):
        """A fitter that succeeds on any input, so only the length check can decide."""
        return {"spec": "a"}, {"spec": "b"}

    monkeypatch.setattr(hermite_fe, "fit_pair_prewarp_als", _fake_als)
    monkeypatch.setattr(hermite_fe, "apply_operand_prewarp", lambda x, spec: np.asarray(x, dtype=np.float64))
    rng = np.random.default_rng(0)
    data = rng.normal(size=(_N, 2))
    y = data[:, 0] * data[:, 1]  # genuine product synergy, so an aligned pair passes held-out validation
    return _fit_prewarp_and_gate_med(
        prospective_pairs={((0, 1), "mul"): None},
        prewarp_enable=True,
        prewarp_y=y,
        prewarp_y_continuous=None,
        prewarp_basis="hermite",
        prewarp_max_degree=3,
        prewarp_min_val_corr=min_val_corr,
        fe_gate_med_enable=False,
        original_cols=[0, 1],
        _use_subsample=False,
        _full_n_rows=_N,
        _sample_idx=None,
        _extval_raw_col=lambda v: data[:operand_rows, v],
    )


@pytest.mark.parametrize("min_val_corr", [0.08, 0.0], ids=["held_out_validation_on", "held_out_validation_off"])
def test_misaligned_operand_is_not_registered(monkeypatch, caplog, min_val_corr):
    """Operands 10 rows short of the target: no warp may be registered, whether or not held-out validation is enabled, and it must warn."""
    with caplog.at_level(logging.WARNING):
        _active, spec_by_var, _gm_active, _gm_medians = _run(monkeypatch, operand_rows=_N - 10, min_val_corr=min_val_corr)
    assert spec_by_var == {}, f"a warp was registered for operands misaligned with the target: {spec_by_var!r}"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and "length" in r.getMessage()], "the misalignment was not reported"


def test_aligned_operands_still_register_the_warp(monkeypatch):
    """Control: the same pair with aligned lengths is admitted, so the rejection above is about the length and nothing else."""
    _active, spec_by_var, _gm_active, _gm_medians = _run(monkeypatch, operand_rows=_N, min_val_corr=0.08)
    assert set(spec_by_var) == {0, 1}
