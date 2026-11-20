"""The resident-operand, raw-redundancy, retention-subsumption and rejection-ledger fallbacks log a WARNING (driven on CPU by stubbing the failing call)."""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def _boom(*args, **kwargs):
    """Fail like a device or estimator fault."""
    raise RuntimeError("boom")


class _BadStr:
    """An object that cannot be stringified."""

    def __str__(self) -> str:
        """Fail like an unprintable operand."""
        raise RuntimeError("no str")

    __repr__ = __str__


def test_resident_assembly_failure_warns_and_uploads_whole_matrix(monkeypatch, caplog):
    """A failing per-column assembly falls back to the whole-matrix upload and warns."""
    from mlframe.feature_selection.filters import _fe_resident_operands as mod

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(mod, "_disabled", lambda: False)

    def _resident(host, key, dtype=None):
        """Fail per column, succeed for the whole matrix."""
        if isinstance(key, tuple):
            raise RuntimeError("column upload broke")
        return host

    monkeypatch.setattr(mod, "resident_operand", _resident)
    with caplog.at_level(logging.WARNING):
        out = mod.assemble_resident_matrix(np.ones((4, 2)), ["a", "b"], "whole")
    assert out.shape == (4, 2)
    assert _warned(caplog, "per-column resident-operand assembly failed")


def test_tail_concentrated_probe_failure_warns(caplog):
    """A failing tail-concentration probe answers not-subsumed and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_helpers as mod

    with caplog.at_level(logging.WARNING):
        assert mod.raw_is_tail_concentrated_subsumed(np.array(["a", "b", "c"]), np.zeros(3), []) is False
    assert _warned(caplog, "raw_is_tail_concentrated_subsumed: probe failed")


def test_ridge_import_failure_warns(monkeypatch, caplog):
    """A missing sklearn skips the held-out Ridge probe and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_helpers as mod

    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)
    with caplog.at_level(logging.WARNING):
        assert mod._heldout_ridge_r2(np.zeros((10, 2)), np.zeros(10)) is None
    assert _warned(caplog, "the held-out Ridge probe is skipped")


def test_no_harm_recheck_failure_warns(caplog):
    """A failing held-out Ridge no-harm re-check leaves the drop standing and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as mod

    n = 100
    st = SimpleNamespace(n_rows=n, drop_names=[])
    y = np.random.default_rng(0).normal(size=n)
    with caplog.at_level(logging.WARNING):
        out = mod._no_harm_guard_result(st, ["a"], None, set(), None, np.zeros((n, 1)), np.zeros(n, dtype=np.int64), y, 0)
    assert out is None
    assert _warned(caplog, "no-harm re-check failed")


def test_informativeness_probe_failure_warns_and_answers_true(caplog):
    """A failing informativeness probe answers True (the subsumption verdict decides) and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as mod

    st = SimpleNamespace(_raw_marginal=_boom)
    with caplog.at_level(logging.WARNING):
        assert mod._some_consumer_adds_information(st, [0], "r", 0) is True
    assert _warned(caplog, "informativeness probe failed for r")


def test_join_support_device_failure_warns(monkeypatch, caplog):
    """A failing device-resident support join falls back to the host join and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as mod
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe

    monkeypatch.setattr(_mi_greedy_cmi_fe, "_renumber_joint_gpu", _boom)
    a = np.array([0, 1, 0, 1], dtype=np.int64)
    with caplog.at_level(logging.WARNING):
        _z, _card, z_dev = mod._join_support([a], [object()])
    assert z_dev is None
    assert _warned(caplog, "device-resident conditioning-support join failed")


def test_linear_leg_failure_warns(monkeypatch, caplog):
    """A failing linear-usability leg leaves the CMI verdict standing and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as mod

    monkeypatch.setattr(mod, "raw_retains_linear_signal_given_children", _boom)
    st = SimpleNamespace(y_arr=np.zeros(5))
    with caplog.at_level(logging.WARNING):
        keep = mod._drop_redundant_raw_o_step2_adversarial_multi_contamination(
            False, False, 0.0, 0.0, True, None, "a", np.zeros((5, 1)), 0, None, st, [0], ["a"], None, 0, 0
        )
    assert keep is False
    assert _warned(caplog, "linear-usability-leg probe failed")


def test_tail_probe_failure_warns(caplog):
    """A failing tail-concentration probe keeps the CMI verdict and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as mod

    with caplog.at_level(logging.WARNING):
        keep = mod._drop_redundant_raw_o_step3_profiles_already_drop(
            True, True, np.array(["x", "y", "z"]), None, "a", np.zeros((3, 1)), 0, [0], ["a"], None, 0.85, 0.7, 0
        )
    assert keep is True
    assert _warned(caplog, "tail-concentration continuous-subsumption probe failed")


def test_group_leak_check_failure_warns(monkeypatch, caplog):
    """A failing group-aware leak check exempts nothing and warns."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as mod
    from mlframe.feature_selection.filters.info_theory import shared

    monkeypatch.setattr(shared, "group_leak_verdict", _boom)
    st = SimpleNamespace(_gmi_payload=(1, 2, 3, 4), drop_names=["a"], _group_leak_names=set())
    with caplog.at_level(logging.WARNING):
        mod._drop_redundant_raw_o_step2_st_gmi_payload(st, np.zeros(5, dtype=np.int64), ["a"], np.zeros((5, 1), dtype=np.int64), 0)
    assert st._group_leak_names == set()
    assert _warned(caplog, "group-aware leak check failed")


def test_retention_device_replay_failure_warns(monkeypatch, caplog):
    """A failing device replay falls back to the host replay and warns."""
    from mlframe.feature_selection.filters import _fe_retention_subsumption as mod
    from mlframe.feature_selection.filters import _gpu_strict_fe

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_strict_resident_enabled", _boom)
    with caplog.at_level(logging.WARNING):
        out = mod.replay_candidate_continuous(SimpleNamespace(kind="unary_binary"), None, lambda r, X: np.arange(3.0))
    assert out.shape == (3,)
    assert _warned(caplog, "device replay of the retention candidate failed")


def test_retention_import_failure_warns(monkeypatch, caplog):
    """A failing import of the permutation-null path retains the candidate and warns."""
    from mlframe.feature_selection.filters import _fe_retention_subsumption as mod

    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters._fe_cmi_redundancy_gate", None)
    with caplog.at_level(logging.WARNING):
        out = mod.retention_form_is_subsumed(cand_continuous=np.arange(10.0), incumbent_continuous=[np.arange(10.0)], y_binned=np.zeros(10, dtype=np.int64))
    assert out is False
    assert _warned(caplog, "import for the conditional-permutation-null path failed")


def test_operands_str_failure_warns(caplog):
    """An unstringifiable operand collection yields a placeholder and warns."""
    from mlframe.feature_selection.filters import _fe_rejection_ledger as mod

    with caplog.at_level(logging.WARNING):
        assert mod._operands_str(_BadStr()) == "<unstringifiable>"
    assert _warned(caplog, "could not stringify operand collection")


def test_admission_cap_warns(caplog):
    """Hitting the admission-ledger cap warns once instead of dropping records silently."""
    from mlframe.feature_selection.filters import _fe_rejection_ledger as mod

    ms = SimpleNamespace(_fe_admission_records_=[{}], fe_rejection_ledger_cap=1)
    with caplog.at_level(logging.WARNING):
        mod.record_fe_admission(ms, candidate="c")
    assert len(ms._fe_admission_records_) == 1
    assert _warned(caplog, "admission ledger hit its record cap")


def test_admission_record_failure_warns(caplog):
    """A failing admission record is skipped and warns."""
    from mlframe.feature_selection.filters import _fe_rejection_ledger as mod

    with caplog.at_level(logging.WARNING):
        mod.record_fe_admission(SimpleNamespace(), candidate=_BadStr())
    assert _warned(caplog, "admission ledger record failed")


def test_rejection_record_failure_warns(caplog):
    """A failing rejection record is skipped and warns."""
    from mlframe.feature_selection.filters import _fe_rejection_ledger as mod

    with caplog.at_level(logging.WARNING):
        mod.record_fe_rejection(SimpleNamespace(), gate="g", candidate=_BadStr())
    assert _warned(caplog, "rejection ledger record failed")
