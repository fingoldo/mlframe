"""Every composite env switch reads the same vocabulary, and the kill switch keeps the caller's other settings (INT-16).

The switches were parsed three ways: ``not os.environ.get(NAME)`` (any non-empty value counted as on, so
``MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS=0`` kept the charts), ``lower() in {"1","true","yes"}`` (``on`` did not disable
anything) and an ad-hoc tuple. They all go through ``env_flag`` now. The kill switch also used to rebuild the discovery
config from ``{"enabled": False}``, discarding every other field the caller had set.
"""

from __future__ import annotations

import pytest

from mlframe.utils.env_flags import env_flag

_SWITCHES = [
    "MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS",
    "MLFRAME_DISABLE_COMPOSITE",
    "MLFRAME_DISCOVERY_CACHE_STRICT",
    "MLFRAME_DISCOVERY_SKIP_TINY_RERANK",
]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On ", "t"])
def test_a_true_spelling_turns_a_switch_on(value, monkeypatch):
    """Anything an operator would reasonably write for "on" reads as on."""
    monkeypatch.setenv("MLFRAME_TEST_FLAG", value)
    assert env_flag("MLFRAME_TEST_FLAG") is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " Off ", "", "f"])
def test_a_false_spelling_turns_a_switch_off(value, monkeypatch):
    """The off vocabulary is honoured, which is the defect: "0" used to mean on for two of these switches."""
    monkeypatch.setenv("MLFRAME_TEST_FLAG", value)
    assert env_flag("MLFRAME_TEST_FLAG", default=True) is False


def test_an_unset_or_unparsable_value_falls_back_to_the_default(monkeypatch):
    """An unset variable and a value in neither vocabulary both mean whatever the caller declared."""
    monkeypatch.delenv("MLFRAME_TEST_FLAG", raising=False)
    assert env_flag("MLFRAME_TEST_FLAG") is False
    assert env_flag("MLFRAME_TEST_FLAG", default=True) is True
    monkeypatch.setenv("MLFRAME_TEST_FLAG", "maybe")
    assert env_flag("MLFRAME_TEST_FLAG", default=True) is True


@pytest.mark.parametrize("name", _SWITCHES)
@pytest.mark.parametrize("value, expected", [("0", False), ("false", False), ("off", False), ("on", True), ("1", True)])
def test_each_composite_switch_reads_the_same_vocabulary(name, value, expected, monkeypatch):
    """The four composite switches agree with each other on what on and off look like."""
    monkeypatch.setenv(name, value)
    assert env_flag(name) is expected


def test_the_kill_switch_only_turns_discovery_off(monkeypatch):
    """``MLFRAME_DISABLE_COMPOSITE`` clears ``enabled`` and leaves every other field the caller set."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core._phase_config_setup import apply_composite_kill_switch

    cfg = CompositeTargetDiscoveryConfig(enabled=True, base_candidates=["b"], transforms=["ratio"], mi_sample_n=123)
    monkeypatch.setenv("MLFRAME_DISABLE_COMPOSITE", "on")
    off = apply_composite_kill_switch(cfg)
    assert off.enabled is False
    assert off.base_candidates == ["b"] and off.transforms == ["ratio"] and off.mi_sample_n == 123

    monkeypatch.setenv("MLFRAME_DISABLE_COMPOSITE", "0")
    assert apply_composite_kill_switch(cfg) is cfg, "0 must leave discovery alone"
