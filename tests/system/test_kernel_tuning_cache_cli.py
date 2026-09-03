"""Coverage for the mlframe-tune-kernels CLI (mlframe.system.kernel_tuning_cache).

X_TEST_SUITE_ARCHITECTURE-6: main + all 6 cmd_* subcommands had zero test coverage,
flagged by a prior audit cycle (F6) and left unaddressed for 15 days. Exercises every
subcommand's dispatch and output via a fake spec + mocked registry/cache calls -- no
real GPU/kernel-tuning work is performed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mlframe.system import kernel_tuning_cache as ktc_cli


def _fake_spec(name="joint_hist_2d", cli_label=None, gpu_capable=True):
    """Build a minimal stand-in for a TunerSpec with just the attributes the CLI reads."""
    return SimpleNamespace(
        kernel_name=name,
        cli_label=cli_label,
        gpu_capable=gpu_capable,
        axes=["n", "k"],
        extra_fns=[],
        salt="s",
        env_key="MLFRAME_KTC_JOINT_HIST_2D",
        equiv_tol=1e-6,
    )


def test_find_spec_by_kernel_name():
    """_find_spec resolves a direct kernel_name key."""
    spec = _fake_spec()
    specs = {"joint_hist_2d": spec}
    assert ktc_cli._find_spec(specs, "joint_hist_2d") is spec


def test_find_spec_by_cli_label():
    """_find_spec falls back to matching cli_label when the key itself doesn't match."""
    spec = _fake_spec(cli_label="jh2d")
    specs = {"joint_hist_2d": spec}
    assert ktc_cli._find_spec(specs, "jh2d") is spec


def test_find_spec_missing_returns_none():
    """_find_spec returns None for an unknown kernel."""
    assert ktc_cli._find_spec({"a": _fake_spec("a")}, "nope") is None


def test_parse_dims_mixed_int_and_str():
    """_parse_dims parses 'a=1,b=x' into {'a': 1, 'b': 'x'}, ignoring malformed pairs."""
    assert ktc_cli._parse_dims("a=1,b=x,garbage") == {"a": 1, "b": "x"}


def test_parse_dims_none_returns_empty():
    """_parse_dims(None) returns an empty dict."""
    assert ktc_cli._parse_dims(None) == {}


def test_cmd_list_prints_specs(capsys):
    """cmd_list returns 0 and prints every discovered kernel name."""
    specs = {"a": _fake_spec("a", gpu_capable=True), "b": _fake_spec("b", gpu_capable=False)}
    rc = ktc_cli.cmd_list(specs)
    out = capsys.readouterr().out
    assert rc == 0
    assert "a" in out and "b" in out
    assert "[GPU]" in out and "[CPU]" in out


def test_cmd_list_empty_specs(capsys):
    """cmd_list on an empty dict still returns 0 with a 'no specs' message."""
    rc = ktc_cli.cmd_list({})
    assert rc == 0
    assert "No specs" in capsys.readouterr().out


def test_cmd_show_known_kernel(capsys):
    """cmd_show prints spec fields and returns 0 for a known kernel."""
    spec = _fake_spec("joint_hist_2d")
    rc = ktc_cli.cmd_show({"joint_hist_2d": spec}, "joint_hist_2d")
    out = capsys.readouterr().out
    assert rc == 0
    assert "joint_hist_2d" in out
    assert "GPU capable" in out


def test_cmd_show_unknown_kernel_returns_1(capsys):
    """cmd_show returns 1 and writes to stderr for an unknown kernel."""
    rc = ktc_cli.cmd_show({}, "nope")
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_cmd_explain_calls_lookup_explain(capsys):
    """cmd_explain resolves the spec, parses --dims, and prints the cache's JSON explanation."""
    spec = _fake_spec("joint_hist_2d")
    with patch.object(ktc_cli, "KernelTuningCache") as mock_cache_cls:
        mock_cache_cls.return_value.lookup_explain.return_value = {"decision": "cached"}
        rc = ktc_cli.cmd_explain({"joint_hist_2d": spec}, "joint_hist_2d", "n=1000")
    out = capsys.readouterr().out
    assert rc == 0
    assert "cached" in out
    mock_cache_cls.return_value.lookup_explain.assert_called_once_with("joint_hist_2d", n=1000)


def test_cmd_refresh_known_kernel_calls_tune_spec(capsys):
    """cmd_refresh forces a tune_spec(force=True) call and reports the persisted region count."""
    spec = _fake_spec("joint_hist_2d")
    with patch.object(ktc_cli, "tune_spec", return_value=3) as mock_tune:
        rc = ktc_cli.cmd_refresh({"joint_hist_2d": spec}, "joint_hist_2d")
    out = capsys.readouterr().out
    assert rc == 0
    assert "3 region" in out
    mock_tune.assert_called_once_with(spec, force=True)


def test_cmd_refresh_unknown_kernel_returns_1(capsys):
    """cmd_refresh returns 1 for an unknown kernel without calling tune_spec."""
    with patch.object(ktc_cli, "tune_spec") as mock_tune:
        rc = ktc_cli.cmd_refresh({}, "nope")
    assert rc == 1
    mock_tune.assert_not_called()


def test_cmd_refresh_all_calls_retune_all(capsys):
    """cmd_refresh_all calls retune_all(force=True) and prints one line per (model, kernel)."""
    with patch.object(ktc_cli, "retune_all", return_value={("cpu", "joint_hist_2d"): 2}) as mock_retune:
        rc = ktc_cli.cmd_refresh_all({"joint_hist_2d": _fake_spec()})
    out = capsys.readouterr().out
    assert rc == 0
    assert "joint_hist_2d" in out and "cpu" in out
    mock_retune.assert_called_once_with(package="mlframe", force=True)


def test_cmd_clear_known_kernel(capsys):
    """cmd_clear evicts the cache entry for a known kernel and reports eviction."""
    spec = _fake_spec("joint_hist_2d")
    with patch.object(ktc_cli, "KernelTuningCache") as mock_cache_cls:
        mock_cache_cls.return_value.evict.return_value = True
        rc = ktc_cli.cmd_clear({"joint_hist_2d": spec}, "joint_hist_2d")
    out = capsys.readouterr().out
    assert rc == 0
    assert "evicted" in out
    mock_cache_cls.return_value.evict.assert_called_once_with("joint_hist_2d")


def test_cmd_clear_no_entry(capsys):
    """cmd_clear reports 'no entry' when evict() finds nothing to remove."""
    with patch.object(ktc_cli, "KernelTuningCache") as mock_cache_cls:
        mock_cache_cls.return_value.evict.return_value = False
        rc = ktc_cli.cmd_clear({}, "unknown_kernel")
    assert rc == 0
    assert "no entry" in capsys.readouterr().out


def test_main_no_command_prints_help(capsys):
    """main() with no subcommand prints help and returns 0 without discovering specs."""
    with patch.object(ktc_cli, "discover_tuners") as mock_discover:
        rc = ktc_cli.main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()
    mock_discover.assert_not_called()


def test_main_explicit_empty_argv_treated_as_no_command(capsys):
    """main(argv=[]) means 'no arguments were given' and must print help, not fall back to
    sys.argv (`argv or sys.argv[1:]` would wrongly treat an explicit [] as falsy)."""
    with patch.object(ktc_cli, "discover_tuners") as mock_discover:
        rc = ktc_cli.main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()
    mock_discover.assert_not_called()


def test_main_no_specs_discovered_returns_1(capsys):
    """main() returns 1 and writes to stderr when discover_tuners finds nothing."""
    with patch.object(ktc_cli, "discover_tuners", return_value={}):
        rc = ktc_cli.main(["list"])
    assert rc == 1
    assert "No specs discovered" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv,expected_cmd",
    [
        (["list"], "cmd_list"),
        (["show", "joint_hist_2d"], "cmd_show"),
        (["explain", "joint_hist_2d"], "cmd_explain"),
        (["refresh", "joint_hist_2d"], "cmd_refresh"),
        (["refresh-all"], "cmd_refresh_all"),
        (["clear", "joint_hist_2d"], "cmd_clear"),
    ],
)
def test_main_dispatches_to_correct_subcommand(argv, expected_cmd):
    """main() routes each subcommand's argv to its matching cmd_* handler."""
    spec = _fake_spec("joint_hist_2d")
    with (
        patch.object(ktc_cli, "discover_tuners", return_value={"joint_hist_2d": spec}),
        patch.object(ktc_cli, expected_cmd, return_value=0) as mock_cmd,
    ):
        rc = ktc_cli.main(argv)
    assert rc == 0
    mock_cmd.assert_called_once()
