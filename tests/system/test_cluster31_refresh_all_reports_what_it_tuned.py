"""Audit 2026-11-20 finding 31-F9: refresh-all reports the tuned results, and an empty result is a failure.

The command used to print the spec count it discovered, but tuned whatever retune_all re-discovered, and always returned 0. The test stubs
retune_all to tune nothing and checks the command returns 1 without printing a count of specs.
"""

from mlframe.system import kernel_tuning_cache as kc


def test_refresh_all_with_nothing_tuned_fails(monkeypatch, capsys):
    """An empty retune result gives exit code 1 and no 'Refreshing all N spec(s)' claim."""
    monkeypatch.setattr(kc, "retune_all", lambda package, force: {})
    code = kc.cmd_refresh_all({"a": object(), "b": object()})
    out = capsys.readouterr().out
    assert code == 1
    assert "spec(s)" not in out
