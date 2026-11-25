"""Cluster 11 regression: the wiring CLI passes its constructor parameters through (audit 2026-11-20, finding 11-F13).

``main()`` never passed ``params`` to plan or apply, so a generated operator had no constructor parameters. The CLI now takes repeatable
``--param NAME:TYPE`` and forwards them.
"""

from unittest.mock import patch

from mlframe.feature_selection._benchmarks.fe_operator_factory import wiring


def test_cli_forwards_params_to_plan():
    """A --param is parsed into a (name, type) tuple and passed to plan (11-F13)."""
    with patch.object(wiring, "plan", return_value=[]) as mock_plan:
        wiring.main(["--kind", "x", "--family", "f", "--flag", "F", "--fn", "fn", "--module", "m", "--prev", "row_stat", "--param", "alpha:float"])
    assert mock_plan.call_args.kwargs["params"] == [("alpha", "float")]
