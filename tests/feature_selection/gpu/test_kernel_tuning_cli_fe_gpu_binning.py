"""The kernel-tuning CLI must expose and roll up the FE binning CPU-vs-GPU sweep."""

from __future__ import annotations

from unittest import mock

from mlframe.feature_selection._benchmarks.kernel_tuning_cache import cli
from mlframe.feature_selection.filters import _gpu_resident_pair_mi as pair_mi


def test_refresh_fe_gpu_binning_runs_the_binning_sweep_with_force_passthrough():
    """The subcommand hands ensure_fe_gpu_binning_tuning and the --force flag to the shared refresh wrapper."""
    with mock.patch.object(cli, "_refresh_generic", return_value=0) as gen:
        assert cli.main(["refresh-fe-gpu-binning", "--force"]) == 0
    label, fn = gen.call_args.args
    assert label == "fe_gpu_binning" and fn is pair_mi.ensure_fe_gpu_binning_tuning
    assert gen.call_args.kwargs == {"force": True}


def test_refresh_all_includes_the_binning_sweep_and_folds_its_failure():
    """refresh-all runs the binning sweep too, and a failing rc turns the rollup non-zero."""
    seen = []

    def _fake(label, fn, force=False):
        """Record the label and fail only the binning sweep."""
        seen.append(label)
        return 1 if label == "fe_gpu_binning" else 0

    handlers = {
        n: (lambda a: 0)
        for n in dir(cli)
        if (n == "_cmd_refresh" or n.startswith("_cmd_refresh_")) and n not in ("_cmd_refresh_all", "_cmd_refresh_fe_gpu_binning")
    }
    with mock.patch.object(cli, "_refresh_generic", side_effect=_fake), mock.patch.multiple(cli, **handlers):
        assert cli.main(["refresh-all"]) == 1
    assert seen == ["fe_gpu_binning"]
