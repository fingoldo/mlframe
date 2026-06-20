"""The kernel-tuning CLI must NOT re-benchmark kernels already validly cached for this host by default.

``refresh-all`` (and every ``refresh-*``) defaults to skip-existing; only ``--force`` re-runs the sweep.
Previously every refresh-* hard-coded ``force=True``, so ``refresh-all`` re-swept everything (minutes of
GPU thrash) even when a valid per-host result was already cached.
"""
from __future__ import annotations

from unittest import mock

import pytest

from mlframe.feature_selection._benchmarks.kernel_tuning_cache import cli


def _parse(argv):
    # Build the same parser main() builds, but stop before dispatch so we just inspect the namespace.
    import argparse

    # Reuse main() via a patched dispatch that returns the parsed args.
    captured = {}

    def _fake_dispatch(args):
        captured["args"] = args
        return 0

    # main() dispatches through a dict comprehension keyed on args.cmd; patch each handler to capture.
    handlers = {
        name: _fake_dispatch
        for name in (
            "_cmd_refresh", "_cmd_refresh_mi", "_cmd_refresh_polyeval",
            "_cmd_refresh_joint_hist_single_perm", "_cmd_refresh_joint_hist_multi_pair",
            "_cmd_refresh_batch_pair_mi", "_cmd_refresh_cat_fe_perm_kernel",
            "_cmd_refresh_rmse_partial_sum", "_cmd_refresh_unary_elementwise",
            "_cmd_refresh_rff_matmul", "_cmd_refresh_knn_hnsw_crossover",
            "_cmd_refresh_discretize_2d_array", "_cmd_refresh_batch_mi_noise_gate",
            "_cmd_refresh_all",
        )
    }
    with mock.patch.multiple(cli, **handlers):
        cli.main(argv)
    return captured["args"]


class TestForceFlagDefault:
    def test_refresh_all_defaults_force_false(self):
        assert _parse(["refresh-all"]).force is False

    def test_refresh_all_force_true_with_flag(self):
        assert _parse(["refresh-all", "--force"]).force is True

    @pytest.mark.parametrize("cmd", [
        "refresh", "refresh-mi", "refresh-batch-pair-mi", "refresh-discretize-2d-array",
        "refresh-batch-mi-noise-gate",
    ])
    def test_each_refresh_defaults_force_false(self, cmd):
        assert _parse([cmd]).force is False

    @pytest.mark.parametrize("cmd", ["refresh", "refresh-mi", "refresh-batch-pair-mi"])
    def test_each_refresh_force_flag(self, cmd):
        assert _parse([cmd, "--force"]).force is True


class TestRefreshGenericSkipsCache:
    """_refresh_generic must pass the operator's force through to ensure_fn (so force=False -> the
    ensure_* returns cached regions WITHOUT re-sweeping)."""

    def test_force_false_forwarded(self):
        seen = {}

        def _ensure(force):
            seen["force"] = force
            return [{"region": 1}]

        rc = cli._refresh_generic("k", _ensure, force=False)
        assert rc == 0 and seen["force"] is False

    def test_force_true_forwarded(self):
        seen = {}

        def _ensure(force):
            seen["force"] = force
            return [{"region": 1}]

        rc = cli._refresh_generic("k", _ensure, force=True)
        assert rc == 0 and seen["force"] is True


class TestRefreshViaRegistrySkipExisting:
    """_refresh_via_new_registry must call tune_spec with skip_existing=True and the operator's force."""

    def test_skip_existing_and_force_forwarded(self):
        rec = {}

        def _tune_spec(spec, *, force, skip_existing):
            rec["force"] = force
            rec["skip_existing"] = skip_existing
            return 3

        fake = mock.MagicMock()
        fake.discover_tuners = lambda package: None
        fake.get_registry = lambda: mock.Mock(get=lambda name: object())
        fake.tune_spec = _tune_spec
        with mock.patch.dict("sys.modules", {"pyutilz.performance.kernel_tuning": fake}):
            rc = cli._refresh_via_new_registry("batch_pair_mi", force=False)
        assert rc == 0 and rec["force"] is False and rec["skip_existing"] is True
