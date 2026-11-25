"""Cluster 10 regression: benchmark harness failures are logged at WARNING, not DEBUG (audit 2026-11-20, finding 10-F6).

The harnesses converted MRMR and scoring failures into NaN metrics and logged the reason at DEBUG, so the default level hid why a row was NaN.
"""

from pathlib import Path

from mlframe.feature_selection._benchmarks import bench_bur_lambda_qual22 as bur
from mlframe.feature_selection._benchmarks import bench_fs_levers_dflip as dflip


def test_bur_lambda_failures_log_at_warning():
    """The MRMR and AUC failure logs are WARNING (10-F6)."""
    src = Path(bur.__file__).resolve().read_text(encoding="utf-8")
    assert 'logger.warning("_eval: MRMR fit/transform failed' in src
    assert 'logger.warning("_downstream: AUC scoring failed' in src
    assert 'logger.debug("_eval: MRMR' not in src


def test_dflip_failure_logs_at_warning():
    """The per-config failure log is WARNING (10-F6)."""
    src = Path(dflip.__file__).resolve().read_text(encoding="utf-8")
    assert 'logger.warning("bench_fs_levers_dflip: config failed' in src
    assert 'logger.debug("bench_fs_levers_dflip: config failed' not in src
