"""Baseline-debt wave 13: closes the remaining genuine broad_except_swallow sites in the dev-only
``_benchmarks`` harness scripts (training/_benchmarks/, feature_selection/_benchmarks/), the last
category of non-MRMR broad_except_swallow debt outside the already-documented false-positive class
(model_card.py / _cb_pool.py / _repr.py / bench_pysr_fe.py / bench_bur_lambda_qual22.py's first
site / bench_mi_correction_miller_madow.py's first site, all of which already surface the
exception via their own return value). Reads source files directly off disk (rather than
importing) since several sibling benchmark scripts have pre-existing, unrelated import-time
issues (e.g. ``_profile_fuzz_1m.py``'s ``_run_suite_profiled`` re-export mismatch) that would
otherwise fail collection through no fault of the wave-13 edits.
"""

from __future__ import annotations

import mlframe

_MLFRAME_SRC_DIR = mlframe.__path__[0]


def _read(relpath: str) -> str:
    """Read a source file under `mlframe/` directly off disk, bypassing module import."""
    import os

    with open(os.path.join(_MLFRAME_SRC_DIR, relpath), encoding="utf-8") as fh:
        return fh.read()


def test_bench_arch_d_free_ram_bytes_logs_on_failure():
    """`_free_ram_bytes` must log on a psutil probe failure."""
    assert "_free_ram_bytes: psutil probe failed" in _read("training/_benchmarks/bench_arch_d.py")


def test_bench_content_fingerprint_rss_mb_logs_on_failure():
    """`_rss_mb` must log on a psutil probe failure."""
    assert "_rss_mb: psutil probe failed" in _read("training/_benchmarks/bench_content_fingerprint.py")


def test_bench_drift_value_counts_microbench_logs_on_failure():
    """`_is_object_array_col` must log on a dtype-probe failure."""
    assert "_is_object_array_col: probe failed" in _read("training/_benchmarks/bench_drift_value_counts_microbench.py")


def test_bench_lgb_dataset_polars_bridge_rss_mb_logs_on_failure():
    """`_rss_mb` must log on a psutil probe failure."""
    assert "_rss_mb: psutil probe failed" in _read("training/_benchmarks/bench_lgb_dataset_polars_bridge.py")


def test_bench_adaptive_nbins_ab_logs_on_failure():
    """The per-method edge computation must log on failure."""
    assert "edge computation failed for method" in _read("feature_selection/_benchmarks/bench_adaptive_nbins_ab.py")


def test_bench_boruta_auto_dispatch_logs_on_failure():
    """`_honest_holdout_auc` must log on a refit/score failure."""
    assert "_honest_holdout_auc: refit/score failed" in _read("feature_selection/_benchmarks/bench_boruta_auto_dispatch.py")


def test_bench_bur_lambda_qual22_logs_on_failure():
    """`_downstream` must log on an AUC-scoring failure."""
    assert "_downstream: AUC scoring failed" in _read("feature_selection/_benchmarks/bench_bur_lambda_qual22.py")


def test_bench_fs_levers_dflip_logs_on_failure():
    """The per-config MRMR fit/score must log on failure."""
    assert "config failed, scoring as nan" in _read("feature_selection/_benchmarks/bench_fs_levers_dflip.py")


def test_bench_mi_correction_miller_madow_logs_on_failure():
    """`_downstream` must log on an AUC-scoring failure."""
    assert "_downstream: AUC scoring failed" in _read("feature_selection/_benchmarks/bench_mi_correction_miller_madow.py")


def test_bench_mrmr_git_sha_and_gpu_model_log_on_failure():
    """`_git_sha` and `_gpu_model` must both log on their respective probe failures."""
    src = _read("feature_selection/_benchmarks/bench_mrmr.py")
    assert "_git_sha: git rev-parse failed" in src
    assert "_gpu_model: cupy device probe failed" in src


def test_bench_mrmr_threading_vs_loky_logs_on_failure():
    """`_peak_rss_mb` must log on a psutil probe failure."""
    assert "_peak_rss_mb: psutil probe failed" in _read("feature_selection/_benchmarks/bench_mrmr_threading_vs_loky.py")


def test_profile_wellbore_mrmr_only_100k_prints_on_failure():
    """The dump-audit hook must print (its own logging convention, gated behind
    ``WELLBORE_DUMP_AUDIT=1``) on introspection failure rather than silently swallowing it."""
    assert "audit hook failed" in _read("feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py")


def test_round4_su_seeded_interactions_bench_logs_on_failure():
    """`_qbin` must log on a quantile-binning failure."""
    assert "_qbin: quantile binning failed" in _read("feature_selection/_benchmarks/fs_hybrid/round4_su_seeded_interactions_bench.py")


def test_round4_synergy_combine_bench_logs_on_failure():
    """`combine_referee` must log on a held-out-split failure."""
    assert "combine_referee: held-out split failed" in _read("feature_selection/_benchmarks/fs_hybrid/round4_synergy_combine_bench.py")


def test_mrmr_largeN_campaign_logs_on_failure_and_reraises_when_requested():
    """The per-cell campaign runner must log on failure, and re-raise when `MRMR_CAMPAIGN_RAISE` is set."""
    src = _read("feature_selection/_benchmarks/fs_quality/mrmr_largeN_campaign.py")
    assert "campaign cell failed, scoring as None" in src
    assert 'if os.environ.get("MRMR_CAMPAIGN_RAISE"):' in src
