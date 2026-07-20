"""Fast smoke coverage for the MRMR ground-truth recall/precision/F1 harness (2026-07-20).

``run_mrmr_fast_subset`` (bench_mdlp_validated_split_suite.py) had NO pytest coverage at all --
this exercises it as a real, fast (~seconds) test so a future change to ``MRMR_BINNING_METHODS``
or ``scen_multicolumn`` that breaks a binning method's end-to-end MRMR.fit path is caught, instead
of silently drifting until someone runs the (uncollected) full sweep by hand.
"""
from __future__ import annotations

from mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite import (
    MRMR_BINNING_METHODS,
    run_mrmr_gt_config,
)


def test_mrmr_fast_subset_runs_every_registered_binning_method():
    """Every method registered in MRMR_BINNING_METHODS must complete a real MRMR.fit without
    raising and report a finite recall/precision/F1 in [0, 1] -- catches a new/renamed strategy
    that silently breaks the ground-truth harness."""
    results = run_mrmr_gt_config(
        n=500, n_relevant=2, n_irrelevant=3, n_redundant=2,
        methods=tuple(MRMR_BINNING_METHODS), seeds=range(1), config_label="fast",
    )
    assert len(results) == len(MRMR_BINNING_METHODS)
    for r in results:
        assert 0.0 <= r.recall <= 1.0, (r.method, r.recall)
        assert 0.0 <= r.precision <= 1.0, (r.method, r.precision)
        assert 0.0 <= r.f1 <= 1.0, (r.method, r.f1)
        assert 0.0 <= r.fpr_noise_redundant <= 1.0, (r.method, r.fpr_noise_redundant)


def test_mrmr_uniform_and_oos_validated_recover_true_signal():
    """The two methods newly wired this session (uniform, mdlp_oos_validated) must actually find
    the true relevant column on an easy, low-noise scenario -- not just run without crashing."""
    results = run_mrmr_gt_config(
        n=800, n_relevant=2, n_irrelevant=2, n_redundant=0,
        methods=("uniform", "mdlp_oos_validated"), seeds=range(3), config_label="signal",
    )
    by_method: dict = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r.recall)
    for method, recalls in by_method.items():
        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= 0.5, f"{method}: mean recall {mean_recall:.2f} too low on an easy 2-relevant/0-redundant scenario"
