"""Cluster 10 regression: local bench helpers that diverge from the shared ones say so (audit 2026-11-20, finding 10-F4).

Each local copy of a shared helper carries a marker naming the intentional divergence, so a later edit to the shared version is not
mistaken for a missed update.
"""

from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks"


@pytest.mark.parametrize(
    "module, marker",
    [
        ("bench_iter31_shap_prefilter.py", "Intentional divergence from _bench_shared.print_stage_table (audit 10-F4)"),
        ("bench_iter33_scaling.py", "Intentional divergence from _bench_shared.print_stage_table (audit 10-F4)"),
        ("bench_iter54_inner_n_jobs_cap.py", "Intentional divergence from _bench_shared.print_stage_table (audit 10-F4)"),
        ("bench_stability_orchestration.py", "Intentional divergence from _bench_shared.build_selector (audit 10-F4)"),
    ],
)
def test_local_copy_declares_its_divergence(module, marker):
    """The local helper definition is preceded by the divergence marker (10-F4)."""
    src = (BENCH / module).read_text(encoding="utf-8")
    assert marker in src
