"""Benchmark suite for mRMR feature selection.

Run with:
    python -m mlframe.feature_selection._benchmarks.bench_mrmr --scenarios all --tag pre-refactor

Aggregate before/after:
    python -m mlframe.feature_selection._benchmarks._aggregate \\
        --before _results/pre-refactor_<sha>.json --after _results/final_<sha>.json
"""

from __future__ import annotations

