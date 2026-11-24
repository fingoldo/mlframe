"""Cluster 39 regression: the iter113 token benchmark times the production function, not a copy (audit 2026-11-20, finding 39-F50).

The NEW arm used to re-implement the token logic inline, so the benchmark could drift from production. It now calls
``_float_canonical_tokens``. The OLD ``np.unique`` arm stays inline on purpose as the baseline.
"""

from pathlib import Path

from mlframe.training.feature_handling import _target_encoders_canon as canon
from mlframe.training.feature_handling._benchmarks import bench_float_canonical_tokens_iter113 as bench


def test_bench_imports_the_production_function():
    """The benchmark's NEW arm is the production _float_canonical_tokens (39-F50)."""
    assert bench._float_canonical_tokens is canon._float_canonical_tokens


def test_bench_no_longer_inlines_the_factorize_arm():
    """The copied factorize-and-tokenise block is gone from the benchmark (39-F50)."""
    src = Path(bench.__file__).resolve().read_text(encoding="utf-8")
    assert "codes, uniqf = pd.factorize(arr, sort=False)" not in src
    assert "_float_canonical_tokens(arr)" in src
