"""count_ceiling is the exact deviance over every row of the law, not an extrapolation from a prefix (audit 2026-11-20, finding 04-F10)."""
import numpy as np

from mlframe.data.datasets._targets_nonbinary import count_ceiling


def test_block_result_equals_the_row_by_row_average_past_the_old_cutoff():
    """The old cutoff at 2001 rows scaled the prefix; rows after it must now move the answer, exactly as a row-by-row loop says."""
    rng = np.random.default_rng(0)
    means = np.exp(rng.normal(0.0, 0.8, 2300))
    per_row = np.mean([count_ceiling(np.array([m]))["poisson_deviance"] for m in means])
    np.testing.assert_allclose(count_ceiling(means)["poisson_deviance"], per_row, rtol=1e-9)
