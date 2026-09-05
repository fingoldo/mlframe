"""Null beds: the target is independent of every column, at four widths.

This is the cheapest and most diagnostic family in the suite, and it belongs first. Without it, a method
that selects generously collects recall on every other bed and never pays for a false discovery -- the
benchmark rewards greed and calls it sensitivity.

What a null bed measures is not accuracy but discipline: how many columns a method claims when the truth is
that none of them matter. An arm that selects more than its nominal false-discovery rate here is flagged in
every subsequent table, because its results elsewhere are partly the same behaviour with a better cover
story.

The widths matter separately. At ``p = 10`` almost nothing goes wrong; at ``p = 1000`` with a few thousand
rows, the best-looking noise column is very good indeed, and any method choosing by a threshold on a
marginal statistic will take it. The curve across widths is the finding, not any single point on it.
"""

from __future__ import annotations

from typing import Tuple

from mlframe.data.datasets.spec import DatasetSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["null_spec", "NULL_WIDTHS"]

#: Widths the null curve is measured at. The interesting region is the last two: a wide null bed is where a
#: threshold on a marginal statistic starts finding structure that is not there.
NULL_WIDTHS: Tuple[int, ...] = (10, 100, 1000, 5000)


def null_spec(width: int = 100, n_samples: int = 5000, seed: int = 0) -> DatasetSpec:
    """Return a spec whose target is independent of every column.

    The link carries no coefficients at all, so the score is a constant and the probability is the
    prevalence. That is the honest construction: drawing a target from an unrelated column would leave a
    faint dependence through the shared stream, and the point of a null bed is that there is none.

    Args:
        width: Number of noise columns.
        n_samples: Rows.
        seed: Root seed; development range for tuning, reserved range for reporting.

    Returns:
        The dataset specification.
    """
    features = tuple(FeatureSpec(name=f"n{i:04d}") for i in range(width))
    return DatasetSpec(
        name=f"null_p{width}",
        n_samples=n_samples,
        root_seed=seed,
        features=features,
        targets=(TargetSpec(name="y", prevalence=0.5, link=LinkSpec(kind="logistic", coefficients={})),),
        edges=(),
        provenance={"family": "null", "purpose": "false-discovery discipline at four widths"},
    )
