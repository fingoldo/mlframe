"""Cross-package API of ``mlframe.evaluation``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.evaluation`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._bootstrap_jackknife import (
    _ci_from_samples as ci_from_samples,
    _jackknife_auc as jackknife_auc,
    _jackknife_ece as jackknife_ece,
    _jackknife_metric as jackknife_metric,
)

__all__ = [
    "ci_from_samples",
    "jackknife_auc",
    "jackknife_ece",
    "jackknife_metric",
]
