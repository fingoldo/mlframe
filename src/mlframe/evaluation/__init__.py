"""Model evaluation: performance reports across cv folds + holdout sets.

Submodules:
    reports   - full evaluation reporting (per-fold, summary tables, plots).
    bootstrap - bootstrap CIs + DeLong AUC test for honest-diagnostics.
"""

from __future__ import annotations


from mlframe.evaluation.reports import *  # noqa: F401,F403
from mlframe.evaluation.bootstrap import bootstrap_metric, delong_test  # noqa: F401
