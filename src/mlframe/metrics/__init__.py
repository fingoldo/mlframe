"""Probabilistic, classification, ranking, and quantile metrics.

Submodules:
    core      - core metric definitions (ICE, ECE, Brier REL/RES/UNC, CMAEW, ...).
    quantile  - quantile-specific metrics (pinball, coverage).
    ranking   - ranking-task metrics (NDCG, MRR, ...).
    scoring   - sklearn-compatible scoring wrappers.
"""

from __future__ import annotations


from mlframe.metrics.core import *  # noqa: F401,F403
from mlframe.metrics.quantile import *  # noqa: F401,F403
from mlframe.metrics.ranking import *  # noqa: F401,F403
from mlframe.metrics.scoring import *  # noqa: F401,F403

# Public re-export so cross-package consumers (importance.py, _eval_helpers.py) can avoid reaching into ``mlframe.metrics.calibration`` internals directly. The underscore-prefixed source remains the implementation; the public name is the documented surface.
from mlframe.metrics.calibration import _show_plots_unless_agg as show_plots_unless_agg  # noqa: F401
# Public re-export of the Brier kernel so reporting consumers (model_card) import it from the package surface instead of the ``_core_auc_brier`` implementation module.
from mlframe.metrics._core_auc_brier import fast_brier_score_loss  # noqa: F401
# Lean full-suite per-target-type aggregator used by per-iteration metric capture (meta-learning / HPO-from-early-observation).
from mlframe.metrics.iteration_metrics import compute_all_metrics  # noqa: F401
