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
# Per-target calibration / classification panel. Documented in README as ``from mlframe.metrics import fast_calibration_report``; make it a first-class export instead of relying on the transitive ``from .core import *`` side-effect. ``CalibrationReport`` is its NamedTuple return type.
from mlframe.metrics.classification._classification_report import (  # noqa: F401
    CalibrationReport,
    fast_calibration_report,
)

# Preserve the historical ``from mlframe.metrics import *`` surface (every public name the star-imports above brought in)
# while GUARANTEEING the documented ``fast_calibration_report`` / ``CalibrationReport`` are part of it as a first-class contract.
__all__ = sorted(
    {name for name in globals() if not name.startswith("_")}
    | {"CalibrationReport", "fast_calibration_report"}
)
