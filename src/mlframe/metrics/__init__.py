"""Probabilistic, classification, ranking, and quantile metrics.

Submodules:
    core      - core metric definitions (ICE, ECE, Brier REL/RES/UNC, CMAEW, ...).
    quantile  - quantile-specific metrics (pinball, coverage).
    ranking   - ranking-task metrics (NDCG, MRR, ...).
    scoring   - sklearn-compatible scoring wrappers.

Import convention: this package's ``__all__`` (below) is the curated EXTERNAL-facing surface -- the
one a caller outside ``metrics/`` who doesn't know the internal module layout should reach for. It is
NOT a mandate that every internal call site rewrite ``from mlframe.metrics.core import X`` to
``from mlframe.metrics import X``: ``core.py`` (unlike the underscore-prefixed internal submodules) is
itself an intentionally public, directly-importable module, and importing straight from it avoids
eagerly loading this package's other submodules (``quantile``/``ranking``/``scoring``/``calibration``/
``classification``/``regression``/``iteration_metrics``) just to reach one ``core`` symbol. Both forms
are supported and bit-identical (the facade re-exports ``core`` in full via ``from .core import *``);
pick whichever import-cost tradeoff fits the call site.
"""

from __future__ import annotations


from mlframe.metrics.core import *
from mlframe.metrics.quantile import *
from mlframe.metrics.ranking import *
from mlframe.metrics.scoring import *

# Public re-export so cross-package consumers (importance.py, _eval_helpers.py) can avoid reaching into ``mlframe.metrics.calibration`` internals directly. The underscore-prefixed source remains the implementation; the public name is the documented surface.
from mlframe.metrics.calibration import _show_plots_unless_agg as show_plots_unless_agg
# Public re-export of the Brier kernel so reporting consumers (model_card) import it from the package surface instead of the ``_core_auc_brier`` implementation module.
from mlframe.metrics._core_auc_brier import fast_brier_score_loss
# Lean full-suite per-target-type aggregator used by per-iteration metric capture (meta-learning / HPO-from-early-observation).
from mlframe.metrics.iteration_metrics import compute_all_metrics
# Per-target calibration / classification panel. Documented in README as ``from mlframe.metrics import fast_calibration_report``; make it a first-class export instead of relying on the transitive ``from .core import *`` side-effect. ``CalibrationReport`` is its NamedTuple return type.
from mlframe.metrics.classification import (
    CalibrationReport,
    fast_calibration_report,
)

# Public re-export of the scalar classification / regression metrics so callers use the documented
# ``from mlframe.metrics import quadratic_weighted_kappa`` surface instead of reaching into the concrete submodule (which
# is exactly what the metric registry itself imports). Only ``core`` is star-imported above, so these submodule names
# were otherwise reachable only as ``mlframe.metrics.classification.*`` / ``.regression.*``. Explicit (not star) to avoid
# shadowing anything the ``core`` star already brought in.
from mlframe.metrics.classification import (
    apply_cutpoints,
    cumulative_gains_curve,
    exploss,
    gains_table,
    lift_curve,
    optimal_ordinal_cutpoints,
    optimal_threshold,
    quadratic_weighted_kappa,
    weighted_kappa,
)
from mlframe.metrics.regression import (
    fast_epsilon_band_accuracy,
    fast_logcosh_loss,
    fast_mrae,
    fast_percent_better,
    fast_rel_mae,
    fast_rmspe,
)

# Preserve the historical ``from mlframe.metrics import *`` surface (every public name the star-imports above brought in)
# while GUARANTEEING the documented ``fast_calibration_report`` / ``CalibrationReport`` are part of it as a first-class contract.
__all__ = sorted({name for name in globals() if not name.startswith("_")} | {"CalibrationReport", "fast_calibration_report"})
