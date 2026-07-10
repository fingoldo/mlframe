"""Model evaluation: performance reports across cv folds + holdout sets.

Submodules:
    reports   - full evaluation reporting (per-fold, summary tables, plots).
    bootstrap - bootstrap CIs + DeLong AUC test for honest-diagnostics.

iter631: ``reports`` is NOT eager-imported. Loading it cascades through
matplotlib + IPython + sklearn (~2.5-3s per process); the honest-diagnostics
suite only needs ``bootstrap`` for CI computation and never touches reports
during a training run. PEP 562 ``__getattr__`` lazy-loads reports symbols
on first attribute access so external ``from mlframe.evaluation import X``
callers keep working at the cost of a one-time deferred import.
"""

from __future__ import annotations


from mlframe.evaluation.bootstrap import auc_ci, auc_variance, bootstrap_metric, delong_test
from mlframe.evaluation.noise_band import cv_score_equivalence_band, is_within_noise_band
from mlframe.evaluation.cv_delta_triage import triage_cv_delta
from mlframe.evaluation.leak_scan import scan_temporal_leak
from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check
from mlframe.evaluation.adversarial_feature_audit import adversarial_validation_feature_audit
from mlframe.evaluation.cv_informativeness import cv_informativeness_check
from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage
from mlframe.evaluation.subgroup_feature_overfit_risk import flag_subgroup_only_feature_overfit_risk
from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan
from mlframe.evaluation.adversarial_fold_selection import build_test_like_validation_fold
from mlframe.evaluation.adversarial_validator import AdversarialValidator
from mlframe.evaluation.compare_cv_schemes import compare_cv_schemes
from mlframe.evaluation.blend_source_selection import check_pairwise_score_correlation


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # ``from . import reports`` would route through ``_handle_fromlist`` ->
    # ``hasattr(package, 'reports')`` which re-enters this __getattr__ and
    # recurses. ``importlib.import_module`` bypasses the package-attribute
    # check and resolves the submodule via sys.modules directly.
    import importlib
    _r = importlib.import_module("mlframe.evaluation.reports")
    try:
        val = getattr(_r, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = val
    return val


def __dir__():
    import importlib
    _r = importlib.import_module("mlframe.evaluation.reports")
    return sorted(set(globals().keys()) | {n for n in dir(_r) if not n.startswith("_")})
