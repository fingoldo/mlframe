"""Meta-linter: production code must use mlframe's own fast metric kernels, not sklearn's.

Policy: the project ships numba-accelerated, sklearn-equivalent kernels
``fast_roc_auc`` / ``fast_brier_score_loss`` / ``fast_log_loss`` /
``fast_log_loss_binary`` (all importable from ``mlframe.metrics.core``). These have
existing sklearn-equivalence tests (``tests/metrics/test_metrics.py`` et al.), so
swapping the sklearn calls for them is behaviour-preserving *and* faster. Production
code that reaches for ``sklearn.metrics.roc_auc_score`` / ``brier_score_loss`` /
``log_loss`` re-introduces the slow, validation-heavy sklearn path.

This linter AST-scans the production directories to which the policy has been applied
(``evaluation/`` and ``feature_selection/``) for any *import of* or *call to* the three
banned sklearn names and fails on any hit that is not in the explicit ALLOWLIST.

The allowlist carries ONLY genuinely-unconvertible sites: the fast kernels are
binary-only (single class-1 probability vector), so a *multiclass* ``log_loss(...,
labels=[...])`` over an ``(n, C>2)`` probability matrix, or a one-vs-rest
``roc_auc_score(..., multi_class="ovr")``, has no drop-in fast equivalent and must
stay on sklearn. Each allowlisted line records a concrete reason.

Note: ``src/mlframe/calibration/`` is intentionally NOT scanned here -- it is covered
by its own calibration swap + tests. This gate guards the directories converted by
the "use our own fast metric kernels" policy pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"

# Production subpackages the fast-kernel policy applies to, tree-wide. Paths are POSIX-relative to _SRC.
_SCANNED_DIRS = (
    "calibration",
    "core",
    "estimators",
    "evaluation",
    "feature_engineering",
    "feature_selection",
    "inference",
    "integrations",
    "metrics",
    "models",
    "preprocessing",
    "reporting",
    "signal",
    "training",
    "utils",
)

# sklearn metric names that HAVE a drop-in fast/own equivalent in mlframe (metrics.core or
# reporting.charts). ``make_scorer`` is deliberately NOT banned -- it is sklearn interop glue,
# not a metric. ``auc`` / ``precision_recall_curve`` / ``mutual_info_score`` are likewise absent
# (no own equivalent yet). Own replacements, in order:
#   roc_auc_score -> fast_roc_auc ; brier_score_loss -> fast_brier_score_loss ;
#   log_loss(binary) -> fast_log_loss_binary ; classification_report -> fast/format_classification_report ;
#   balanced_accuracy_score -> balanced_accuracy_binary ; r2_score -> fast_r2_score ;
#   mean_absolute_error -> fast_mean_absolute_error ; mean_squared_error -> fast_mean_squared_error ;
#   average_precision_score -> mlframe.metrics.core.average_precision_score (own PR-AUC kernel) ;
#   precision_score -> fast_precision ; recall_score -> precision_recall_f1_from_counts ;
#   accuracy_score -> accuracy_ratio / fast_classification_report accuracy ;
#   confusion_matrix -> confusion_matrix_counts ; roc_curve -> fast_roc_curve ;
#   ConfusionMatrixDisplay -> plot_confusion_matrix (mlframe.reporting.charts).
_BANNED = {
    "roc_auc_score",
    "brier_score_loss",
    "log_loss",
    "classification_report",
    "balanced_accuracy_score",
    "r2_score",
    "mean_absolute_error",
    "mean_squared_error",
    "average_precision_score",
    "precision_score",
    "recall_score",
    "accuracy_score",
    "confusion_matrix",
    "roc_curve",
    "ConfusionMatrixDisplay",
}

# Allowlist of (posix-relative-path, banned-name) pairs left on sklearn on purpose,
# each with a concrete, verifiable reason. Line numbers are deliberately NOT pinned so
# the allowlist survives edits above the site; the (file, name) granularity is enough
# because every allowlisted file uses the name ONLY in its documented multiclass path.
_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "feature_selection/filters/_usability_aware_selection.py",
        "log_loss",
    ): "multiclass CV log-loss: log_loss(proba(n,C>2), labels=arange(n_classes)); fast_log_loss is binary-only.",
    (
        "feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_loss.py",
        "log_loss",
    ): "multiclass honest-loss path: log_loss(proba(n,C), labels=range(C)); fast_log_loss is binary-only (binary path already swapped).",
    (
        "feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_loss.py",
        "roc_auc_score",
    ): "one-vs-rest macro AUC: roc_auc_score(proba, multi_class='ovr'); fast_roc_auc is binary-only (binary path already swapped).",
    (
        "evaluation/reports.py",
        "balanced_accuracy_score",
    ): "multiclass balanced accuracy (macro recall over C>2 classes); balanced_accuracy_binary is binary-only. Binary path uses balanced_accuracy_binary; sklearn only on the nclasses>2 branch.",
    (
        "training/_helpers_training_configs.py",
        "roc_auc_score",
    ): "one-vs-rest multiclass AUC: neg_ovr_roc_auc_score wraps roc_auc_score(..., multi_class='ovr') as an XGB eval_metric; fast_roc_auc is binary-only.",
    (
        "training/_partial_fit_es_wrapper.py",
        "log_loss",
    ): "multiclass log-loss on the (n,C>2) proba matrix (binary (n,) path uses fast_log_loss_binary via shape dispatch); fast_log_loss is binary-only.",
    (
        "training/_partial_fit_es_wrapper.py",
        "roc_auc_score",
    ): "one-vs-rest multiclass AUC on the (n,C>2) proba matrix (binary (n,) path uses fast_roc_auc via shape dispatch); fast_roc_auc is binary-only.",
    (
        "training/automl.py",
        "roc_auc_score",
    ): "one-vs-rest multiclass test AUC (K>2 branch): roc_auc_score(..., multi_class='ovr'); binary branch uses fast_roc_auc; fast_roc_auc is binary-only.",
    (
        "training/baselines/_dummy_bootstrap.py",
        "log_loss",
    ): "log-loss bootstrap CI with labels-pinned degenerate resamples + multilabel macro path (per-column labels=[0,1]); fast_log_loss is binary-only and has no labels= single-class-resample handling.",
    (
        "training/baselines/_dummy_metrics_pick_plot.py",
        "log_loss",
    ): "multiclass log_loss(y, proba(n,C), labels=arange(C)) + multilabel per-label labels=[0,1] path; binary AUC uses fast_roc_auc; fast_log_loss is binary-only.",
    (
        "training/baselines/_dummy_metrics_pick_plot.py",
        "roc_auc_score",
    ): "one-vs-rest multiclass macro AUC (roc_auc_score(..., multi_class='ovr', average='macro')); binary branch uses fast_roc_auc; fast_roc_auc is binary-only.",
    (
        "training/reporting/_reporting_probabilistic.py",
        "classification_report",
    ): "multilabel-indicator / non-1D targets + exception safety-net fallback; format_classification_report handles the single-label path and is used there, but is single-label-only (nclasses scalar), so multilabel keeps sklearn.",
    (
        "training/reporting/_reporting.py",
        "classification_report",
    ): "shared single-source sklearn fallback re-exported for the sibling probabilistic report's multilabel-indicator / exception path; format_classification_report is single-label-only, so multilabel keeps sklearn.",
    (
        "feature_selection/functional_adapters.py",
        "accuracy_score",
    ): "generic default scoring for greedy_backward_elimination/iterative_zero_importance_pruning, dispatched on the caller's own target (binary or multiclass); mlframe has no fast accuracy kernel.",
    (
        "feature_selection/functional_adapters.py",
        "r2_score",
    ): "generic default scoring for greedy_backward_elimination/iterative_zero_importance_pruning's regression branch; mlframe has no fast r2 kernel.",
    (
        "training/composite/feature_subset_bagging.py",
        "r2_score",
    ): "OOF weighting for correlation-cluster regression subsets (regression-only feature); mlframe has no fast r2 kernel.",
    (
        "training/core/_diagnostics_registry.py",
        "r2_score",
    ): "default diagnostics metric_fn shared across classification and regression targets (accuracy-style metrics reject the caller's continuous leaked-dummy prediction); mlframe has no fast r2 kernel.",
    (
        "training/core/_phase_finalize_calibration.py",
        "balanced_accuracy_score",
    ): "default threshold-optimizer metric_fn, overridable by the caller but must work for both binary and multiclass by default; balanced_accuracy_binary is binary-only.",
    (
        "training/composite/classification_discovery.py",
        "log_loss",
    ): "multiclass CV/holdout log-loss: log_loss(proba(n,C), labels=classes) over an arbitrary class count; fast_log_loss is binary-only.",
}


def _iter_prod_files() -> list[Path]:
    """Every ``.py`` file under the scanned production directories, excluding benchmarks/caches."""
    files: list[Path] = []
    for d in _SCANNED_DIRS:
        base = _SRC / d
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts or "_benchmarks" in p.parts:
                continue
            files.append(p)
    return files


def _banned_hits(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, name)] for each import of / call to a banned sklearn metric name.

    Detects: ``from sklearn.metrics import roc_auc_score`` (incl. aliased / multi-name
    forms), ``sklearn.metrics.log_loss`` attribute imports, and bare/attribute *calls*
    of the banned names (``brier_score_loss(...)``, ``_auc(...)`` bound to a banned import).
    Comments and docstrings never match -- this walks the AST, not the text.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []

    hits: list[tuple[int, str]] = []
    # Local names bound to a banned sklearn metric via import (handles `as` aliases).
    aliased: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "sklearn.metrics" or mod.startswith("sklearn.metrics."):
                for a in node.names:
                    if a.name in _BANNED:
                        hits.append((node.lineno, a.name))
                        aliased[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            # `import sklearn.metrics as m` -> m.log_loss(...) caught by the call branch below on attr.
            pass

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # sklearn.metrics.roc_auc_score(...) attribute call
        if isinstance(func, ast.Attribute) and func.attr in _BANNED:
            hits.append((node.lineno, func.attr))
        # bare call of an imported/aliased banned name
        elif isinstance(func, ast.Name) and func.id in aliased:
            hits.append((node.lineno, aliased[func.id]))

    return hits


def test_no_sklearn_metrics_in_production() -> None:
    """No production file may call a banned sklearn metric outside _ALLOWLIST."""
    offenders: list[str] = []
    for path in _iter_prod_files():
        rel = path.relative_to(_SRC).as_posix()
        for lineno, name in _banned_hits(path):
            if (rel, name) in _ALLOWLIST:
                continue
            offenders.append(f"{rel}:{lineno}  sklearn.metrics.{name}")

    assert not offenders, (
        "Production code must use mlframe's own fast metric kernels, not sklearn's "
        "roc_auc_score / brier_score_loss / log_loss. Import the drop-in replacements "
        "from mlframe.metrics.core: fast_roc_auc, fast_brier_score_loss, fast_log_loss "
        "(binary-only). For genuinely multiclass calls that have no fast equivalent, add "
        "the (path, name) pair to _ALLOWLIST with a concrete reason. Offenders:\n  " + "\n  ".join(sorted(offenders))
    )


def test_allowlist_entries_are_still_live() -> None:
    """Guard against a stale allowlist: every allowlisted (file, name) must still occur in
    that file, else the entry is dead and should be removed (keeps the allowlist honest --
    a resolved leftover must not linger and mask a future re-introduction elsewhere)."""
    dead: list[str] = []
    for rel, name in _ALLOWLIST.keys():
        path = _SRC / rel
        if not path.exists():
            dead.append(f"{rel} (file missing)")
            continue
        names = {n for _ln, n in _banned_hits(path)}
        if name not in names:
            dead.append(f"{rel}:{name}")
    assert not dead, "Stale _ALLOWLIST entries (site no longer uses the sklearn metric -- remove them):\n  " + "\n  ".join(sorted(dead))
