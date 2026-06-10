"""Panel-token catalogue for the reporting DSL (INV-58).

``describe_available_panels()`` lists, per task type, every panel token a
``ReportingConfig.*_panels`` / ``compose_*_figure`` template accepts, with a
one-line description. Tokens are sourced from each chart module's
``ALLOWED_*_PANEL_TOKENS`` frozenset, so a token added to a chart without a
description here surfaces immediately (the function flags it ``(no description)``
rather than silently omitting it).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from mlframe.reporting.charts.binary import ALLOWED_BINARY_PANEL_TOKENS
from mlframe.reporting.charts.ltr import ALLOWED_LTR_PANEL_TOKENS
from mlframe.reporting.charts.multiclass import ALLOWED_MULTICLASS_PANEL_TOKENS
from mlframe.reporting.charts.multilabel import ALLOWED_MULTILABEL_PANEL_TOKENS
from mlframe.reporting.charts.quantile import ALLOWED_QUANTILE_PANEL_TOKENS
from mlframe.reporting.charts.regression import ALLOWED_REGRESSION_PANEL_TOKENS

# One-line descriptions per token. Keyed by task type so a token name reused across task types (ROC, PR_F1,
# CALIB_GRID) can carry a task-specific blurb.
_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "binary_classification": {
        "ROC": "TPR vs FPR with the chance diagonal; AUC in the title.",
        "PR": "Precision vs recall with the prevalence no-skill baseline; AP in the title.",
        "SCORE_DIST": "Overlaid class-conditional score histograms (y=0 vs y=1) with a threshold marker.",
        "KS": "Class-conditional score ECDFs with the max-gap (KS statistic) marked.",
        "THRESHOLD": "Precision/recall/F1 vs threshold + queue-rate on a secondary axis (operating-point picker).",
        "GAIN": "Cumulative-gain curve (positives captured vs population) with the lift area shaded.",
        "PIT": "Probability-integral-transform histogram; uniform = perfect calibration.",
    },
    "multiclass_classification": {
        "CONFUSION": "Row-normalised confusion-matrix heatmap (P(pred | true)).",
        "CONFUSED_PAIRS": "Top-N most-confused (true -> pred) class pairs as horizontal bars.",
        "PR_F1": "Per-class precision / recall / F1 grouped bar.",
        "ROC": "Per-class one-vs-rest ROC curves overlaid.",
        "PR_CURVES": "Per-class one-vs-rest precision-recall curves overlaid.",
        "CALIB_GRID": "Per-class reliability curves overlaid (perfect = diagonal).",
        "PROB_DIST": "Per-true-class violin of P(y = true_class | x).",
        "TOP_K_ACC": "Top-k accuracy curve for k = 1..K.",
    },
    "multilabel_classification": {
        "PR_F1": "Per-label precision / recall / F1 grouped bar.",
        "ROC": "Per-label ROC curves overlaid.",
        "CALIB_GRID": "Per-label reliability curves overlaid.",
        "COOCCURRENCE": "True x predicted label co-occurrence heatmap.",
        "CARDINALITY": "Distribution of #labels per row (pred vs true grouped bar).",
        "JACCARD_DIST": "Per-row Jaccard-score histogram.",
        "HAMMING_DIST": "Per-row Hamming-distance histogram.",
    },
    "learning_to_rank": {
        "NDCG_K": "NDCG@k curve for k = 1..max_per_query.",
        "NDCG_DIST": "Per-query NDCG@10 distribution (violin).",
        "NDCG_BY_QSIZE": "Mean NDCG@10 binned by query size (exposes small-group inflation).",
        "LIFT": "Cumulative relevance vs rank position (lift / gain curve over queries).",
        "MRR_DIST": "Per-query reciprocal-rank histogram.",
        "SCORE_BY_REL": "Predicted-score distribution per relevance grade.",
        "TOP1_BY_QSIZE": "Top-1 accuracy as a function of query size.",
    },
    "quantile_regression": {
        "RELIABILITY": "Empirical vs nominal coverage per alpha (diagonal = calibrated).",
        "COVERAGE": "Empirical vs nominal interval coverage per symmetric pair; mean width on a secondary axis.",
        "PINBALL_BY_ALPHA": "Mean pinball loss per alpha (which tail the model is worst at).",
        "INTERVAL_BAND": "Per-row median line + filled lo..hi band, y_true as markers.",
        "WIDTH_DIST": "Histogram of interval widths (sharpness diagnostic).",
        "PIT_HIST": "PIT histogram (uniform = calibrated); needs K >= 3 alphas.",
        "QUANTILE_RELIABILITY": "Per-tau isotonic-recalibrated observed coverage vs nominal tau (CORP-style).",
        "PINBALL_DECOMP": "CORP additive pinball decomposition (miscal - discr + uncert) per tau.",
        "QUANTILE_CROSSING": "Per adjacent-tau-pair fraction of rows with q_lo > q_hi (monotonicity violations).",
    },
    "regression": {
        "SCATTER": "Predictions vs true with y=x, robust trend line, and worst-K residuals highlighted red.",
        "RESID_HIST": "Residual histogram + fitted-Normal overlay (noise hypothesis + suggested loss).",
        "RESID_VS_PRED": "Residuals vs predicted with a running-median + IQR band (heteroscedasticity / bias).",
        "ERR_BY_DECILE": "Per-target-decile mean |residual| + mean signed residual (GBM compression pathology).",
    },
}

# Task type -> its ALLOWED_*_PANEL_TOKENS frozenset (the source of truth for which tokens exist).
_TASK_TOKENS: Dict[str, frozenset] = {
    "binary_classification": ALLOWED_BINARY_PANEL_TOKENS,
    "multiclass_classification": ALLOWED_MULTICLASS_PANEL_TOKENS,
    "multilabel_classification": ALLOWED_MULTILABEL_PANEL_TOKENS,
    "learning_to_rank": ALLOWED_LTR_PANEL_TOKENS,
    "quantile_regression": ALLOWED_QUANTILE_PANEL_TOKENS,
    "regression": ALLOWED_REGRESSION_PANEL_TOKENS,
}


def available_panels() -> Dict[str, List[Tuple[str, str]]]:
    """Return ``{task_type: [(token, description), ...]}`` for every task type, tokens sorted alphabetically.

    Tokens come from the chart modules' frozensets; a token with no description here is paired with
    ``"(no description)"`` so the gap is visible rather than the token silently dropped.
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    for task, tokens in _TASK_TOKENS.items():
        descs = _DESCRIPTIONS.get(task, {})
        out[task] = [(tok, descs.get(tok, "(no description)")) for tok in sorted(tokens)]
    return out


def describe_available_panels(*, file=None) -> Dict[str, List[Tuple[str, str]]]:
    """Print the panel-token catalogue per task type and return the same structured mapping.

    Each task type's section lists its tokens (alphabetical) with a one-line description. ``file`` defaults to stdout;
    pass a stream to capture. Console output is ASCII-only. Returns the ``available_panels()`` mapping so callers can
    introspect programmatically too.
    """
    catalogue = available_panels()
    lines: List[str] = ["Available reporting panel tokens (per task type):"]
    for task in _TASK_TOKENS:
        lines.append("")
        lines.append(f"[{task}]")
        for tok, desc in catalogue[task]:
            lines.append(f"  {tok:<22} {desc}")
    print("\n".join(lines), file=file)
    return catalogue


__all__ = [
    "available_panels",
    "describe_available_panels",
]
