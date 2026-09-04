"""Why the 5-signal/15-noise ranking benchmark loses downstream AUC, and which knob causes it.

Reproduces the fixture from
``tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_quality_metrics.py::TestRankingQuality
::test_top_k_precision_5_signals_15_noise`` and measures downstream 5-fold ROC-AUC for five arms.

The result (see ``audits/full_audit_2026-09-01/known_complications.md``) is that the loss is NOT caused by
engineering composite features. It is caused by ``fe_drop_redundant_raw_operands`` dropping the raw operand
columns afterwards: with FE still fully on and only that rule disabled, the selection recovers the
five-raw-signal baseline exactly. A mixture like ``add(add(neg(sig0),neg(sig1)),add(neg(sig2),neg(sig4)))``
preserves the sum and destroys the individual contributions, which is what a downstream linear model needs
when the signals carry different coefficients.

Run: ``python benchmarks/mrmr_fe_operand_drop_diagnostic.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# Each arm is (label, MRMR kwargs). The empty dict is the shipped default.
ARMS: tuple[tuple[str, dict], ...] = (
    ("default (full mode, FE on)", {}),
    ("use_simple_mode=True (FE off)", {"use_simple_mode": True}),
    ("fe_drop_redundant_raw_operands=False", {"fe_drop_redundant_raw_operands": False}),
    ("fe_raw_retention_max_n=25", {"fe_raw_retention_max_n": 25}),
)


def build() -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the benchmark fixture verbatim (seed 200): 5 weighted signals, 15 pure-noise columns."""
    rng = np.random.default_rng(200)
    n = 2500
    sig_cols = {f"sig{k}": rng.standard_normal(n) for k in range(5)}
    # Coefficients DIFFER per signal (0.8, 0.7, ... 0.4). That is what makes a lossy sum-style composite
    # strictly worse than the operands: the individual contributions cannot be recovered from the sum.
    y_lin = sum(sig_cols[f"sig{k}"] * (0.8 - 0.1 * k) for k in range(5)) + 0.5 * rng.standard_normal(n)
    y = pd.Series((y_lin > 0).astype(np.int64))
    noise_cols = {f"noise{k}": rng.standard_normal(n) for k in range(15)}
    return pd.DataFrame({**sig_cols, **noise_cols}), y


def auc(frame: pd.DataFrame, y: pd.Series) -> float:
    """5-fold ROC-AUC of a logistic model on the given columns."""
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=2000), frame, y, cv=5, scoring="roc_auc")))


def main() -> None:
    """Run every arm against the five-raw-signal baseline and print the comparison table."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = build()
    base = auc(X[[f"sig{k}" for k in range(5)]], y)
    print(f"{'baseline (5 raw signals)':38s} AUC={base:.4f}")

    for label, kwargs in ARMS:
        sel = MRMR(verbose=0, **kwargs).fit(X, y)
        names = list(sel.get_feature_names_out())
        got = auc(pd.DataFrame(sel.transform(X)), y)
        print(f"{label:38s} AUC={got:.4f}  gap={base - got:+.4f}  n_selected={len(names)}")
        for nm in names:
            print(f"      {nm}")


if __name__ == "__main__":
    main()
