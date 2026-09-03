"""TRAINING_COMPOSITE_CORE_A-4 regression test: per-candidate-column median imputation in
classification_discovery.py must use the TRAIN fold's own median, not a median computed over the whole
screening split (train + that fold's held-out val rows) before the CV loop.

The bug (fixed): both _stage1_screen and _stage2_paired computed `_impute_column(...)` ONCE over the
whole screening split before the StratifiedKFold loop -- each fold's train-side imputed values
incorporated information from that fold's own held-out validation rows, a train/val boundary leak of a
summary statistic (the median) that mildly inflates margin_gain/cv_gain. Fixed via
`_impute_column_fold(col, tr)`, computing the median from `tr` only, called fresh inside each fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.classification_discovery import _impute_column, _impute_column_fold

pytestmark = pytest.mark.fast


def test_fold_median_uses_only_train_rows_not_val():
    """The imputed value for a missing entry must come from the TRAIN fold's median, and change when
    the val-fold rows (excluded from the median) change, proving no val-row leakage into the median."""
    # 10 rows: rows 0-6 are "train" (median computable from these alone), rows 7-9 are "val".
    col = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0, np.nan, 5000.0, 5000.0, 5000.0])
    tr = np.arange(7)
    imputed_a = _impute_column_fold(col, tr)
    train_median = np.median([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    assert imputed_a[6, 0] == pytest.approx(train_median)

    # Change ONLY the val-fold rows (7-9) to something wildly different; if the old whole-split-median
    # bug were still present, this would shift the imputed value for the NaN in the train rows.
    col_b = col.copy()
    col_b[7:10] = -999999.0
    imputed_b = _impute_column_fold(col_b, tr)
    assert imputed_b[6, 0] == pytest.approx(train_median), "val-fold rows must not influence the train-fold median"
    assert imputed_a[6, 0] == imputed_b[6, 0]


def test_whole_split_impute_column_would_have_leaked_for_contrast():
    """Sanity/contrast: the OLD _impute_column (whole-split median, still present for other legitimate
    uses) DOES shift when val-fold rows change -- confirming the fold-aware fix actually changes behavior."""
    col = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0, np.nan, 5000.0, 5000.0, 5000.0])
    imputed_whole_a = _impute_column(col)
    col_b = col.copy()
    col_b[7:10] = -999999.0
    imputed_whole_b = _impute_column(col_b)
    assert imputed_whole_a[6, 0] != imputed_whole_b[6, 0], "whole-split median SHOULD shift with val-row changes (that's the leak the fix avoids)"
