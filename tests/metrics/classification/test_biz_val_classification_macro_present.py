"""qual-7 biz_value + regression: macro precision/recall/F1 averaged over PRESENT classes (sklearn semantics).

``fast_classification_report``'s macro averages divided by ``nclasses``, so a class declared in ``nclasses`` but
absent from both ``y_true`` and ``y_pred`` contributed a zeroed P/R/F1 and DEFLATED the macro means. sklearn's
``classification_report`` macro-averages over the union of classes present in y_true or y_pred. ``macro_over_present``
(default True) restores that; ``macro_over_present=False`` keeps the legacy divide-by-nclasses behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support

from mlframe.metrics.core import fast_classification_report


def _sklearn_macro(yt, yp):
    labels = np.unique(np.concatenate([yt, yp]))
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=labels, average="macro", zero_division=0)
    return np.array([p, r, f])


def test_macro_over_present_matches_sklearn_with_absent_class():
    yt = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    yp = np.array([0, 1, 1, 1, 2, 2, 0, 2], dtype=np.int64)
    # nclasses=4 declares a class (3) that appears in neither array.
    macro = np.array(fast_classification_report(yt, yp, nclasses=4)[8])
    np.testing.assert_allclose(macro, _sklearn_macro(yt, yp), atol=1e-12)


def test_legacy_macro_over_all_deflates_and_is_opt_in():
    yt = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    yp = np.array([0, 1, 1, 1, 2, 2, 0, 2], dtype=np.int64)
    legacy = np.array(fast_classification_report(yt, yp, nclasses=4, macro_over_present=False)[8])
    gt = _sklearn_macro(yt, yp)
    # Legacy divides by nclasses=4 (incl. absent class 3): 3/4 of the present mean.
    np.testing.assert_allclose(legacy, gt * 0.75, atol=1e-12)
    assert legacy.mean() < gt.mean() - 0.1  # materially deflated


def test_no_absent_class_default_equals_legacy():
    # When every declared class is present, present-mask is all-True so both policies coincide.
    yt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    yp = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
    new = np.array(fast_classification_report(yt, yp, nclasses=3)[8])
    legacy = np.array(fast_classification_report(yt, yp, nclasses=3, macro_over_present=False)[8])
    np.testing.assert_allclose(new, legacy, atol=1e-12)


def test_biz_val_macro_present_beats_legacy_vs_sklearn_majority():
    """Macro-over-present must match sklearn (|err|~0) and beat the legacy divide-by-nclasses on the MAJORITY
    of (scenario x seed) cells with declared-but-absent classes. Measured bench: 48/48 cells, new |err|=0.0 vs
    legacy 0.07-0.38. Floor here: new beats legacy in >=90% of 24 cells AND new |err| < 1e-9 everywhere."""
    rng_seeds = range(8)
    win_new = 0
    n_cells = 0
    for nclasses in (6, 10):
        for absent in (1, 2, nclasses // 2):
            active = max(2, nclasses - absent)
            for seed in rng_seeds:
                rng = np.random.default_rng(seed * 17 + nclasses * 3 + absent)
                yt = rng.integers(0, active, size=600).astype(np.int64)
                yp = yt.copy()
                flip = rng.random(600) < 0.35
                yp[flip] = rng.integers(0, active, size=int(flip.sum()))
                gt = _sklearn_macro(yt, yp)
                new = np.array(fast_classification_report(yt, yp, nclasses=nclasses)[8])
                legacy = np.array(fast_classification_report(yt, yp, nclasses=nclasses, macro_over_present=False)[8])
                en = float(np.mean(np.abs(new - gt)))
                el = float(np.mean(np.abs(legacy - gt)))
                assert en < 1e-9, f"new macro must match sklearn; err={en}"
                if en < el:
                    win_new += 1
                n_cells += 1
    assert win_new >= int(0.9 * n_cells), f"new wins {win_new}/{n_cells}, expected >=90%"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-cov"]))
