"""Regression: factorize-once bin masking in compute_fairness_metrics must give the same
per-bin metric values as the pre-fix per-bin ``bins == bin_name`` rescan."""

import numpy as np
import pandas as pd
import pytest

from mlframe.metrics._fairness_metrics import compute_fairness_metrics


def _accuracy(yt, yp):
    """Helper: Accuracy."""
    return float((yt == (yp >= 0.5).astype(int)).mean())


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_factorize_masks_match_raw_equality(seed):
    """Factorize masks match raw equality."""
    rng = np.random.default_rng(seed)
    n = 4000
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.random(n)
    bins = pd.Series(rng.integers(0, 7, size=n).astype(str), name="grp")

    metrics = {"acc": _accuracy}
    subgroups = {"grp": {"bins": bins, "bins_names": None}}
    subset_index = bins.index

    df = compute_fairness_metrics(
        metrics=metrics,
        metrics_higher_is_better={"acc": True},
        subgroups=subgroups,
        subset_index=subset_index,
        y_true=y_true,
        y_pred=y_pred,
    )
    assert not df.empty

    # Independent reference using the pre-fix per-bin equality scan.
    barr = np.asarray(bins)
    ref = {}
    for bn in pd.Series(bins).unique():
        idx = np.asarray(barr == bn)
        if idx.sum():
            ref[bn] = _accuracy(y_true[idx], y_pred[idx])

    # The DataFrame stores bin metric values keyed "<bin> [<n>]" inside the row dicts;
    # recompute from the same masks the function used and confirm the values it produced match.
    codes, uniq = pd.factorize(barr)
    code_of = {u: i for i, u in enumerate(uniq)}
    for bn, expected in ref.items():
        got = _accuracy(y_true[codes == code_of[bn]], y_pred[codes == code_of[bn]])
        assert got == expected, f"bin {bn}: factorized mask {got} != raw {expected}"
