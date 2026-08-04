"""``report_regression_model_perf``'s ConcordanceIndex must match the standalone
``fast_concordance_index`` bit-for-bit.

Formerly this metric was derived from the Kendall tau-b already computed on the same
(targets, preds) via the closed-form ``(tau_b + 1) / 2`` -- a real bug: that identity only
holds when ``preds`` has NO ties, and is measurably wrong once it does (e.g. a tree-ensemble
risk score with repeated leaf outputs). ``report_regression_model_perf`` now calls
``fast_concordance_index`` directly instead of re-deriving it from Kendall's tau.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.regression._regression_extras import fast_concordance_index
from mlframe.training.reporting._reporting_regression import report_regression_model_perf


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("n", [600, 5000, 20000])
def test_report_cindex_byte_identical_to_standalone_concordance(seed: int, n: int):
    """Report cindex byte identical to standalone concordance."""
    rng = np.random.default_rng(seed)
    yt = np.abs(rng.standard_normal(n)) * 10.0 + 5.0
    yp = yt + 0.3 * rng.standard_normal(n)

    m: dict = {}
    report_regression_model_perf(
        targets=yt,
        columns=["a"],
        model_name="m",
        model=None,
        preds=yp,
        metrics=m,
        print_report=False,
        show_perf_chart=False,
    )

    ref = fast_concordance_index(yt, yp)
    assert m["ConcordanceIndex"] == ref, f"ConcordanceIndex {m['ConcordanceIndex']!r} != standalone fast_concordance_index {ref!r} (seed={seed}, n={n})"
    # (tau_b + 1) / 2 only coincides with the true C-index in the tie-free case, and even then
    # only up to floating-point rounding (two independent algorithms, not the same expression
    # reused) -- a loose approximate check, not the byte-identical assertion this test used to
    # make when ConcordanceIndex was literally computed AS that expression (the bug).
    assert m["ConcordanceIndex"] == pytest.approx((m["Kendall"] + 1.0) / 2.0, abs=1e-9)


def test_report_cindex_diverges_from_kendall_identity_under_prediction_ties():
    """ConcordanceIndex must diverge from the buggy (Kendall + 1) / 2 identity once preds have ties."""
    # The regression guard for the actual bug: with ties in preds, (tau_b + 1) / 2 is measurably
    # wrong, so ConcordanceIndex must NOT equal it (pre-fix, ConcordanceIndex WAS this identity).
    rng = np.random.default_rng(3)
    n = 400
    yt = rng.standard_normal(n)
    yp = np.round(yt + 0.4 * rng.standard_normal(n), 1)  # coarse rounding manufactures real ties

    m: dict = {}
    report_regression_model_perf(
        targets=yt, columns=["a"], model_name="m", model=None, preds=yp, metrics=m, print_report=False, show_perf_chart=False,
    )
    tau_derived = (m["Kendall"] + 1.0) / 2.0
    assert m["ConcordanceIndex"] == fast_concordance_index(yt, yp)
    assert abs(m["ConcordanceIndex"] - tau_derived) > 1e-6, "ConcordanceIndex should diverge from the buggy tau-derived identity once preds carry ties"
