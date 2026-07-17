"""iter56 perf-loop regression sensor.

``report_regression_model_perf`` derives ConcordanceIndex as ``(Kendall + 1) / 2``
from the Kendall tau-b it already computed on the SAME (targets, preds), instead
of calling ``fast_concordance_index`` (which recomputed the full tau-b -- a
duplicate O(N log N) scipy.kendalltau / O(N^2) numba pass on identical inputs).

C-index == (tau_b + 1) / 2 exactly, so the stamped ConcordanceIndex MUST stay
byte-identical to the standalone ``fast_concordance_index`` it folds into.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.regression._regression_extras import fast_concordance_index
from mlframe.training.reporting._reporting_regression import report_regression_model_perf


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("n", [600, 5000, 20000])
def test_report_cindex_byte_identical_to_standalone_concordance(seed: int, n: int):
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
    # And the closed-form identity itself.
    assert m["ConcordanceIndex"] == (m["Kendall"] + 1.0) / 2.0
