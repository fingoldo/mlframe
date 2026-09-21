"""The composite value report must build when the suite has no group key.

The MoE phase passes ``group_ids=None`` for a non-grouped suite - the default - and ``pd.factorize(np.asarray(None))``
raised "len() of unsized object", so the report was dropped with a warning on every such run. ``None`` now means one
group covering every row.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite._value_report import build_composite_value_report, render_composite_value_report


def _preds(n: int = 200, seed: int = 0):
    """Truth, a raw prediction and a better composite prediction."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return y, y + rng.normal(scale=0.5, size=n), y + rng.normal(scale=0.2, size=n)


def test_the_report_builds_without_groups():
    """No group ids: one group, every row counted, and the composite's lower RMSE shows in the aggregate."""
    y, raw, comp = _preds()
    report = build_composite_value_report(y, raw, comp, None)
    assert report["n_groups"] == 1
    assert report["n_rows"] == len(y)
    agg = report["aggregate"]
    assert agg, "the aggregate block must be populated"
    assert render_composite_value_report(report)


def test_no_groups_equals_one_explicit_group():
    """``None`` is exactly one group holding every row."""
    y, raw, comp = _preds()
    implicit = build_composite_value_report(y, raw, comp, None)
    explicit = build_composite_value_report(y, raw, comp, np.full(len(y), "all", dtype=object))
    assert implicit == explicit
