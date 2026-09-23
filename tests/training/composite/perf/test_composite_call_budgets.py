"""Expensive discovery primitives stay within their call budget.

Each defect here was a primitive called far more often than the work needed: the screen feature matrix rebuilt by every
gate, one correlation build per base context instead of per target, a LightGBM Dataset per (spec, seed, fold) over at most
a handful of distinct fold matrices, a data signature computed on every fit. The IDEAL count is the budget; today's excess
is recorded in ``_composite_call_budget_baseline.json`` with a note, may only go down, and any other primitive over its
ideal fails.
"""

from __future__ import annotations

import orjson
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytestmark = pytest.mark.slow  # one full discovery fit (~60 s)

_BASELINE = Path(__file__).resolve().parent / "_composite_call_budget_baseline.json"
# Ideal calls per default discovery fit of the fixture below.
_IDEAL = {"build_feature_matrix": 1, "near_collinear_keep_mask": 1, "data_signature": 0, "generate_interaction_bases": 1, "lgb_Dataset": 0}


@pytest.fixture(scope="module")
def counts():
    """Call counts of one default-config discovery fit (n=600, two bases, a group column), run single-threaded."""
    import lightgbm as lgb

    import mlframe.training.composite.cache as cache
    import mlframe.training.composite.discovery._eval_stats as es
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.composite.transforms import interaction_bases as ib
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    from ._call_budget import CallBudget

    rng = np.random.default_rng(0)
    n = 600
    g = np.arange(n) % 12
    b1, b2 = rng.uniform(1.0, 10.0, n) + 0.2 * g, rng.uniform(1.0, 5.0, n)
    df = pd.DataFrame({"b1": b1, "b2": b2, "x": rng.normal(size=n), "g": g, "y": 2.0 * b1 + b2 + rng.normal(0.0, 0.5, n)})
    # Serial, so the counts are deterministic: the shared fold-dataset cache is per thread by design (set_label mutates it),
    # so a thread pool builds one copy per worker and the count varies with scheduling (141-156 measured in parallel).
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, base_candidates=["b1", "b2"], group_column="g",
                                         discovery_n_jobs=1, tiny_model_n_jobs=1, tiny_rerank_n_jobs=1)
    targets = {"near_collinear_keep_mask": es.near_collinear_keep_mask, "data_signature": cache.data_signature,
               "generate_interaction_bases": ib.generate_interaction_bases, "lgb_Dataset": lgb.Dataset.__init__,
               "build_feature_matrix": CompositeTargetDiscovery._build_feature_matrix}
    owners = {"lgb_Dataset": (lgb.Dataset, "__init__"), "build_feature_matrix": (CompositeTargetDiscovery, "_build_feature_matrix")}
    with warnings.catch_warnings(), CallBudget(targets, owners) as calls:
        warnings.simplefilter("ignore")
        CompositeTargetDiscovery(cfg).fit(df, "y", ["b1", "b2", "x", "g"], np.arange(n))
        return dict(calls)


@pytest.mark.parametrize("primitive", sorted(_IDEAL))
def test_a_primitive_stays_within_its_budget(counts, primitive: str):
    """At most the ideal count, or the recorded excess; a count below the record means the record must be lowered."""
    recorded = orjson.loads(_BASELINE.read_text(encoding="utf-8")).get(primitive)
    got, ideal = counts[primitive], _IDEAL[primitive]
    if recorded is None:
        assert got <= ideal, f"{primitive}: {got} calls per fit, over its ideal {ideal}"
        return
    assert got <= recorded["measured"], f"{primitive}: {got} calls, over the recorded {recorded['measured']} ({recorded['note']})"
    assert got == recorded["measured"] or got > ideal, f"{primitive} improved to {got}; lower its baseline entry (or drop it at the ideal)"
    assert got == recorded["measured"], f"{primitive} improved from {recorded['measured']} to {got}: lower the baseline to lock the gain in"


def test_every_baseline_entry_is_a_budgeted_primitive():
    """The baseline only records known primitives, each with a note on why it exceeds the ideal."""
    base = orjson.loads(_BASELINE.read_text(encoding="utf-8"))
    assert set(base) <= set(_IDEAL)
    assert all(str(v.get("note", "")).strip() for v in base.values())
