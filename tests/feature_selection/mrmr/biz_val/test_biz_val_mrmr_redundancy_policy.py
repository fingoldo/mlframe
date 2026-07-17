"""biz_value: the ``redundancy_policy`` knob governs whether subsumed raw operands survive.

On ``y = a**2 / b`` MRMR engineers a ratio that info-subsumes raw ``a`` and ``b``. A feature
selector must not destroy linearly-usable raw signal, so the default ``emit_both`` keeps the raws
ALONGSIDE the engineered ratio; the opt-in ``drop`` prunes them (the minimal-set / tree-oriented
behaviour). This pins both verdicts on the same fixture so a regression in either flips the win.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

N = 5000
SEED = 42


def _ratio_fixture(seed: int = SEED, n: int = N):
    """Ratio fixture."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    e = rng.normal(0.0, 1.0, n)  # pure noise
    y = a**2 / b
    df = pd.DataFrame({"a": a, "b": b, "e": e})
    return df, pd.Series(y, name="y")


def _covers_ratio(eng_names) -> bool:
    """Covers ratio."""
    return any({"a", "b"} <= set(nm) for nm in eng_names)


def test_biz_val_mrmr_redundancy_policy_emit_both_keeps_raw_operands():
    """``emit_both`` (default): the engineered ratio is selected AND raw ``a``, ``b`` survive."""
    df, y = _ratio_fixture()
    fs = MRMR(verbose=0, random_seed=SEED, redundancy_policy="emit_both")
    fs.fit(df, y)
    out = list(fs.get_feature_names_out())
    eng = [nm for nm in out if nm not in {"a", "b", "e"}]
    assert _covers_ratio(eng), f"no a**2/b engineered ratio recovered: {out}"
    assert "a" in out and "b" in out, f"emit_both must keep raw operands a,b alongside the engineered ratio: {out}"


def test_biz_val_mrmr_redundancy_policy_drop_prunes_raw_operands():
    """``drop``: the engineered ratio is selected and the subsumed raw operands are pruned."""
    df, y = _ratio_fixture()
    fs = MRMR(verbose=0, random_seed=SEED, redundancy_policy="drop")
    fs.fit(df, y)
    out = list(fs.get_feature_names_out())
    eng = [nm for nm in out if nm not in {"a", "b", "e"}]
    assert _covers_ratio(eng), f"no a**2/b engineered ratio recovered: {out}"
    subsumed = {nm for nm in out if nm in {"a", "b"}}
    assert subsumed == set(), f"drop must prune subsumed raw operands a,b: {out}"
