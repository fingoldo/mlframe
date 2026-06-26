"""Param-robustness guard for F2 single-compound recovery (2026-06-26).

The F2 golden goal (one clean fused compound covering a/b AND c/d, no noise 'e', no fragments) must be
ROBUST to the permutation-null counts, not balanced on a single tuned point. A 2026-06-26 investigation
established this (the earlier "64/64 bloats" alarm was a measurement artifact: a different fixture where 'e'
was a real 0.3*e contributor + leaving FE knobs at the fast-search default). This pins the robustness so a
future change that re-introduces perm-count sensitivity in the FE admission / fusion is caught.
"""
from __future__ import annotations

import re
import warnings

import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from tests.feature_selection.mrmr.test_f2_single_compound_across_distributions import _make, _classify, _cols

_ID = re.compile(r"[a-zA-Z_]\w*")


@pytest.mark.parametrize("full,baseline", [(3, 2), (10, 20), (64, 64)])
def test_f2_recovery_robust_to_permutation_counts(full, baseline):
    """One clean fused compound on the canonical F2 fixture across a wide perm-count range -- the recovered
    compound must not depend on full/baseline_npermutations (FE knobs at the golden config)."""
    df, y = _make("uniform", n=10_000, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=42, n_jobs=1, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05,
                  full_npermutations=full, baseline_npermutations=baseline).fit(df, y)
    names = [str(s) for s in fs.get_feature_names_out()]
    full_c, frag_ab, frag_cd = _classify(names)
    assert all("e" not in _cols(nm) for nm in names), f"[full={full},base={baseline}] noise 'e': {names}"
    assert len(full_c) == 1, f"[full={full},base={baseline}] expected ONE fused compound, got {len(full_c)}: {names}"
    assert not frag_ab and not frag_cd, f"[full={full},base={baseline}] fragment(s): ab={frag_ab} cd={frag_cd}"
    assert "c" in _cols(full_c[0]), f"[full={full},base={baseline}] compound dropped log(c): {full_c[0]}"
