"""Param-robustness guard for F2 single-compound recovery (2026-06-26).

The F2 golden goal (one clean fused compound covering a/b AND c/d, no noise 'e', no fragments) must be
ROBUST to the permutation-null counts, not balanced on a single tuned point. A 2026-06-26 investigation
established this (the earlier "64/64 bloats" alarm was a measurement artifact: a different fixture where 'e'
was a real 0.3*e contributor + leaving FE knobs at the fast-search default). This pins the robustness so a
future change that re-introduces perm-count sensitivity in the FE admission / fusion is caught.

Each config fits in a FRESH SUBPROCESS (the canonical _run_user_case pattern): MRMR.fit consumes global
np.random AND prior in-process GPU/kernel-tuning-cache state perturbs the razor-sensitive single-compound
selection, so in-process this pin is order-dependent (it passed alone but failed after the CUDA batcher
tests in the same process). Subprocess isolation + CPU-forced + sweep-disabled makes the verdict reproducible.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

_ID = re.compile(r"[a-zA-Z_]\w*")
_RAW = {"a", "b", "c", "d", "e"}


def _cols(nm):
    return set(_ID.findall(nm)) & _RAW


def _classify(names):
    full, frag_ab, frag_cd = [], [], []
    for nm in names:
        cs = _cols(nm)
        has_ab, has_cd = bool(cs & {"a", "b"}), bool(cs & {"c", "d"})
        if has_ab and has_cd:
            full.append(nm)
        elif has_ab:
            frag_ab.append(nm)
        elif has_cd:
            frag_cd.append(nm)
    return full, frag_ab, frag_cd


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _run_f2(full: int, baseline: int, n: int = 10_000, seed: int = 42) -> list:
    """Fit the canonical F2 (uniform) in a fresh CPU-forced subprocess; return selected feature names. Uses
    the SAME _make('uniform') fixture as the F2 single-compound test (sample_operands), not an inline draw."""
    src = (
        "import os, sys, json, warnings, numpy as np\n"
        "warnings.simplefilter('ignore')\n"
        f"sys.path.insert(0, {_REPO_ROOT!r})\n"
        f"np.random.seed({seed})\n"
        "from tests.feature_selection.mrmr.core.test_f2_single_compound_across_distributions import _make\n"
        f"df, y = _make('uniform', {n}, {seed})\n"
        "from mlframe.feature_selection.filters.mrmr import MRMR\n"
        f"fs=MRMR(verbose=0, random_seed={seed}, n_jobs=1, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05,"
        f" full_npermutations={full}, baseline_npermutations={baseline}).fit(df, y)\n"
        "print('RESULT_JSON='+json.dumps(list(fs.get_feature_names_out())))\n"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""        # force CPU -> no prior-GPU-state contamination
    env["MLFRAME_DISABLE_HNSW"] = "1"
    env["PYUTILZ_KERNEL_DISABLE_SWEEP"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=600, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    raise AssertionError(
        f"subprocess fit returned no selection (rc={proc.returncode}); stderr tail:\n"
        + "\n".join(proc.stderr.splitlines()[-15:])
    )


@pytest.mark.timeout(700)
@pytest.mark.parametrize("full,baseline", [(3, 2), (10, 20), (64, 64)])
def test_f2_recovery_robust_to_permutation_counts(full, baseline):
    """One clean fused compound on the canonical F2 fixture across a wide perm-count range -- the recovered
    compound must not depend on full/baseline_npermutations (FE knobs at the golden config)."""
    names = [str(s) for s in _run_f2(full, baseline)]
    full_c, frag_ab, frag_cd = _classify(names)
    assert all("e" not in _cols(nm) for nm in names), f"[full={full},base={baseline}] noise 'e': {names}"
    assert len(full_c) == 1, f"[full={full},base={baseline}] expected ONE fused compound, got {len(full_c)}: {names}"
    assert not frag_ab and not frag_cd, f"[full={full},base={baseline}] fragment(s): ab={frag_ab} cd={frag_cd}"
    assert "c" in _cols(full_c[0]), f"[full={full},base={baseline}] compound dropped log(c): {full_c[0]}"
