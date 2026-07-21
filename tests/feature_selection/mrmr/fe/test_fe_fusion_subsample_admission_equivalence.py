"""mrmr_audit_2026-07-20 B-20: the GPU-resident additive-fusion twin decides fusion
admission on a strided subsample above ``MLFRAME_FE_FUSION_MAX_ROWS`` (default 250,000
rows), while the CPU sibling :func:`propose_additive_fusions` always decides on the full
n. ``test_fe_fusion_scoring_subsample.py`` pins the STRIDE FORMULA and the full-n OUTPUT
invariant, but nothing compared the two backends' actual ADMISSION VERDICT above the
250k threshold -- this file closes that gap directly on the CPU decision primitives (no
cupy/GPU dependency), verifying that scoring on a strided subsample reproduces the SAME
admit/reject decision as scoring on the full n, on realistic above-cap row counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_additive_fusion import propose_additive_fusions
from mlframe.feature_selection.filters.engineered_recipes import build_orth_univariate_recipe
from mlframe.feature_selection.filters.mrmr import MRMR


def _fusion_stride(n, max_rows):
    """The exact strided-subsample formula used in propose_additive_fusions_gpu."""
    return int(n // max_rows) if max_rows > 0 and n > max_rows else 1


def _build_fusion_inputs(n, seed):
    """Two disjoint, additively-separable engineered halves (He_2(c), He_2(d)) whose sum
    carries genuine joint signal with y -- a clean ADMIT case for the fusion razor."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
    h_c = c**2 - 1.0  # He_2(c)
    h_d = d**2 - 1.0  # He_2(d)
    y_cont = h_c + h_d + 0.05 * rng.standard_normal(n)
    y_dense = pd.qcut(y_cont, q=8, labels=False, duplicates="drop").astype(np.int64)

    name_c = "c__He2"
    name_d = "d__He2"
    engineered_recipes = {
        name_c: build_orth_univariate_recipe(name=name_c, src_name="c", basis="hermite", degree=2),
        name_d: build_orth_univariate_recipe(name=name_d, src_name="d", basis="hermite", degree=2),
    }
    engineered_continuous = {name_c: h_c, name_d: h_d}
    X = pd.DataFrame({"c": c, "d": d})
    return {
        "engineered_recipes": engineered_recipes,
        "engineered_continuous": engineered_continuous,
        "newly_engineered_names": [name_c, name_d],
        "raw_name_set": {"c", "d"},
        "cols": ["c", "d"],
        "classes_y": y_dense,
        "X": X,
        "nbins": 10,
        "seed": 0,
    }


class TestFusionAdmissionSubsampleEquivalence:
    """mrmr_audit_2026-07-20 B-20: the fusion admission VERDICT (admit vs reject) must
    match between full-n scoring and strided-subsample scoring above the row cap --
    exactly the property the GPU-resident twin relies on to stay selection-equivalent."""

    def test_strided_subsample_reproduces_full_n_admission_verdict(self):
        """At n=1,000,000 (stride=4 under the default 250k cap), scoring the SAME clean
        additive-fusion fixture on the full n and on the stride-4 subsample must reach the
        SAME admit/reject verdict for the (c__He2, d__He2) pair."""
        n = 1_000_000
        max_rows = 250_000
        stride = _fusion_stride(n, max_rows)
        assert stride == 4, "fixture assumption: n=1e6 under a 250k cap strides by 4"

        inputs_full = _build_fusion_inputs(n, seed=0)
        fs = MRMR(verbose=0)

        admitted_full, subsumed_full, _ = propose_additive_fusions(fs, **inputs_full)
        assert admitted_full, "clean additive-separable fixture must be admitted at full n (fixture sanity check)"
        assert {"c__He2", "d__He2"} <= subsumed_full

        # Build the SAME data, then take every `stride`-th row BEFORE calling propose_additive_fusions --
        # mirrors what the GPU-resident twin's scoring subsample sees (it also scores the halves AND the
        # fused pair on the same strided rows).
        rng = np.random.default_rng(0)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        h_c = c**2 - 1.0
        h_d = d**2 - 1.0
        y_cont = h_c + h_d + 0.05 * rng.standard_normal(n)
        y_dense = pd.qcut(y_cont, q=8, labels=False, duplicates="drop").astype(np.int64)

        sc = slice(None, None, stride)
        inputs_sc = {
            "engineered_recipes": inputs_full["engineered_recipes"],
            "engineered_continuous": {"c__He2": h_c[sc], "d__He2": h_d[sc]},
            "newly_engineered_names": ["c__He2", "d__He2"],
            "raw_name_set": {"c", "d"},
            "cols": ["c", "d"],
            "classes_y": y_dense[sc],
            "X": pd.DataFrame({"c": c[sc], "d": d[sc]}),
            "nbins": 10,
            "seed": 0,
        }
        admitted_sc, subsumed_sc, _ = propose_additive_fusions(fs, **inputs_sc)
        assert admitted_sc, "B-20 regression: subsample-scored decision diverged from full-n (rejected where full-n admits)"
        assert {"c__He2", "d__He2"} <= subsumed_sc, "B-20 regression: subsample scoring subsumed a different half set than full-n"

    def test_strided_subsample_reproduces_full_n_rejection_verdict(self):
        """The rejection side of the same property: two INDEPENDENT (non-additive) halves
        must stay REJECTED under both full-n and stride-4 subsample scoring."""
        n = 1_000_000
        stride = _fusion_stride(n, 250_000)
        assert stride == 4

        rng = np.random.default_rng(1)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        h_c = c**2 - 1.0
        h_d = d**2 - 1.0
        # y depends on h_c ONLY -- h_d is pure noise relative to y, so add(h_c, h_d) buys no
        # genuine joint uplift over the stronger half alone; the razor must reject the fusion.
        y_cont = h_c + 0.05 * rng.standard_normal(n)
        y_dense = pd.qcut(y_cont, q=8, labels=False, duplicates="drop").astype(np.int64)

        name_c, name_d = "c__He2", "d__He2"
        engineered_recipes = {
            name_c: build_orth_univariate_recipe(name=name_c, src_name="c", basis="hermite", degree=2),
            name_d: build_orth_univariate_recipe(name=name_d, src_name="d", basis="hermite", degree=2),
        }
        fs = MRMR(verbose=0)

        def _run(sc):
            """Run propose_additive_fusions on the given row slice."""
            return propose_additive_fusions(
                fs,
                engineered_recipes=engineered_recipes,
                engineered_continuous={name_c: h_c[sc], name_d: h_d[sc]},
                newly_engineered_names=[name_c, name_d],
                raw_name_set={"c", "d"},
                cols=["c", "d"],
                classes_y=y_dense[sc],
                X=pd.DataFrame({"c": c[sc], "d": d[sc]}),
                nbins=10,
                seed=0,
            )

        admitted_full, _, _ = _run(slice(None))
        admitted_sc, _, _ = _run(slice(None, None, stride))
        fused_full = {a["name"] for a in admitted_full if name_c in a["name"] and name_d in a["name"]}
        fused_sc = {a["name"] for a in admitted_sc if name_c in a["name"] and name_d in a["name"]}
        assert not fused_full, "fixture sanity check: independent-halves pair must NOT fuse at full n"
        assert not fused_sc, "B-20 regression: subsample scoring admitted a fusion full-n correctly rejects"
