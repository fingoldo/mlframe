"""Dump what the benchmark fit selects, to compare two trees for selection equivalence.

``python -m mlframe.feature_selection._benchmarks.profiling.fit_selection_dump OUT.json [rows] [cols]`` runs the ``all_families_wall`` configuration (every FE family, strict-resident GPU, mdlp, seed 0,
a warm-up fit first) and writes the selected column indices, ``n_features_`` and the engineered features to ``OUT.json``. Run it on two checkouts and compare the files: identical files mean the changes between
them did not change what the fit selects (2026-11-20: the launch-reduction series, 6aeb3c516 against 8dcb8971e, identical).
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np


def main(out_path: str, n_rows: int = 100000, n_cols: int = 100) -> dict:
    """Fit the benchmark configuration and write the selection dump to ``out_path``; returns the dump."""
    warnings.simplefilter("ignore")
    for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
        os.environ.setdefault(k, v)
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
    data = F._make(n_rows, n_cols, 0)
    MRMR._FIT_CACHE.clear()
    m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw)
    m.fit(*data)
    out = {
        "support": [int(i) for i in np.asarray(m.support_).ravel()],
        "n_features": int(m.n_features_),
        "engineered": sorted(str(x) for x in (getattr(m, "_engineered_features_", None) or [])),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    dump = main(sys.argv[1], *(int(a) for a in sys.argv[2:4]))
    print("DUMP support", len(dump["support"]), "engineered", len(dump["engineered"]))
