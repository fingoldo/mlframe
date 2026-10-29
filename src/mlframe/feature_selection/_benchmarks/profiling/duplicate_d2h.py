"""Device-to-host copies of one warm fit that repeat the same content, with the call sites of every copy.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.duplicate_d2h [rows] [extra MRMR kwargs as key=value, ...]``. Runs the F2 example once warm under
``duplicate_transfer_audit`` (GPU strict-resident mode, as in ``fit_wall.py``) and prints the transfers grouped by content. A dense float64 column and its float32 copy, or int8 and int64
codes of one column, are one group. Run it on a quiet box: the groups depend on which backends the kernel-tuning cache picks.
"""

from __future__ import annotations

import ast
import os
import sys
import warnings

for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)
warnings.simplefilter("ignore")

from mlframe.feature_selection._benchmarks.offset_product.ab_fit import _make
from mlframe.feature_selection.filters._gpu_strict_fe import duplicate_transfer_audit
from mlframe.feature_selection.filters.mrmr import MRMR


def main() -> None:
    """Warm up on a small frame, fit the requested size under the audit, print the report."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000
    extra = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    kw = dict(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1) | extra
    df, y = _make(5_000, 1)
    MRMR(random_seed=1, **kw).fit(df, y)
    df, y = _make(n, 2)
    with duplicate_transfer_audit() as rep:
        MRMR(random_seed=2, **kw).fit(df, y)
    print(rep.summary(limit=20), flush=True)


if __name__ == "__main__":
    main()
