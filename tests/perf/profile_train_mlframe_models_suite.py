"""CODE-LOW-6: cProfile harness for ``train_mlframe_models_suite`` end-to-end.

Runs a tiny synthetic single-target regression suite, captures a cProfile dump under
``tests/perf/results/train_mlframe_models_suite.prof``, and prints the top-30 cumulative
hotspots. This is a sibling of ``profile_training_core.py`` with the canonical filename the
audit table calls out (``profile_train_mlframe_models_suite.py``); kept as a thin wrapper so
both names resolve to the same harness logic.

Usage::

    python -m tests.perf.profile_train_mlframe_models_suite
    python -m tests.perf.profile_train_mlframe_models_suite --n-rows 5000
    python -m tests.perf.profile_train_mlframe_models_suite --output custom.prof

cProfile attribution overhead inflates pandas-internal call timings -- treat the dump as a
relative-ranking tool, not an absolute cost report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the existing harness logic so a single source of truth ranks the hotspots.
HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from profile_training_core import profile  # type: ignore[import-not-found]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "train_mlframe_models_suite.prof",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    out = profile(n_rows=args.n_rows, output_path=args.output, top=args.top)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
