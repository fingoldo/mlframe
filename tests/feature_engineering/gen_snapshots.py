"""Regenerate DIGESTS for TestCreateAggregatedFeaturesSnapshot in test_coverage_fill.py.

Why this script exists:
- The snapshot tests assert digest(feats, names) == DIGESTS[scenario][:16].
- When numpy / scipy / numba versions shift, the produced floats can differ at
  the 12-15th decimal place. That changes the bytes-digest while the feature
  STRUCTURE (length, naming, ordering) stays identical -- not a bug, just a
  numerical-precision drift caused by upstream stack movement.
- Run this script after a deliberate libdep bump or a confirmed-clean run on
  a representative machine. Output prints the canonical DIGESTS literal you
  paste back into test_coverage_fill.py.

Usage:
    cd <mlframe repo root>
    PYTHONPATH=src python tests/feature_engineering/gen_snapshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/feature_engineering/gen_snapshots.py` from repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tests" / "feature_engineering") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests" / "feature_engineering"))


def main() -> None:
    # Reuse the test module's own helpers + scenarios so we never drift away
    # from what the test actually compares against.
    from test_coverage_fill import (  # type: ignore[import-not-found]
        DIGESTS,
        SNAPSHOTS,
        _scenario_kwargs,
        _snap_df,
        _snapshot_digest,
    )
    from mlframe.feature_engineering.timeseries import create_aggregated_features

    print("DIGESTS = {")
    max_name = max(len(repr(n)) for n in SNAPSHOTS)
    for scenario in SNAPSHOTS:
        df = _snap_df()
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            **_scenario_kwargs(scenario),
        )
        digest = _snapshot_digest(feats, names)[:16]
        prev = DIGESTS.get(scenario, "<missing>")
        flag = " # CHANGED" if digest != prev else ""
        print(f"    {repr(scenario):<{max_name}}: {repr(digest)},{flag}")
    print("}")


if __name__ == "__main__":
    main()
