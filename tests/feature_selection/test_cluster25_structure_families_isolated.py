"""Audit 2026-11-20 finding 25-F21: one failing structure detector family does not skip the others, and is named in the report.

The four families used to share one try block, so a crash in the first left the rest unrun and the report looked like "nothing
found". The test patches the modular family to raise and checks that the report says so, and that the gate family still ran.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection import structure_discovery as sd


def test_one_failed_family_is_named_and_the_others_still_run(monkeypatch):
    """The modular family raises; the report's skipped field names it, and gate still contributes its call."""
    calls = []

    def boom(*args, **kwargs):
        """Stand in for the modular detector failing."""
        raise RuntimeError("modular detector broke")

    real_gate = sd._gate_relations

    def counting_gate(*args, **kwargs):
        """Record that the gate family ran, then delegate to the real detector."""
        calls.append("gate")
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(sd, "_modular_relations", boom)
    monkeypatch.setattr(sd, "_gate_relations", counting_gate)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 4, size=(200, 3)), columns=["a", "b", "c"])
    y = rng.integers(0, 2, size=200)
    report = sd.discover_structure(X, y, include=("modular", "gate"), significance_n_perm=5)
    assert calls == ["gate"]
    assert report.skipped is not None and "modular" in report.skipped
