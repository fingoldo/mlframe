"""A pair scored on the per-pair batched path (no cross-pair chunk buffer) still hands its configuration codes to the family-wise gate."""

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core as core
import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score as score


def test_per_pair_path_gives_the_family_gate_its_codes(monkeypatch):
    """Without a chunk buffer the gate used to see no codes and silently skip every pair."""
    monkeypatch.setattr(core, "_check_prospective__only_worth_chunking_least", lambda *a, **k: (None, False, {}))
    seen = []
    real = score.apply_pair_family_gate_from_state

    def spy(st, classes_y, n_bins, alpha=0.01):
        """Record how many configuration codes the gate was handed for this pair."""
        seen.append(len(st.__dict__.get("var_pairs_codes") or {}))
        return real(st, classes_y, n_bins, alpha)

    monkeypatch.setattr(score, "apply_pair_family_gate_from_state", spy)
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(5)})
    y = ((X["f0"] * X["f1"] + X["f2"] / (1 + X["f3"].abs()) + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0).fit(X, y)
    assert seen and min(seen) > 0, seen
