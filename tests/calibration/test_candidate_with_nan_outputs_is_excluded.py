"""A calibrator with non-finite OOF outputs is not a candidate (audit 2026-11-20, finding 22-F2)."""
import numpy as np

import mlframe.calibration._policy_pick_helpers as helpers
import mlframe.calibration.policy as policy


def test_nan_producing_candidate_is_excluded(monkeypatch):
    """The NaN candidate used to be scored, its ECE computed over the finite rows only, and could win the pick."""
    def fake_fit(name, p, y):
        """Return an identity calibrator for 'good' and a NaN-producing one for 'bad'."""
        if name == "bad":
            return lambda x: np.full_like(np.asarray(x, dtype=np.float64), np.nan)
        return lambda x: np.asarray(x, dtype=np.float64)

    monkeypatch.setattr(policy, "_fit_calibrator", fake_fit)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.uniform(size=400) < p).astype(int)
    idx_matrix = rng.integers(0, p.size, size=(20, p.size))
    results = helpers.score_candidates(("good", "bad"), p, y, None, idx_matrix, 10, 0.1)
    assert "bad" not in results
    assert "good" in results
