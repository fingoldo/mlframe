"""The binary-classification usability greedy scores its shortlist with one batched Newton fit per fold and selects exactly what the per-candidate sklearn refit selects."""
import numpy as np
import pytest

from mlframe.feature_selection.filters import _usability_greedy_clf_host_batched as hb
from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate, usability_greedy


def _pool(seed, n=1500, p=14):
    """Fourteen raw columns plus four near-duplicate tanh forms; the target depends on four of them and one product."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    logit = sum((1 if i % 2 else -1) * X[:, i] for i in range(4)) + 0.5 * X[:, 4] * X[:, 5]
    y = (logit + rng.logistic(size=n) > 0).astype(int)
    pool = [UsableCandidate(name=f"c{i}", values=X[:, i].copy(), mi=float(abs(np.corrcoef(X[:, i], y)[0, 1]))) for i in range(p)]
    for j in range(4):
        v = np.tanh(X[:, j]) + 1e-3 * rng.standard_normal(n)
        pool.append(UsableCandidate(name=f"t{j}", values=v, mi=float(abs(np.corrcoef(v, y)[0, 1]))))
    return pool, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_newton_selects_what_the_exact_refit_selects(seed, monkeypatch):
    """Same selected names, same order, with the batched fit and with the per-candidate refit forced."""
    pool, y = _pool(seed)
    monkeypatch.setenv("_USAB_FORCE_FULL_REFIT", "1")
    exact = [c.name for c in usability_greedy(pool, y, classification=True, seed=seed)]
    monkeypatch.delenv("_USAB_FORCE_FULL_REFIT")
    fast = [c.name for c in usability_greedy(pool, y, classification=True, seed=seed)]
    assert fast == exact and exact


def test_a_failing_batched_fit_falls_back_to_the_exact_refit(monkeypatch):
    """A fold the Newton fit cannot solve leaves the selection to the sklearn refit: identical result."""
    pool, y = _pool(0)
    monkeypatch.setenv("MLFRAME_USAB_CLF_NEWTON", "0")
    reference = [c.name for c in usability_greedy(pool, y, classification=True, seed=0)]
    monkeypatch.delenv("MLFRAME_USAB_CLF_NEWTON")

    def _boom(*_a, **_k):
        """Stand-in for a singular or non-convergent fit."""
        raise hb.HostBatchedFallbackError()

    monkeypatch.setattr(hb, "cv_candidates_binary_host", _boom)
    assert [c.name for c in usability_greedy(pool, y, classification=True, seed=0)] == reference
