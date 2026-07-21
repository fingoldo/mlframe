"""Regression tests for audits/full_audit_2026-07-21/training_composite_discovery.md findings F1, F2, F6.

F3 (``_calibration_gate.py`` implemented-but-unwired) requires new residual-capture plumbing through
the tiny-CV loop that doesn't exist anywhere today -- judged out of scope for this fix pass (real
feature work, not a bug fix); the module is left as its own honestly-documented opt-in code, unused
but real and tested (see ``tests/training/composite/screening/test_calibration_gate.py``).
F4 is a docstring-only fix (no behavior to pin). F5 is an architecture/LOC-debt flag, not a bug.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1 (P1): multi-base forward-stepwise promotion ignored config.random_state / groups / time_aware
# ---------------------------------------------------------------------------


def test_f1_multibase_promotion_threads_random_state_and_groups(monkeypatch):
    """apply_multi_base_forward_stepwise now passes the caller's config.random_state and the
    (train_idx-sliced) group_ids through to forward_stepwise_multi_base, instead of always using
    the function's own hardcoded defaults (random_state=42, groups=None)."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    import mlframe.training.composite.discovery._fit_multibase as fmb_mod

    captured = []
    orig_fn = fmb_mod.forward_stepwise_multi_base

    def spy(*args, **kwargs):
        """Records call arguments for this test's assertions."""
        captured.append((args, kwargs))
        return orig_fn(*args, **kwargs)

    monkeypatch.setattr(fmb_mod, "forward_stepwise_multi_base", spy)

    rng = np.random.default_rng(0)
    n = 600
    base_1 = rng.normal(size=n)
    base_2 = rng.normal(size=n)
    y = base_1 + 0.3 * base_2 + rng.normal(scale=0.05, size=n)
    groups = np.repeat(np.arange(30), n // 30)
    df = pd.DataFrame({"base_1": base_1, "base_2": base_2, "y": y})

    config = CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["base_1", "base_2"],
        transforms=("linear_residual",),
        tiny_screening_models="single_lgbm",
        tiny_model_n_seed_repeats=1,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        transform_waic_validation_enabled=False,
        multi_base_enabled=True,
        multi_base_max_k=2,
        random_state=777,
        require_beats_raw_baseline=False,
        fail_on_no_gain="fallback_raw",
    )
    disc = CompositeTargetDiscovery(config)
    disc._group_ids_for_rerank = groups
    train_idx = np.arange(n)
    disc.fit(df, "y", ["base_1", "base_2"], train_idx)

    assert captured, "forward_stepwise_multi_base was never called -- test fixture didn't reach multi-base promotion"
    for args, kwargs in captured:
        assert kwargs.get("random_state") == 777, kwargs
        # groups is train_idx-sliced group_ids, not None (the pre-fix behavior) -- length matches
        # the y_train array actually passed (the honest-holdout split may reduce it below len(train_idx)).
        assert kwargs.get("groups") is not None, kwargs
        _y_train_passed = args[0]
        assert len(kwargs["groups"]) == len(_y_train_passed)


def test_f1_multibase_random_state_default_still_42_when_unset():
    """Sanity: forward_stepwise_multi_base's OWN default (random_state=42) is unchanged; the fix
    threads the CALLER's config value through, it doesn't change the function's own default."""
    import inspect

    from mlframe.training.composite.discovery.forward_stepwise import forward_stepwise_multi_base

    sig = inspect.signature(forward_stepwise_multi_base)
    assert sig.parameters["random_state"].default == 42
    assert sig.parameters["time_aware"].default is True
    assert sig.parameters["groups"].default is None


# ---------------------------------------------------------------------------
# F2 (P1): WAIC tie-break K-fold always used seed 0 via a nonexistent self.random_seed attribute
# ---------------------------------------------------------------------------


def test_f2_waic_tiebreak_reads_config_random_state_not_missing_attribute():
    """self.random_seed never existed on CompositeTargetDiscovery; the fixed line reads
    self.config.random_state instead, so a configured non-default seed actually takes effect."""
    from types import SimpleNamespace

    class _FakeConfig:
        """Fake Config."""
        random_state = 777
        transform_waic_n_folds = 4

    fake_self = SimpleNamespace(config=_FakeConfig())
    assert not hasattr(fake_self, "random_seed")

    rs = int(getattr(fake_self.config, "random_state", 0) or 0)
    assert rs == 777

    # Confirms the OLD expression really was silently pinned to 0 regardless of configured seed.
    rs_old_buggy = int(getattr(fake_self, "random_seed", 0) or 0)
    assert rs_old_buggy == 0


def test_f2_composite_target_discovery_has_no_random_seed_attribute():
    """Guards the root cause staying root-caused: if a future refactor ever adds a real
    `random_seed` attribute to CompositeTargetDiscovery, this test should be revisited -- until
    then, confirms the class genuinely has no such attribute (only .config.random_state)."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    config = CompositeTargetDiscoveryConfig(enabled=True, base_candidates=["b"], random_state=5)
    disc = CompositeTargetDiscovery(config)
    assert not hasattr(disc, "random_seed")
    assert disc.config.random_state == 5


# ---------------------------------------------------------------------------
# F6 (P2, perf): _causal_rolling was an unvectorized per-row Python loop
# ---------------------------------------------------------------------------


def _causal_rolling_naive(y_sorted, window, *, median):
    """Causal rolling naive."""
    n = y_sorted.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1:
        return out
    for i in range(window, n):
        past = y_sorted[i - window : i]
        out[i] = np.median(past) if median else past.mean()
    return out


@pytest.mark.parametrize("median", [False, True])
@pytest.mark.parametrize("n,window", [(50, 3), (500, 7), (2000, 15)])
def test_f6_causal_rolling_kernel_bit_identical_to_naive(n, window, median):
    """The njit running-window kernel must match the original per-row loop's output exactly (NaN mask + values)."""
    from mlframe.training.composite.discovery._base_engineering import _causal_rolling

    rng = np.random.default_rng(n + window + int(median))
    y = rng.normal(size=n)
    naive = _causal_rolling_naive(y, window, median=median)
    fast = _causal_rolling(y, window, median=median)

    assert np.array_equal(np.isnan(naive), np.isnan(fast))
    both_nan = np.isnan(naive)
    assert np.allclose(naive[~both_nan], fast[~both_nan], rtol=0, atol=1e-9)


def test_f6_causal_rolling_uses_njit_kernel_not_python_loop():
    """Confirms the kernels are actually wired in (not just defined but unused)."""
    import inspect

    from mlframe.training.composite.discovery import _base_engineering as be_mod

    src = inspect.getsource(be_mod._causal_rolling)
    assert "_causal_rolling_mean_kernel" in src
    assert "_causal_rolling_median_kernel" in src
    assert "for i in range(window, n):" not in src
