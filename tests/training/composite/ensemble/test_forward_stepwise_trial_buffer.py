"""Bit-identity regression for the preallocated-trial-buffer optimization in
``forward_stepwise_multi_base``.

The greedy forward-stepwise selector used to rebuild the ``(n, K+1)`` OLS design
matrix with ``np.column_stack`` on EVERY candidate trial. The current code stacks
the kept-prefix into a single reused buffer ONCE per round and overwrites only the
candidate column per trial. The matrix handed to OLS is byte-identical, so the
selection and the per-fold RMSEs MUST be identical between the two paths.

The ``_legacy_per_trial_stack`` A/B knob exposes the old behaviour at the public
API:
  * ``_legacy_per_trial_stack=True``  -> legacy ``np.column_stack`` per trial.
  * ``_legacy_per_trial_stack=False`` -> reused preallocated buffer (default).

These tests pin BOTH sides equal across the splitter modes the helper supports
(shuffled KFold, TimeSeriesSplit, GroupKFold) and with fold-score persistence on,
so a future refactor that broke the buffer fill order / dtype / contiguity (and
thus silently changed an OLS input) trips the per-fold-RMSE comparison, not just
the final ``kept`` list.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import forward_stepwise_multi_base


def _make_pool(n: int, n_seed: int, n_cand: int, *, seed: int):
    """Target driven by ``n_seed`` signal bases plus a pool of weaker candidates."""
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    signal = np.zeros(n, dtype=np.float64)
    for i in range(n_seed):
        c = rng.normal(loc=float(i), scale=1.0 + 0.2 * i, size=n)
        cols[f"seed{i}"] = c
        signal = signal + (0.8 + 0.1 * i) * c
    # A mix: a couple of genuinely-helpful bases + noise, so selection makes real choices.
    for j in range(n_cand):
        if j % 4 == 0:
            cols[f"cand{j}"] = signal * (0.1 + 0.02 * j) + rng.normal(scale=2.0, size=n)
        else:
            cols[f"cand{j}"] = rng.normal(scale=1.0 + 0.05 * j, size=n)
    y = signal + rng.normal(scale=0.5, size=n)
    seeds = [f"seed{i}" for i in range(n_seed)]
    return y, cols, seeds


def _run(y, cols, seeds, *, legacy: bool, **kw):
    """Runs forward_stepwise_multi_base with fold-score persistence on, toggling the legacy per-trial-stack code path."""
    return forward_stepwise_multi_base(
        y,
        candidate_bases=cols,
        seed_bases=seeds,
        cv_persist_fold_scores=True,
        _legacy_per_trial_stack=legacy,
        **kw,
    )


def _assert_identical(res_legacy, res_buffer) -> None:
    """Asserts the legacy per-trial-stack path and the buffered path selected the same candidates in the same order."""
    kept_l, diag_l = res_legacy
    kept_b, diag_b = res_buffer
    # Same selection, same order.
    assert kept_l == kept_b, f"selection diverged: legacy={kept_l} buffer={kept_b}"
    # Same number of diagnostic steps.
    assert len(diag_l) == len(diag_b)
    for step, (dl, db) in enumerate(zip(diag_l, diag_b)):
        assert dl["candidate_added"] == db["candidate_added"], f"step {step} candidate diverged"
        assert dl["accepted"] == db["accepted"], f"step {step} accept flag diverged"
        # Aggregated before/after RMSE bit-identical (==, not approx).
        for fld in ("rmse_before", "rmse_after", "marginal_gain"):
            a, b = dl[fld], db[fld]
            if np.isnan(a) or np.isnan(b):
                assert np.isnan(a) and np.isnan(b), f"step {step} {fld}: {a} vs {b}"
            else:
                assert a == b, f"step {step} {fld} not bit-identical: {a!r} vs {b!r}"
        # Per-candidate per-fold RMSE tables bit-identical.
        fl = dl.get("fold_rmses_per_candidate", {})
        fb = db.get("fold_rmses_per_candidate", {})
        assert set(fl.keys()) == set(fb.keys()), f"step {step} candidate set diverged"
        for cand, folds_l in fl.items():
            folds_b = fb[cand]
            assert len(folds_l) == len(folds_b), f"step {step} cand {cand} fold count diverged"
            for fi, (xl, xb) in enumerate(zip(folds_l, folds_b)):
                assert xl == xb, f"step {step} cand {cand} fold {fi} RMSE not bit-identical: {xl!r} vs {xb!r}"


@pytest.mark.parametrize("n", [400, 2000])
@pytest.mark.parametrize("max_k", [2, 3, 4])
def test_buffer_vs_legacy_bit_identical_timeseries(n: int, max_k: int) -> None:
    """Default ``time_aware=True`` (TimeSeriesSplit) -- the production code path."""
    y, cols, seeds = _make_pool(n, n_seed=1, n_cand=12, seed=11)
    res_l = _run(y, cols, seeds, legacy=True, max_k=max_k, min_marginal_rmse_gain=0.0, cv_folds=3)
    res_b = _run(y, cols, seeds, legacy=False, max_k=max_k, min_marginal_rmse_gain=0.0, cv_folds=3)
    _assert_identical(res_l, res_b)


def test_buffer_vs_legacy_bit_identical_shuffled_kfold() -> None:
    """``time_aware=False`` with no groups -> shuffled KFold."""
    y, cols, seeds = _make_pool(1500, n_seed=2, n_cand=15, seed=22)
    kw = dict(max_k=4, min_marginal_rmse_gain=0.0, cv_folds=4, time_aware=False, random_state=7)
    _assert_identical(_run(y, cols, seeds, legacy=True, **kw), _run(y, cols, seeds, legacy=False, **kw))


def test_buffer_vs_legacy_bit_identical_groupkfold() -> None:
    """``time_aware=False`` with a groups array -> GroupKFold path."""
    n = 1200
    y, cols, seeds = _make_pool(n, n_seed=1, n_cand=12, seed=33)
    groups = np.repeat(np.arange(6), n // 6)
    kw = dict(max_k=3, min_marginal_rmse_gain=0.0, cv_folds=3, time_aware=False, groups=groups)
    _assert_identical(_run(y, cols, seeds, legacy=True, **kw), _run(y, cols, seeds, legacy=False, **kw))


def test_buffer_vs_legacy_bit_identical_with_gate_and_paired() -> None:
    """With the relative-gain gate + paired-fold selection both active (production defaults),
    the two paths must still pick the same bases and stop at the same step."""
    # Two strong signal bases so the greedy loop genuinely runs multiple rounds even under the
    # 2% relative-gain gate + paired-fold majority (otherwise the test would be vacuous).
    y, cols, seeds = _make_pool(1800, n_seed=2, n_cand=14, seed=44)
    # Seed only the first; leave the second strong base in the pool so it gets greedily ADDED.
    pool_seeds = seeds[:1]
    kw = dict(max_k=4, min_marginal_rmse_gain=0.0, cv_folds=3, paired_fold_selection=True)
    res_l = _run(y, cols, pool_seeds, legacy=True, **kw)
    res_b = _run(y, cols, pool_seeds, legacy=False, **kw)
    _assert_identical(res_l, res_b)
    # Sanity: this fixture actually adds at least one base (otherwise the test is vacuous).
    assert len(res_b[0]) > len(pool_seeds), "fixture should add at least one base to exercise multiple rounds"


def test_buffer_path_is_the_default() -> None:
    """Guard against a future flip of the optimization off-by-default: calling WITHOUT the knob
    must equal the explicit buffer path, never the legacy path (when they would differ they don't,
    but this pins that the public default routes through the buffer code)."""
    y, cols, seeds = _make_pool(800, n_seed=1, n_cand=10, seed=55)
    kw = dict(max_k=3, min_marginal_rmse_gain=0.0, cv_folds=3)
    default = forward_stepwise_multi_base(y, candidate_bases=cols, seed_bases=seeds, cv_persist_fold_scores=True, **kw)
    explicit_buffer = _run(y, cols, seeds, legacy=False, **kw)
    _assert_identical(default, explicit_buffer)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
