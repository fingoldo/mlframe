"""biz_value + activation tests for ``MRMR(bur_lambda=...)`` (MRwMR-BUR unique-relevance bonus, Gao 2022).

The BUR term adds ``lambda * max(0, I(X;Y) - max_j I(X; X_j))`` to the post-Fleuret MRMR gain (``evaluation.py:713``), rewarding a candidate whose marginal-y
relevance cannot be explained by any already-selected feature. These tests pin (a) that the knob is ACTIVE (the bonus codepath fires when ``bur_lambda>0`` --
NOT a dead param like the pre-fix ``mi_correction``), and (b) the qual-22 verdict that BUR does not improve known-relevant-set selection over the default on the
production path, so a future "flip it on" cannot slip through.

Verdict (qual-22): NOT a default flip -- the default Fleuret conditional-MI redundancy term already de-prioritizes redundant clusters and retains the unique
driver (baseline unique-driver recall = 1.000 on every tested scenario), so the BUR bonus is ~0 for cluster members (their relevance IS explained by a selected
sibling) and the unique driver is already selected. See mlframe/feature_selection/_benchmarks/bench_bur_lambda_qual22.py. Kept opt-in (REJECTED!=DELETED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _redundant_cluster_unique(seed: int, n: int = 800):
    """Strong latent z replicated into a redundant cluster + a unique moderate driver u; the regime BUR is designed for."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    cluster = np.column_stack([z + 0.30 * rng.standard_normal(n) for _ in range(4)])
    u = rng.standard_normal(n)
    noise = rng.standard_normal((n, 4))
    X = pd.DataFrame(np.column_stack([cluster, u, noise]), columns=[f"f{i}" for i in range(9)])
    y = 1.6 * z + 1.1 * u + 0.4 * rng.standard_normal(n)
    return X, y, 4  # unique-driver column index


def test_biz_val_bur_lambda_default_is_zero():
    """The BUR bonus default is OFF (0.0): qual-22 found no known-relevant-set win, so the default must not carry it."""
    from mlframe.feature_selection.filters import MRMR

    assert MRMR().bur_lambda == 0.0


def test_biz_val_bur_lambda_activates_codepath_when_positive():
    """Dead-knob guard: with ``bur_lambda>0`` the ``get_bur_lambda()`` thread-local read in the candidate scorer must actually fire during fit.

    This is the qual-21 lesson applied: a validated+stored param that never reaches the fit path is a latent bug. A regression that drops the
    ``set_bur_lambda``/``get_bur_lambda`` wiring would make this counter stay 0 and fail.
    """
    import mlframe.feature_selection.filters.evaluation as ev
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.info_theory import get_bur_lambda as _real

    hits = {"pos": 0}
    orig = ev.get_bur_lambda

    def _traced():
        """Helper that traced."""
        v = _real()
        if v > 0:
            hits["pos"] += 1
        return v

    ev.get_bur_lambda = _traced
    try:
        X, y, _ = _redundant_cluster_unique(0)
        MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=1, bur_lambda=5.0).fit(X, y)
    finally:
        ev.get_bur_lambda = orig
    assert hits["pos"] > 0, "bur_lambda>0 must reach the candidate-scorer BUR codepath during fit (knob must not be dead)"


def test_biz_val_bur_lambda_thread_local_resets_after_fit():
    """The BUR thread-local must reset to 0.0 after fit so a subsequent default-MRMR fit is not silently BUR-weighted."""
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.info_theory import get_bur_lambda

    X, y, _ = _redundant_cluster_unique(1)
    MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=1, bur_lambda=1.0).fit(X, y)
    assert get_bur_lambda() == 0.0, "fit must reset the BUR thread-local to 0.0 on completion"


def test_biz_val_bur_lambda_does_not_beat_default_on_known_relevant_set():
    """Verdict pin (qual-22): BUR must NOT measurably improve unique-driver recall over the default on the redundant-cluster-vs-unique synthetic.

    The Fleuret redundancy term already keeps the unique driver (baseline recall 1.0); a future change that made BUR materially help here would mean the
    default machinery had regressed -- this test would then flag it. Fails only if BUR's unique-driver recall EXCEEDS the default's by a material margin
    (i.e. the default newly drops the driver), which is the signal to reconsider the flip.
    """
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.filters import MRMR

    def _urec(bur, seed):
        """Helper that urec."""
        X, y, uidx = _redundant_cluster_unique(seed)
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed)
        m = MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=1, bur_lambda=bur).fit(Xtr, ytr)
        chosen = {int(c[1:]) for c in m.transform(Xtr).columns if c.startswith("f") and c[1:].isdigit()}
        return 1.0 if uidx in chosen else 0.0

    seeds = range(4)
    base = [_urec(0.0, s) for s in seeds]
    bur = [_urec(1.0, s) for s in seeds]
    assert sum(base) == len(base), f"sanity: default machinery already recovers the unique driver every seed; got {base}"
    # BUR must not OUT-recall the default (it can only tie here); a strict majority win would invalidate the qual-22 verdict.
    bur_strict_wins = sum(1 for b, k in zip(base, bur) if k > b)
    assert bur_strict_wins == 0, f"qual-22 verdict: BUR should not beat default unique-driver recall on this synthetic; got {bur_strict_wins} strict wins"
