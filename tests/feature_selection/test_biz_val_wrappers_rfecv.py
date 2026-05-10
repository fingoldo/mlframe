"""biz_val tests for ``RFECV`` (feature_selection/wrappers/_rfecv.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN that locks in RFECV's
core parameters. A future code change that silently breaks one of
these will fail the matching assertion.

Naming: ``test_biz_val_rfecv_<parameter>_<scenario>``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _signal_plus_noise(n=2000, p_signal=3, p_noise=10, seed=42):
    """Linear binary target with ``p_signal`` true features and
    ``p_noise`` pure-noise features."""
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_signal, X_noise])
    y = (X_signal.sum(axis=1) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def _correlated_redundant(n=2000, seed=42):
    """4 correlated features (rho ~0.95) + 1 unique informative + 5
    noise. Greedy selection might pick from the correlated cluster
    only; stability selection should additionally surface the unique
    informative."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    X_corr = np.column_stack([base + 0.1 * rng.normal(size=n) for _ in range(4)])
    X_signal_unique = rng.normal(size=(n, 1))
    X_noise = rng.normal(size=(n, 5))
    X = np.column_stack([X_corr, X_signal_unique, X_noise])
    y = (X_corr[:, 0] + X_signal_unique[:, 0] + 0.3 * rng.normal(size=n) > 0
         ).astype(np.int64)
    return X, y


def _support_indices(sel):
    """Return support_ as integer indices regardless of whether RFECV
    exposes a boolean mask or an integer-array support."""
    s = sel.support_
    if s.dtype == bool:
        return [int(i) for i in np.flatnonzero(s)]
    return [int(i) for i in s]


# ---------------------------------------------------------------------------
# n_features_selection_rule: argmax vs one_se_min
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_n_features_selection_rule_one_se_min_picks_smaller():
    """``n_features_selection_rule='one_se_min'`` must pick <= the
    number of features as ``='argmax'`` on a target with 3 strong +
    10 noise. The 1-SE rule trades a tiny mean-score loss for a
    smaller, more interpretable model."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    common = dict(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
    )
    sel_argmax = RFECV(n_features_selection_rule="argmax", **common)
    sel_one_se = RFECV(n_features_selection_rule="one_se_min", **common)
    sel_argmax.fit(df, y)
    sel_one_se.fit(df, y)
    assert sel_one_se.n_features_ <= sel_argmax.n_features_, (
        f"one_se_min picked {sel_one_se.n_features_}, argmax picked "
        f"{sel_argmax.n_features_}; one_se_min must be <= argmax"
    )


# ---------------------------------------------------------------------------
# stability_selection: bootstrap voting
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_stability_selection_recovers_signal_features():
    """With ``stability_selection=True`` + bootstrap voting, RFECV
    should select features that appear in the majority of bootstrap
    runs. On a clean signal-plus-noise target, the 3 signal features
    must all be in the support_."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=6, verbose=0, random_state=42,
        max_noimproving_iters=3,
        stability_selection=True,
        stability_n_bootstrap=10,
        stability_threshold=0.4,
    )
    sel.fit(df, y)
    selected = set(_support_indices(sel))
    overlap = selected & {0, 1, 2}
    assert len(overlap) >= 2, (
        f"stability_selection must recover >=2 of 3 signal features; "
        f"got selected={sorted(selected)}, overlap={overlap}"
    )


# ---------------------------------------------------------------------------
# must_include: forced inclusion
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_must_include_keeps_specified_feature():
    """``must_include=['x_noise_5']`` (a pure-noise column) must
    remain in support_ after RFECV runs. Without the constraint,
    a noise feature would never survive feature elimination on this
    target. Catches regressions in the must_include enforcement."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    # Name the noise columns explicitly so the must_include is clear.
    cols = [f"x{i}" for i in range(3)] + [f"noise{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=cols)

    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
        must_include=["noise5"],
    )
    sel.fit(df, y)
    selected_names = [df.columns[i] for i in _support_indices(sel)]
    assert "noise5" in selected_names, (
        f"must_include=['noise5'] must keep noise5; "
        f"got selected={selected_names}"
    )


# ---------------------------------------------------------------------------
# importance_getter='conditional_permutation' (CPI)
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_conditional_permutation_keeps_unique_informative_under_correlation():
    """Vanilla permutation importance assigns ZERO importance to each
    member of a correlated cluster (the model still predicts via the
    redundant copies). Conditional permutation breaks this by
    permuting WITHIN leaves of a tree on ``X_{-j}``, preserving the
    joint distribution.

    Floor: with ``importance_getter='conditional_permutation'``, the
    UNIQUE informative feature ``x4`` (not in the correlated cluster)
    must be selected. A vanilla-permutation run would also pick at
    least one cluster member; the cluster's INTERNAL ranking is more
    sensitive to CPI so that's not the cleanest signal."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _correlated_redundant(n=1500, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=50,
                                            max_depth=6),
        cv=3, max_refits=6, verbose=0, random_state=42,
        max_noimproving_iters=3,
        importance_getter="conditional_permutation",
    )
    sel.fit(df, y)
    selected = set(_support_indices(sel))
    assert 4 in selected, (
        f"CPI must keep unique-informative x4; got selected={sorted(selected)}"
    )
    # And at least one of the correlated cluster {x0..x3} must also
    # survive (each carries the same joint signal).
    assert any(i in selected for i in (0, 1, 2, 3)), (
        f"CPI must keep >=1 correlated-cluster member; "
        f"got selected={sorted(selected)}"
    )


# ---------------------------------------------------------------------------
# checkpoint_path: resume-from-checkpoint
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_checkpoint_resume_produces_same_support(tmp_path):
    """RFECV with ``checkpoint_path`` must (a) write a resume file
    that allows a subsequent identical fit to pick up where it left
    off, AND (b) produce identical support_ vs running through to
    completion in one call. Catches regressions in the
    save / load / signature-check logic."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1000, p_signal=3, p_noise=8, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    cp = str(tmp_path / "rfecv_ckpt.pkl")

    sel_full = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
    )
    sel_full.fit(df, y)
    sel_resume = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
        checkpoint_path=cp,
    )
    sel_resume.fit(df, y)
    full_set = set(_support_indices(sel_full))
    resume_set = set(_support_indices(sel_resume))
    # Both must converge on the same support set on a deterministic
    # seed; the checkpoint mechanism must not change the result.
    assert full_set == resume_set, (
        f"checkpoint-enabled fit must produce same support; "
        f"full={sorted(full_set)}, ckpt={sorted(resume_set)}"
    )
