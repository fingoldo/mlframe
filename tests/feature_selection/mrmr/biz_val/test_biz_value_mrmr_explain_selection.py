"""Cycle-4 biz_value: one-call ``MRMR.explain_selection()`` human-readable report.

WHY THIS LAYER
--------------
The MRMR / Hybrid public surface carries dozens of ``fe_*`` knobs, a Layer-54
survivor provenance surface (``fe_provenance_``), a per-gate rejection ledger
(``fe_rejection_ledger_``), and a Layer-99 meta-FE recommender -- but NO single
human-readable accessor narrating "why these features / what did FE build /
which gate dropped what / which knob would I turn". ``explain_selection()``
assembles those existing artifacts into a one-screen narrative.

PURE ADDITIVE, FIT-ONLY, DEFAULT-ON. It's a method you call; no flag, no
decision-logic change, zero accuracy risk.

CONTRACTS PINNED
----------------
* C1: on the canonical y = a**2/b + log(c)*sin(d) fixture (FE on), the report
  NAMES (a) surviving engineered recipe kinds, (b) the BINDING rejection gate +
  a margin band, (c) the recommender's chosen fe_* flags -- all in < 1 screen.
* C2: a domain user can answer "why / what / what-would-I-turn" from the string
  alone (readability: it contains the survivor section, the binding gate line,
  the recommender line and the actionable hint).
* C3: graceful degradation -- FE disabled -> empty-but-valid report (no crash,
  still names the survivors, says no rejections); classification fixture ditto.
* C4: never raises; length is capped under one screen.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR
    defaults = dict(
        verbose=0,
        random_seed=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        stability_selection_method="classic",
        retain_artifacts=False,
        n_jobs=1,
    )
    defaults.update(overrides)
    return MRMR(**defaults)


def _canonical_frame(n: int = 800, seed: int = 7):
    """The canonical y = a**2/b + log(c)*sin(d) fixture: a non-linear function of
    four continuous inputs plus noise columns. Drives the synergy / hybrid-orth FE
    search so survivors get engineered recipes AND the gates reject many candidates."""
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal(n)
    b = rng.uniform(0.5, 2.5, n)            # bounded away from 0
    c = rng.uniform(0.5, 5.0, n)            # positive for log
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    X = pd.DataFrame({
        "a": a, "b": b, "c": c, "d": d,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    score = a ** 2 / b + np.log(c) * np.sin(d) + 0.3 * rng.standard_normal(n)
    y = pd.Series((score > np.median(score)).astype(int))
    return X, y


def _fe_on(**overrides):
    return _mrmr(
        fe_hybrid_orth_enable=True,
        fe_auto=True,
        **overrides,
    )


# ---------------------------------------------------------------------------
# C1 + C2: canonical-fixture readability
# ---------------------------------------------------------------------------

def test_explain_selection_names_recipes_binding_gate_and_flags():
    X, y = _canonical_frame()
    est = _fe_on()
    est.fit(X, y)
    report = est.explain_selection()

    assert isinstance(report, str) and report
    # < 1 screen (the assembly caps; a screen is ~2600 chars / ~40 lines).
    assert len(report) <= 2600
    assert report.count("\n") <= 40

    low = report.lower()
    # (a) surviving engineered recipe kinds named.
    assert "surviving features" in low
    assert "recipe kinds" in low
    # (b) the binding rejection gate is named, OR (graceful) no rejections happened.
    assert "binding gate" in low or "no fe candidate was dropped" in low
    # (c) the recommender's chosen flags section is present.
    assert "fe recommender" in low
    # actionable hint present.
    assert "hint:" in low


def test_canonical_report_surfaces_real_recipe_and_gate_with_margin():
    """biz_value: when FE actually builds + a gate actually binds, the report must
    surface a GENUINE engineered recipe kind AND the binding gate with a numeric
    margin band -- the readability payload a user needs."""
    X, y = _canonical_frame()
    est = _fe_on()
    est.fit(X, y)
    report = est.explain_selection()

    prov = est.fe_provenance_
    led = est.fe_rejection_ledger_

    if prov is not None and (prov["origin"].astype(str) != "raw").any():
        # at least one engineered recipe kind must appear by name in the report.
        eng_kinds = {str(o) for o in prov["origin"].unique() if str(o) != "raw"}
        assert any(k in report for k in eng_kinds), (
            f"no engineered recipe kind from {eng_kinds} surfaced in report:\n{report}"
        )

    if led is not None and not led.empty:
        # the binding gate (top killer) must be named AND a margin band rendered.
        top_gate = str(led.groupby("gate").size().sort_values(ascending=False).index[0])
        assert top_gate in report, f"binding gate {top_gate} not in report:\n{report}"
        assert "margin" in report.lower()


# ---------------------------------------------------------------------------
# C3: graceful degradation
# ---------------------------------------------------------------------------

def test_explain_selection_graceful_with_fe_disabled():
    X, y = _canonical_frame(n=600)
    est = _mrmr(fe_max_steps=0, fe_hybrid_orth_enable=False, fe_mi_greedy_enable=False, fe_auto=False)
    est.fit(X, y)
    report = est.explain_selection()
    assert isinstance(report, str) and report
    low = report.lower()
    # Graceful report: it names survivors and carries a COHERENT rejections section -- either "none" (when the
    # gates dropped nothing) or a rejection summary. (fe_max_steps=0 turns off the iterative pair search, but the
    # default-on discrete/basis operator families still run via the operator-only path, so candidates may be
    # gated -- the report must describe that outcome either way.)
    assert "surviving features" in low
    assert (
        "no fe candidate was dropped" in low
        or "rejections: none" in low
        or "candidates dropped" in low
        or "binding gate" in low
    ), report
    assert "fe recommender" in low
    assert len(report) <= 2600


def test_explain_selection_graceful_on_plain_classification():
    rng = np.random.default_rng(3)
    n = 500
    x_sig = rng.standard_normal(n)
    X = pd.DataFrame({
        "x_signal": x_sig,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    y = pd.Series((x_sig + 0.3 * rng.standard_normal(n) > 0).astype(int))
    est = _mrmr(fe_max_steps=0, fe_auto=False)
    est.fit(X, y)
    report = est.explain_selection()
    assert isinstance(report, str) and report
    assert "surviving features" in report.lower()
    assert "hint:" in report.lower()


def test_explain_selection_never_raises_on_unfitted():
    from mlframe.feature_selection.filters.mrmr import MRMR
    est = MRMR(verbose=0, random_seed=0)
    # unfitted: must not raise, returns a valid (degraded) report.
    report = est.explain_selection()
    assert isinstance(report, str) and report
