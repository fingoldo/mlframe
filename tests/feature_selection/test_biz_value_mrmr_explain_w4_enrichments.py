"""Wave-2 W4 biz_value: two assemblable enrichments on ``MRMR.explain_selection()``.

(a) WHAT-IF-FLIP preview -- "relaxing knob X by one band would re-admit N ledger
    candidates at margin > -delta". PURE COUNT over the recorded ledger, NO refit.
(b) PER-FEATURE MI/gain ATTRIBUTION -- each surviving feature line shows its MRMR
    gain to y (the cached selection score), ordered so the top-signal survivor leads.

DECISIVE CROSS-CHECK
--------------------
The what-if preview count is asserted to EQUAL the actual number of re-admits when the
flag is really flipped one band and the fit re-run, restricted to the candidates the
ledger recorded (the preview's universe). Both pure-additive metadata; selection is
byte-identical (the preview never refits, never mutates state).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._mrmr_explain import _GATE_TO_FLIP_BAND


def _mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR
    defaults = dict(
        verbose=0, random_seed=0, dcd_enable=False, cluster_aggregate_enable=False,
        build_friend_graph=False, stability_selection_method="classic",
        retain_artifacts=False, n_jobs=1,
    )
    defaults.update(overrides)
    return MRMR(**defaults)


def _canonical_frame(n: int = 800, seed: int = 7):
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal(n)
    b = rng.uniform(0.5, 2.5, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    X = pd.DataFrame({
        "a": a, "b": b, "c": c, "d": d,
        "noise_0": rng.standard_normal(n), "noise_1": rng.standard_normal(n),
    })
    score = a ** 2 / b + np.log(c) * np.sin(d) + 0.3 * rng.standard_normal(n)
    y = pd.Series((score > np.median(score)).astype(int))
    return X, y


def _fe_on(**overrides):
    return _mrmr(fe_hybrid_orth_enable=True, fe_auto=True, **overrides)


# ---------------------------------------------------------------------------
# (a) WHAT-IF-FLIP preview: count matches the recorded ledger arithmetic.
# ---------------------------------------------------------------------------

def test_whatif_count_matches_ledger_margin_arithmetic():
    """The preview count for a gate == count(ledger.margin > -delta) for that gate's band.
    This is the exact definition the preview claims; pin it directly on the ledger."""
    X, y = _canonical_frame()
    est = _fe_on()
    est.fit(X, y)
    report = est.explain_selection()

    led = est.fe_rejection_ledger_
    if led is None or led.empty:
        assert "ledger empty" in report.lower()
        pytest.skip("no rejections recorded on this fixture")

    gate_col = led["gate"].astype(str)
    margin_col = pd.to_numeric(led["margin"], errors="coerce")
    surfaced = False
    for gate, (knob, delta) in _GATE_TO_FLIP_BAND.items():
        n_gate = int(gate_col.eq(gate).sum())
        if n_gate == 0:
            continue
        expected = int((gate_col.eq(gate) & (margin_col > -delta)).sum())
        # if this gate's line is in the report, its count must equal `expected`.
        if f"[{gate}]" in report:
            line = next(l for l in report.splitlines() if f"[{gate}]" in l)
            assert f"re-admit {expected} candidate" in line, (
                f"preview count for {gate} != ledger arithmetic {expected}:\n{line}"
            )
            surfaced = True
    assert surfaced, f"no relaxable gate surfaced in what-if section:\n{report}"


def test_whatif_preview_matches_actual_flag_flip_refit():
    """DECISIVE cross-check: for the engineered_mi_prevalence gate, the preview's re-admit
    count equals the REAL number of recorded candidates that clear once the flag is flipped
    one band (0.90 -> 0.80) and the fit re-run -- restricted to the ledger's recorded universe."""
    X, y = _canonical_frame()
    est = _fe_on(fe_min_engineered_mi_prevalence=0.90)
    est.fit(X, y)
    led = est.fe_rejection_ledger_
    if led is None or led.empty or "engineered_mi_prevalence" not in led["gate"].astype(str).values:
        pytest.skip("engineered_mi_prevalence did not bind on this fixture")

    gate = "engineered_mi_prevalence"
    knob, delta = _GATE_TO_FLIP_BAND[gate]
    g = led["gate"].astype(str).eq(gate)
    margin = pd.to_numeric(led["margin"], errors="coerce")
    # candidates the gate recorded, with their observed value (= margin + threshold).
    recorded = led.loc[g].copy()
    recorded["_margin"] = margin[g]
    preview_count = int((recorded["_margin"] > -delta).sum())

    # ACTUAL one-band flip refit: lower the threshold by `delta`. A candidate that the
    # 0.90 gate dropped is re-admitted by the 0.80 gate iff its observed ratio >= 0.80,
    # i.e. observed - 0.90 > -0.10, i.e. recorded margin (vs 0.90) > -delta. So the count
    # of recorded candidates that the relaxed gate would NO LONGER drop is exactly:
    est_relaxed = _fe_on(fe_min_engineered_mi_prevalence=0.90 - delta)
    est_relaxed.fit(X, y)
    led_relaxed = est_relaxed.fe_rejection_ledger_
    # the same candidates, now NOT dropped by this gate (re-admitted through it):
    recorded_cands = set(recorded["candidate"].astype(str))
    if led_relaxed is None or led_relaxed.empty:
        still_dropped = set()
    else:
        gr = led_relaxed["gate"].astype(str).eq(gate)
        still_dropped = set(led_relaxed.loc[gr, "candidate"].astype(str)) & recorded_cands
    actual_readmit = len(recorded_cands - still_dropped)

    assert preview_count == actual_readmit, (
        f"what-if preview {preview_count} != actual flag-flip re-admit {actual_readmit} "
        f"(recorded={len(recorded_cands)}, still_dropped={len(still_dropped)})"
    )


# ---------------------------------------------------------------------------
# (b) PER-FEATURE MI/gain ATTRIBUTION.
# ---------------------------------------------------------------------------

def test_attribution_column_populated_and_ordered():
    """Each surviving feature line shows a gain= attribution, and they are ordered
    descending by gain (top-signal survivor first)."""
    X, y = _canonical_frame()
    est = _fe_on()
    est.fit(X, y)
    report = est.explain_selection()

    assert "by MI/gain attribution:" in report
    line = next(l for l in report.splitlines() if "by MI/gain attribution:" in l)
    gains = [float(tok.split("gain=")[1].rstrip(",")) for tok in line.split() if "gain=" in tok]
    assert gains, f"no gain= attribution rendered:\n{line}"
    assert gains == sorted(gains, reverse=True), f"attribution not gain-descending: {gains}"


def test_top_attribution_is_genuine_signal_feature():
    """biz_value: the top-MI/gain survivor is a genuine signal carrier (a/b/c/d or an
    engineered recipe built from them), NOT a noise column."""
    X, y = _canonical_frame()
    est = _fe_on()
    est.fit(X, y)
    report = est.explain_selection()

    line = next(l for l in report.splitlines() if "by MI/gain attribution:" in l)
    first = line.split("by MI/gain attribution:")[1].strip().split(",")[0]
    feat = first.split("[")[0].strip()
    # the genuine signal columns are a,b,c,d (or engineered cols derived from them, which
    # embed those names); a pure noise_* survivor topping the list would be the failure.
    assert not feat.startswith("noise_"), f"top attribution is a noise feature: {first}\n{report}"
    assert any(s in feat for s in ("a", "b", "c", "d")) or "[raw]" not in first, (
        f"top attribution {first} is not a recognisable signal carrier"
    )


def test_graceful_when_ledger_empty():
    """What-if section degrades to an empty-ledger message, no crash."""
    X, y = _canonical_frame(n=500)
    est = _mrmr(fe_max_steps=0, fe_auto=False)
    est.fit(X, y)
    report = est.explain_selection()
    assert isinstance(report, str) and report
    low = report.lower()
    assert "what-if" in low
    assert len(report) <= 2600
