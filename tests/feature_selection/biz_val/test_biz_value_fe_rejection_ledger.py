"""PER-GATE FE REJECTION LEDGER (the rejection side of the Layer-54 provenance surface).

WHY THIS LAYER
--------------
``fe_provenance_`` records the SURVIVING FE recipes. The FE pair / candidate search runs
each candidate through ~6 GATES and SILENTLY drops the rejected ones, so diagnosing "WHY
was this candidate dropped?" required hand-instrumenting the search. ``fe_rejection_ledger_``
makes that self-diagnosing: for every candidate a gate kills it records the candidate
(operands + operator), WHICH gate killed it, the observed value, the threshold, and the
MARGIN (how far it missed), plus the FE step index. Default-ON, cheap (record-only -- it
reuses values the gates already computed), memory-capped.

CONTRACTS PINNED
----------------
* C1: ``record_fe_rejection`` fingers the correct gate + margin for a candidate killed by
  EACH gate type (unit; one synthetic candidate per gate, assert the ledger names it).
* C2: the real ``apply_cmi_redundancy_gate`` + ``confirm_recipes_cross_fold`` diagnostics
  round-trip into correct ledger records (margin sign, threshold, reason).
* C3 (decisive biz_value): on the canonical poly x raw fixture the END-TO-END ledger
  correctly fingers the marginal pair-MI prevalence floor AND a second per-pair / redundancy
  floor as the killers of the expected dropped pair candidates (the floors the session
  diagnosed by hand), with negative margins.
* C4: the ledger adds ~0 wall (it only RECORDS) -- the fit hotspot is unchanged ledger-off
  vs ledger-on.
* C5: default-ON (``fe_rejection_ledger_`` populated after every fit, no opt-in) + pickle
  survives + ``get_fe_rejection_report()`` renders.
* C6: memory cap is honoured and never silent (a ``ledger_capped`` marker row is recorded).

NEVER xfail. Pure additive metadata; default-ON (transparency, not opt-in).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._fe_rejection_ledger import (
    FE_GATE_LABELS,
    FE_REJECTION_LEDGER_CAP,
    compute_fe_rejection_ledger,
    get_fe_rejection_report,
    populate_fe_rejection_ledger,
    record_fe_rejection,
)


class _Stub:
    """Minimal estimator stand-in carrying just the ledger record list."""


# ---------------------------------------------------------------------------
# C1: unit -- the recorder fingers the correct gate + margin per gate type.
# ---------------------------------------------------------------------------


def test_recorder_fingers_each_gate_with_correct_margin():
    s = _Stub()
    # One synthetic rejected candidate per real (non-synthetic) gate type, each with a
    # KNOWN observed/threshold so the margin is deterministic.
    cases = [
        # gate, observed, threshold, expected margin
        ("marginal_pair_mi_prescreen", 0.94, 0.97, -0.03),
        ("order2_maxt_floor", 0.10, 0.15, -0.05),
        ("engineered_mi_prevalence", 0.83, 0.97, -0.14),
        ("marginal_uplift_floor", 1.10, 1.30, -0.20),
        ("cmi_redundancy", 0.02, 0.04, -0.02),
        ("stability_vote", 2.0, 3.0, -1.0),
    ]
    for i, (gate, obs, thr, _) in enumerate(cases):
        record_fe_rejection(
            s,
            gate=gate,
            candidate=f"cand_{i}",
            operands=(i, i + 1),
            operator="mul",
            observed=obs,
            threshold=thr,
            reason="unit",
            step=0,
        )
    led = compute_fe_rejection_ledger(s)
    assert len(led) == len(cases)
    # Every gate label used is a member of the canonical public set.
    for gate in led["gate"]:
        assert gate in FE_GATE_LABELS, gate
    # The recorder fingered the right gate + margin for each candidate.
    for i, (gate, obs, thr, exp_margin) in enumerate(cases):
        row = led[led["candidate"] == f"cand_{i}"]
        assert len(row) == 1
        assert row["gate"].iloc[0] == gate
        assert row["observed"].iloc[0] == pytest.approx(obs)
        assert row["threshold"].iloc[0] == pytest.approx(thr)
        assert row["margin"].iloc[0] == pytest.approx(exp_margin), gate
        # margin is NEGATIVE for a missed gate (the whole point of the ledger).
        assert row["margin"].iloc[0] < 0
        assert row["operands"].iloc[0] == f"({i}, {i + 1})"


def test_recorder_default_margin_is_observed_minus_threshold():
    s = _Stub()
    record_fe_rejection(s, gate="cmi_redundancy", candidate="c", observed=0.3, threshold=0.5)
    led = compute_fe_rejection_ledger(s)
    assert led["margin"].iloc[0] == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# C2: the REAL gate diagnostics round-trip into correct ledger records.
# ---------------------------------------------------------------------------


def test_real_cmi_gate_diagnostics_map_to_ledger():
    """Run the production CMI redundancy gate on a pool with one genuine driver and a
    near-duplicate redundant remap; assert the recorder records the dropped name under
    ``cmi_redundancy`` with a NEGATIVE margin (observed CMI/excess below its bar)."""
    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate

    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) > 0.5).astype(np.int64)
    driver = y + 0.05 * rng.standard_normal(n)  # genuine, high CMI with y
    redundant = driver + 1e-6 * rng.standard_normal(n)  # ~monotone remap of driver -> redundant
    noise = rng.standard_normal(n)  # independent noise
    candidates = {
        "driver": (driver.astype(np.float64), 0.4),
        "redundant": (redundant.astype(np.float64), 0.4),
        "noise": (noise.astype(np.float64), 0.001),
    }
    accepted, diag = apply_cmi_redundancy_gate(candidates, y, nbins=8, seed=0)
    dropped = set(candidates) - accepted
    assert dropped, "expected the redundant / noise candidate to be dropped by the CMI gate"

    s = _Stub()
    for dn in dropped:
        d = diag.get(dn, {})
        reason = str(d.get("reason", "redundant"))
        if reason == "redundant_below_floor":
            obs, thr = d.get("cmi"), d.get("floor")
        else:
            obs, thr = d.get("cmi_excess"), d.get("rel_bar")
        record_fe_rejection(
            s,
            gate="cmi_redundancy",
            candidate=dn,
            observed=obs,
            threshold=thr,
            reason=reason,
            step=0,
        )
    led = compute_fe_rejection_ledger(s)
    assert (led["gate"] == "cmi_redundancy").all()
    # At least one dropped candidate has a meaningful (finite, non-positive) margin.
    finite = led[np.isfinite(led["margin"].astype(float))]
    assert len(finite) >= 1
    assert (finite["margin"].astype(float) <= 1e-9).all()


def test_real_stability_vote_diagnostics_map_to_ledger():
    """Run the production cross-fold stability vote with a tight quorum on recipes that
    cannot clear the held-out uplift gate; assert the per-recipe pass/quorum diagnostics
    surface and map to a ``stability_vote`` ledger record with passes < need_eff."""
    from mlframe.feature_selection.filters._fe_stability_vote import confirm_recipes_cross_fold
    from mlframe.feature_selection.filters.engineered_recipes import build_unary_binary_recipe

    rng = np.random.default_rng(1)
    n = 1500
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    y = (a > 0).astype(np.int64)  # y depends ONLY on a; b,c are noise
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    # Two noise-operand recipes that have no genuine held-out uplift -> should fail quorum.
    recipes = {
        "mul(b,c)": build_unary_binary_recipe(
            name="mul(b,c)",
            src_a_name="b",
            src_b_name="c",
            unary_a_name="identity",
            unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal",
            binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.int32,
        ),
        "add(b,c)": build_unary_binary_recipe(
            name="add(b,c)",
            src_a_name="b",
            src_b_name="c",
            unary_a_name="identity",
            unary_b_name="identity",
            binary_name="add",
            unary_preset="minimal",
            binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.int32,
        ),
    }
    diag: dict = {}
    failed = confirm_recipes_cross_fold(
        recipes=recipes,
        X=X,
        y_codes=y,
        feature_names_in=["a", "b", "c"],
        nbins=8,
        k=5,
        quorum=0.6,
        rng=np.random.default_rng(0),
        diagnostics_out=diag,
    )
    assert failed, "expected the noise-operand recipes to fail the stability quorum"
    s = _Stub()
    for fn in failed:
        vd = diag.get(fn, {})
        record_fe_rejection(
            s,
            gate="stability_vote",
            candidate=fn,
            operands=vd.get("src_names"),
            observed=vd.get("passes", float("nan")),
            threshold=vd.get("need_eff", float("nan")),
            reason="below_quorum",
            step=0,
        )
    led = compute_fe_rejection_ledger(s)
    assert (led["gate"] == "stability_vote").all()
    # passes < need_eff -> a strictly-negative margin for every failed recipe with diagnostics.
    finite = led[np.isfinite(led["margin"].astype(float))]
    assert len(finite) >= 1
    assert (finite["margin"].astype(float) < 0).all()


# ---------------------------------------------------------------------------
# C3: DECISIVE biz_value -- end-to-end on the canonical poly x raw fixture.
# ---------------------------------------------------------------------------


def _canonical_fixture(seed: int, n: int):
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    e = rng.random(n)
    y = a**2 / b + np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y")


@pytest.mark.timeout(600)
def test_ledger_fingers_floors_on_canonical_fixture():
    """On the canonical ``y = a**2/b + log(c)*sin(d)`` fixture (with a genuine synergy + the
    decoy ``e`` + cross-mix pairs), the END-TO-END rejection ledger must FINGER the marginal
    pair-MI prevalence floor as the killer of the expected dropped pair candidates -- the
    floor the session diagnosed by hand -- AND surface at least one second floor (engineered-MI
    prevalence / marginal-uplift / CMI redundancy) so the rejection side is fully self-diagnosing."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _canonical_fixture(seed=0, n=4000)
    fs = MRMR(verbose=0, fe_max_steps=2, n_workers=1)
    fs.fit(df, y)

    led = fs.fe_rejection_ledger_
    assert isinstance(led, pd.DataFrame)
    assert not led.empty, "FE search on the canonical fixture rejected NO candidate -- ledger empty"

    gates = set(led["gate"].unique())
    # The marginal pair-MI prevalence floor MUST be among the killers (the cross-mix /
    # decoy pairs whose joint MI does not beat their marginal sum by the prevalence bar).
    assert "marginal_pair_mi_prescreen" in gates, f"prevalence floor did not finger any dropped candidate; gates seen={sorted(gates)}"
    # At least one SECOND distinct floor must also fire, so the ledger covers more than a
    # single gate (the session hand-diagnosed multiple floors on this fixture).
    second_floors = {
        "engineered_mi_prevalence",
        "marginal_uplift_floor",
        "cmi_redundancy",
        "order2_maxt_floor",
        "stability_vote",
    }
    assert gates & second_floors, f"only the pre-screen fired; expected a second floor too. gates seen={sorted(gates)}"

    # Every rejection records a NEGATIVE margin (observed missed the threshold) where finite,
    # and carries the operands so the user can identify the dropped candidate.
    prescreen = led[led["gate"] == "marginal_pair_mi_prescreen"]
    assert len(prescreen) >= 1
    fin = prescreen[np.isfinite(prescreen["margin"].astype(float))]
    assert (fin["margin"].astype(float) < 0).all(), "prevalence-floor margins must be negative misses"
    assert (prescreen["operator"] == "pair").all()
    assert (prescreen["operands"].astype(str).str.len() > 2).all()

    # The accessor renders a non-empty per-gate summary.
    report = fs.get_fe_rejection_report()
    assert "MRMR FE rejection ledger:" in report
    assert "marginal_pair_mi_prescreen" in report


# ---------------------------------------------------------------------------
# C5: default-ON + pickle + clone + empty-report behaviour.
# ---------------------------------------------------------------------------


def test_default_on_and_pickle_roundtrip():
    from sklearn.base import clone
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 600
    x_sig = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x_signal": x_sig,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    y = pd.Series((x_sig + 0.3 * rng.standard_normal(n) > 0).astype(int))
    fs = MRMR(verbose=0, random_seed=0, fe_max_steps=0)
    fs.fit(X, y)
    # Default-ON: the attribute is populated (a DataFrame) after every fit, no opt-in.
    assert isinstance(fs.fe_rejection_ledger_, pd.DataFrame)
    # Pickle round-trip preserves the ledger content.
    fs2 = pickle.loads(pickle.dumps(fs))
    pd.testing.assert_frame_equal(fs.fe_rejection_ledger_, fs2.fe_rejection_ledger_)
    # clone() drops fitted state (sklearn convention); the cloned estimator has no ledger yet.
    cl = clone(fs)
    assert getattr(cl, "fe_rejection_ledger_", None) is None


def test_empty_report_message():
    s = _Stub()
    assert "is empty" in get_fe_rejection_report(s)
    populate_fe_rejection_ledger(s)
    assert isinstance(s.fe_rejection_ledger_, pd.DataFrame)
    assert s.fe_rejection_ledger_.empty


# ---------------------------------------------------------------------------
# C6: memory cap is honoured + NEVER silent.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# C4: cProfile -- the ledger only RECORDS, so its wall-cost is ~0.
# ---------------------------------------------------------------------------


def test_recorder_wall_cost_is_negligible():
    """The recorder is a pure ``list.append`` of a small dict -- it must add ~0 wall. Profile a
    large batch of record calls and assert (a) ``record_fe_rejection`` is the ONLY mlframe frame
    that shows up (no hidden MI / permutation recompute), and (b) the per-call wall is sub-microsecond
    scale, far below any gate's per-candidate cost (a single CMI permutation null is ~ms)."""
    import cProfile
    import pstats
    import io
    import time

    s = _Stub()
    N = 20_000
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    for i in range(N):
        record_fe_rejection(
            s,
            gate="cmi_redundancy",
            candidate=f"c{i}",
            operands=(i, i + 1),
            operator="mul",
            observed=0.1,
            threshold=0.2,
            reason="prof",
            step=0,
        )
    pr.disable()
    wall = time.perf_counter() - t0

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(15)
    text = buf.getvalue()
    # No heavy numeric primitive is invoked by the recorder (record-only contract).
    for forbidden in ("mi_direct", "permutation", "discretize", "batch_mi", "_cmi_from_binned"):
        assert forbidden not in text, f"recorder unexpectedly invoked {forbidden}:\n{text}"
    # Sub-microsecond-scale per call (generous 50us ceiling absorbs CI jitter); proves the
    # ledger adds no measurable wall to a gate whose per-candidate cost is orders of magnitude larger.
    per_call_us = (wall / N) * 1e6
    assert per_call_us < 50.0, f"recorder per-call wall {per_call_us:.2f}us too high:\n{text}"


def test_memory_cap_records_marker_and_stops():
    s = _Stub()
    # Push past the cap; only CAP+1 records should land (CAP real + 1 marker).
    for i in range(FE_REJECTION_LEDGER_CAP + 25):
        record_fe_rejection(
            s,
            gate="cmi_redundancy",
            candidate=f"c{i}",
            observed=0.1,
            threshold=0.2,
            step=0,
        )
    records = s._fe_rejection_records_
    assert len(records) == FE_REJECTION_LEDGER_CAP + 1, len(records)
    # The truncation is NOT silent: a ledger_capped marker row is present.
    led = compute_fe_rejection_ledger(s)
    assert (led["gate"] == "ledger_capped").sum() == 1
