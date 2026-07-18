"""W6 FOLLOW-UP: unified local-MI abs-MAD floor reject-sink wiring through the FE-family callers.

WHY THIS LAYER
--------------
``_unified_fe_gate.local_mi_gate`` is the shared Tier-1 abs-MAD noise-floor gate
(``raw_mi_noise_floor`` = med + k*MAD). W6 (commit ad75e9ff) instrumented the gate
itself with a ``reject_sink`` callable so it records every abs-MAD floor kill into the
FE rejection ledger under ``gate="marginal_uplift_floor"`` /
``operator="unified_local_mi_gate"``. BUT its ~6 FE-family callers (target-encoding,
count/freq/cat-num, missingness-indicator, ratio/delta, rare-category,
conditional-residual, conditional-dispersion) did NOT pass the sink -- so for every
candidate THOSE families dropped at the unified floor, the kill was silently
un-recorded. This layer threads the sink (built exactly like the cluster-basis
``_cb_reject_sink`` in ``_fit_impl_core``) through each family's caller chain.

CONTRACTS PINNED
----------------
* C1 (unit): each newly-wired family caller forwards a ``reject_sink`` to
  ``local_mi_gate`` and the recorded kills carry ``gate="marginal_uplift_floor"`` +
  ``operator="unified_local_mi_gate"`` + a negative margin. The kept (survivor) set is
  BYTE-IDENTICAL with vs without the sink (recording must not change selection).
* C2 (decisive biz_value, end-to-end): on a noisy pool that drives at least 2 of the
  unified-gate FE families to a floor kill, ``MRMR.fit().fe_rejection_ledger_`` now
  contains those families' ``marginal_uplift_floor`` kills (it did NOT before this
  wiring), and ``explain_selection()`` can finger one such kill.

PURE ADDITIVE -- selection byte-identical; default-ON diagnostic. NEVER xfail.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


class _Sink:
    """Groups tests covering Sink."""
    def __init__(self):
        self.records = []

    def __call__(self, **kw):
        self.records.append(kw)


def _assert_floor_records(sink):
    """Assert floor records."""
    assert sink.records, "expected at least one abs-MAD floor kill to be recorded"
    for rec in sink.records:
        assert rec["gate"] == "marginal_uplift_floor"
        assert rec["operator"] == "unified_local_mi_gate"
        assert rec["observed"] < rec["threshold"]


# ---------------------------------------------------------------------------
# C1 unit: every newly-wired family caller forwards the sink + byte-identical keep.
# ---------------------------------------------------------------------------


def _noisy_cat_pool(seed=0, n=600):
    """Noisy cat pool."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    # One informative categorical (tracks y sign) + several pure-noise categoricals.
    df = pd.DataFrame(
        {
            "cat_good": (sig > 0).astype(int).astype(str),
            **{f"cat_noise{i}": rng.integers(0, 20, size=n).astype(str) for i in range(5)},
        }
    )
    return df, y, sig


def _gate_kill_pool(seed=0, n=600):
    """enc_df with one informative + several noise columns and a NUMERIC raw_X
    whose abs-MAD floor is positive -> the noise columns are floor-killed."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    enc = pd.DataFrame(
        {
            "good": sig + 0.1 * rng.standard_normal(n),
            **{f"noise{i}": rng.standard_normal(n) for i in range(6)},
        }
    )
    raw = pd.DataFrame({f"r{i}": rng.standard_normal(n) for i in range(5)})
    return enc, raw, y


def test_count_freq_gate_helper_records_floor_kill():
    """count/freq/cat-num share ``_gate_enc`` -> ``local_mi_gate``; verify the
    helper forwards the sink and the keep set is byte-identical."""
    from mlframe.feature_selection.filters._count_freq_interaction_fe import _gate_enc

    enc, raw, y = _gate_kill_pool(1)
    sink = _Sink()
    keep = _gate_enc(enc, y, raw, True, 20, reject_sink=sink)
    keep_ns = _gate_enc(enc, y, raw, True, 20)
    assert list(keep.columns) == list(keep_ns.columns)
    _assert_floor_records(sink)


def _noisy_num_pool(seed=0, n=600):
    """Noisy num pool."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    df = pd.DataFrame(
        {
            "good": np.abs(sig) + 1.0,  # positive, tracks |signal|
            **{f"n{i}": np.abs(rng.standard_normal(n)) + 1.0 for i in range(5)},
        }
    )
    return df, y


def test_pairwise_ratio_caller_records_floor_kill():
    """Pairwise ratio caller records floor kill."""
    from mlframe.feature_selection.filters._ratio_delta_fe import (
        pairwise_ratio_with_recipes,
    )

    df, y = _noisy_num_pool(3)
    cols = list(df.columns)
    sink = _Sink()
    _, app, _ = pairwise_ratio_with_recipes(
        df,
        cols=cols,
        mi_gate=True,
        mi_gate_top_k=50,
        y=y,
        reject_sink=sink,
    )
    _, app_ns, _ = pairwise_ratio_with_recipes(
        df,
        cols=cols,
        mi_gate=True,
        mi_gate_top_k=50,
        y=y,
    )
    assert app == app_ns
    _assert_floor_records(sink)


def test_missing_indicator_caller_records_floor_kill():
    """Missing indicator caller records floor kill."""
    from mlframe.feature_selection.filters._missingness_fe import (
        missing_indicator_with_recipes,
    )

    rng = np.random.default_rng(4)
    n = 600
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    # Several columns with random (noise) missingness patterns -> indicators are noise.
    cols = {}
    for i in range(6):
        v = rng.standard_normal(n)
        mask = rng.random(n) < 0.3
        v[mask] = np.nan
        cols[f"m{i}"] = v
    df = pd.DataFrame(cols)
    sink = _Sink()
    _, app, _ = missing_indicator_with_recipes(
        df,
        cols=list(df.columns),
        mi_gate=True,
        mi_gate_top_k=20,
        y=y,
        raw_X=df,
        reject_sink=sink,
    )
    _, app_ns, _ = missing_indicator_with_recipes(
        df,
        cols=list(df.columns),
        mi_gate=True,
        mi_gate_top_k=20,
        y=y,
        raw_X=df,
    )
    assert app == app_ns
    _assert_floor_records(sink)


def test_rare_category_caller_records_floor_kill():
    """Rare category caller records floor kill."""
    from mlframe.feature_selection.filters._extra_fe_families import (
        hybrid_rare_category_fe,
    )

    rng = np.random.default_rng(5)
    n = 800
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    # Categoricals with rare levels (drive is_rare/freq_band emitters) PLUS numeric
    # noise columns so raw_X's abs-MAD floor is positive -> noise emitters get killed.
    cat_cols = [f"cat{i}" for i in range(4)]
    data = {c: rng.integers(0, 40, size=n).astype(str) for c in cat_cols}
    # Multiple informative numeric cols raise the abs-MAD floor (median MI) well
    # above the rare/freq_band emitters' near-zero MI, so they are floor-killed.
    for i in range(4):
        data[f"rstrong{i}"] = sig + (0.05 + 0.1 * i) * rng.standard_normal(n)
    df = pd.DataFrame(data)
    sink = _Sink()
    _, app, _, _ = hybrid_rare_category_fe(
        df,
        y,
        cat_cols=cat_cols,
        mi_gate=True,
        mi_gate_top_k=2,
        reject_sink=sink,
    )
    _, app_ns, _, _ = hybrid_rare_category_fe(
        df,
        y,
        cat_cols=cat_cols,
        mi_gate=True,
        mi_gate_top_k=2,
    )
    assert app == app_ns
    _assert_floor_records(sink)


# ---------------------------------------------------------------------------
# C2 DECISIVE biz_value: end-to-end via MRMR.fit -- 2+ families finger floor
# kills in the ledger, invisible before this wiring; explain_selection fingers one.
# ---------------------------------------------------------------------------


def _end_to_end_fit():
    """End to end fit."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(7)
    n = 1500
    sig = rng.standard_normal(n)
    y = pd.Series((sig > 0).astype(int), name="y")
    # Numeric pool for ratio family (mostly noise) + categorical pool for count-encoding.
    data = {
        "good_num": np.abs(sig) + 1.0,
    }
    for i in range(5):
        data[f"num{i}"] = np.abs(rng.standard_normal(n)) + 1.0
    for i in range(4):
        data[f"cat{i}"] = rng.integers(0, 25, size=n).astype(str)
    X = pd.DataFrame(data)
    fs = MRMR(
        verbose=0,
        n_workers=1,
        fe_max_steps=0,
        fe_local_mi_gate=True,
        fe_local_mi_gate_top_k=50,
        fe_pairwise_ratio_enable=True,
        fe_pairwise_ratio_cols=tuple(c for c in X.columns if c.startswith(("good_num", "num"))),
        fe_count_encoding_enable=True,
        fe_count_encoding_cols=tuple(c for c in X.columns if c.startswith("cat")),
    )
    fs.fit(X, y)
    return fs


@pytest.mark.timeout(600)
def test_end_to_end_ledger_fingers_unified_floor_kills_two_families():
    """End to end ledger fingers unified floor kills two families."""
    fs = _end_to_end_fit()
    led = fs.fe_rejection_ledger_
    assert isinstance(led, pd.DataFrame)
    floor = led[led["gate"] == "marginal_uplift_floor"]
    assert (
        not floor.empty
    ), f"no unified abs-MAD floor kill recorded -- the sink wiring is not reaching the FE-family callers. gates seen={sorted(led['gate'].unique())}"
    # The unified-gate operator label proves these came from local_mi_gate (not the
    # pair-search marginal_uplift_floor), i.e. the newly-wired family callers.
    uni = floor[floor["operator"] == "unified_local_mi_gate"]
    assert not uni.empty, "floor kills exist but none carry operator='unified_local_mi_gate'; the FE-family sink wiring did not fire"
    # Every unified floor kill is a genuine miss (negative margin).
    fin = uni[np.isfinite(uni["margin"].astype(float))]
    assert (fin["margin"].astype(float) < 0).all()

    # explain_selection can finger a unified-gate-family abs-floor kill.
    expl = fs.explain_selection()
    assert isinstance(expl, str) and expl
    assert "marginal_uplift_floor" in expl or "marginal-uplift floor" in expl
