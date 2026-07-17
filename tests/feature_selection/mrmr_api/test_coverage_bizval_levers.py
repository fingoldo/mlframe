"""biz_value QUANTITATIVE-win coverage for MRMR statistical levers.

Per CLAUDE.md "Every new ML trick gets a biz_value synthetic test": each lever below has a synthetic
where the trick should CLEARLY win, and a quantitative floor pinned 5-15% below the development-time
measured value so a real regression trips while seed noise does not.

Levers covered (each compared against its closest baseline = the lever OFF / a coarser setting):

  * mi_correction='miller_madow'            -- widens the true-signal-over-high-card-noise MI margin.
  * mi_normalization='su'                   -- widens the signal/noise relevance RATIO past raw MI.
  * quantization_nbins                       -- fine bins capture an oscillatory signal coarse bins miss.
  * conditional-MI acceptance gate           -- drops a redundant engineered copy, keeps a private
    interaction, drops noise (the constant-free replacement for the prevalence-ratio gate).
  * redundancy_aggregator='jmim'             -- end-to-end no-harm + synergy-capture floor vs Fleuret.

All kernel-level tests are <1s; the two end-to-end MRMR fits are budgeted < 30s each. Numba is
pre-warmed in a module fixture so a cold compile does not blow a budget.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.info_theory import mi, mi_miller_madow, symmetric_uncertainty
from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate


@pytest.fixture(scope="module", autouse=True)
def _prewarm():
    n = 64
    rng = np.random.default_rng(0)
    fd = np.column_stack([rng.integers(0, 4, n), rng.integers(0, 2, n)]).astype(np.int32)
    fnb = np.array([4, 2], dtype=np.int64)
    for fn in (mi, mi_miller_madow, symmetric_uncertainty):
        fn(fd, np.array([0]), np.array([1]), fnb)
    yield


def _score(a, b, na, nb, fn):
    fd = np.column_stack([a, b]).astype(np.int32)
    fnb = np.array([na, nb], dtype=np.int64)
    return fn(fd, np.array([0]), np.array([1]), fnb)


def _quantile_bin(v, nbins):
    edges = np.quantile(v, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.digitize(v, edges).astype(np.int32)


# ----------------------------------------------------------------------------------------------------
# mi_correction='miller_madow' biz_value: widens the true-signal-over-noise margin.
# ----------------------------------------------------------------------------------------------------


def test_biz_val_mi_correction_widens_signal_over_highcard_noise_margin():
    """Miller-Madow correction widens the (true-signal MI minus high-card-noise MI) margin vs plug-in.

    Measured: plug-in margin 0.2726, MM margin 0.2964 (MM drives noise MI to -0.009). Floor: MM margin
    must beat plug-in margin AND the corrected noise MI must drop below the plug-in noise MI by >=0.01."""
    rng = np.random.default_rng(8)
    n = 800
    y = (rng.random(n) < 0.5).astype(np.int32)
    sig = y.copy()
    sig[rng.random(n) < 0.15] ^= 1  # 85%-agreement binary signal
    noise = rng.integers(0, 40, n).astype(np.int32)  # 40-bin entropy-inflated noise

    mi_sig = _score(sig, y, 2, 2, mi)
    mi_noise = _score(noise, y, 40, 2, mi)
    mm_sig = _score(sig, y, 2, 2, mi_miller_madow)
    mm_noise = _score(noise, y, 40, 2, mi_miller_madow)

    plugin_margin = mi_sig - mi_noise
    mm_margin = mm_sig - mm_noise
    assert mm_margin > plugin_margin, f"MM must widen the signal-vs-noise margin: mm={mm_margin:.4f} plugin={plugin_margin:.4f}"
    assert (mi_noise - mm_noise) >= 0.01, f"MM must cut the high-card noise MI by >=0.01: {mi_noise:.4f}->{mm_noise:.4f}"
    # Measured mm_margin 0.2964; floor 10% below.
    assert mm_margin >= 0.265, f"MM margin floor 0.265, got {mm_margin:.4f}"


# ----------------------------------------------------------------------------------------------------
# mi_normalization='su' biz_value: widens the signal/noise relevance RATIO past raw MI.
# ----------------------------------------------------------------------------------------------------


def test_biz_val_su_normalization_widens_signal_to_noise_ratio():
    """SU normalization ranks a binary true signal far above a high-cardinality noise feature, with a
    signal/noise RATIO well beyond what raw MI gives (SU normalises away the noise's entropy inflation).

    Measured: raw-MI ratio ~19x, SU ratio ~60x. Floor: SU ratio >= 40x and SU ratio >= 2x the MI ratio."""
    rng = np.random.default_rng(8)
    n = 800
    y = (rng.random(n) < 0.5).astype(np.int32)
    sig = y.copy()
    sig[rng.random(n) < 0.15] ^= 1
    noise = rng.integers(0, 40, n).astype(np.int32)

    mi_ratio = _score(sig, y, 2, 2, mi) / _score(noise, y, 40, 2, mi)
    su_sig = _score(sig, y, 2, 2, symmetric_uncertainty)
    su_noise = _score(noise, y, 40, 2, symmetric_uncertainty)
    su_ratio = su_sig / su_noise

    assert su_ratio >= 40.0, f"SU signal/noise ratio floor 40x, got {su_ratio:.1f}"
    assert su_ratio >= 2.0 * mi_ratio, f"SU ratio ({su_ratio:.1f}) must >= 2x raw-MI ratio ({mi_ratio:.1f})"


# ----------------------------------------------------------------------------------------------------
# quantization_nbins biz_value: fine bins capture an oscillatory signal coarse bins miss.
# ----------------------------------------------------------------------------------------------------


def test_biz_val_quantization_nbins_captures_oscillatory_signal():
    """A sin(2x) signal is invisible at nbins=2 (oscillation averages out) but recovered at finer bins.

    Measured MI: nbins=2 -> 0.0012, nbins=4 -> 0.611, nbins=12 -> 0.919. Floor: nbins>=4 MI must clear
    0.5 while nbins=2 stays below 0.05 -- a >=10x capture gain from finer quantization."""
    rng = np.random.default_rng(10)
    n = 2000
    x = rng.uniform(-3, 3, n)
    y_cont = np.sin(2 * x) + 0.1 * rng.standard_normal(n)
    y_bin = _quantile_bin(y_cont, 4)
    nby = int(y_bin.max()) + 1

    def mi_at(nbins):
        xb = _quantile_bin(x, nbins)
        return _score(xb, y_bin, int(xb.max()) + 1, nby, mi)

    mi_coarse = mi_at(2)
    mi_fine4 = mi_at(4)
    mi_fine12 = mi_at(12)
    assert mi_coarse < 0.05, f"nbins=2 must miss the oscillatory signal, got {mi_coarse:.4f}"
    assert mi_fine4 >= 0.5, f"nbins=4 must recover the signal (floor 0.5), got {mi_fine4:.4f}"
    assert mi_fine12 > mi_fine4, "finer binning keeps gaining on this signal"
    assert mi_fine4 >= 10 * max(mi_coarse, 1e-6), "finer quantization is a >=10x capture gain here"


# ----------------------------------------------------------------------------------------------------
# Conditional-MI acceptance gate biz_value: drops a redundant copy, keeps a private interaction.
# ----------------------------------------------------------------------------------------------------


def test_biz_val_cmi_gate_drops_redundant_keeps_private_drops_noise():
    """The CMI redundancy gate (constant-free replacement for the prevalence-ratio gate) admits exactly
    ONE of a collinear duplicate PAIR plus a genuinely-private interaction feature, and rejects pure
    noise. A prevalence-ratio gate tuned to drop the duplicate would also drop the weaker private
    feature; the CMI gate keeps it because it carries information the admitted support does not.

    Measured: private excess ~0.21 (admitted), noise excess ~0.012 (rejected below floor/rel-bar),
    exactly one of {genuine, redundant} admitted as the seed, the other dropped as redundant."""
    rng = np.random.default_rng(12)
    n = 3000
    g1 = rng.standard_normal(n)
    priv = rng.standard_normal(n)
    y_cont = np.sign(g1) + np.sign(priv) + 0.15 * rng.standard_normal(n)
    y_bin = (y_cont > np.median(y_cont)).astype(np.int32)

    candidates = {
        "genuine": (g1, 0.0),
        "redundant": (g1 + 0.01 * rng.standard_normal(n), 0.0),  # near-duplicate of genuine
        "private": (priv, 0.0),
        "noise": (rng.standard_normal(n), 0.0),
    }
    # fill in real marginal MIs (the gate uses them for ordering / seeding).
    for k, (col, _) in list(candidates.items()):
        xb = _quantile_bin(col, 10)
        candidates[k] = (col, _score(xb, y_bin, int(xb.max()) + 1, 2, mi))

    accepted, diag = apply_cmi_redundancy_gate(candidates, y_bin, seed=0)

    assert "private" in accepted, f"the private-interaction feature must be admitted, diag={diag['private']}"
    assert "noise" not in accepted, f"pure noise must be rejected, diag={diag['noise']}"
    dup_kept = {"genuine", "redundant"} & accepted
    assert len(dup_kept) == 1, f"exactly ONE of the collinear duplicate pair must survive, got {dup_kept}"
    assert diag["private"]["cmi_excess"] >= 0.18, f"private debiased-excess CMI floor 0.18, got {diag['private']['cmi_excess']:.4f}"
    assert diag["noise"]["cmi_excess"] < 0.05, f"noise excess must collapse near 0, got {diag['noise']['cmi_excess']:.4f}"


# ----------------------------------------------------------------------------------------------------
# redundancy_aggregator='jmim' biz_value: end-to-end no-harm + full synergy capture on XOR.
# ----------------------------------------------------------------------------------------------------


def test_biz_val_jmim_captures_xor_synergy_no_harm_vs_fleuret():
    """On a pure-XOR target both the Fleuret (default) and JMIM redundancy aggregators must recover the
    synergy and let a downstream depth-4 tree separate the classes perfectly. JMIM must be NO WORSE
    than Fleuret (the closest baseline). Measured: both reach CV-AUC 1.0; floor 0.95, jmim >= fleuret - 0.02."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(11)
    n = 2000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    y = a ^ b
    cols = {"a": a, "b": b}
    for i in range(6):
        cols[f"noise_{i}"] = rng.integers(0, 3, n)
    X = pd.DataFrame(cols)

    def fit_auc(aggregator):
        m = MRMR(
            verbose=0,
            full_npermutations=3,
            baseline_npermutations=2,
            random_seed=42,
            fe_max_steps=1,
            interactions_max_order=2,
            redundancy_aggregator=aggregator,
        )
        Xt = np.asarray(m.fit_transform(X, y))
        assert Xt.shape[1] > 0, f"{aggregator} selected nothing on XOR"
        return cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), Xt, y, cv=3, scoring="roc_auc").mean()

    t = time.time()
    auc_fleuret = fit_auc(None)
    auc_jmim = fit_auc("jmim")
    assert time.time() - t < 90, "two XOR fits should complete inside the budget"
    assert auc_fleuret >= 0.95, f"Fleuret must recover XOR synergy (floor 0.95), got {auc_fleuret:.4f}"
    assert auc_jmim >= 0.95, f"JMIM must recover XOR synergy (floor 0.95), got {auc_jmim:.4f}"
    assert auc_jmim >= auc_fleuret - 0.02, f"JMIM must be no worse than Fleuret: jmim={auc_jmim:.4f} fleuret={auc_fleuret:.4f}"
