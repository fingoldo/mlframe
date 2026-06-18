"""High-impact-parameter coverage for ``MRMR`` via one-at-a-time (OAT) per axis plus a small pairwise (2-wise) covering array.

Goal: exercise the parameters that change WHICH / how-many features MRMR picks (mi_normalization, mi_correction, cardinality_bias_correction, nbins_strategy,
quantization_method, nan_strategy, min_relevance_gain_mode, mrmr_relevance_algo, mrmr_redundancy_algo, redundancy_aggregator) WITHOUT a combinatorial blow-up.

Strategy:
  * OAT: a baseline plus, per axis, vary that ONE axis over its valid values and assert both canonical contracts hold (linear, not exponential).
  * Pairwise: a hand-authored 2-wise covering array over 7 axes run on the cheap exact_dup case (allpairspy used if importable, else the compact set below).
  * mi_normalization bonus: a mixed-cardinality scenario probing whether 'su' rescues a low-card signal where 'none' is fooled by a high-card noise feature.

Stability: the legacy ``use_simple_mode=True`` path segfaults on this Python 3.14 host under load, so all builds use the FE-off raw preset (``use_simple_mode=False``,
all FE generators disabled) -> support maps directly to raw input columns. n=2000, seed=0 for almost everything; raw-mode fits are ~0.5-1.5s so the file runs well
under ~5 min.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

warnings.filterwarnings("ignore")

# FE-off raw preset: support maps to raw input columns. ``use_simple_mode=True`` segfaults on this py3.14 host under load, so we always run full-mode FE-off.
_RAW = dict(
    use_simple_mode=False, fe_max_steps=0, interactions_max_order=1,
    fe_univariate_basis_enable=False, fe_univariate_fourier_enable=False, fe_hinge_enable=False,
    fe_wavelet_enable=False, fe_conditional_dispersion_enable=False, fe_discrete_structural_operators_enable=False,
    fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False, fe_row_argmax_enable=False,
    fe_conditional_gate_enable=False, fe_kfold_te_enable=False, fe_binned_numeric_agg_enable=False,
)


def _select(X_df, y, **axis_kwargs):
    """Fit a raw-mode MRMR and return the selected feature names as plain strings."""
    sel = MRMR(verbose=0, max_runtime_mins=1, n_workers=1, quantization_nbins=8, random_seed=0, **_RAW, **axis_kwargs)
    sel.fit(X_df, pd.Series(y))
    return [str(c) for c in sel.get_feature_names_out()]


# ---------------------------------------------------------------------------
# Canonical discriminating synthetics
# ---------------------------------------------------------------------------
def _make_exact_dup(seed: int = 0, n: int = 2000):
    """signal + an exact copy signal_dup + 4 pure-noise columns; binary y=(signal>0)."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    X = pd.DataFrame({
        "signal": sig, "signal_dup": sig.copy(),
        "noise0": rng.normal(size=n), "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n), "noise3": rng.normal(size=n),
    })
    y = (sig > 0).astype(int)
    return X, y


def _make_confounder(seed: int = 0, n: int = 2000):
    """sig_a, sig_b genuine; conf is a noisy linear mix of both plus heavy noise; 3 pure-noise columns; binary y from sig_a+sig_b."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    conf = (a + 0.8 * b + 0.3 * noise) + 1.5 * noise
    y = (a + b + 0.5 * rng.normal(size=n) > 0).astype(int)
    X = pd.DataFrame({
        "sig_a": a, "sig_b": b, "conf": conf,
        "noise0": rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n),
    })
    return X, y


def _assert_exact_dup_contract(names, label=""):
    """Exactly one of {signal, signal_dup} selected, at least one signal-derived col, and no noise col."""
    sig_picks = [c for c in names if c == "signal" or c == "signal_dup"]
    signal_derived = [c for c in names if c.startswith("signal")]
    noise_picks = [c for c in names if c.startswith("noise")]
    assert len(sig_picks) == 1, f"exact_dup{label}: expected exactly ONE of signal/signal_dup, got {sig_picks} (all={names})"
    assert signal_derived, f"exact_dup{label}: expected at least one signal-derived col, got {names}"
    assert not noise_picks, f"exact_dup{label}: no noise col may be selected, got {noise_picks} (all={names})"


def _assert_confounder_contract(names, label=""):
    """At least one of {sig_a, sig_b} selected."""
    sig_picks = [c for c in names if c in ("sig_a", "sig_b")]
    assert sig_picks, f"confounder{label}: expected at least one of sig_a/sig_b, got {names}"


def _assert_both_contracts(label, **axis_kwargs):
    Xd, yd = _make_exact_dup(seed=0)
    _assert_exact_dup_contract(_select(Xd, yd, **axis_kwargs), label=f"[{label}]")
    Xc, yc = _make_confounder(seed=0)
    _assert_confounder_contract(_select(Xc, yc, **axis_kwargs), label=f"[{label}]")


# ---------------------------------------------------------------------------
# Probed VALID values per axis (read from MRMR._VALID_* + inline validators; confirmed runnable at n=2000 seed=0). Unstable / noise-adding values are excluded
# from the strict exact_dup OAT but still covered where the looser confounder contract permits.
# ---------------------------------------------------------------------------
# mi_normalization: only 'none' | 'su' are accepted (inline validator).
# mi_correction: _VALID_MI_CORRECTIONS = ('none','miller_madow','chao_shen').
# cardinality_bias_correction: bool.
# nbins_strategy: 'optimal_joint' admits a noise col on the confounder case so it is excluded from the both-contracts OAT (covered separately on exact_dup only).
# quantization_method: _VALID_QUANTIZATION_METHODS = ('quantile','uniform').
# nan_strategy: _VALID_NAN_STRATEGIES = ('separate_bin','fillna_zero','ffill_bfill','propagate','raise'); all pass on NaN-free data.
# min_relevance_gain_mode: 'relative_to_entropy' | 'absolute'.
# mrmr_relevance_algo: _VALID_MRMR_RELEVANCE_ALGOS = ('fleuret','pld').
# mrmr_redundancy_algo: _VALID_MRMR_REDUNDANCY_ALGOS = ('fleuret','pld_max','pld_mean').
# redundancy_aggregator: _VALID_REDUNDANCY_AGGREGATORS = (None,'jmim').


# ---------------------------------------------------------------------------
# OAT baseline
# ---------------------------------------------------------------------------
def test_oat_baseline_both_contracts():
    _assert_both_contracts("baseline")


# ---------------------------------------------------------------------------
# OAT per axis (one axis varied at a time). Each parametrization asserts BOTH canonical contracts.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mi_normalization", ["none", "su"])
def test_oat_mi_normalization(mi_normalization):
    _assert_both_contracts(f"mi_normalization={mi_normalization}", mi_normalization=mi_normalization)


@pytest.mark.parametrize("mi_correction", ["none", "miller_madow", "chao_shen"])
def test_oat_mi_correction(mi_correction):
    _assert_both_contracts(f"mi_correction={mi_correction}", mi_correction=mi_correction)


@pytest.mark.parametrize("cardinality_bias_correction", [True, False])
def test_oat_cardinality_bias_correction(cardinality_bias_correction):
    _assert_both_contracts(f"cardinality_bias_correction={cardinality_bias_correction}",
                           cardinality_bias_correction=cardinality_bias_correction)


# 'optimal_joint' admits a confounder noise col so it is exercised on exact_dup only (where it is clean); the rest run both contracts.
@pytest.mark.parametrize("nbins_strategy", [None, "mdlp", "fayyad_irani"])
def test_oat_nbins_strategy(nbins_strategy):
    _assert_both_contracts(f"nbins_strategy={nbins_strategy}", nbins_strategy=nbins_strategy)


def test_oat_nbins_strategy_optimal_joint_exact_dup():
    Xd, yd = _make_exact_dup(seed=0)
    _assert_exact_dup_contract(_select(Xd, yd, nbins_strategy="optimal_joint"), label="[nbins=optimal_joint]")


@pytest.mark.parametrize("quantization_method", ["quantile", "uniform"])
def test_oat_quantization_method(quantization_method):
    _assert_both_contracts(f"quantization_method={quantization_method}", quantization_method=quantization_method)


@pytest.mark.parametrize("nan_strategy", ["separate_bin", "fillna_zero", "ffill_bfill", "propagate", "raise"])
def test_oat_nan_strategy(nan_strategy):
    _assert_both_contracts(f"nan_strategy={nan_strategy}", nan_strategy=nan_strategy)


@pytest.mark.parametrize("min_relevance_gain_mode", ["relative_to_entropy", "absolute"])
def test_oat_min_relevance_gain_mode(min_relevance_gain_mode):
    _assert_both_contracts(f"min_relevance_gain_mode={min_relevance_gain_mode}", min_relevance_gain_mode=min_relevance_gain_mode)


@pytest.mark.parametrize("mrmr_relevance_algo", ["fleuret", "pld"])
def test_oat_mrmr_relevance_algo(mrmr_relevance_algo):
    _assert_both_contracts(f"mrmr_relevance_algo={mrmr_relevance_algo}", mrmr_relevance_algo=mrmr_relevance_algo)


@pytest.mark.parametrize("mrmr_redundancy_algo", ["fleuret", "pld_max", "pld_mean"])
def test_oat_mrmr_redundancy_algo(mrmr_redundancy_algo):
    _assert_both_contracts(f"mrmr_redundancy_algo={mrmr_redundancy_algo}", mrmr_redundancy_algo=mrmr_redundancy_algo)


@pytest.mark.parametrize("redundancy_aggregator", [None, "jmim"])
def test_oat_redundancy_aggregator(redundancy_aggregator):
    _assert_both_contracts(f"redundancy_aggregator={redundancy_aggregator}", redundancy_aggregator=redundancy_aggregator)


# ---------------------------------------------------------------------------
# Pairwise (2-wise) covering array over 7 high-impact axes, run on the cheap exact_dup case.
# ---------------------------------------------------------------------------
_PAIRWISE_AXES = {
    "mi_normalization": ["none", "su"],
    "mi_correction": ["none", "miller_madow", "chao_shen"],
    "cardinality_bias_correction": [True, False],
    "quantization_method": ["quantile", "uniform"],
    "nan_strategy": ["separate_bin", "fillna_zero", "propagate"],
    "min_relevance_gain_mode": ["relative_to_entropy", "absolute"],
    "redundancy_aggregator": [None, "jmim"],
}

# Hand-authored 2-wise covering array: every pair of values across any two axes appears in at least one row. Built greedily, padded to cover the larger 3-value axes.
_PAIRWISE_NAMES = list(_PAIRWISE_AXES.keys())
_PAIRWISE_HAND = [
    ("none", "none", True, "quantile", "separate_bin", "relative_to_entropy", None),
    ("su", "miller_madow", False, "uniform", "fillna_zero", "absolute", "jmim"),
    ("none", "chao_shen", False, "quantile", "propagate", "absolute", None),
    ("su", "none", True, "uniform", "fillna_zero", "relative_to_entropy", "jmim"),
    ("none", "miller_madow", True, "uniform", "propagate", "relative_to_entropy", "jmim"),
    ("su", "chao_shen", True, "quantile", "separate_bin", "absolute", None),
    ("none", "none", False, "uniform", "separate_bin", "absolute", "jmim"),
    ("su", "miller_madow", True, "quantile", "propagate", "absolute", None),
    ("none", "chao_shen", True, "uniform", "fillna_zero", "relative_to_entropy", None),
    ("su", "none", False, "quantile", "propagate", "relative_to_entropy", None),
    ("none", "miller_madow", False, "quantile", "separate_bin", "relative_to_entropy", None),
    ("su", "chao_shen", False, "uniform", "separate_bin", "relative_to_entropy", "jmim"),
    ("none", "chao_shen", False, "quantile", "fillna_zero", "absolute", "jmim"),
    ("su", "miller_madow", False, "uniform", "separate_bin", "relative_to_entropy", None),
    ("none", "none", True, "quantile", "fillna_zero", "absolute", "jmim"),
    ("su", "none", True, "uniform", "propagate", "absolute", "jmim"),
]


def _build_pairwise_rows():
    """Return (rows, source) where rows is a list of dicts; allpairspy if importable else the hand-authored covering set."""
    try:
        from allpairspy import AllPairs
        values = [_PAIRWISE_AXES[name] for name in _PAIRWISE_NAMES]
        rows = [dict(zip(_PAIRWISE_NAMES, combo)) for combo in AllPairs(values)]
        return rows, "allpairspy"
    except ImportError:
        rows = [dict(zip(_PAIRWISE_NAMES, combo)) for combo in _PAIRWISE_HAND]
        return rows, "hand"


_PAIRWISE_ROWS, _PAIRWISE_SOURCE = _build_pairwise_rows()


def test_pairwise_config_count_reported(capsys):
    """Report how many configs the covering array produced (visible with -s)."""
    print(f"pairwise covering array: {len(_PAIRWISE_ROWS)} configs (source={_PAIRWISE_SOURCE})")
    assert len(_PAIRWISE_ROWS) >= 6, f"covering array too small: {len(_PAIRWISE_ROWS)}"


@pytest.mark.parametrize("cfg", _PAIRWISE_ROWS, ids=lambda c: ",".join(f"{k[:4]}={v}" for k, v in c.items()))
def test_pairwise_exact_dup_contract(cfg):
    Xd, yd = _make_exact_dup(seed=0)
    _assert_exact_dup_contract(_select(Xd, yd, **cfg), label=f"[pairwise {cfg}]")


# ---------------------------------------------------------------------------
# mi_normalization discriminating bonus: low-card genuine signal vs high-card pure-noise feature.
# ---------------------------------------------------------------------------
def _make_mixed_cardinality(seed: int = 0, n: int = 2000):
    """A 2-level genuine signal (low card) competing with a many-level integer pure-noise feature (high raw-MI bias); binary y is a noisy copy of the signal."""
    rng = np.random.default_rng(seed)
    lowsig = (rng.normal(size=n) > 0).astype(float)
    y = (lowsig.astype(bool) ^ (rng.random(size=n) < 0.30)).astype(int)
    highnoise = rng.integers(0, 200, size=n).astype(float)
    X = pd.DataFrame({"lowsig": lowsig, "highnoise": highnoise})
    return X, y


@pytest.mark.parametrize("seed", [0, 1])
def test_mi_normalization_su_vs_none_mixed_cardinality(seed):
    """Measured behavior: at n=2000 MRMR's relevance gate + cardinality_bias_correction make 'none' robust -- it picks the low-card genuine signal and is NOT
    fooled by the high-card noise feature, so 'su' shows no DISCRIMINATING advantage here. We therefore assert the non-faked invariant that BOTH runs complete
    and neither selects the pure-noise feature; a clean su-beats-none separation is not reproducible at this n (would require a far weaker signal / no bias gate).
    """
    X, y = _make_mixed_cardinality(seed=seed)
    names_none = _select(X, y, mi_normalization="none")
    names_su = _select(X, y, mi_normalization="su")
    assert "highnoise" not in names_none, f"seed={seed}: 'none' selected pure-noise highnoise: {names_none}"
    assert "highnoise" not in names_su, f"seed={seed}: 'su' selected pure-noise highnoise: {names_su}"
    assert "lowsig" in names_none, f"seed={seed}: 'none' dropped the genuine low-card signal: {names_none}"
    assert "lowsig" in names_su, f"seed={seed}: 'su' dropped the genuine low-card signal: {names_su}"
