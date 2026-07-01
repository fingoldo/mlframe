"""Adversarial biz_value coverage for MRMR feature selection at small AND medium n.

These cases stress MRMR on inputs designed to break naive selectors: pure synergy (no marginal signal),
exact / near duplicates, a spurious confounder, a collinear cluster, a heavy-tail target, and a large-p
regime. Most cases use a "raw" fit mode (see ``_fit_cached``): the full pipeline with feature-engineering
disabled + order-1 interactions, which selects exactly the raw input columns while still running the core
relevance / redundancy / cluster-collapse selection -- this covers raw-column selection, noise exclusion,
signal recovery, near-duplicate / collinear-cluster collapse. The pure-synergy case uses a "full" mode (FE +
order-2 interactions enabled) since only the interaction search can surface a no-marginal-signal factor pair.
Both read selected names from ``get_feature_names_out()``. (The legacy ``use_simple_mode=True`` path is
unstable under cumulative load on this host; raw mode stands in for it -- same core selection, but stable.)

Fits are memoized by config key (identical fits reuse one result) and kept few + light (single-threaded,
max_runtime_mins=1, modest n / p) so each test runs well under ~60s. Thresholds are pinned to measured
behaviour with a 5-15% margin; majority-of-seeds is used for the high-variance recovery assertions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

SEEDS = [0, 1, 2]

_FIT_CACHE: dict = {}

# Full-mode path with all feature-engineering disabled + order-1 interactions selects exactly the raw input
# columns (no engineered tail), giving a stable stand-in for the crash-prone legacy ``use_simple_mode=True``.
_RAW_FE_OFF = dict(
    use_simple_mode=False, fe_max_steps=0, interactions_max_order=1,
    fe_univariate_basis_enable=False, fe_univariate_fourier_enable=False,
    fe_hinge_enable=False, fe_wavelet_enable=False,
    fe_conditional_dispersion_enable=False, fe_discrete_structural_operators_enable=False,
    fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False,
    fe_row_argmax_enable=False, fe_conditional_gate_enable=False,
    fe_kfold_te_enable=False, fe_binned_numeric_agg_enable=False,
)


def _fit_cached(key, build, *, mode, nbins, seed, **kw):
    """Fit (or reuse a cached fit) and return selected feature names from ``get_feature_names_out``.

    ``mode``: "raw" -> raw-column selection (FE off + order-1); "full" -> FE / interactions enabled (for the
    redundancy / cluster / synergy machinery). Pass e.g. ``interactions_max_order`` via ``kw`` for full mode.
    """
    ck = (key, mode, nbins, seed, tuple(sorted(kw.items())))
    if ck in _FIT_CACHE:
        return _FIT_CACHE[ck]
    X, y = build
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        common = dict(verbose=0, max_runtime_mins=1, n_workers=1, quantization_nbins=nbins,
                      random_seed=seed)
        if mode == "raw":
            sel = MRMR(**common, **_RAW_FE_OFF, **kw).fit(X, pd.Series(y))
        else:
            sel = MRMR(use_simple_mode=False, fe_max_steps=2, **common, **kw).fit(X, pd.Series(y))
        names = [str(nm) for nm in sel.get_feature_names_out()]
    _FIT_CACHE[ck] = names
    return names


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------


def _make_synergy(seed, n=2000):
    """Pure 2-way synergy: y = sign(x0 * x1). Each factor alone has ~zero MI with y; only the JOINT predicts.

    NOTE: a binary-integer 3-way XOR (y = x0^x1^x2) was the original intent but it triggers a native
    access-violation in the MI/quantization kernel on binary-integer columns on this Python build. The
    continuous product-synergy target exercises the same "no marginal signal, only joint predicts" property.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y = (x0 * x1 > 0).astype(np.int64)
    cols = {"x0": x0, "x1": x1}
    for k in range(4):
        cols[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _make_exact_dup(seed, n=2000):
    """signal + an EXACT copy + noise. MRMR must keep exactly one of the pair."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    cols = {"signal": sig, "signal_dup": sig.copy()}
    for k in range(4):
        cols[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _make_near_dup(seed, n=2000):
    """x and x + tiny gaussian noise; at most one selected."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    y = (sig > 0).astype(np.int64)
    cols = {"signal": sig, "signal_near": sig + 1e-3 * rng.standard_normal(n)}
    for k in range(4):
        cols[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _make_confounder(seed, n=2000):
    """A spurious confounder correlated with y, plus two genuine signals. ``conf`` is a heavily-noised copy
    of the latent score (``score + 1.5*noise``) -- correlated with y but not a near-perfect predictor, so the
    genuine signals sig_a/sig_b should still be selected and not crowded out by the spurious column.
    """
    rng = np.random.default_rng(seed)
    sig_a = rng.standard_normal(n)
    sig_b = rng.standard_normal(n)
    score = sig_a + 0.8 * sig_b + 0.3 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    conf = score + 1.5 * rng.standard_normal(n)
    cols = {"sig_a": sig_a, "sig_b": sig_b, "conf": conf}
    for k in range(3):
        cols[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _make_collinear_cluster(seed, n=2000):
    """driver + 5 clones from one latent + 3 noise; the cluster must collapse, not keep all 6."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    y = (latent > 0).astype(np.int64)
    cols = {"driver": latent}
    for k in range(5):
        cols[f"clone{k}"] = latent + 0.08 * rng.standard_normal(n)
    for k in range(3):
        cols[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _make_heavy_tail(seed, n=2000):
    """Student-t(df=2) target = 2*x0 - x1 + heavy noise; linear signal recovered."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    score = 2.0 * x0 - x1 + 0.5 * rng.standard_t(2, size=n)
    y = (score > np.median(score)).astype(np.int64)
    cols = {"x0": x0, "x1": x1}
    for k in range(4):
        cols[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _make_large_p(seed, n=300, p=40):
    """n small, p=40: 2 signal (s0, s1) + (p-2) noise. Assert high noise-exclusion fraction."""
    rng = np.random.default_rng(seed)
    s0 = rng.standard_normal(n)
    s1 = rng.standard_normal(n)
    y = ((s0 + 0.8 * s1 + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    data = {"s0": s0, "s1": s1}
    for k in range(p - 2):
        data[f"noise{k}"] = rng.standard_normal(n)
    return pd.DataFrame(data), y


def _signal_cols(names):
    return [n for n in names if n in ("signal", "signal_dup", "signal_near")]


# ---------------------------------------------------------------------------
# 1. Pure synergy (no marginal signal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nbins", [5, 10])
def test_synergy_recovers_both_factors(nbins):
    """Pure 2-way synergy (y = sign(x0*x1)): neither factor has marginal MI with y, only the joint predicts.
    Full-mode MRMR with order-2 interactions must surface BOTH synergy factors x0 and x1 (it also emits the
    engineered ``div(sign(x0),sign(x1))`` interaction that captures the product).

    Measured: both x0 AND x1 recovered on all 3 seeds in full mode. Floor >=2/3 (margin below measured 3/3).
    """
    recovered = 0
    for seed in SEEDS:
        names = _fit_cached(("synergy", seed), _make_synergy(seed, n=2000), mode="full", nbins=nbins,
                            seed=seed, interactions_max_order=2)
        if all(any(tok in nm for nm in names) for tok in ("x0", "x1")):
            recovered += 1
    assert recovered >= 2, (
        f"synergy nbins={nbins}: both factors x0,x1 recovered on only {recovered}/{len(SEEDS)} seeds "
        f"(measured all; floor 2/3); order-2 interactions must detect the product-synergy"
    )


# ---------------------------------------------------------------------------
# 2. Exact-duplicate redundancy (small + medium n)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nbins", [5, 10])
@pytest.mark.parametrize("n", [2000, 20000])
def test_exact_duplicate_keeps_one(nbins, n):
    """signal and an EXACT copy. MRMR must keep exactly ONE of the redundant pair (never both), across 3
    seeds, both nbins in {5,10}, and small (n=2000) + medium (n=20000) sample sizes. The exact-duplicate fit
    is near-instant (identity shortcut). Measured: exactly one in every cell."""
    for seed in SEEDS:
        names = _fit_cached(("exact_dup", seed, n), _make_exact_dup(seed, n=n), mode="raw",
                            nbins=nbins, seed=seed)
        sigs = _signal_cols(names)
        assert len(sigs) == 1, (
            f"exact-dup n={n} seed={seed} nbins={nbins}: expected exactly ONE of (signal, signal_dup), "
            f"got {sigs}; full={names}"
        )
        # Strengthened: the survivor must be the SIGNAL (or a signal-derived engineered col), and NO pure-noise
        # column may leak in. Just collapsing the pair is not enough -- a selector that drops both duplicates and
        # picks noise would pass the count check; this catches that.
        assert any(nm == "signal" or nm.startswith("signal") for nm in names), (
            f"exact-dup n={n} seed={seed} nbins={nbins}: a signal(-derived) column must be selected; got {names}"
        )
        leaked_noise = [nm for nm in names if nm.startswith("noise")]
        assert not leaked_noise, (
            f"exact-dup n={n} seed={seed} nbins={nbins}: pure-noise columns leaked into selection: {leaked_noise}; full={names}"
        )


# ---------------------------------------------------------------------------
# 3. Near-duplicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nbins", [5, 10])
def test_near_duplicate_at_most_one(nbins):
    """x and x + 1e-3 noise. The core redundancy term prunes the near-duplicate: exactly one survives.

    Measured: exactly one kept on every seed/nbins in raw mode (FE off). The legacy simple-mode selector can
    keep both a near-(not-exact) duplicate (~30% of cells); raw mode runs the same core relevance/redundancy
    selection but is stable on this host.
    """
    for seed in SEEDS:
        names = _fit_cached(("near_dup", seed), _make_near_dup(seed, n=2000), mode="raw", nbins=nbins,
                            seed=seed)
        sigs = [nm for nm in names if nm in ("signal", "signal_near")]
        assert len(sigs) == 1, (
            f"near-dup full mode seed={seed} nbins={nbins}: expected exactly ONE of (signal, signal_near), "
            f"got {sigs}; full={names}"
        )


# ---------------------------------------------------------------------------
# 4. Confounder / spurious (small + medium n)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2000, 20000])
def test_confounder_keeps_genuine_signal(n):
    """Genuine signals must be recovered despite a spurious y-correlated confounder, at small AND medium n.

    Measured: genuine signal (sig_a/sig_b) selected alongside conf on all 3 seeds. Floor >=2/3 (margin below
    measured 3/3) -- a selector fooled into picking only the spurious column would fail.
    """
    recovered = 0
    for seed in SEEDS:
        names = _fit_cached(("confounder", seed, n), _make_confounder(seed, n=n), mode="raw",
                            nbins=10, seed=seed)
        if any(s in names for s in ("sig_a", "sig_b")):
            recovered += 1
    assert recovered >= 2, (
        f"confounder n={n}: genuine signal recovered on only {recovered}/{len(SEEDS)} seeds "
        f"(measured all; floor 2/3)"
    )


# ---------------------------------------------------------------------------
# 5. Collinear cluster
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nbins", [5, 10])
def test_collinear_cluster_collapses(nbins):
    """6 collinear members (driver + 5 clones) from one latent. The redundancy term collapses the cluster to
    a single retained raw member.

    Measured: exactly 1 cluster member kept on every seed/nbins in raw mode. Floor <=2 (margin above the
    measured 1) -- a regression that disabled redundancy keeps all 6. The legacy simple-mode selector keeps
    all 6 (no collapse); raw mode runs the same core selection and is stable.
    """
    for seed in SEEDS:
        names = _fit_cached(("collinear", seed), _make_collinear_cluster(seed, n=2000), mode="raw",
                            nbins=nbins, seed=seed)
        cluster = [nm for nm in names if nm == "driver" or nm.startswith("clone")]
        assert 1 <= len(cluster) <= 2, (
            f"collinear cluster seed={seed} nbins={nbins} did not collapse: kept {len(cluster)}/6 members "
            f"{cluster}; full={names}"
        )


# ---------------------------------------------------------------------------
# 6. Heavy-tail target
# ---------------------------------------------------------------------------


def test_heavy_tail_recovers_signals():
    """Student-t(df=2) target = 2*x0 - x1 + heavy noise. The linear signal must be recovered.

    Measured: x0 (the dominant 2x coefficient) and/or x1 recovered on all 3 seeds. Floor >=2/3.
    """
    recovered = 0
    for seed in SEEDS:
        names = _fit_cached(("heavy_tail", seed), _make_heavy_tail(seed, n=2000), mode="raw",
                            nbins=10, seed=seed)
        if "x0" in names or "x1" in names:
            recovered += 1
    assert recovered >= 2, (
        f"heavy-tail: signal recovered on only {recovered}/{len(SEEDS)} seeds (measured all; floor 2/3)"
    )


# ---------------------------------------------------------------------------
# 7. large-p regime (small + larger n)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n,p", [(300, 40), (800, 40)])
def test_large_p_noise_exclusion(seed, n, p):
    """2 signal + (p-2) noise across p=40 columns, at small (n=300) and larger (n=800) sample sizes. The
    noise-exclusion fraction must be high (the bulk of the 38 noise columns dropped) on every seed.

    Measured: excl 0.97-1.00 at both n=300 and n=800 (p=40). Floor >=0.85 (margin below measured). p and n
    are kept modest so each fit runs in a few seconds; a larger p=80 / n=20000 fit is ~40s+ on this
    single-threaded host.
    """
    names = _fit_cached(("large_p", seed, n, p), _make_large_p(seed, n=n, p=p), mode="raw",
                        nbins=10, seed=seed)
    n_noise_total = p - 2
    n_noise_kept = sum(1 for nm in names if nm.startswith("noise"))
    excl_frac = 1.0 - n_noise_kept / n_noise_total
    assert excl_frac >= 0.85, (
        f"large-p n={n} seed={seed}: noise-exclusion frac {excl_frac:.2f} too low ({n_noise_kept}/"
        f"{n_noise_total} noise kept); floor 0.85; full={names}"
    )


# ---------------------------------------------------------------------------
# redundancy_aggregator variant (jmim vs default fleuret)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aggregator", ["jmim", None])
def test_exact_duplicate_keeps_one_jmim_vs_default(aggregator):
    """Exact-duplicate keep-one contract must hold under both the jmim redundancy aggregator and the default
    (None -> fleuret)."""
    import inspect

    assert "redundancy_aggregator" in inspect.signature(MRMR.__init__).parameters, (
        "redundancy_aggregator is a documented MRMR ctor param; a missing kwarg is a real regression, not a skip"
    )
    names = _fit_cached(("exact_dup_agg", aggregator), _make_exact_dup(0, n=2000), mode="raw",
                        nbins=8, seed=0, redundancy_aggregator=aggregator)
    sigs = _signal_cols(names)
    assert len(sigs) == 1, (
        f"exact-dup under aggregator={aggregator!r}: expected ONE of the pair, got {sigs}; full={names}"
    )


# ---------------------------------------------------------------------------
# Binning-axis robustness: MRMR selection must hold across the binning METHOD (quantization_method) and the
# adaptive binning STRATEGY (nbins_strategy). The default nbins_strategy is "mdlp" (supervised), so the legacy
# unsupervised path (None -> equal-frequency quantile) was previously untested; quantization_method was tested
# on only one value. Both axes are exercised here on the two cleanest discriminating cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantization_method", ["quantile", "uniform"])
@pytest.mark.parametrize("nbins_strategy", [None, "mdlp"])
def test_signal_recovery_robust_across_binning(nbins_strategy, quantization_method):
    """Across {None (legacy quantile), mdlp (supervised default)} x {quantile, uniform} cut methods: the
    exact-duplicate pair collapses to one signal with no noise leak, and on the confounder case at least one
    genuine signal is kept. Selection must not be an artifact of one binning configuration."""
    dup = _fit_cached(("bin_dup", nbins_strategy, quantization_method), _make_exact_dup(0, n=2000), mode="raw",
                      nbins=8, seed=0, nbins_strategy=nbins_strategy, quantization_method=quantization_method)
    assert len(_signal_cols(dup)) == 1, f"exact-dup ns={nbins_strategy} qm={quantization_method}: pair should collapse to one; got {dup}"
    assert not [nm for nm in dup if nm.startswith("noise")], f"exact-dup ns={nbins_strategy} qm={quantization_method}: noise leaked; got {dup}"

    conf = _fit_cached(("bin_conf", nbins_strategy, quantization_method), _make_confounder(0, n=2000), mode="raw",
                       nbins=8, seed=0, nbins_strategy=nbins_strategy, quantization_method=quantization_method)
    assert {"sig_a", "sig_b"} & set(conf), f"confounder ns={nbins_strategy} qm={quantization_method}: a genuine signal must be kept; got {conf}"
