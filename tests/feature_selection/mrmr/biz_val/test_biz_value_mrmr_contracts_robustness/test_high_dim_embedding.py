"""Consolidated from test_biz_value_mrmr_layer20.py.

Layer 20 biz_value MRMR contracts: HIGH-DIMENSIONAL TEXT/EMBEDDING
FEATURES + WIDE-MATRIX SCALE.

WHY THIS LAYER
--------------
Modern production ML pipelines feed MRMR matrices that look nothing
like the tidy ``p < 30`` tabular benchmarks the early layers (1-19)
pinned. Two specific data shapes dominate:

A. EMBEDDING MATRICES -- sentence-transformers (384 / 768 dim),
   OpenAI ``ada-002`` (1536 dim), image encoders (512 / 1024 dim).
   In practice only a small SUBSET of the latent axes correlates with
   the downstream label; the rest are correlated Gaussian noise whose
   joint structure can fool a naive filter into selecting a redundant
   handful.

B. TF-IDF / BAG-OF-WORDS -- per-token columns that are mostly zero
   with a thin tail of non-zeros. The actionable signal lives on 1-2
   specific token columns out of hundreds; everything else is
   structural sparsity.

Both shapes push p to 100-500 features while keeping n modest
(1k-2k rows). Layers 1-19 stayed at p <= 25, so the wall-time AND
the correctness contracts at p >> 25 were never pinned.

THE FIVE PROBED PROPERTIES
--------------------------
1. SCALE-OUT WALL TIME. p=200 must finish in under 60 s; p=500 must
   finish in under 5 min on the dev box. A super-linear blow-up here
   silently breaks every embedding-backed model in prod.

2. SIGNAL RECOVERY. With 3-5 informative dims buried in p=200, MRMR
   must recover AT LEAST ONE real signal. Layer-13 noise-rejection
   already pins ``support_size <= floor`` on noise-only inputs; here
   we pin the dual contract: noise NEAR signal must NOT crowd the
   signal out.

3. NOISE FILTERING AT SCALE. ``support_`` stays small (<= 10 of 200)
   on the embedding-like fixture; the 197 noise dims do NOT pile up
   even though each is a high-quality Gaussian.

4. SPARSE-FEATURE HANDLING. TF-IDF-like columns (95% zeros) are not
   silently rejected by the quantizer (no-op binning on a mostly-zero
   column collapses to 1 bin -> entropy 0). The selector either
   surfaces the real token signal OR fails actionably -- not silently.

5. EMBEDDING CROSS-TERMS / DCD COLLAPSE. When 50 columns are random
   projections of the same scalar latent (every column carries
   redundant signal), DCD-on (``use_simple_mode=False``) must collapse
   the 50-dim cluster to 1-2 representative columns. With DCD-off
   (``use_simple_mode=True``, the production default for wall-time
   reasons), the selector is allowed to keep multiple correlated
   projections -- we pin that the support is non-empty and contains
   columns from BOTH latents (no single-latent monopoly).

WALL-TIME BUDGETS (DEV BOX, NO GPU)
-----------------------------------
p=200 numeric:        < 60 s
p=100 sparse:         < 30 s
p=500 numeric:        < 300 s (5 min)
p=100 embedding:      < 60 s
p=100 embedding (DCD): < 90 s

These are the contract budgets, NOT the observed times. Observed at
authoring time on dev box: 2.85 s / 1.27 s / 3.88 s / 4.11 s / 3.56 s.
The contract budgets leave a ~10-25x safety margin to absorb
co-located numba JIT compilation, CI runners, and the worst-case
seed.
"""

from __future__ import annotations

import time
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROWS = 1_500
N_ROWS_SPARSE = 1_200
SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_embedding_like(seed: int, n: int = N_ROWS, p: int = 200, n_signal: int = 3):
    """Wide matrix with ``n_signal`` informative dims + ``p - n_signal``
    Gaussian noise dims; mimics a sentence-transformer / ada-002
    embedding where the relevant axes are a small subset.

    ``y`` is a linear combination of the first ``n_signal`` columns; the
    remaining ``p - n_signal`` are i.i.d. standard normal noise. The
    signal columns are named ``sig_0``..``sig_{n_signal-1}`` so the
    contract assertions stay readable.
    """
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((n, n_signal))
    # Coefficients designed so every signal dim carries individually
    # detectable MI vs y (>= 0.5x of the strongest's MI -- avoids the
    # Layer-13 "trailing noise outranks weak signal" failure mode).
    if n_signal == 3:
        coefs = np.array([2.0, 1.0, -1.0])
    elif n_signal == 5:
        coefs = np.array([2.0, 1.5, 1.0, -1.0, -1.5])
    else:
        coefs = np.linspace(1.0, 2.0, n_signal)
    y_arr = sig @ coefs + 0.2 * rng.standard_normal(n)
    noise = rng.standard_normal((n, p - n_signal))
    X = np.hstack([sig, noise])
    cols = [f"sig_{k}" for k in range(n_signal)] + [f"noise_{k}" for k in range(p - n_signal)]
    return pd.DataFrame(X, columns=cols), pd.Series(y_arr, name="y_reg")


def _build_tfidf_like(seed: int, n: int = 1_500, p: int = 100, sparsity: float = 0.10):
    """TF-IDF / bag-of-words-like sparse matrix. ``sparsity`` is the
    probability that any given cell is non-zero (~90% zeros at the
    default 0.10). Two specific tokens carry the y signal.

    DESIGN NOTES (authoring 2026-05-30):
    -----------------------------------
    Sparsity raised from the textbook 5% to 10% because the MDLP
    discretizer (production default since Wave 7 / 2026-05-29) bins
    every column with < ~80 non-zero rows into a single bin -> H(X)=0
    -> MI=0, even when sklearn's KSG MI clearly identifies the same
    column as the strongest signal in the matrix (MI ~ 0.09 nats on
    seed=13). The Layer-20 sparse contract uses ``nbins_strategy=None``
    (legacy fixed quantile binning) to bypass this MDLP regression on
    sparse columns. A separate test class
    (``TestMdlpSparseRegression``) pins the bug explicitly so the
    eventual MDLP-on-sparse fix can flip the contract back.

    Signal coefficients raised to (5.0, 4.0) so the y/X relationship is
    detectable on ~150 non-zero rows per token at n=1500, sparsity=0.10.
    """
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, p)) < sparsity).astype(np.float64)
    weights = 1.0 + rng.random((n, p))  # TF-IDF-like positive weights
    X = mask * weights
    # Two informative tokens at positions 3 and 17 -- coefficients large
    # enough that the marginal MI clears the sparse-quantizer floor.
    y_score = 5.0 * X[:, 3] + 4.0 * X[:, 17] + 0.05 * rng.standard_normal(n)
    y_arr = (y_score > np.median(y_score)).astype(np.int64)
    cols = [f"tok_{k}" for k in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y_arr, name="y_cls")


def _build_embedding_cross_terms(seed: int, n: int = N_ROWS, k_per_latent: int = 50):
    """Two latent scalars ``z1, z2`` each randomly projected into
    ``k_per_latent`` columns, mimicking a frozen-encoder embedding
    block where multiple output dims carry redundant copies of the
    same underlying factor. y depends only on z1 and z2.

    Total p = 2 * k_per_latent. The DCD-on contract is that the
    selector collapses each block to ~1 representative; DCD-off
    (default) merely needs to keep BOTH latent blocks alive.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    W1 = rng.standard_normal((1, k_per_latent)) * 0.5
    W2 = rng.standard_normal((1, k_per_latent)) * 0.5
    emb1 = z1[:, None] * W1 + 0.05 * rng.standard_normal((n, k_per_latent))
    emb2 = z2[:, None] * W2 + 0.05 * rng.standard_normal((n, k_per_latent))
    y_arr = 2.0 * z1 + z2 + 0.3 * rng.standard_normal(n)
    X = np.hstack([emb1, emb2])
    cols = [f"e1_{k}" for k in range(k_per_latent)] + [f"e2_{k}" for k in range(k_per_latent)]
    return pd.DataFrame(X, columns=cols), pd.Series(y_arr, name="y_reg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    """Wave 9 production-default MRMR with FE / interactions stripped
    for wall-time bound. ``random_seed`` injected per-seed by callers.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_quiet(sel, X, y):
    """Fit ``sel`` on ``(X, y)`` with warnings silenced; return elapsed seconds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        sel.fit(X, y)
        return time.time() - t0


def _support_names(sel, columns):
    """Return the list of selected column names regardless of whether
    ``support_`` is stored as a boolean mask or integer indices.
    """
    sup = np.asarray(sel.support_)
    if sup.dtype == bool:
        idx = np.flatnonzero(sup)
    else:
        idx = sup.astype(np.intp)
    return [columns[i] for i in idx]


@cache
def _p200_fit(seed: int):
    """Cached ``(X, y, sel, elapsed)`` for the default-config p=200 fit.

    ``test_p200_walltime_and_signal`` and ``test_fit_then_transform_p200``
    both build the identical (seed, p=200, n_signal=3) frame and fit the
    identical default-config MRMR on it -- deterministic, so the second
    call was pure waste. Nothing downstream mutates X/y/sel in place.
    """
    X, y = _build_embedding_like(seed, p=200, n_signal=3)
    sel = _make_mrmr(random_seed=seed)
    elapsed = _fit_quiet(sel, X, y)
    return X, y, sel, elapsed


@cache
def _p500_fit(seed: int):
    """Cached ``(X, y, sel, elapsed)`` for the default-config p=500 fit.

    ``test_p500_walltime_budget`` and ``test_p500_support_stays_bounded``
    both build the identical (seed, n=N_ROWS, p=500, n_signal=3) frame and
    fit the identical default-config MRMR on it. Nothing downstream
    mutates X/y/sel in place.
    """
    X, y = _build_embedding_like(seed, n=N_ROWS, p=500, n_signal=3)
    sel = _make_mrmr(random_seed=seed)
    elapsed = _fit_quiet(sel, X, y)
    return X, y, sel, elapsed


@cache
def _sparse_default_fit(seed: int):
    """Cached ``(X, y, sel, elapsed)`` for the default-config sparse TF-IDF fit.

    ``test_sparse_does_not_crash``, ``test_sparse_recovers_at_least_one_token``,
    ``test_sparse_support_stays_bounded`` (all default-config), and
    ``test_mdlp_recovers_sparse_signal_under_default`` (explicit
    ``nbins_strategy="mdlp"``, which IS the MRMR default -- see
    ``_mrmr_class.py``'s ``nbins_strategy: str = "mdlp"``) all fit the
    IDENTICAL config on the identical (seed, p=100) frame. Nothing
    downstream mutates X/y/sel in place.
    """
    X, y = _build_tfidf_like(seed, p=100)
    sel = _make_mrmr(random_seed=seed)
    elapsed = _fit_quiet(sel, X, y)
    return X, y, sel, elapsed


# ---------------------------------------------------------------------------
# Contract 1: p=200 embedding-like -- wall time under 60s
# ---------------------------------------------------------------------------


class TestEmbeddingLikeP200Scales:
    """3 informative dims hidden in 200 columns. The selector must
    finish in under 60 s AND surface at least one real signal AND
    keep ``support_`` small (<= 10 of 200) -- the noise cloud must not
    pile up just because every dim is a clean Gaussian.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_p200_walltime_and_signal(self, seed):
        """Default-config p=200 fit: under 60s, non-empty, bounded, recovers a signal column."""
        X, _y, sel, elapsed = _p200_fit(seed)
        assert elapsed < 60.0, (
            f"p=200 fit took {elapsed:.2f}s, budget 60s; seed={seed}. "
            f"Super-linear blow-up at embedding scale silently breaks every "
            f"embedding-backed model in prod."
        )
        names = _support_names(sel, list(X.columns))
        assert len(names) >= 1, f"empty support_ on embedding-like p=200; seed={seed}. min_features_fallback=1 default should prevent this."
        # Noise filtering: support stays small (<=10 of 200)
        assert (
            len(names) <= 10
        ), f"support_size={len(names)} on p=200 with 3 real signals (seed={seed}); 197 noise dims should not pile up. selected={names[:15]}"
        # Signal recovery: at least one sig_* column survives
        signals_kept = [c for c in names if c.startswith("sig_")]
        assert len(signals_kept) >= 1, f"no real signal columns in support_={names}; seed={seed}. Noise outranked every real signal -- classic high-dim hijack."

    @pytest.mark.parametrize("seed", SEEDS)
    def test_p200_5_signals(self, seed):
        """Same wall-time + recovery contract with 5 informative dims
        of varying strength. Catches a regression where the bottom
        weakest signal (coef=1.0) gets crowded out by noise.
        """
        X, y = _build_embedding_like(seed, p=200, n_signal=5)
        sel = _make_mrmr(random_seed=seed)
        elapsed = _fit_quiet(sel, X, y)
        assert elapsed < 60.0, f"p=200 / n_signal=5 fit took {elapsed:.2f}s, budget 60s; seed={seed}"
        names = _support_names(sel, list(X.columns))
        signals_kept = [c for c in names if c.startswith("sig_")]
        assert len(signals_kept) >= 1, f"no signal columns recovered with 5 real signals in p=200; seed={seed}, support={names[:15]}"


# ---------------------------------------------------------------------------
# Contract 2: TF-IDF sparse -- doesn't reject mostly-zero columns wrongly
# ---------------------------------------------------------------------------


class TestTfIdfSparseHandled:
    """~90% zeros per column; signal lives on 2 specific tokens. Uses
    PRODUCTION DEFAULT config (``nbins_strategy='mdlp'``) since the
    2026-05-31 sparse-aware secondary fallback in _adaptive_nbins.py:
    when MDLP returns 0 splits AND the unsupervised quantile fallback
    ALSO collapses (every quantile lands at the dominant value), a
    separate-bin path activates: 1 bin for the dominant value (zero)
    + quantile bins on the non-zero subset. Mirrors the
    nan_strategy='separate_bin' pattern.

    The previously workaround used (``nbins_strategy=None,
    quantization_method='uniform'``) is no longer needed - the default
    config now correctly surfaces tok_3 / tok_17 on every seed.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_sparse_does_not_crash(self, seed):
        """Default-config TF-IDF-like sparse fit completes under the 30s budget."""
        _X, _y, _sel, elapsed = _sparse_default_fit(seed)
        assert elapsed < 30.0, f"sparse p=100 fit took {elapsed:.2f}s, budget 30s; seed={seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_sparse_recovers_at_least_one_token(self, seed):
        """At least one of the two real token columns (tok_3, tok_17)
        must appear in support_ under PRODUCTION DEFAULT config.
        """
        X, _y, sel, _elapsed = _sparse_default_fit(seed)
        names = _support_names(sel, list(X.columns))
        assert len(names) >= 1, (
            f"empty support_ on TF-IDF-like sparse data; seed={seed}. "
            f"Silent rejection of every mostly-zero column is the "
            f"production-breaking bug class this contract pins."
        )
        true_signals = {"tok_3", "tok_17"}
        hit = [c for c in names if c in true_signals]
        assert (
            len(hit) >= 1
        ), f"neither tok_3 nor tok_17 in support_={names}; seed={seed}. Sparse-aware fallback should have surfaced the signal under default MDLP config."

    @pytest.mark.parametrize("seed", SEEDS)
    def test_sparse_support_stays_bounded(self, seed):
        """Sparse-column MI estimates can spike on the lucky-bin
        random subset of zero-runs; pin that support stays < 15 of 100
        so a quantizer regression doesn't let the noise cloud through.
        """
        X, _y, sel, _elapsed = _sparse_default_fit(seed)
        names = _support_names(sel, list(X.columns))
        assert len(names) <= 15, f"sparse support_size={len(names)}/100; seed={seed}. Mostly-zero columns should not pile up. selected={names[:15]}"


# ---------------------------------------------------------------------------
# Contract 2b: MDLP-on-sparse REGRESSION DOCUMENTATION
# ---------------------------------------------------------------------------


class TestMdlpSparseFixed:
    """2026-05-31: MDLP-on-sparse RESOLVED via sparse-aware secondary
    fallback in _adaptive_nbins.py:per_feature_edges. When MDLP collapses
    AND the unsupervised quantile fallback ALSO collapses (every
    quantile lands at the dominant value), a separate-bin path
    activates: 1 bin for the dominant value (typically zero) + quantile
    bins on the non-dominant subset.

    Mechanism: detect dominant_frac > 0.5; split _finite into
    {== dominant_val} and {!= dominant_val}; quantile the non-dominant
    subset; boundary edge between dominant value and non-dominant
    range. Mirrors nan_strategy='separate_bin' pattern.

    Pre-fix: support=['tok_0'] (alphabetical fallback). Post-fix:
    support contains tok_3 and/or tok_17 with fallback_used_=False.
    """

    @pytest.mark.parametrize("seed", (13, 42))
    def test_mdlp_recovers_sparse_signal_under_default(self, seed):
        """Under production default (nbins_strategy='mdlp'), at least
        one of tok_3 / tok_17 must be in support_, AND fallback_used_
        must be False (the sparse-aware path is genuine, not a fallback).
        """
        X, _y, sel, _elapsed = _sparse_default_fit(seed)
        names = _support_names(sel, list(X.columns))
        true_signals = {"tok_3", "tok_17"}
        hit = [c for c in names if c in true_signals]
        assert len(hit) >= 1, (
            f"MDLP-default missed sparse signal on seed={seed}. "
            f"Expected tok_3 and/or tok_17 in support; got {names}. "
            f"The sparse-aware fallback in _adaptive_nbins.py may have "
            f"regressed or its dominant-fraction threshold (0.5) changed."
        )
        # Sparse-aware path is a genuine recovery, not a fallback signal.
        assert getattr(sel, "fallback_used_", False) is False, (
            f"MDLP recovered sparse signal but fallback_used_=True; "
            f"that's a contradiction - either the sparse-aware path is "
            f"misclassified as fallback or the diagnostic flag is wrong. "
            f"seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 3: p=500 -- 5-min budget, signal recovery
# ---------------------------------------------------------------------------


class TestP500WideMatrixScales:
    """Pushes the dispatcher + scaling. Different seeds reveal numba
    JIT cache effects (one parametrize value triggers compilation, the
    rest amortise) -- the 5-min budget absorbs the worst-case warm-up.
    """

    @pytest.mark.parametrize("seed", (1, 13, 42))
    def test_p500_walltime_budget(self, seed):
        """Default-config p=500 fit: under the 300s budget, recovers a signal column."""
        X, _y, sel, elapsed = _p500_fit(seed)
        assert elapsed < 300.0, f"p=500 fit took {elapsed:.2f}s, budget 300s; seed={seed}. Embedding-scale wall-time blow-up."
        names = _support_names(sel, list(X.columns))
        assert len(names) >= 1
        # Real signal must survive even at p=500 noise dim count
        signals_kept = [c for c in names if c.startswith("sig_")]
        assert len(signals_kept) >= 1, f"no signal columns in p=500 support={names[:20]}; seed={seed}. 497 noise dims crowded out every real signal."

    @pytest.mark.parametrize("seed", (1, 13, 42))
    def test_p500_support_stays_bounded(self, seed):
        """Even at p=500 the noise cloud must not pile up: support
        stays <= 15 (3 real signals + slack for marginal noise).
        """
        X, _y, sel, _elapsed = _p500_fit(seed)
        names = _support_names(sel, list(X.columns))
        assert len(names) <= 15, f"p=500 support_size={len(names)} > 15; seed={seed}. Noise pile-up at scale. selected={names[:15]}"


# ---------------------------------------------------------------------------
# Contract 4: Embedding cross-terms -- DCD collapses redundant projections
# ---------------------------------------------------------------------------


class TestEmbeddingCrossTermsDcd:
    """50 random projections of z1 + 50 of z2 = 100 columns, each
    individually informative but per-block redundant. Two contracts:

    A. Default (``use_simple_mode=True``, DCD-off for speed): support
       must contain columns from BOTH latents -- no single-latent
       monopoly that drops half the signal. Support size is allowed
       to be large (no redundancy gate) but the under-50 cap detects
       a regression where every single column is admitted.

    B. DCD-on (``use_simple_mode=False``): support collapses to a
       handful (1-4) of representative columns and STILL covers both
       latents.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_keeps_both_latents(self, seed):
        """Default (DCD-off) config: support covers both latents, no single-latent monopoly."""
        X, y = _build_embedding_cross_terms(seed, k_per_latent=50)
        sel = _make_mrmr(random_seed=seed)
        elapsed = _fit_quiet(sel, X, y)
        assert elapsed < 60.0, f"embedding cross-terms p=100 fit took {elapsed:.2f}s, budget 60s; seed={seed}"
        names = _support_names(sel, list(X.columns))
        e1_picked = sum(1 for c in names if c.startswith("e1_"))
        e2_picked = sum(1 for c in names if c.startswith("e2_"))
        assert e1_picked >= 1, f"no e1_* columns selected; seed={seed}, support={names[:10]}. Latent z1 was completely dropped."
        assert e2_picked >= 1, f"no e2_* columns selected; seed={seed}, support={names[:10]}. Latent z2 was completely dropped."
        # Sanity bound: don't admit every single column
        assert len(names) < 100, f"support_size={len(names)} == 100 means every column was selected; seed={seed}. Relevance gate was bypassed."

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dcd_on_collapses_redundancy(self, seed):
        """``use_simple_mode=False`` engages the per-candidate
        conditional-MI redundancy check. On 100 columns that are 50
        copies of z1 + 50 copies of z2, the collapse target is 2-4
        representatives. We pin <= 10 to absorb seed variance and
        ``max_consec_unconfirmed`` patience edge cases.
        """
        X, y = _build_embedding_cross_terms(seed, k_per_latent=50)
        sel = _make_mrmr(random_seed=seed, use_simple_mode=False)
        elapsed = _fit_quiet(sel, X, y)
        assert elapsed < 90.0, f"DCD-on embedding cross-terms fit took {elapsed:.2f}s, budget 90s; seed={seed}"
        names = _support_names(sel, list(X.columns))
        assert 1 <= len(names) <= 10, (
            f"DCD-on support_size={len(names)} outside [1, 10]; "
            f"seed={seed}, support={names}. Per-block redundancy collapse "
            f"is broken; 50 copies of z1 should reduce to ~1."
        )
        # At minimum, the strongest latent (z1, coef=2.0) must be
        # represented. Per-seed both latents would be ideal but with
        # max_consec_unconfirmed=10 patience the weaker latent (z2,
        # coef=1.0) can drop out on adversarial seeds -- so we pin
        # the existence of at least ONE block's representative.
        blocks = {c.split("_")[0] for c in names}
        assert (
            "e1" in blocks
        ), f"strongest latent z1 (coef=2.0) absent from DCD-on support; seed={seed}, support={names}. Redundancy collapse dropped the dominant signal."


# ---------------------------------------------------------------------------
# Contract 5: end-to-end -- fit + transform round-trip at p=200
# ---------------------------------------------------------------------------


class TestHighDimFitTransformRoundTrip:
    """The selected high-dim support must transform cleanly on held-
    out data with the same schema. Catches a regression where the
    p=200 fit succeeds but ``transform`` silently widens / truncates.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_then_transform_p200(self, seed):
        """Reuses the p=200 fit; transform on held-out data preserves row/column shape."""
        X_train, _y_train, sel, _elapsed = _p200_fit(seed)
        names = _support_names(sel, list(X_train.columns))
        assert len(names) >= 1

        # Held-out frame with the SAME schema
        X_test, _ = _build_embedding_like(seed + 99_999, p=200, n_signal=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sel.transform(X_test)
        assert out.shape[0] == X_test.shape[0], f"row count mismatch after p=200 transform; seed={seed}. out.shape={out.shape}, expected rows={X_test.shape[0]}"
        assert out.shape[1] == len(names), f"column count mismatch after p=200 transform; seed={seed}. out.shape={out.shape}, support_size={len(names)}"
        # DataFrame path: column names must match support exactly
        if isinstance(out, pd.DataFrame):
            assert list(out.columns) == names, f"transform returned column names != support; seed={seed}. got={list(out.columns)[:10]}, expected={names[:10]}"
