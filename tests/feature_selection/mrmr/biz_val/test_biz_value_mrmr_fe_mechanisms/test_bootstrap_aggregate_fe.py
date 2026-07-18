"""Layer 36 biz_value: STABILITY-AWARE FE -- bootstrap-aggregate MRMR FE

Consolidated verbatim from test_biz_value_mrmr_layer36.py (per audit finding test_code_quality-16).
selection (Meinshausen-Buhlmann stability selection applied to the
hybrid FE pipeline).

Why
---
MRMR's hybrid FE pipeline emits engineered columns via six constructor
stages (Layer 35 union). Each constructor's MI-greedy gate is a noisy
two-gate statistic on a single fit: borderline candidates flip in /
out of the support based on random fold / sampling noise.

For HIGH-STAKES production use, an analyst building a permanent
externally-audited model wants engineered columns that CONSISTENTLY
appear across bootstrap subsamples -- not whichever set survived a
single noisy fit. Layer 36 wraps MRMR with subsampling-without-
replacement stability selection: refit n_bootstraps times on
size-fraction subsamples, count selection frequency per engineered
NAME (union of all six FE attribute lists), surface the columns whose
freq clears a support threshold.

Contracts pinned
----------------
* TestStableSignalHasHighFrequency
    On y = sign(x^2 - 1), the orth-poly Hermite-2 term ``x__He2``
    appears in >= 80% of bootstraps. With this signal strength the
    constructor reliably catches the quadratic shape; the wrapper must
    not erode that reliability.

* TestNoiseTransformsHaveLowFrequency
    Any engineered name derived from a pure-noise source column
    (``noise*__He2`` / ``noise*__He3`` / unary-of-noise / ...) appears
    in <= 30% of bootstraps -- ideally 0 (MRMR's MI gate is
    conservative enough that pure noise transforms never reach the
    engineered list). The 30% cap is the high-stakes threshold below
    which an analyst can confidently discard the column.

* TestStabilityRanksMatchOrBeatSingleFit
    Across 5 distinct data seeds, the proportion of seeds where
    ``x__He2`` enters the stable_set is >= the proportion where it
    enters a single MRMR fit's ``hybrid_orth_features_``. Stability
    is at LEAST as reliable as the single-fit point estimate; the
    common case (strong signal) is a tie at 100%, and the wrapper does
    not regress that. Crucially, the same 5-seed sweep also confirms
    NO noise transform ever enters the stable_set on ANY of the
    seeds -- the biz value an analyst pays for.

* TestPickleAndCloneAllParamsPreserved
    The wrapper survives ``pickle.loads(pickle.dumps(m))`` and
    ``sklearn.base.clone(m)`` with every constructor arg
    (base_mrmr_params, n_bootstraps, sample_fraction,
    support_threshold, random_state) preserved + fitted state
    (frequencies_, stable_set_, per_bootstrap_engineered_, full_mrmr_)
    surviving pickle round-trip with identical transform output.

NEVER xfail. Real MRMR fits, real bootstrap variance. If a contract
breaks: fix prod / recalibrate the fixture -- per the standing
biz_value rules.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings
from functools import cache

import numpy as np
import pandas as pd

from sklearn.base import clone

warnings.filterwarnings("ignore")


HEADLINE_SEED = 0
MULTI_SEEDS = (21, 22, 23, 24, 25)
N_BOOTSTRAPS = 10
SAMPLE_FRACTION = 0.75
SUPPORT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_mrmr_params() -> dict:
    """Single source of truth for the MRMR config used across layer 36
    contracts. orth-poly is the only FE stage enabled so we can pin
    He2 / He3 names without cross-stage interference; other FE
    mechanisms have their own layer-22 / layer-26 / layer-32-34 /
    layer-35 contracts.
    """
    return dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        fe_ntop_features=15,
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=6,
    )


def _make_strong_signal(seed: int, n: int = 1500):
    """y = sign(x^2 - 1) + a touch of noise contribution.

    Strong enough that x__He2 enters the engineered list reliably on a
    single MRMR fit; bootstrap aggregation should keep that reliability
    AND confirm noise transforms never make the stable set.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = rng.standard_normal((n, 6))
    logit = 3.0 * np.sign(x**2 - 1.0) + 0.2 * noise[:, 0]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame(
        {
            "x": x,
            **{f"noise{i}": noise[:, i] for i in range(6)},
        }
    )
    return X, pd.Series(y, name="y")


def _make_moderate_signal(seed: int, n: int = 600):
    """Smaller n + slightly weaker logit; still strong enough that
    x__He2 reliably surfaces in MRMR's engineered list. Used by the
    multi-seed test to keep total bootstrap runtime manageable.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = rng.standard_normal((n, 6))
    logit = 1.5 * np.sign(x**2 - 1.0) + 0.3 * noise[:, 0]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame(
        {
            "x": x,
            **{f"noise{i}": noise[:, i] for i in range(6)},
        }
    )
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: stable signal feature has high frequency
# ---------------------------------------------------------------------------


@cache
def _strong_signal_stability_result():
    """Cached ``(X, y, result)`` for the headline-seed strong-signal fit.

    Contracts 1 and 2 (``TestStableSignalHasHighFrequency`` /
    ``TestNoiseTransformsHaveLowFrequency``) both call ``stability_select_fe``
    with IDENTICAL data and params to check different assertions on the SAME
    result -- compute the bootstrap-aggregate MRMR fit (the expensive step)
    once instead of twice. Nothing downstream mutates X/y/result in place.
    """
    from mlframe.feature_selection.filters._stability_fe import stability_select_fe

    X, y = _make_strong_signal(seed=HEADLINE_SEED)
    result = stability_select_fe(
        X,
        y,
        base_mrmr_params=_base_mrmr_params(),
        n_bootstraps=N_BOOTSTRAPS,
        sample_fraction=SAMPLE_FRACTION,
        support_threshold=SUPPORT_THRESHOLD,
        random_state=HEADLINE_SEED,
    )
    return X, y, result


class TestStableSignalHasHighFrequency:
    """Contract 1: x__He2 appears in >= 80% of bootstraps on strong signal."""

    def test_x_he2_appears_in_at_least_80pct_of_bootstraps(self):
        """x__He2 selection_frequency >= 0.8 and it enters the stable_set."""
        _X, _y, result = _strong_signal_stability_result()
        freq = result["frequencies"]
        assert "x__He2" in freq["engineered_name"].tolist(), f"x__He2 did not appear in any bootstrap; freqs=\n{freq}"
        x_he2_freq = float(freq.loc[freq["engineered_name"] == "x__He2", "selection_frequency"].iloc[0])
        assert x_he2_freq >= 0.8, f"x__He2 selection_frequency {x_he2_freq:.2f} below the 0.8 contract;\nfull freq table:\n{freq}"
        # And the stable set should include it.
        assert (
            "x__He2" in result["stable_set"]
        ), f"x__He2 freq={x_he2_freq:.2f} >= 0.8 but did not enter the stable_set (threshold={SUPPORT_THRESHOLD}); stable_set={result['stable_set']}"


# ---------------------------------------------------------------------------
# Contract 2: noise transforms have low frequency
# ---------------------------------------------------------------------------


class TestNoiseTransformsHaveLowFrequency:
    """Contract 2: any noise-derived engineered name has selection_frequency <= 30%."""

    def test_no_noise_transform_exceeds_30pct_frequency(self):
        """No noise*__He<k> name exceeds 30% frequency, and none enters the stable_set."""
        _X, _y, result = _strong_signal_stability_result()
        freq = result["frequencies"]
        # A "noise transform" is any engineered name whose underlying source
        # is one of the noise* columns. The orth-poly naming convention is
        # ``<source>__He<k>``, so ``startswith('noise')`` catches the entire
        # family. MI-greedy / kfold_te / count / freq / cat-num are off in
        # this test config, so the only possible engineered names are
        # ``<source>__He<k>``.
        offenders = []
        for _, row in freq.iterrows():
            name = str(row["engineered_name"])
            f = float(row["selection_frequency"])
            if name.startswith("noise") and f > 0.3:
                offenders.append((name, f))
        assert not offenders, f"Noise transforms exceeded the 30% frequency cap: {offenders};\nfull freq table:\n{freq}"
        # And the stable set must contain ZERO noise transforms.
        bad_in_stable = [c for c in result["stable_set"] if c.startswith("noise")]
        assert not bad_in_stable, f"Noise transforms leaked into stable_set: {bad_in_stable}; full stable_set={result['stable_set']}"


# ---------------------------------------------------------------------------
# Contract 3: stability ranks >= MRMR single-fit across 5 seeds
# ---------------------------------------------------------------------------


class TestStabilityRanksMatchOrBeatSingleFit:
    """Contract 3: across 5 seeds, stability hit-rate >= single-fit hit-rate for x__He2."""

    def test_stable_x_he2_hit_rate_at_least_single_fit(self):
        """Across 5 distinct data realisations:

        Single-fit MRMR: count seeds where x__He2 enters hybrid_orth_features_.
        Stability:        count seeds where x__He2 enters stable_set.

        Contract: stability_hits >= single_hits. The common-case strong
        signal saturates both at 5 / 5 (tie); when the signal is
        borderline, single-fit flips while stability holds. The wrapper
        must not degrade either case.

        Also confirm: NO noise transform makes the stable_set on ANY of
        the 5 seeds. This is the headline biz value an analyst pays
        for -- stable engineered columns that are reproducibly real
        across bootstrap subsamples.
        """
        from mlframe.feature_selection.filters._stability_fe import (
            stability_select_fe,
        )
        from mlframe.feature_selection.filters.mrmr import MRMR

        base = _base_mrmr_params()
        single_hits = 0
        stab_hits = 0
        per_seed_hits = []
        noise_in_stable_seeds = []
        for seed in MULTI_SEEDS:
            X, y = _make_moderate_signal(seed=seed)
            # Single-fit
            m = MRMR(**base)
            m.fit(X, y)
            single_eng = list(getattr(m, "hybrid_orth_features_", []) or [])
            single_hit = "x__He2" in single_eng
            if single_hit:
                single_hits += 1
            # Stability
            result = stability_select_fe(
                X,
                y,
                base_mrmr_params=base,
                n_bootstraps=N_BOOTSTRAPS,
                sample_fraction=SAMPLE_FRACTION,
                support_threshold=SUPPORT_THRESHOLD,
                random_state=seed,
            )
            stable = result["stable_set"]
            stab_hit = "x__He2" in stable
            if stab_hit:
                stab_hits += 1
            per_seed_hits.append((seed, int(single_hit), int(stab_hit)))
            noisy = [c for c in stable if c.startswith("noise")]
            if noisy:
                noise_in_stable_seeds.append((seed, noisy))
        assert stab_hits >= single_hits, (
            f"Stability hit rate {stab_hits}/{len(MULTI_SEEDS)} < single-fit hit rate "
            f"{single_hits}/{len(MULTI_SEEDS)}; the wrapper regressed reliability "
            f"(per-seed single/stable hits: {per_seed_hits})"
        )
        assert not noise_in_stable_seeds, f"Noise transforms leaked into stable_set on at least one seed: {noise_in_stable_seeds}"


# ---------------------------------------------------------------------------
# Contract 4: pickle + clone preserve all params + fitted state
# ---------------------------------------------------------------------------


class TestPickleAndCloneAllParamsPreserved:
    """Contract 4: pickle/clone round-trips preserve every constructor param and fitted state."""

    def _build(self):
        """Build an unfitted StabilityFESelector with the layer-36 default params."""
        from mlframe.feature_selection.filters._stability_fe import (
            StabilityFESelector,
        )

        return StabilityFESelector(
            base_mrmr_params=_base_mrmr_params(),
            n_bootstraps=N_BOOTSTRAPS,
            sample_fraction=SAMPLE_FRACTION,
            support_threshold=SUPPORT_THRESHOLD,
            random_state=HEADLINE_SEED,
        )

    def test_clone_preserves_all_params(self):
        """sklearn.base.clone preserves every constructor param on an unfitted clone."""
        m = self._build()
        m2 = clone(m)
        p1 = m.get_params()
        p2 = m2.get_params()
        for key in (
            "base_mrmr_params",
            "n_bootstraps",
            "sample_fraction",
            "support_threshold",
            "random_state",
        ):
            assert p1[key] == p2[key], f"clone lost param '{key}': orig={p1[key]!r} clone={p2[key]!r}"
        # Clone is unfitted.
        assert not hasattr(m2, "frequencies_")
        assert not hasattr(m2, "stable_set_")
        assert not hasattr(m2, "full_mrmr_")

    def test_pickle_roundtrip_preserves_fitted_state(self):
        """pickle.loads(pickle.dumps(m)) preserves fitted state and transform output bit-for-bit."""
        X, y = _make_strong_signal(seed=HEADLINE_SEED)
        m = self._build()
        m.fit(X, y)
        pre_freq = m.frequencies_.copy()
        pre_stable = list(m.stable_set_)
        pre_out = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object, not untrusted input

        # Fitted attrs survive.
        assert hasattr(m2, "frequencies_")
        assert hasattr(m2, "stable_set_")
        assert hasattr(m2, "full_mrmr_")
        assert hasattr(m2, "per_bootstrap_engineered_")
        # frequencies_ DataFrame identical in shape + content.
        post_freq = m2.frequencies_
        pd.testing.assert_frame_equal(pre_freq, post_freq)
        # stable_set_ identical.
        assert list(m2.stable_set_) == pre_stable
        # Transform output bit-identical for numeric cols.
        post_out = m2.transform(X)
        assert list(post_out.columns) == list(pre_out.columns), f"pickle changed transform columns: pre={list(pre_out.columns)} post={list(post_out.columns)}"
        for col in pre_out.columns:
            if pd.api.types.is_numeric_dtype(pre_out[col]):
                np.testing.assert_allclose(
                    np.asarray(pre_out[col], dtype=np.float64),
                    np.asarray(post_out[col], dtype=np.float64),
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg=f"pickle changed values of column {col!r}",
                )
