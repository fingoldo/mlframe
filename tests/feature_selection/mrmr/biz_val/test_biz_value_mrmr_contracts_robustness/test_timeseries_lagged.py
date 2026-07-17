"""Consolidated from test_biz_value_mrmr_layer9.py.

Layer 9 biz_value MRMR contracts: TIME-SERIES LAGGED FEATURES.

WHY THIS LAYER
--------------
Every nowcasting / forecasting / streaming-anomaly pipeline in
production builds a feature matrix from a base time series by emitting:

* Lags ``x_t0, x_t1, ..., x_tk`` (window of past values)
* Rolling aggregates ``mean_3``, ``mean_7``, ``std_5`` etc.

These columns are all strongly correlated with each other (they share
the same underlying AR(1) / autoregressive signal). A naive selector
either:

* Keeps every lag (drowns downstream model in 10-50 collinear cols), or
* Keeps the lag with the highest marginal MI and drops good aggregates
  that summarize different temporal scales, or
* Picks a useless lag (e.g. ``x_t4``) because of weak permutation power
  + redundancy noise.

MRMR's redundancy term should:

1. Find the RIGHT lag (the one y actually depends on, ``x_t1``).
2. Prune at least some of the 5-lag collinear cluster.
3. Still evaluate rolling aggregates honestly (not auto-rejecting them
   just because they correlate with the kept lag).

CONTRACTS PINNED
----------------
1. Right lag found: ``x_t1`` in support_ AND ranks first (target-
   aligned lag must be MRMR's #1 pick).
2. Selection retains the full predictive signal: a downstream LogReg
   on the MRMR selection is within a small band of one on the full
   lag+aggregate matrix (rebaselined from the old ">=1 rolling mean
   kept" name-membership check, which was simple-mode specific:
   full-mode redundancy correctly prunes the means as adding ~zero MI
   conditional on x_t1 -- see TestLaggedAggregatesConsidered).
3. At least one feature pruned: support_ is strictly smaller than the
   full input column set (selection actually filters something).
4. Stability across 5 seeds: x_t1 ranks #1 every time.

OBSERVED + DOCUMENTED (not pinned tightly; behaviour of the full-mode
default ``use_simple_mode=False``):

* The selection is COMPACT. Under full-mode Fleuret conditional-MI
  redundancy the collinear lag cluster + rolling aggregates collapse
  hard: conditional on the selected x_t1, the sibling lags and the
  rolling means add ~zero MI, so the observed support is often just
  ``['x_t1']``. This is correct, not over-pruning -- x_t1 carries the
  full signal (downstream LogReg AUC parity is pinned in
  TestLaggedAggregatesConsidered). (Under the legacy ``use_simple_mode``
  selector every lag scored a positive marginal relevance-minus-
  redundancy and all 5 lags were typically kept; that was a simple-mode
  artifact, not a richer selection.)
* When more than x_t1 survives, ``std_5`` (a variance summary, not a
  mean-tracking signal) is consistently the FIRST feature pruned
  across seeds -- weakest MI with y = sign(x_t1).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _build_lagged_ts(n: int = 2500, seed: int = 9001, phi: float = 0.85):
    """Synthetic AR(1) base signal with lag-k shifted copies + rolling
    aggregates. y depends ONLY on lag-1 (``x_t1``).

    Columns:
      * x_t0, x_t1, x_t2, x_t3, x_t4 - lag-k of an AR(1) process
      * mean_3, mean_7 - rolling means of the base series
      * std_5          - rolling std of the base series

    y = sign(x_t1 + 0.3 * noise) so only the 1-step lag carries true
    signal; the rest are collinear/aggregate views of the same process.
    """
    rng = np.random.default_rng(seed)
    # Generate AR(1) base series of length n + buffer for lags.
    buf = 16
    eps = rng.standard_normal(n + buf)
    base = np.empty(n + buf, dtype=np.float64)
    base[0] = eps[0]
    for t in range(1, n + buf):
        base[t] = phi * base[t - 1] + eps[t]
    # Lag columns aligned to the last n rows; x_tk[t] = base[t - k].
    # Anchor t-index spans [buf, buf+n). Lag-k slice is [buf-k, buf-k+n).
    cols = {}
    for k in range(5):
        cols[f"x_t{k}"] = base[buf - k : buf - k + n]
    # Rolling aggregates of the base series, anchored at the same t-index.
    s_base = pd.Series(base[buf - 1 : buf - 1 + n + 1])  # one-step-ahead window anchor
    # We just use a pandas rolling over the windowed base, then take last n.
    full = pd.Series(base)
    cols["mean_3"] = full.rolling(window=3, min_periods=1).mean().to_numpy()[buf : buf + n]
    cols["mean_7"] = full.rolling(window=7, min_periods=1).mean().to_numpy()[buf : buf + n]
    cols["std_5"] = full.rolling(window=5, min_periods=2).std().fillna(0.0).to_numpy()[buf : buf + n]
    X = pd.DataFrame(cols)
    # Target: only lag-1 carries signal.
    noise = rng.standard_normal(n)
    y_logit = cols["x_t1"] + 0.3 * noise
    y = pd.Series((y_logit > 0).astype(np.int64), name="y")
    return X, y


class TestLaggedBasics:
    """The target-aligned lag must be found AND ranked first."""

    def test_target_lag_in_support(self):
        """``x_t1`` is the only true signal. It must appear in support_.
        If absent, the entire lag-tracking story is broken.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_t1" in names, f"Target lag x_t1 missing from support; support={names}"

    def test_target_lag_ranks_first(self):
        """Across seeds, ``x_t1`` consistently ranks #1. This is the
        strong contract: MRMR's relevance term picks the right lag from
        the collinear cluster as the top feature. If a sibling lag
        (x_t0, x_t2, ..) outranks x_t1, MRMR is randomly picking from
        the cluster, not finding the true target-aligned lag.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) >= 1, f"Empty selection; {names}"
        assert names[0] == "x_t1", f"x_t1 must rank #1 (only true signal); got #1={names[0]}, full={names}"

    def test_at_least_one_feature_pruned(self):
        """We have 8 input features and 1 true signal. The selector
        must reject AT LEAST ONE feature - returning the full input
        verbatim means MRMR's pruning logic is a no-op.

        Observed: ``std_5`` is consistently dropped (variance-aware
        feature, weak MI with sign(x_t1) target). This is the realistic
        bound under the default config; tighter lag-cluster pruning is
        intentionally a downstream wrapper's job and is documented in
        the module docstring rather than xfailed here.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) < 8, f"Selector returned ALL 8 inputs; pruning is a no-op; support={names}"


class TestLaggedAggregatesConsidered:
    """The selection must retain the FULL predictive signal of the
    lag+aggregate matrix -- it must not throw away information by
    over-pruning. We pin this by downstream-AUC parity rather than by
    forcing a specific aggregate column to survive, because under the
    full-mode (Fleuret conditional-MI) default the rolling means add
    ~zero MI conditional on x_t1 and are correctly dropped."""

    def test_selection_retains_full_signal(self):
        """Downstream-AUC parity: a LogReg on the MRMR selection must be
        within a small band of a LogReg on the full lag+aggregate matrix.

        Rebaselined from the old "at least one of mean_3/mean_7 in
        support_" name-membership assertion, which was simple-mode
        specific: simple-mode MRMR kept marginally-relevant-but-redundant
        columns, so an aggregate that correlates with x_t1 survived.
        Full-mode MRMR (the new default) computes redundancy as Fleuret
        conditional MI: the rolling means are linear combinations of
        recent lags, so conditional on the selected x_t1 they add ~zero
        MI and are correctly pruned. Measured (seed 9001): x_t1 alone
        scores 5-fold ROC-AUC 0.9905, identical to all-8-columns 0.9903,
        so dropping the aggregates costs NO downstream value -- forcing
        one to survive would pin a simple-mode artifact, not a real win.
        This AUC-parity contract is still falsifiable: if MRMR dropped
        x_t1 itself (the only true signal) the selection AUC would
        collapse toward 0.5 and this assertion would fire.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        from tests.feature_selection._biz_val_synth import downstream_auc
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X, y = _build_lagged_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) >= 1, f"Empty selection; support={names}"

        auc_sel = downstream_auc(sel, X, y.to_numpy(), cv=5)
        auc_full = cross_val_score(
            LogisticRegression(max_iter=400),
            X.to_numpy(),
            y.to_numpy(),
            cv=5,
            scoring="roc_auc",
        ).mean()
        assert auc_sel >= auc_full - 0.03, (
            f"MRMR selection {names} lost downstream signal: "
            f"selection AUC={auc_sel:.4f} vs full-matrix AUC={auc_full:.4f} "
            f"(gap > 0.03). Over-pruning destroyed predictive information."
        )


class TestLaggedSeedRobustness:
    """The 'target lag found AND ranks first' contract must hold across
    multiple seeds. This is the strong stability claim - getting x_t1
    right on one lucky seed isn't a real win."""

    @pytest.mark.parametrize("seed", [9001, 9002, 9003, 9004, 9005])
    def test_target_lag_first_across_seeds(self, seed):
        """Check test target lag first across seeds."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert names and names[0] == "x_t1", f"seed={seed}: x_t1 not ranked #1; support={names}"

    @pytest.mark.parametrize("seed", [9001, 9002, 9003, 9004, 9005])
    def test_pruning_active_across_seeds(self, seed):
        """Pruning must remove AT LEAST ONE feature across seeds.
        Observed: std_5 is dropped on every seed (variance feature has
        weakest MI with sign(x_t1)). Bound at <8 (strictly smaller than
        full input) keeps the contract honest without overpinning.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) < 8, f"seed={seed}: pruning no-op; size={len(names)}/8 selected; support={names}"


class TestLaggedTargetLagPriority:
    """Stronger ordering claims: x_t1 must outrank std_5 (the
    variance-summary aggregate has weakest MI with sign(x_t1)), and
    x_t1 must outrank the most distant lag x_t4 (lag-4 is the
    weakest MI-bearer of the lag cluster on an AR(1)-with-phi=0.85
    process)."""

    def test_target_lag_outranks_std_aggregate(self):
        """If both x_t1 and std_5 end up in support_, x_t1 must rank
        ABOVE std_5. (In the observed case std_5 is dropped entirely;
        this test bites if a future change starts keeping std_5 above
        x_t1.)
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_t1" in names, f"x_t1 missing; support={names}"
        if "std_5" not in names:
            return  # std_5 dropped entirely - already optimal.
        assert names.index("x_t1") < names.index("std_5"), (
            f"x_t1 (rank {names.index('x_t1')}) ranks below std_5 "
            f"(rank {names.index('std_5')}); variance-summary aggregate "
            f"beating target-aligned lag is a real bug; support={names}"
        )

    def test_target_lag_outranks_distant_lag(self):
        """If both x_t1 and x_t4 are in support_, x_t1 must rank above
        x_t4. On AR(1) with phi=0.85, lag-4 has correlation phi**3 ~
        0.61 with x_t1, so its MI with y is real but lower than x_t1's
        direct MI. If x_t4 outranks x_t1, MRMR's relevance scoring is
        flipping the lag ordering.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_lagged_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_t1" in names, f"x_t1 missing; support={names}"
        if "x_t4" not in names:
            return  # x_t4 dropped - already optimal.
        assert names.index("x_t1") < names.index("x_t4"), (
            f"x_t1 (rank {names.index('x_t1')}) ranks below x_t4 (rank {names.index('x_t4')}); MRMR flipped the lag ordering; support={names}"
        )
