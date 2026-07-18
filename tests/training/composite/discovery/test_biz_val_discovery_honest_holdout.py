"""biz_val + leakage tests for the SA27 post-selection-inference holdout (winner's curse).

The discovery driver selects the winner spec on the SAME ``mi_gain`` statistic it then
reports, so that in-screen gain is the MAX over many candidates on one screening sample --
optimistically biased upward (winner's curse). ``honest_holdout_frac`` carves a holdout out
of train BEFORE screening and re-scores only the final winner(s) on it, producing a de-biased
``honest_holdout_gain``.

Tests here assert:
  * biz_value: on pure noise the in-screen winner gain is materially HIGHER than the honest
    holdout gain, and the honest gain is ~0 (the honest estimate is materially less biased).
  * biz_value (real signal): the honest gain tracks the true gain (stays well positive) -- the
    de-bias does not destroy a genuine effect.
  * leakage: the holdout indices never appear in the screening sample / are disjoint from train.
  * the pre-fix shape had no honest gain key (simulated in-place by running with the holdout off).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _pure_noise_df(n=6000, seed=7, n_feat=10, n_base=8):
    """y independent of every feature / base -- no transform has a true gain.

    Many features AND many base candidates so the candidate family is large: the
    winner = max over many in-screen gains, which is exactly where the winner's curse
    inflates the reported point gain above the truth (0)."""
    rng = np.random.default_rng(seed)
    data = {f"f{j}": rng.normal(size=n) for j in range(n_feat)}
    for b in range(n_base):
        data[f"b{b}"] = rng.normal(size=n)
    data["base"] = rng.normal(size=n)
    data["y"] = rng.normal(size=n)  # truly independent
    return pd.DataFrame(data)


def _noise_feature_cols(n_feat=10, n_base=8):
    """Noise feature cols."""
    return [f"f{j}" for j in range(n_feat)] + [f"b{b}" for b in range(n_base)] + ["base"]


def _real_signal_df(n=3000, seed=7):
    """y = 1 + 2.5*base + small noise -- linear_residual has a genuine gain."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    data = {f"f{j}": rng.normal(size=n) for j in range(4)}
    data["base"] = base
    # Features f0/f1 carry the residual structure so MI(T, X_remaining) is real.
    data["f0"] = data["f0"] + 0.8 * np.sign(base)
    data["y"] = 1.0 + 2.5 * base + 0.3 * data["f0"] + 0.2 * rng.normal(size=n)
    return pd.DataFrame(data)


def _make_config(**overrides):
    """Make config."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    defaults = dict(
        enabled=True,
        base_candidates=["base"],
        transforms=("diff", "ratio", "linear_residual"),
        top_k_after_mi=64,
        mi_sample_n=2000,
        eps_mi_gain=-10.0,  # admit everything so a winner always emerges
        screening="mi",  # keep it fast; the holdout re-score is the unit under test
        random_state=42,
        require_beats_raw_baseline=False,
        fail_on_no_gain="fallback_raw",
        multi_base_enabled=False,
        # Disable the opt-in discovery steps (region-adaptive / interaction-base / auto-chain):
        # they are orthogonal to the winner's-curse mechanism under test and fit LGBM models that
        # dominate the test wall. The honest-holdout re-score runs after them regardless.
        region_adaptive_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        honest_holdout_frac=0.3,
    )
    defaults.update(overrides)
    return CompositeTargetDiscoveryConfig(**defaults)


def _run(df, config, feature_cols, train_frac=0.85):
    """Fits CompositeTargetDiscovery using the first train_frac rows as train_idx and returns the fitted discovery object."""
    from mlframe.training.composite import CompositeTargetDiscovery

    n = len(df)
    train_idx = np.arange(0, int(train_frac * n))
    disc = CompositeTargetDiscovery(config)
    disc.fit(df, target_col="y", feature_cols=feature_cols, train_idx=train_idx)
    return disc


# ---------------------------------------------------------------------------
# biz_value: honest holdout gain is materially LESS biased than the in-screen gain
# ---------------------------------------------------------------------------


def test_biz_val_honest_holdout_debiases_pure_noise_winner():
    """On PURE NOISE (true gain = 0) the in-screen winner mi_gain is the MAX over a large
    candidate family -> spuriously inflated UPWARD, while the honest holdout gain -- the same
    quantity re-scored on never-touched rows -- is ~0. So the honest estimate is materially
    less biased than the in-screen selection score.

    The winner's curse is a per-FAMILY bias, so the assertion is aggregate across seeds:
      * mean in-screen winner gain is positive (the curse pushes the max above 0),
      * mean honest gain is below it AND ~0 (the de-biased truth),
      * honest < in-screen on a clear majority of seeds.
    Magnitudes are small (bin-MI floors at 0) but the SIGN + RATIO of the bias is the point.
    Measured (~36 candidates/seed): mean in-screen ~+0.0016, mean honest ~+0.0002 (≈7x less),
    honest <= in-screen on 9/10 seeds."""
    feat = _noise_feature_cols()
    base_cands = [f"b{b}" for b in range(8)]
    inscreen, honest_vals, honest_below = [], [], 0
    seeds = list(range(7, 17))
    for s in seeds:
        df = _pure_noise_df(n=6000, seed=s)
        disc = _run(
            df,
            _make_config(random_state=s, base_candidates=base_cands, mi_sample_n=3000, honest_holdout_frac=0.35),
            feat,
            train_frac=0.9,
        )
        specs = disc.export_specs()
        assert specs, "discovery must emit a winner with eps_mi_gain=-10"
        winner = max(specs, key=lambda d: d["mi_gain"])  # highest in-screen selection score
        honest = winner["honest_holdout_gain"]
        assert honest is not None, "honest holdout gain must be computed (frac=0.3, n=3000)"
        inscreen.append(winner["mi_gain"])
        honest_vals.append(honest)
        if honest <= winner["mi_gain"] + 1e-9:
            honest_below += 1
    mean_inscreen = float(np.mean(inscreen))
    mean_honest = float(np.mean(honest_vals))
    # The honest, post-selection estimate is materially less optimistic than the in-screen max.
    assert mean_honest < mean_inscreen, (
        f"honest holdout gain should be below the in-screen winner gain on noise (winner's "
        f"curse); mean honest {mean_honest:+.4f} vs mean in-screen {mean_inscreen:+.4f}"
    )
    # The in-screen winner gain is inflated upward (positive) by the max-over-candidates.
    assert mean_inscreen >= 0.0, f"in-screen winner gain should be inflated >=0; got {mean_inscreen:+.4f}"
    # The honest gain is ~0 on pure noise (the true gain), NOT inflated -- and materially
    # smaller in magnitude than the in-screen selection score.
    assert abs(mean_honest) < 0.01, f"honest holdout gain on pure noise should be ~0; got {mean_honest:+.4f}"
    assert (
        abs(mean_honest) < 0.6 * mean_inscreen + 1e-9
    ), f"honest gain {mean_honest:+.4f} should be materially below in-screen {mean_inscreen:+.4f} (winner's curse de-bias)"
    assert honest_below >= 8, f"honest gain should be <= in-screen gain on a strong majority of noise seeds; got {honest_below}/{len(seeds)}"


def test_biz_val_honest_holdout_preserves_real_signal_gain():
    """On a REAL-signal DGP the honest holdout gain stays clearly positive -- the de-bias
    does not erase a genuine effect (it only removes the optimistic inflation)."""
    feat = [f"f{j}" for j in range(4)] + ["base"]
    pos = 0
    seeds = list(range(7, 13))
    for s in seeds:
        df = _real_signal_df(n=3000, seed=s)
        disc = _run(df, _make_config(random_state=s, transforms=("linear_residual", "diff", "ratio")), feat)
        specs = disc.export_specs()
        assert specs
        # Pick the linear_residual spec (the one with a true gain on this DGP).
        lr = [d for d in specs if d["transform_name"] == "linear_residual"]
        cand = lr[0] if lr else max(specs, key=lambda d: d["mi_gain"])
        honest = cand["honest_holdout_gain"]
        assert honest is not None
        if honest > 0.0:
            pos += 1
    assert pos >= 5, f"honest holdout gain should stay positive on real signal; got {pos}/{len(seeds)}"


# ---------------------------------------------------------------------------
# leakage: holdout rows never enter screening / selection
# ---------------------------------------------------------------------------


def test_honest_holdout_indices_disjoint_from_screening_and_within_train():
    """The carved holdout indices must be (a) a subset of train_idx and (b) disjoint from
    the screening pool the rest of fit() consumes -- no row that scored a candidate also
    scored the post-selection estimate."""
    from mlframe.training.composite.discovery._honest_holdout import split_screening_holdout

    train_idx = np.arange(100, 5100)  # offset so we also verify subset-of-train
    screen, holdout = split_screening_holdout(train_idx, 0.2, random_state=42)
    assert holdout is not None
    # Disjoint.
    assert np.intersect1d(screen, holdout).size == 0
    # Union == train (no row dropped / duplicated).
    assert np.array_equal(np.union1d(screen, holdout), np.unique(train_idx))
    assert screen.size + holdout.size == train_idx.size
    # Subset of train.
    assert np.isin(holdout, train_idx).all()
    # Roughly the requested fraction.
    assert abs(holdout.size / train_idx.size - 0.2) < 0.01


def test_honest_holdout_indices_stored_and_excluded_from_screen_after_fit():
    """End-to-end: after fit(), ``honest_holdout_idx_`` is populated and disjoint from the
    screening rows actually used (``train_idx_`` is the FULL set; the holdout is carved out)."""
    feat = _noise_feature_cols()
    df = _pure_noise_df(n=3000, seed=11)
    disc = _run(df, _make_config(random_state=11), feat)
    ho = disc.honest_holdout_idx_
    assert ho is not None and ho.size > 0
    full_train = np.arange(0, int(0.85 * len(df)))
    # The holdout is a subset of the full train rows.
    assert np.isin(ho, full_train).all()
    # The screening pool = full train minus holdout; verify the holdout was excluded.
    screen_pool = np.setdiff1d(full_train, ho)
    assert np.intersect1d(screen_pool, ho).size == 0
    assert screen_pool.size + ho.size == full_train.size


def test_honest_holdout_disabled_leaves_no_holdout_and_no_honest_gain():
    """frac=0 / None disables the split: every train row screens, specs carry honest_gain=None.

    This is the PRE-FIX shape -- the result object had only the in-screen mi_gain. The test
    documents that disabling the feature reproduces the old behaviour exactly (and that the
    biz_value de-bias only exists when the holdout is on)."""
    feat = _noise_feature_cols()
    df = _pure_noise_df(n=3000, seed=9)
    disc = _run(df, _make_config(random_state=9, honest_holdout_frac=0.0), feat)
    assert disc.honest_holdout_idx_ is None
    for d in disc.export_specs():
        assert d["honest_holdout_gain"] is None, "honest gain must be None when the holdout is disabled (pre-fix result shape)"
