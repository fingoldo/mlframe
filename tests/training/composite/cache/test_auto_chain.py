"""Unit + biz_value tests for the auto-chain discovery prototype.

``_auto_chain.discover_chains`` composes residual->tail-compress chains on the
fly and surfaces only those that beat BOTH single stages on y-scale OOF RMSE.

biz_value (the headline): on a synthetic whose generating process IS
residual-then-tail-compress (``y = alpha*base + z**3`` with ``z`` linear in the
features), the auto-discovered ``linres+{cbrt|sp}`` chain beats both the single
residual and the single unary on y-scale OOF RMSE across the MAJORITY of seeds.
A regression that breaks the chain machinery drops the win and trips the floor.

The unit tests pin: the chain Transform is well-formed + round-trips, MI-gain is
monotone-blind to the second stage (the documented reason MI cannot rank chains),
the "beats both singles" gate is enforced, and an unwinnable target yields [].
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._auto_chain import (
    ChainCandidate,
    build_chain_transform,
    discover_chains,
    _y_scale_cv_rmse,
)
from mlframe.training.composite.transforms import Transform, TRANSFORMS_REGISTRY


pytest.importorskip("lightgbm")


def _synth_chain_target(seed: int, n: int = 3000):
    """y = 2*base + z**3, z linear in two held-out features -> chain should win."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    z = 0.9 * f1 + 0.7 * f2 + 0.25 * rng.normal(size=n)
    y = 2.0 * base + z**3
    x_matrix = np.column_stack([f1, f2])
    return y, base, x_matrix


# ----------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------


def test_build_chain_transform_is_wellformed_and_roundtrips():
    """Build chain transform is wellformed and roundtrips."""
    tf = build_chain_transform("linear_residual", "cbrt")
    assert isinstance(tf, Transform)
    assert tf.name == "chain_linear_residual_cbrt"
    assert tf.requires_base is True
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    base = rng.normal(size=500)
    params = tf.fit(y, base)
    t = tf.forward(y, base, params)
    y_back = tf.inverse(t, base, params)
    # The chain is an exact bijection on its in-domain rows: forward then inverse
    # recovers y (both stages are invertible point maps).
    assert np.allclose(y_back, y, atol=1e-6)


def test_build_chain_transform_rejects_unknown_stage():
    """Build chain transform rejects unknown stage."""
    with pytest.raises(KeyError):
        build_chain_transform("not_a_residual", "cbrt")
    with pytest.raises(KeyError):
        build_chain_transform("linear_residual", "not_a_unary")


def test_mi_gain_is_monotone_blind_to_second_stage():
    """MI(chain) == MI(residual) under the bin estimator (cbrt is monotone).

    This is WHY MI-gain cannot rank tail-compression chains -- the documented
    reason the scorer is y-scale RMSE, not MI. If this ever stops holding the
    module docstring's central claim is wrong, so we pin it.
    """
    from mlframe.training.composite.discovery._auto_chain import _mi_gain_of
    from mlframe.training.composite.discovery.screening import _mi_to_target

    y, base, x = _synth_chain_target(0, n=2000)
    mi_y = _mi_to_target(x, y, n_neighbors=3, random_state=0, estimator="bin", nbins=16)
    kw = dict(y=y, base=base, x_matrix=x, mi_y=mi_y, mi_estimator="bin", mi_nbins=16, mi_n_neighbors=3, random_state=0)
    res_gain = _mi_gain_of(TRANSFORMS_REGISTRY["linear_residual"], **kw)
    chain_gain = _mi_gain_of(build_chain_transform("linear_residual", "cbrt"), **kw)
    assert np.isfinite(res_gain) and np.isfinite(chain_gain)
    assert abs(chain_gain - res_gain) < 1e-6, "binned MI must be invariant to the monotone cbrt second stage"


def test_discover_chains_returns_empty_when_no_chain_wins():
    """Pure-linear target: residual already optimal, no tail to compress.

    y = 2*base + linear(f) + light gaussian noise. The single residual cannot be
    improved by any tail compression -> discover_chains returns []. A high margin
    requirement makes the gate strict.
    """
    rng = np.random.default_rng(3)
    n = 3000
    base = rng.normal(size=n)
    f1 = rng.normal(size=n)
    y = 2.0 * base + 1.5 * f1 + 0.3 * rng.normal(size=n)
    x = f1.reshape(-1, 1)
    out = discover_chains(
        y=y,
        base=base,
        x_matrix=x,
        residual_names=["linear_residual"],
        unary_names=["cbrt", "yj", "sp"],
        min_rmse_margin=0.05,
        random_state=3,
    )
    assert out == []


def test_discover_chains_candidates_carry_fitted_params_and_beat_singles():
    """Discover chains candidates carry fitted params and beat singles."""
    y, base, x = _synth_chain_target(1)
    out = discover_chains(
        y=y,
        base=base,
        x_matrix=x,
        residual_names=["linear_residual"],
        unary_names=["cbrt", "yj", "sp"],
        random_state=1,
    )
    assert out, "expected at least one winning chain on a chain-shaped target"
    best = out[0]
    assert isinstance(best, ChainCandidate)
    assert best.fitted_params  # non-empty params usable downstream
    assert best.rmse < best.residual_rmse
    assert best.rmse < best.unary_rmse
    assert best.margin > 0.0
    # sorted best-first by ascending RMSE
    assert all(out[i].rmse <= out[i + 1].rmse for i in range(len(out) - 1))


def test_y_scale_cv_rmse_raw_baseline_finite():
    """Y scale cv rmse raw baseline finite."""
    y, base, x = _synth_chain_target(2, n=1000)
    raw_rmse, vf = _y_scale_cv_rmse(
        None,
        y=y,
        base=base,
        x_matrix=x,
        cv_folds=4,
        random_state=2,
        family="lgb",
        n_estimators=40,
        num_leaves=15,
        learning_rate=0.1,
    )
    assert np.isfinite(raw_rmse) and raw_rmse > 0
    assert vf == 1.0


# ----------------------------------------------------------------------
# biz_value: chain beats both singles on the MAJORITY of seeds
# ----------------------------------------------------------------------


def test_biz_val_auto_chain_beats_both_singles_majority_of_seeds():
    """Floor: chain beats BOTH singles on >=6 of 8 seeds (measured 8/8).

    The 8/8 measured win has comfortable headroom; the >=6 floor absorbs seed
    noise while still failing hard if the chain machinery regresses (a broken
    chain drops to 0 wins). This is the production-correctness lock for the
    auto-chaining feature.
    """
    wins = 0
    margins = []
    for seed in range(8):
        y, base, x = _synth_chain_target(seed)
        out = discover_chains(
            y=y,
            base=base,
            x_matrix=x,
            residual_names=["linear_residual"],
            unary_names=["cbrt", "yj", "sp"],
            random_state=seed,
        )
        if out:
            best = out[0]
            # strictly beats both singles by construction of the gate
            assert best.rmse < best.residual_rmse
            assert best.rmse < best.unary_rmse
            wins += 1
            margins.append(best.margin)
    assert wins >= 6, f"chain should beat both singles on >=6/8 seeds, got {wins}"
    assert np.mean(margins) > 0.02, f"mean RMSE margin over best single too small: {np.mean(margins):.4f}"


def test_biz_val_chain_also_beats_raw_y():
    """The winning chain must also beat untransformed raw-y OOF RMSE.

    A target that needs the chain is, by construction, hard for a model on raw y;
    the chain's whole value is making it learnable. Pin that the chain RMSE is
    materially below raw_rmse (measured ~2.47 vs ~3.27, a >20% reduction).
    """
    y, base, x = _synth_chain_target(4)
    out = discover_chains(
        y=y,
        base=base,
        x_matrix=x,
        residual_names=["linear_residual"],
        unary_names=["cbrt", "yj", "sp"],
        random_state=4,
    )
    assert out
    best = out[0]
    assert best.rmse < 0.9 * best.raw_rmse, f"chain rmse {best.rmse:.3f} should be <0.9x raw {best.raw_rmse:.3f}"
