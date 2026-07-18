"""Regression tests for the P13 bit-identical gather-reuse in
``mlframe.training.composite.discovery._eval.eval_one_transform``.

P13 (2026-06-11) hoisted the ``y_train[valid]`` / ``base_train[valid]`` gather
so the SAME arrays feed both ``transform.fit`` and the residual-std probe's
``transform.forward``, instead of re-fancy-indexing the rows a second time. The
subtlety: the T15 fitted-domain refinement can NARROW ``valid`` AFTER the fit,
so the cached gather goes stale and must be re-taken on the narrowed mask. A
careless implementation that reused the STALE (pre-T15) gather would feed the
residual-std probe the wrong rows -> wrong ``T_std`` / ``y_std`` ratio -> a
different reject/keep decision and a wrong ``n_train_rows``.

These tests pin BOTH sides:

* ``_valid_stale=False`` (no fitted-domain hook): the probe ``forward`` sees
  exactly ``valid.sum()`` rows and the spec's ``n_train_rows`` matches.
* ``_valid_stale=True`` (a ``domain_check_fitted`` hook that drops rows): the
  probe ``forward`` sees the NARROWED count, NOT the pre-T15 count, and
  ``n_train_rows`` reflects the narrowed mask.

The stale-gather bug is decisively caught because the recorded probe-forward
length would equal the pre-T15 count instead of the post-T15 count.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._eval import eval_one_transform
from mlframe.training.composite.discovery.screening import (
    _prebin_feature_columns,
)
from mlframe.training.composite.transforms import Transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class _StubDiscovery:
    """Minimal stand-in exposing just what ``eval_one_transform`` reads:
    ``self.config`` and ``self._reject``."""

    def __init__(self, config):
        self.config = config

    def _reject(self, base, transform_name, mi_y, valid_frac, reason):
        """Reject."""
        return {
            "spec": None,
            "kept": False,
            "rejected": True,
            "base": base,
            "transform_name": transform_name,
            "valid_domain_frac": valid_frac,
            "mi_y": mi_y,
            "reason": reason,
        }


def _make_config():
    """Make config."""
    return CompositeTargetDiscoveryConfig(
        enabled=True,
        mi_estimator="bin",
        mi_nbins=8,
        mi_aggregation="mean",
        min_valid_domain_frac=0.5,
        mi_gain_bootstrap_n=0,
    )


def _build_context(n, n_feat, nbins, rng):
    """Build a ``base_contexts[base]`` dict matching the ``_fit.py`` contract."""
    x_remaining = rng.normal(size=(n, n_feat)).astype(np.float64)
    x_prebinned = _prebin_feature_columns(x_remaining, nbins=nbins)
    base_train = rng.normal(size=n).astype(np.float64)
    return {
        "base_train": base_train,
        "base_screen": base_train,  # screen == train sample here
        "x_remaining_matrix": x_remaining,
        "_x_prebinned": x_prebinned,
        "mi_y_for_base": 0.0,
        "_mi_kwargs": {"nbins": nbins, "aggregation": "mean"},
    }


def _make_recording_transform(*, with_fitted_hook, drop_tail_frac=0.0):
    """A simple additive transform ``T = y - base`` that records every array
    length passed to ``forward``. Optionally exposes a ``domain_check_fitted``
    that drops the trailing ``drop_tail_frac`` of rows so the T15 narrowing
    path fires (``_valid_stale=True``)."""
    forward_lengths: list[int] = []

    def _domain_check(y, base):
        """Domain check."""
        return np.isfinite(y) & np.isfinite(base)

    def _fit(y, base):
        # Trivial linear-residual-like fit; non-empty params dict.
        """Fit."""
        return {"alpha": 1.0, "beta": float(np.mean(y - base))}

    def _forward(y, base, params):
        """Forward."""
        forward_lengths.append(int(np.asarray(y).shape[0]))
        return np.asarray(y, dtype=np.float64) - np.asarray(base, dtype=np.float64) - params["beta"]

    def _inverse(t, base, params):
        """Inverse."""
        return np.asarray(t) + np.asarray(base) + params["beta"]

    domain_fitted = None
    if with_fitted_hook:

        def _domain_fitted(y, base, params):
            """Domain fitted."""
            y = np.asarray(y)
            mask = np.ones(y.shape[0], dtype=bool)
            if drop_tail_frac > 0.0:
                k = int(y.shape[0] * drop_tail_frac)
                if k > 0:
                    mask[-k:] = False
            return mask

        domain_fitted = _domain_fitted

    t = Transform(
        name="stub_resid",
        forward=_forward,
        inverse=_inverse,
        fit=_fit,
        domain_check=_domain_check,
        description="stub",
        domain_check_fitted=domain_fitted,
        requires_base=True,
    )
    return t, forward_lengths


def test_p13_probe_forward_sees_full_valid_when_not_stale():
    """``_valid_stale=False`` path: with no fitted-domain hook, the residual-std
    probe forwards over exactly ``valid.sum()`` rows and the spec records that
    count as ``n_train_rows``."""
    rng = np.random.default_rng(0)
    n, n_feat, nbins = 400, 4, 8
    ctx = _build_context(n, n_feat, nbins, rng)
    y_train = rng.normal(size=n).astype(np.float64) * 5.0 + 100.0
    y_screen = y_train  # screen == train
    transform, fwd_lengths = _make_recording_transform(with_fitted_hook=False)

    disc = _StubDiscovery(_make_config())
    out = eval_one_transform(
        disc,
        "lag1",
        "stub_resid",
        transform,
        base_contexts={"lag1": ctx},
        y_train=y_train,
        y_screen=y_screen,
        target_col="y",
    )
    assert len(out) == 1
    spec = out[0]["spec"]
    assert spec is not None, f"unexpected reject: {out[0].get('reason')}"
    # First forward call is the residual-std probe over y_train[valid]; all rows
    # are finite so valid.sum() == n.
    assert fwd_lengths, "probe forward never called"
    assert fwd_lengths[0] == n, f"probe forwarded {fwd_lengths[0]} rows, expected the full valid count {n}"
    assert spec.n_train_rows == n


def test_p13_probe_forward_uses_narrowed_valid_when_stale():
    """``_valid_stale=True`` path: a ``domain_check_fitted`` that drops the
    trailing 30% of rows must NARROW the probe's forward input AND
    ``n_train_rows``. A stale-gather reuse would forward the full n rows."""
    rng = np.random.default_rng(1)
    n, n_feat, nbins = 400, 4, 8
    ctx = _build_context(n, n_feat, nbins, rng)
    y_train = rng.normal(size=n).astype(np.float64) * 5.0 + 100.0
    y_screen = y_train
    transform, fwd_lengths = _make_recording_transform(
        with_fitted_hook=True,
        drop_tail_frac=0.30,
    )

    disc = _StubDiscovery(_make_config())
    out = eval_one_transform(
        disc,
        "lag1",
        "stub_resid",
        transform,
        base_contexts={"lag1": ctx},
        y_train=y_train,
        y_screen=y_screen,
        target_col="y",
    )
    assert len(out) == 1
    spec = out[0]["spec"]
    assert spec is not None, f"unexpected reject: {out[0].get('reason')}"
    expected_narrowed = n - int(n * 0.30)  # 280
    # Decisive: the probe forward (first forward call) must see the NARROWED
    # count, not the pre-T15 full count. A stale-gather reuse would record n.
    assert fwd_lengths, "probe forward never called"
    assert fwd_lengths[0] == expected_narrowed, (
        f"probe forwarded {fwd_lengths[0]} rows; expected the narrowed valid "
        f"count {expected_narrowed} (n={n}). A value of {n} means the stale "
        f"pre-T15 gather was reused (P13 regression)."
    )
    assert spec.n_train_rows == expected_narrowed


def test_p13_outcome_is_deterministic_across_repeats():
    """The gather-reuse must not introduce any nondeterminism: identical inputs
    -> identical spec mi values across repeated calls."""
    rng = np.random.default_rng(2)
    n, n_feat, nbins = 500, 5, 8
    y_train = rng.normal(size=n).astype(np.float64) * 3.0 + 50.0
    y_screen = y_train

    results = []
    for _ in range(3):
        ctx = _build_context(n, n_feat, nbins, np.random.default_rng(7))
        transform, _ = _make_recording_transform(with_fitted_hook=False)
        disc = _StubDiscovery(_make_config())
        out = eval_one_transform(
            disc,
            "lag1",
            "stub_resid",
            transform,
            base_contexts={"lag1": ctx},
            y_train=y_train,
            y_screen=y_screen,
            target_col="y",
        )
        spec = out[0]["spec"]
        assert spec is not None
        results.append((spec.mi_t, spec.mi_y, spec.mi_gain, spec.n_train_rows))
    assert results[0] == results[1] == results[2], results


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
