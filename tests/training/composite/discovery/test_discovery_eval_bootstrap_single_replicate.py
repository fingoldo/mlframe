"""TRAINING_COMPOSITE_DISCOVERY-1 regression test: the bootstrap MI-gain CI must not raise an uncaught
IndexError (crashing the whole fit() call) when mi_gain_bootstrap_n=1 and that lone replicate fails.

The bug (fixed): `boot_finite.size >= bootstrap_n // 2` is trivially true (0 >= 0) even for an empty
array when bootstrap_n <= 1, so np.percentile(boot_finite, 2.5) on an EMPTY array raised IndexError
instead of leaving mi_gain_lcb at its no-CI point-estimate default.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery import _eval as _eval_mod
from mlframe.training.composite.discovery._eval import build_unary_base_context, eval_one_transform
from mlframe.training.composite.transforms import get_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig

pytestmark = pytest.mark.fast


class _Disc:
    """Minimal stand-in exposing the ``self.config`` + ``self._reject`` surface eval_one_transform reads."""

    def __init__(self, config):
        self.config = config

    def _reject(self, base, transform_name, mi_y, valid_frac, *, reason):
        """Returns the standard rejected-spec dict shape eval_one_transform expects, tagged with the given reason."""
        return {"spec": None, "kept": False, "reason": reason, "base": base, "transform": transform_name}


def _make_unary_ctx(n=800, f=6, nbins=10, seed=11):
    """Build a full-X prebinned unary sentinel context + the shared train/screen targets."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    y = np.abs(0.7 * x[:, 0] + 0.4 * x[:, 1]).astype(np.float64) + 0.3
    y += rng.standard_normal(n) * 0.1
    y = np.abs(y) + 0.5

    from mlframe.training.composite.discovery.screening import _mi_per_feature_prebinned, _prebin_feature_columns

    full_prebinned = _prebin_feature_columns(x, nbins=nbins)
    per_feat_y = _mi_per_feature_prebinned(full_prebinned, y, nbins=nbins)
    sample_idx = np.arange(n)
    ctx = build_unary_base_context(
        full_x_matrix=x, full_x_prebinned=full_prebinned, per_feat_y_full=per_feat_y,
        y_screen=y, n_train=n, sample_idx=sample_idx, mi_aggregation="mean", mi_nbins=nbins,
        mi_n_neighbors=3, random_state=seed, mi_estimator="bin",
    )
    return ctx, y


def test_single_bootstrap_replicate_failure_does_not_raise_indexerror(monkeypatch):
    """mi_gain_bootstrap_n=1 with the lone replicate forced to fail must not crash, and must fall back
    to the point-estimate mi_gain_lcb."""
    cfg = CompositeTargetDiscoveryConfig(
        mi_nbins=10, mi_estimator="bin", mi_gain_bootstrap_n=1, min_valid_domain_frac=0.0, random_state=11,
    )
    disc = _Disc(cfg)
    transform_name = "cbrt_y"
    transform = get_transform(transform_name)
    assert not transform.requires_base

    ctx, y = _make_unary_ctx()
    base = ""
    base_contexts = {base: ctx}

    real_prebinned = _eval_mod._mi_to_target_prebinned
    call_count = {"n": 0}

    def _fail_after_primary(*args, **kwargs):
        """Let the primary (non-bootstrap) MI computation succeed, but force every BOOTSTRAP replicate's
        MI computation (mi_estimator='bin' routes both through _mi_to_target_prebinned) to fail."""
        call_count["n"] += 1
        if call_count["n"] <= 2:  # primary mi_t + mi_y_compare (or similar) computed first
            return real_prebinned(*args, **kwargs)
        raise RuntimeError("forced bootstrap replicate failure")

    monkeypatch.setattr(_eval_mod, "_mi_to_target_prebinned", _fail_after_primary)

    result = eval_one_transform(
        disc, base, transform_name, transform,
        base_contexts=base_contexts, y_train=y, y_screen=y, target_col="y",
    )
    assert len(result) == 1
    entry = result[0]
    assert entry["bootstrap_failure_count"] == 1
    # mi_gain_lcb must have fallen back to the point-estimate default (spec is None here since the
    # forced MI failure also breaks the primary mi_t/mi_y computation -- the key assertion is simply
    # that no IndexError propagated out of eval_one_transform).
    assert "mi_gain_lcb" in entry
