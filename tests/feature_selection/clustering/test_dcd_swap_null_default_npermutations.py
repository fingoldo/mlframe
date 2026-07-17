"""Regression guard for the DCD swap permutation-null sample-size bug
(audit 2026-06-03: dcd-core-1 / dcd-swap-null-1 / dcd-swap-null-2).

THE BUG
-------
The DCD swap permutation null reused the screening-confidence
``full_npermutations`` (MRMR default 3) as its draw count B. The swap
p-value is ``(n_exceed + 1) / (B + 1)``, so at B=3 the smallest attainable
p-value is ``(0 + 1) / (3 + 1) = 0.25``. Acceptance requires
``perm_p_value < swap_alpha`` (0.05), and ``0.25 < 0.05`` is arithmetically
impossible -- so EVERY aggregate swap was rejected, and the member-swap
branch (which rejects on ``member_p >= swap_alpha``, i.e. ``0.25 >= 0.05``,
always true) was equally dead. Under shipped defaults the entire supervised
swap subsystem committed zero swaps; the only tests that proved swaps fire
all overrode ``full_npermutations`` to 20/50.

THE FIX
-------
A dedicated ``dcd_swap_npermutations`` (default 199 -> min-p 0.005) governs
the swap null, decoupled from ``full_npermutations`` (which now only acts as
the on/off switch). ``evaluate_swap_candidate`` additionally auto-raises B to
``ceil(1/swap_alpha)`` so the null can never be structurally un-rejectable.

These tests assert a swap fires on the canonical redundancy cluster at the
DEFAULT ``full_npermutations`` (no override) -- the exact path that was dead
pre-fix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _three_dups_plus_strong_frame(n: int = 1500, seed: int = 0):
    """Strong unrelated col + 3 perfectly-correlated duplicates + 1 noise.

    Mirrors the Layer 41 Probe C fixture: at ``tau_cluster=0.5`` DCD anchors
    on the first duplicate and grows the cluster to anchor + 2 members, so
    ``dcd_cluster_size_threshold=2`` makes the cluster swap-eligible.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong": other,
            "dup_a": latent + 0.01 * rng.standard_normal(n),
            "dup_b": latent + 0.01 * rng.standard_normal(n),
            "dup_c": latent + 0.01 * rng.standard_normal(n),
            "noise_0": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


class TestSwapNullKnobExists:
    def test_dcd_swap_npermutations_in_signature_default_199(self):
        import inspect

        from mlframe.feature_selection.filters.mrmr import MRMR

        sig = inspect.signature(MRMR.__init__)
        assert "dcd_swap_npermutations" in sig.parameters, (
            "dcd_swap_npermutations must be a public MRMR constructor param so the swap null is tunable and decoupled from full_npermutations."
        )
        assert sig.parameters["dcd_swap_npermutations"].default == 199
        # store_params_in_object must persist it onto the instance.
        assert int(MRMR().dcd_swap_npermutations) == 199

    def test_clone_round_trips_the_knob(self):
        from sklearn.base import clone

        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(dcd_swap_npermutations=333)
        assert int(clone(m).dcd_swap_npermutations) == 333


class TestSwapFiresAtDefaultNpermutations:
    def test_swap_fires_at_default_full_npermutations(self):
        """The headline regression: with DEFAULT full_npermutations (=3) and
        the opt-in dcd_cluster_size_threshold=2, a swap must fire on the
        3-dup fixture. Pre-fix the B=3 null made this impossible (n_swaps=0).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            # NOTE: full_npermutations is NOT overridden -> the shipped default
            # (3) that previously made every swap un-acceptable.
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert int(m.dcd_["n_swaps"]) >= 1, (
            f"With the decoupled swap null, a swap must fire on the 3-dup "
            f"fixture at the DEFAULT full_npermutations; got "
            f"n_swaps={m.dcd_['n_swaps']}, swap_log={m.dcd_.get('swap_log')}. "
            f"If this is 0, the swap null is again tied to full_npermutations=3 "
            f"(min-p 0.25 >> swap_alpha 0.05)."
        )

    def test_tiny_swap_npermutations_is_auto_raised(self):
        """Backstop: even if a user pins dcd_swap_npermutations below the
        value needed to resolve swap_alpha, evaluate_swap_candidate auto-raises
        B to ceil(1/swap_alpha) so the null stays passable and the swap fires.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_npermutations=3,  # structurally un-passable without the guard
            dcd_swap_alpha=0.05,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert int(m.dcd_["n_swaps"]) >= 1, (
            f"The auto-raise backstop must lift B to ceil(1/0.05)=20 so a swap can fire even with dcd_swap_npermutations=3; got n_swaps={m.dcd_['n_swaps']}."
        )

    def test_full_npermutations_zero_still_skips_swap_null(self):
        """The on/off semantics are preserved: full_npermutations=0 means the
        caller opted out of every null, so the swap commits on the
        deterministic gate alone (B_eff=0). This guards against the fix
        accidentally forcing a null when none was requested.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=0,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        # Deterministic gate alone still accepts the denoising swap on perfect dups.
        assert int(m.dcd_["n_swaps"]) >= 1, (
            f"full_npermutations=0 must still allow swaps via the deterministic gate (no null requested); got n_swaps={m.dcd_['n_swaps']}."
        )
