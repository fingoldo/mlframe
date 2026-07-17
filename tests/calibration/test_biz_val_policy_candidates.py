"""biz_value test for the ``candidates`` knob of ``pick_best_calibrator`` (calibration/policy.py).

``candidates`` restricts the calibrator family the policy considers. It is decision-influencing: the chosen
calibrator is the one whose held-out ECE is lowest WITHIN the restricted set, and that calibrator's transform is
what downstream consumers apply to the reported probabilities -> different ``candidates`` -> different probabilities.

The measurable claims locked here:
1. The restriction is HONORED: with ``candidates=("Sigmoid",)`` the chosen method is ``Sigmoid`` and the alternatives
   pool contains only restricted names (never Isotonic/Beta/Spline).
2. On a genuinely sigmoid-distorted miscalibration (the Platt/Sigmoid family's natural domain), restricting to the
   matched ``Sigmoid`` family delivers a low held-out ECE that is NOT materially worse than letting the full,
   over-flexible candidate set compete -- so a deployment that pins the matched family pays no honest-ECE penalty.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.fast]


def _make_sigmoid_miscalibrated(n: int, seed: int = 7):
    """Helper that make sigmoid miscalibrated."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=n)
    # True prob is a steeper sigmoid of raw -> the distortion is exactly what a Sigmoid/Platt calibrator inverts.
    true_p = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.int64)
    return raw, y


def test_biz_val_policy_candidates_restriction_is_honored():
    """``candidates=("Sigmoid",)`` must force the chosen method to Sigmoid and never surface a non-restricted
    alternative. Catches a regression that ignores the restriction and benches the full pool."""
    from mlframe.calibration.policy import pick_best_calibrator

    raw, y = _make_sigmoid_miscalibrated(n=2000, seed=11)
    out = pick_best_calibrator(
        probs=None,
        y=None,
        oof_probs=raw,
        oof_y=y,
        candidates=("Sigmoid",),
        n_bootstrap=200,
        random_state=11,
        selection="same_oof",
    )
    assert out["chosen"] == "Sigmoid", out
    allowed = {"Sigmoid"}
    assert set(out["alternatives"]).issubset(allowed), out["alternatives"]


def test_biz_val_policy_candidates_matched_family_no_ece_penalty():
    """On a sigmoid-distorted target, pinning ``candidates=("Sigmoid",)`` must not yield a materially worse honest
    held-out ECE than the unrestricted full-pool pick. The DELTA vs the full pool is the locked quantity."""
    from mlframe.calibration.policy import pick_best_calibrator

    raw, y = _make_sigmoid_miscalibrated(n=2000, seed=11)
    common = dict(probs=None, y=None, oof_probs=raw, oof_y=y, n_bootstrap=200, random_state=11, selection="inner_cv")

    restricted = pick_best_calibrator(candidates=("Sigmoid",), **common)
    full = pick_best_calibrator(candidates=None, **common)

    ece_restricted = float(restricted["ece_mean"])
    ece_full = float(full["ece_mean"])
    assert np.isfinite(ece_restricted) and ece_restricted >= 0.0

    # The matched Sigmoid family should be within a small absolute ECE of the best the full pool can do; it must not
    # be materially worse (the distortion is in Sigmoid's natural domain). Floor absorbs bootstrap/inner-CV noise.
    assert ece_restricted <= ece_full + 0.02, f"restricted Sigmoid ECE {ece_restricted:.4f} should be within 0.02 of full-pool ECE {ece_full:.4f}"
