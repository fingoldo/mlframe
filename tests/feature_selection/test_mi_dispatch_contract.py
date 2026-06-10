"""Contract tests for ``score_pair_mi`` -- the single documented entry point routing to all 9 MI estimators.

Finding param_axes-04: every estimator is biz_value-tested through its standalone module, but the routing seam itself
(``estimator`` dispatch, ``estimator_kwargs`` forwarding, the float64 coercion, the unknown-id ValueError) is never
exercised by any test. A typo in one routing branch (wrong fn in ``fn_map``, dropped kwargs) would pass the whole suite.

These are behavioral assertions only: each estimator is driven through the public ``score_pair_mi`` seam on a signal pair
and a noise pair, and the dispatched result is checked to separate them. Floors are calibrated 5-15% below a measured run
(see ``audit/fs_tests_audit_2026_06_10/param_axes-04`` recipe): at seed 0 / n=800 the measured signal-minus-noise deltas are
plug_in 0.489, mixed_ksg 0.479, ksg_lnc 0.479, fastmi 0.325, median 0.486, genie 0.480 -- all robust across seeds 0-4
(min over seeds: fastmi 0.324, the rest >=0.467). The proposal's +0.05 floor therefore carries large margin for the binned /
k-NN / panel estimators; fastmi (copula FFT-KDE) is the smallest and gets the same +0.05 floor (still ~0.27 of headroom).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_dispatch import score_pair_mi
from tests.feature_selection.conftest import fast_subset


# ---------------------------------------------------------------------------
# Estimator inventory. Fast ids run unconditionally; neural ids are slow + need
# torch (+ a downloaded checkpoint for infonet / the mist-statinf package for mist).
# ---------------------------------------------------------------------------

FAST_ESTIMATORS = ["plug_in", "mixed_ksg", "ksg_lnc", "fastmi", "median", "genie"]
NEURAL_ESTIMATORS = ["mine", "infonet", "mist"]

# Calibrated floor: +0.05 above the noise MI (per param_axes-04 recipe). Measured smallest fast delta is fastmi ~0.324 at
# seed 0, so a 0.05 separation floor is ~6.5x below the measured value -- it trips on a broken routing branch but never on
# seed noise. Neural estimators that actually run (mine) clear ~0.45; the same floor applies.
SIGNAL_NOISE_FLOOR = 0.05

# A single fixed seed (proposal uses default_rng(0)) keeps every dispatched MI deterministic, so the fixed floor is exact.
_SEED = 0
_N = 800


def _signal_noise_pair():
    """Return ``(x, y_sig, y_noise)``: x continuous; y_sig a thresholded copy of x (strong MI); y_noise independent."""
    rng = np.random.default_rng(_SEED)
    x = rng.normal(size=_N)
    y_sig = (x + 0.3 * rng.normal(size=_N) > 0).astype(np.int64)
    y_noise = rng.integers(0, 2, _N)
    return x, y_sig, y_noise


def _assert_signal_exceeds_noise(estimator: str):
    """Drive ``estimator`` through the public dispatcher on a signal pair and a noise pair; pin the separation floor."""
    x, y_sig, y_noise = _signal_noise_pair()
    mi_s = score_pair_mi(x, y_sig, estimator=estimator)
    mi_n = score_pair_mi(x, y_noise, estimator=estimator)
    assert np.isfinite(mi_s), f"{estimator}: signal MI must be finite, got {mi_s!r}"
    assert np.isfinite(mi_n), f"{estimator}: noise MI must be finite, got {mi_n!r}"
    assert mi_s >= 0.0, f"{estimator}: MI is clamped at 0 (nats), signal MI was {mi_s}"
    assert mi_n >= 0.0, f"{estimator}: MI is clamped at 0 (nats), noise MI was {mi_n}"
    assert mi_s > mi_n + SIGNAL_NOISE_FLOOR, (
        f"{estimator}: dispatched signal MI {mi_s:.4f} must exceed noise MI {mi_n:.4f} by > {SIGNAL_NOISE_FLOOR} "
        f"(delta {mi_s - mi_n:.4f}); a routing typo or dropped kwargs collapses this gap"
    )


@pytest.mark.parametrize("estimator", fast_subset(FAST_ESTIMATORS, n=1))
def test_score_pair_mi_signal_exceeds_noise_fast(estimator):
    """Each fast estimator routed through ``score_pair_mi`` separates a signal pair from a noise pair.

    ``fast_subset`` keeps one representative (plug_in) under ``MLFRAME_FAST=1`` so the routing seam stays covered cheaply.
    """
    _assert_signal_exceeds_noise(estimator)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", NEURAL_ESTIMATORS)
def test_score_pair_mi_signal_exceeds_noise_neural(estimator):
    """Neural estimators (mine / infonet / mist) routed through ``score_pair_mi`` separate signal from noise.

    Guarded by torch + slow marker. ``infonet`` needs a downloaded checkpoint and ``mist`` needs the ``mist-statinf``
    package; when those optional assets are absent the routing layer raises a clear RuntimeError / ImportError, which we
    treat as a legitimate environment skip (the ``importorskip`` pattern), NOT a reason to weaken the win assertion.
    """
    pytest.importorskip("torch")
    try:
        _assert_signal_exceeds_noise(estimator)
    except (ImportError, RuntimeError, FileNotFoundError, OSError) as exc:
        msg = str(exc).lower()
        if any(tok in msg for tok in ("checkpoint", "not installed", "not found", "download", "no module")):
            pytest.skip(f"{estimator}: optional neural asset/package unavailable in this env -- {exc}")
        raise


def test_unknown_estimator_raises():
    """A bad estimator id is rejected by the dispatcher with a ValueError naming the unknown estimator."""
    rng = np.random.default_rng(_SEED)
    x = rng.normal(size=200)
    y = rng.integers(0, 2, 200)
    with pytest.raises(ValueError, match="unknown estimator"):
        score_pair_mi(x, y, estimator="does_not_exist")


def test_estimator_kwargs_forwarded_to_routed_fn():
    """``estimator_kwargs`` reach the routed estimator: mixed_ksg with k=3 vs k=15 yields different finite MI values.

    If forwarding were dropped, both calls would use the routed fn's default k=5 and return the identical value. The two
    k settings produce a measured ~0.012 gap at seed 0 (k3 0.4868 vs k15 0.4991); we only require they differ and stay
    finite, which is enough to prove the kwargs crossed the dispatch boundary.
    """
    x, y_sig, _ = _signal_noise_pair()
    mi_k3 = score_pair_mi(x, y_sig, estimator="mixed_ksg", estimator_kwargs={"k": 3})
    mi_k15 = score_pair_mi(x, y_sig, estimator="mixed_ksg", estimator_kwargs={"k": 15})
    assert np.isfinite(mi_k3), f"k=3 MI must be finite, got {mi_k3!r}"
    assert np.isfinite(mi_k15), f"k=15 MI must be finite, got {mi_k15!r}"
    assert mi_k3 != mi_k15, (
        f"estimator_kwargs not forwarded: mixed_ksg k=3 ({mi_k3}) and k=15 ({mi_k15}) returned the same value, "
        "so k never reached the routed fn"
    )
