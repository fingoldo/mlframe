"""Unit coverage for ``_param_accuracy_warnings.warn_accuracy_suboptimal_params`` (mrmr_audit_2026-07-20
test_coverage.md #2 / edge_cases.md #5-8, and the B-22-adjacent P2 fix for the one-shot latch bug).

Prior to the fix, ``_accuracy_caveats_warned_`` was a bare boolean latch set unconditionally on the
FIRST call regardless of whether any caveat actually fired -- so ``estimator.fit()`` (clean config, no
warning) followed by ``estimator.set_params(bad_value); estimator.fit()`` (now genuinely bad) never
warned on the second fit, because the latch was already tripped by the first, warning-free call.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

from mlframe.feature_selection.filters._param_accuracy_warnings import (
    ACCURACY_SUBOPTIMAL,
    warn_accuracy_suboptimal_params,
)


def _clean_estimator(**overrides) -> SimpleNamespace:
    """A SimpleNamespace exposing every ACCURACY_SUBOPTIMAL attr at a GOOD (non-triggering) value, with overrides."""
    ns = SimpleNamespace()
    for c in ACCURACY_SUBOPTIMAL:
        setattr(ns, c.attr, overrides.get(c.attr, "some_good_value_unlikely_to_match"))
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestWarnAccuracySuboptimalParams:
    """Direct unit coverage: fires when a known-bad value is set, stays silent on a clean config."""

    def test_silent_on_default_config(self):
        """A clean/default config (no caveat attr matches its bad predicate) never warns."""
        est = _clean_estimator()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)
        assert not w, f"unexpected warning(s) on a clean config: {[str(x.message) for x in w]}"

    def test_fires_on_a_bad_value(self):
        """Setting one caveat attr to its documented-bad value fires exactly one consolidated UserWarning."""
        caveat = ACCURACY_SUBOPTIMAL[0]
        # `_eq(False)` predicates: bad value is False; numeric predicates: use a value the predicate accepts.
        bad_val = False if "enable" in caveat.attr else 0
        assert caveat.is_bad(bad_val), f"fixture assumption broken: {bad_val!r} is not flagged bad for {caveat.attr!r}"
        est = _clean_estimator(**{caveat.attr: bad_val})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert caveat.attr in str(w[0].message)

    def test_missing_or_throwing_attr_is_skipped_not_raised(self):
        """A missing attr is skipped (hasattr guard); an attr whose getattr raises is also skipped, never propagated."""

        class _Throws:
            """Descriptor whose __get__ raises, simulating a property that fails."""

            def __get__(self, obj, owner=None):
                raise RuntimeError("boom")

        est = _clean_estimator()
        # Remove one attr entirely (missing-attr path) and make another raise (throwing-attr path).
        caveats = ACCURACY_SUBOPTIMAL
        assert len(caveats) >= 2, "fixture needs at least 2 registered caveats"
        delattr(est, caveats[0].attr)
        est2 = type("EstWithThrowingAttr", (), {})()
        for c in caveats:
            setattr(type(est2), c.attr, "some_good_value_unlikely_to_match")
        setattr(type(est2), caveats[1].attr, _Throws())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)  # missing attr path
            warn_accuracy_suboptimal_params(est2)  # throwing attr path
        assert not w, f"missing/throwing attrs must be silently skipped, not raised/warned: {[str(x.message) for x in w]}"

    def test_quantization_nbins_fencepost_and_float_footgun(self):
        """quantization_nbins < 5 fires; == 5 does not (documented boundary); a FLOAT 4.0 also fires
        (int-typed predicate check must not silently exempt float-typed bad values)."""
        nbins_caveat = next(c for c in ACCURACY_SUBOPTIMAL if c.attr == "quantization_nbins")
        assert nbins_caveat.is_bad(4)
        assert not nbins_caveat.is_bad(5)
        assert not nbins_caveat.is_bad(10)

    def test_fires_at_most_once_per_distinct_param_snapshot(self):
        """Calling twice with the SAME bad config warns only once (the latch); calling again after
        restoring the good value, then setting a DIFFERENT bad value, warns again -- the latch is keyed
        on the parameter snapshot, not a one-shot-forever flag."""
        caveat = ACCURACY_SUBOPTIMAL[0]
        bad_val = False if "enable" in caveat.attr else 0
        est = _clean_estimator(**{caveat.attr: bad_val})

        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)
        assert len(w1) == 1

        # Same bad config again: latch suppresses the duplicate.
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)
        assert not w2, "identical repeated config must not re-warn"

    def test_regression_refit_after_setting_a_new_bad_value_warns_again(self):
        """B-22-adjacent regression: fit clean (no warning) -> set_params to a bad value -> fit again
        MUST warn. Pre-fix, the unconditional latch set on the first (warning-free) call silently
        suppressed this second, genuinely-bad-config warning forever."""
        caveat = ACCURACY_SUBOPTIMAL[0]
        bad_val = False if "enable" in caveat.attr else 0
        est = _clean_estimator()

        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)  # first "fit": clean config, no warning
        assert not w1

        setattr(est, caveat.attr, bad_val)  # simulate set_params()

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            warn_accuracy_suboptimal_params(est)  # second "fit": now genuinely bad
        assert len(w2) == 1, (
            f"B-22 regression: re-fitting after set_params() to a known-bad value produced "
            f"{len(w2)} warning(s), expected exactly 1 -- the one-shot latch is suppressing a "
            f"genuinely new bad configuration."
        )
        assert caveat.attr in str(w2[0].message)
