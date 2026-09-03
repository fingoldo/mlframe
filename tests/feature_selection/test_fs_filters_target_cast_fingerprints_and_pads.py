"""Ten defects across the filters cluster, mostly of one shape: a shortcut that is right in one regime and
silently wrong in another, with nothing in the output to tell them apart.

The sharpest is the target cast. Three public entry points in `_conditional_gate_fe` did
`np.asarray(y).astype(np.int64)` -- the continuous-target truncation trap `_y_encoding.py` exists to prevent.
On a regression target in [0, 1) that collapses every row to class 0, so every MI reads exactly 0.0, the gate
never fires, and the family emits zero features with no error. On a target in [0, 10) it is worse: MI is
non-zero but measures a truncated target.

Then two cache-identity defects that end in `_fit_identity_shortcut` -- "select everything" for a frame that was
never scored -- one from a 10-cell content sample that an outlier clip preserves, one from an `id()`-derived key
that CPython hands out again after a collection. Two additive epsilons in denominators that are then inverted
and cubed. A "registered" flag set before the registration was attempted. A float32 sort key with thousands of
ties per row. And a deadline check that no configuration could reach.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestAContinuousTargetIsNotTruncatedToClasses:
    """`astype(np.int64)` on a target in [0, 1) makes every MI in the module exactly 0.0."""

    def _module(self):
        """The conditional-gate FE family."""
        from mlframe.feature_selection.filters import _conditional_gate_fe

        return _conditional_gate_fe

    def test_no_bare_int64_target_cast_survives(self):
        """The three module-boundary casts the finding names."""
        import inspect
        import re

        src = inspect.getsource(self._module())
        offenders = re.findall(r"^\s*yi = np\.asarray\(y\)\.astype\(np\.int64\)\s*$", src, re.M)
        assert offenders == [], offenders

    def test_the_encoder_is_used_instead(self):
        """Idempotent on already-dense integer codes, so the classification path is unaffected."""
        import inspect

        src = inspect.getsource(self._module())
        assert src.count("yi = encode_y_for_classif_mi(y)") == 3, src.count("yi = encode_y_for_classif_mi(y)")

    def test_a_unit_interval_target_does_not_collapse_to_one_class(self):
        """The concrete consequence: `astype(np.int64)` leaves exactly one distinct class, so MI is 0.0."""
        from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

        rng = np.random.default_rng(0)
        y = rng.random(2000)  # log-returns, a normalised label, a probability
        assert len(np.unique(np.asarray(y).astype(np.int64))) == 1, "the fixture no longer reproduces the trap"
        assert len(np.unique(encode_y_for_classif_mi(y))) > 1

    def test_an_integer_target_is_passed_through_unchanged(self):
        """The encoder must not disturb the classification path the module was already right about."""
        from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

        y = np.array([0, 1, 2, 1, 0, 2] * 100)
        assert sorted(np.unique(encode_y_for_classif_mi(y)).tolist()) == [0, 1, 2]


class TestTheIdentityCacheCannotCollideOrGuess:
    """A hit makes `_fit_identity_shortcut` return `support_ = arange(n_cols)` for an unscored frame."""

    def _fp(self, X):
        """The X-side identity fingerprint."""
        from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_x_fingerprint

        return _mrmr_compute_x_fingerprint(X)

    def _frames(self):
        """A frame and its outlier-clipped variant -- identical schema, identical boundary cells, different content."""
        import pandas as pd

        rng = np.random.default_rng(0)
        base = pd.DataFrame(rng.normal(0, 1, size=(5000, 4)), columns=list("abcd"))
        clipped = base.clip(lower=-1.5, upper=1.5)
        return base, clipped

    def test_an_outlier_clipped_variant_gets_a_different_fingerprint(self):
        """The canonical preprocessing-sweep case the 10-cell sample could not tell apart."""
        base, clipped = self._frames()
        assert self._fp(base) != self._fp(clipped)

    def test_the_same_frame_still_fingerprints_the_same(self):
        """A cache that never hits is as useless as one that hits wrongly."""
        base, _ = self._frames()
        assert self._fp(base) == self._fp(base.copy())

    def test_both_fingerprints_use_the_same_sample_size(self):
        """One rule, one constant -- the divergence is what left this side unfixed."""
        import inspect

        from mlframe.feature_selection.filters import _mrmr_fingerprints

        src = inspect.getsource(_mrmr_fingerprints)
        assert "_CELL_SAMPLE_POSITIONS = 1024" in src
        assert "min(10, n_rows)" not in src

    def test_a_fingerprint_failure_yields_a_never_matching_token(self):
        """An `id()` key is reused after a collection; a fingerprint failure must disable the cache, not risk a hit."""
        import inspect

        from mlframe.feature_selection.filters import _mrmr_fingerprints

        src = inspect.getsource(_mrmr_fingerprints)
        assert "fp_uncacheable_" in src
        assert 'f"fp_id{id(X):x}"' not in src

    def test_two_failed_fingerprints_never_match_each_other(self):
        """The property that matters, stated directly."""
        from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_x_fingerprint

        class Hostile:
            """An X whose every attribute access raises, forcing the fallback path."""

            def __getattr__(self, name):
                raise RuntimeError("no")

        a, b = _mrmr_compute_x_fingerprint(Hostile()), _mrmr_compute_x_fingerprint(Hostile())
        assert a != b, (a, b)


class TestMomentPadsThatGetInvertedAndCubed:
    """An additive pad on a std that is then raised to the third and fourth powers."""

    def _moments(self, x):
        """mean, std, skew, excess kurtosis, min, max."""
        from mlframe.feature_selection.filters.hermite_fe import _moment_fingerprint_njit

        return _moment_fingerprint_njit(np.ascontiguousarray(x, dtype=np.float64))

    def test_skew_is_scale_invariant(self):
        """Skew is dimensionless, so rescaling the same shape must not change it. The pad made it scale-dependent."""
        rng = np.random.default_rng(0)
        shape = rng.gamma(1.5, 1.0, 4000)  # genuinely right-skewed
        big = self._moments(shape)[2]
        small = self._moments(shape * 1e-11)[2]
        assert small == pytest.approx(big, rel=1e-6), (small, big)

    def test_a_heavy_skew_at_tiny_scale_still_reads_as_heavy(self):
        """`basis_route_by_moments` branches on `abs(skew) > 1.5`; the pad scaled a 2.0 down to 0.25."""
        rng = np.random.default_rng(1)
        shape = rng.gamma(0.5, 1.0, 4000) * 1e-11
        assert abs(self._moments(shape)[2]) > 1.5

    def test_the_routing_decision_no_longer_depends_on_scale(self):
        """The consequence the moments feed."""
        from mlframe.feature_selection.filters.hermite_fe import basis_route_by_moments

        rng = np.random.default_rng(2)
        shape = rng.gamma(0.5, 1.0, 4000)
        assert basis_route_by_moments(shape) == basis_route_by_moments(shape * 1e-11)

    def test_a_constant_column_reports_no_shape_rather_than_noise(self):
        """Zero std has no skew; inventing one from the pad is worse than saying nothing."""
        _, std, skew, kurt, _, _ = self._moments(np.full(1000, 7.0))
        assert std <= 1e-12 and skew == 0.0 and kurt == 0.0


class TestTheKernelTuningDefaultsAreRetriedAndAnnounced:
    """The flag was set before the attempt, so one transient failure disabled the shipped defaults for good."""

    def _src(self):
        """The registration module's source."""
        import inspect

        from mlframe.feature_selection.filters import _kernel_tuning

        return inspect.getsource(_kernel_tuning)

    def test_the_flag_is_not_set_before_the_attempt(self):
        """It was assigned above the `try`, so one transient failure disabled the shipped defaults permanently."""
        src = self._src()
        body = src[src.index("with _DEFAULTS_LOCK:") : src.index("register_default_cache(_DEFAULT_TUNING_JSON)")]
        assert "_DEFAULTS_REGISTERED = True  # never re-attempt" not in body
        # The only assignments before the attempt are the two genuinely-permanent cases: no file, no package.
        assert body.count("_DEFAULTS_REGISTERED = True") == 2, body

    def test_a_registration_failure_warns(self):
        """It logged at debug, which production logging does not emit."""
        src = self._src()
        assert "logger.warning(" in src and "NOT registered" in src

    def test_the_flag_is_set_on_the_success_path(self):
        """An `else:` on the try, so a success still fires exactly once per process."""
        assert "        else:\n            _DEFAULTS_REGISTERED = True" in self._src()


class TestThePrunedCountSumKernelMatchesTheFullOne:
    """Pass 1 discarded s2, s3 and s4, paying 2.5x the arithmetic and three unused allocations per call."""

    def test_it_is_bit_identical_to_the_full_kernels_first_two_outputs(self):
        """Same row order, same additions -- identity by construction, asserted anyway."""
        from mlframe.feature_selection.filters._binned_numeric_agg_fe import _per_cell_count_sum_njit, _per_cell_raw_moments_njit

        rng = np.random.default_rng(0)
        n, n_cells = 5000, 37
        # Non-negative, as `np.searchsorted` always produces at the two call sites; the full kernel has no
        # negative-code guard either, so mirroring it exactly is the bit-identity contract.
        codes = rng.integers(0, n_cells, size=n).astype(np.int64)
        v = rng.normal(0, 100, n)
        cnt_ref, s1_ref, *_ = _per_cell_raw_moments_njit(codes, v, n_cells)
        cnt, s1 = _per_cell_count_sum_njit(codes, v, n_cells)
        assert np.array_equal(cnt, cnt_ref) and np.array_equal(s1, s1_ref)

    def test_the_stable_path_uses_the_pruned_kernel(self):
        """An unused fast variant delivers no speedup."""
        import inspect

        from mlframe.feature_selection.filters import _binned_numeric_agg_fe

        src = inspect.getsource(_binned_numeric_agg_fe._per_cell_moments_stable)
        assert "_per_cell_count_sum_njit" in src
        assert "cnt, s1, _, _, _" not in src


def test_the_device_shuffle_keys_are_float64():
    """float32 uniforms have ~1.68e7 grid points, so at n=600k each row carries ~10,700 tied keys whose relative
    order `argsort` resolves by index -- a small positive correlation with the identity permutation, not a
    uniform draw, and a different estimator from the CPU Fisher-Yates floor."""
    import inspect

    from mlframe.feature_selection.filters import _permutation_null_resident

    src = inspect.getsource(_permutation_null_resident)
    assert "dtype=cp.float64)" in src
    assert "_rng.random((nperm, n), dtype=cp.float32)" not in src


def test_the_polynom_deadline_is_passed_explicitly_into_the_workers():
    """`_fe_deadline`'s state is a `threading.local`, which crosses neither the loky process boundary nor the
    big-stack sub-thread `_eval_one_pair` runs the impl on -- so the check was unreachable in EVERY
    configuration, serial included, not only under `n_jobs > 1` as the module assumed."""
    import inspect

    from mlframe.feature_selection.filters import polynom_pair_fe

    src = inspect.getsource(polynom_pair_fe)
    assert "def _eval_one_pair(raw_vars_pair, X_arr, y_arr, fe_deadline=None):" in src
    assert "def _eval_one_pair_impl(raw_vars_pair, X_arr, y_arr, fe_deadline=None):" in src
    assert "set_fe_deadline(fe_deadline)" in src
    assert src.count("_fe_deadline_value") >= 4  # resolved once on the main thread, forwarded to all three call sites


def test_the_target_encoding_moment_divisions_are_guarded():
    """`np.where` evaluates both branches, so a zero-variance category computes 0.0/0.0 and warns per fold per
    column -- noise rather than wrongness, but it would hard-fail any suite under `-W error`."""
    import inspect

    from mlframe.feature_selection.filters import _target_encoding_fe

    src = inspect.getsource(_target_encoding_fe)
    assert src.count('with np.errstate(divide="ignore", invalid="ignore"):') >= 2
