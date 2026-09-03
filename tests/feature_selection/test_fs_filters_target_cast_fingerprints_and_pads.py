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

import ast

import numpy as np
import pytest


class TestAContinuousTargetIsNotTruncatedToClasses:
    """`astype(np.int64)` on a target in [0, 1) makes every MI in the module exactly 0.0."""

    def _module(self):
        """The conditional-gate FE family."""
        from mlframe.feature_selection.filters import _conditional_gate_fe

        return _conditional_gate_fe

    def test_every_target_boundary_goes_through_the_encoder(self):
        """No module-boundary cast may reach for `np.asarray(y).astype(np.int64)` directly.

        Structural: a bare int64 cast and the encoder agree on already-dense integer codes, so the two are
        indistinguishable on the classification path -- the divergence only shows on the targets the sibling
        behavioural tests below cover. What this pins is that no THIRD boundary reintroduces the raw cast,
        which no single call can demonstrate. Asserted on the parsed module rather than its text.
        """
        import ast

        from tests._source_ast import called_names, module_ast

        tree = module_ast(self._module())
        calls = called_names(tree)
        assert calls.count("encode_y_for_classif_mi") >= 3, f"expected every target boundary to use the encoder, saw {calls.count('encode_y_for_classif_mi')}"

        bare_casts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "astype"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "asarray"
            and any(isinstance(a, ast.Name) and a.id == "y" for a in node.func.value.args)
        ]
        assert not bare_casts, f"a bare np.asarray(y).astype(...) target cast is back at line(s) {[n.lineno for n in bare_casts]}"

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
        """One rule, one constant -- the divergence is exactly what left one side unfixed.

        Structural: two fingerprints sampling different numbers of positions agree on most inputs, so the
        defect shows only on the frames that fall between the two sample sizes. Pin the shared constant.
        """
        import ast

        from mlframe.feature_selection.filters import _mrmr_fingerprints
        from tests._source_ast import module_ast

        tree = module_ast(_mrmr_fingerprints)
        sample_consts = {
            t.id: node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for t in node.targets
            if isinstance(t, ast.Name) and t.id == "_CELL_SAMPLE_POSITIONS"
        }
        assert sample_consts.get("_CELL_SAMPLE_POSITIONS") == 1024, f"the shared sample-position constant is missing or changed: {sample_consts}"
        assert "_CELL_SAMPLE_POSITIONS" in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}, "the constant is defined but never read"

    def test_a_failed_y_fingerprint_also_refuses_to_key_on_id(self):
        """The y fingerprint had the SAME id()-fallback the X side was fixed for, and still logged at debug.

        Found by widening the structural check below from the one X-side literal it named to any f-string
        keyed on `id(...)`. Two failures must never collide, and the token must not be derived from an
        address: CPython reuses addresses after collection, so a y built once its predecessor was dropped
        very commonly lands on the same id -- and the identity cache then serves the earlier target's fit.
        """
        from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_y_fingerprint_sample

        class _Unfingerprintable:
            """A y whose array conversion raises, driving the failure branch."""

            def __array__(self, *args, **kwargs):
                """Refuse conversion so the fingerprint falls back."""
                raise ValueError("no array for you")

        a, b = _Unfingerprintable(), _Unfingerprintable()
        tok_a = _mrmr_compute_y_fingerprint_sample(a)
        tok_b = _mrmr_compute_y_fingerprint_sample(b)
        assert tok_a != tok_b, "two failed y fingerprints collided, so the identity cache can serve the wrong fit"
        assert _mrmr_compute_y_fingerprint_sample(a) != tok_a, "a failed y fingerprint is stable across calls, i.e. it can still produce a cache HIT"
        for tok in (tok_a, tok_b):
            assert "uncacheable" in tok, f"the failure token should announce that the cache is disabled, got {tok!r}"
            assert f"{id(a):x}" not in tok and f"{id(b):x}" not in tok, f"the failure token is derived from an object address: {tok!r}"

    def test_a_failed_y_fingerprint_is_audible(self, caplog):
        """It logged at debug, which production logging does not emit, while silently changing which fit runs."""
        import logging

        from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_y_fingerprint_sample

        class _Unfingerprintable:
            """A y whose array conversion raises, driving the failure branch."""

            def __array__(self, *args, **kwargs):
                """Refuse conversion so the fingerprint falls back."""
                raise ValueError("no array for you")

        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._mrmr_fingerprints"):
            _mrmr_compute_y_fingerprint_sample(_Unfingerprintable())
        assert any("disabling the identity cache" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]

    def test_a_fingerprint_failure_yields_a_never_matching_token(self):
        """A failed fingerprint must disable the cache, never key on `id()`, which is reused after a collection.

        Structural: the sibling below already shows two failures never match each other, but that holds for an
        `id()` key too whenever the two objects happen to be alive at once -- the dangerous case is an id
        REUSED after a collection, which a test cannot force deterministically.
        """
        import ast

        from mlframe.feature_selection.filters import _mrmr_fingerprints
        from tests._source_ast import called_names, module_ast, string_literals

        tree = module_ast(_mrmr_fingerprints)
        assert any("fp_uncacheable_" in s for s in string_literals(tree)), "the never-matching failure token is gone"
        id_keyed = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            for part in node.values
            if isinstance(part, ast.FormattedValue)
            for call in ast.walk(part.value)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "id"
        ]
        assert not id_keyed, f"a fingerprint is being keyed on id() again at line(s) {[n.lineno for n in id_keyed]}"
        assert "id" not in set(called_names(tree)) or not id_keyed, "id() is used to build a cache key"

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

    def test_a_transient_failure_does_not_latch_the_defaults_off(self, monkeypatch, caplog):
        """A registration that RAISES must leave the flag unset, so the next call retries.

        Driven through the module's own reset hook rather than read out of its source: the flag used to be
        assigned above the `try`, so one transient fault -- a locked file, a momentary import error --
        disabled the shipped per-hardware kernel-tuning defaults for the entire process, silently, and every
        later fit quietly ran on the fallback.
        """
        import logging

        from mlframe.feature_selection.filters import _kernel_tuning as kt

        kt._reset_for_tests()
        calls = {"n": 0}

        def _boom(*_a, **_k):
            """Fail the first registration attempt the way a transient fault would."""
            calls["n"] += 1
            raise RuntimeError("transient registration fault")

        monkeypatch.setattr("pyutilz.performance.kernel_tuning.cache.register_default_cache", _boom, raising=False)
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._kernel_tuning"):
            kt._register_default_tuning_cache()

        assert calls["n"] == 1, f"registration was not attempted exactly once, got {calls['n']}"
        assert kt._DEFAULTS_REGISTERED is False, "a transient failure latched the flag, so the defaults are disabled for the whole process"
        assert any("NOT registered" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]

        # ...and the next call really does retry rather than short-circuiting on the flag.
        kt._register_default_tuning_cache()
        assert calls["n"] == 2, "the second call did not retry, so the failure latched after all"
        kt._reset_for_tests()

    def test_a_successful_registration_fires_exactly_once(self, monkeypatch):
        """Success DOES latch: the shipped defaults are registered once per process, not once per call."""
        from mlframe.feature_selection.filters import _kernel_tuning as kt

        kt._reset_for_tests()
        calls = {"n": 0}

        def _ok(*_a, **_k):
            """Register successfully."""
            calls["n"] += 1

        monkeypatch.setattr("pyutilz.performance.kernel_tuning.cache.register_default_cache", _ok, raising=False)
        kt._register_default_tuning_cache()
        kt._register_default_tuning_cache()

        assert calls["n"] == 1, f"a successful registration re-ran; it should latch. calls={calls['n']}"
        assert kt._DEFAULTS_REGISTERED is True
        kt._reset_for_tests()


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
        """An unused fast variant delivers no speedup, and the discarded outputs are the whole point.

        Structural: both kernels return identical counts and sums -- the sibling above proves that -- so which
        one the stable path calls is invisible in the OUTPUT and visible only in the work done. Asserted on
        the parsed function.
        """
        from mlframe.feature_selection.filters import _binned_numeric_agg_fe
        from tests._source_ast import called_names, function_ast

        fn = function_ast(_binned_numeric_agg_fe, "_per_cell_moments_stable")
        calls = called_names(fn)
        assert "_per_cell_count_sum_njit" in calls, f"the stable path is not calling the pruned kernel; calls={sorted(set(calls))}"
        assert "_per_cell_raw_moments_njit" not in calls, "the stable path still calls the full kernel and discards three of its outputs"


def test_the_device_shuffle_keys_are_float64():
    """float32 uniforms have ~1.68e7 grid points, so at n=600k each row carries ~10,700 tied keys whose relative
    order `argsort` resolves by index -- a small positive correlation with the identity permutation, not a
    uniform draw, and a different estimator from the CPU Fisher-Yates floor."""
    from mlframe.feature_selection.filters import _permutation_null_resident

    from tests._source_ast import module_ast

    # Structural: this is a cupy path, so the observable difference needs a GPU AND n large enough for the
    # float32 grid to collide (~1.68e7 points; at n=600k each row carries ~10,700 tied keys, which argsort
    # resolves by index -- a small positive correlation with the identity permutation rather than a uniform
    # draw). What is checkable everywhere is that the keys are drawn at float64.
    tree = module_ast(_permutation_null_resident)
    dtypes = {
        kw.value.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute)
    }
    assert "float64" in dtypes, "the device shuffle keys are no longer drawn at float64"
    assert "float32" not in dtypes, f"a float32 dtype is back in the shuffle-key path: {sorted(dtypes)}"


def test_the_polynom_deadline_is_passed_explicitly_into_the_workers():
    """`_fe_deadline`'s state is a `threading.local`, which crosses neither the loky process boundary nor the
    big-stack sub-thread `_eval_one_pair` runs the impl on -- so the check was unreachable in EVERY
    configuration, serial included, not only under `n_jobs > 1` as the module assumed."""
    from mlframe.feature_selection.filters import polynom_pair_fe

    from tests._source_ast import called_names, module_ast

    # Structural: the deadline has to survive a loky process boundary AND the big-stack sub-thread the impl
    # runs on, so observing it requires standing up both -- and the defect was that the check was unreachable
    # in EVERY configuration, serial included, which is precisely why no behavioural test caught it.
    # Both are NESTED functions, so they are unreachable via getattr and only visible in the parsed module.
    tree = module_ast(polynom_pair_fe)
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("_eval_one_pair")}
    for fn_name in ("_eval_one_pair", "_eval_one_pair_impl"):
        assert fn_name in defs, f"{fn_name} is gone; this test needs updating if the worker was restructured"
        params = {a.arg for a in defs[fn_name].args.args} | {a.arg for a in defs[fn_name].args.kwonlyargs}
        assert "fe_deadline" in params, f"{fn_name} no longer accepts fe_deadline, so it cannot be forwarded across the boundary"

    assert "set_fe_deadline" in called_names(module_ast(polynom_pair_fe)), "the worker no longer re-establishes the deadline in its own thread, so the check is unreachable again"


def test_the_target_encoding_moment_divisions_are_guarded():
    """`np.where` evaluates both branches, so a zero-variance category computes 0.0/0.0 and warns per fold per
    column -- noise rather than wrongness, but it would hard-fail any suite under `-W error`."""
    from mlframe.feature_selection.filters import _target_encoding_fe
    from tests._source_ast import module_ast

    # Structural: `np.errstate` suppresses a WARNING, and a warning is not part of any return value -- the
    # numbers are identical with and without it. What it prevents is a suite run under `-W error` hard-failing
    # per fold per column on a zero-variance category, where `np.where` evaluates both branches and the dead
    # one computes 0.0/0.0.
    guards = [
        node
        for node in ast.walk(module_ast(_target_encoding_fe))
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == "errstate"
    ]
    assert len(guards) >= 2, f"the moment divisions are no longer wrapped in np.errstate (found {len(guards)} guard(s))"
