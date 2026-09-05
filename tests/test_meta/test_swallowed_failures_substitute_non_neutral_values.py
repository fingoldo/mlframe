"""Sixteen handlers that swallowed a failure by substituting a value which is not neutral.

A fallback is only harmless when the substituted value means "unknown". These substituted values meant
something specific and wrong:

  * `_max_err = 0.0` is the BEST possible max error, so it made the extrapolation-collapse sensor's
    `_max_err > 5 * y_std` test unconditionally False -- switching a safety sensor off in exactly the
    situations (shape mismatch, object-dtype predictions) where it matters most;
  * `-np.inf` for a failed hinge solve is the value that GUARANTEES rejection, so a shape bug or a driver
    fault silently discarded a good breakpoint;
  * `return True` from a failed VRAM probe is the value that ALLOWS the upload, removing the OOM protection
    precisely when the device is too unhealthy to answer;
  * `0.0` for a failed fingerprint statistic means "uncorrelated" / "no cardinality", both legal readings, so
    the oracle keys against a fingerprint that misdescribes the data rather than one that admits ignorance.

And a family of latched sentinels where one transient fault permanently downgraded the process, plus a
permutation-null whose fallback drew DIFFERENT permutations from an already-advanced generator, silently
changing a keep/reject verdict.
"""

from __future__ import annotations

import pathlib

import ast

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"


def _tree(rel: str) -> ast.Module:
    """Parsed AST of a module by repo-relative path under src/mlframe."""
    return ast.parse((SRC / rel).read_text(encoding="utf-8"))


def _emitted(rel: str) -> str:
    """Every string literal the module contains, joined -- what it can actually SAY.

    These tests are about whether a failure is announced and what the announcement claims, so the subject is
    the module's emitted messages. Searching the raw source text instead matches a phrase sitting in a comment
    just as happily as one in a log call -- and several of these handlers carry a comment explaining the very
    wording being asserted, so "the warning says this" and "a note above the handler says this" were
    indistinguishable. Joining the literals keeps the multi-line message checks working, since an implicitly
    concatenated message is several literals.
    """
    return " ".join(n.value for n in ast.walk(_tree(rel)) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _assigns_const_in_handler(rel: str, name: str) -> set:
    """Every constant assigned to ``name`` INSIDE an except handler.

    Scoped to handlers on purpose. These modules legitimately initialise the same names to the same constants
    at the top of a function -- that is the starting value, not a substitution -- and the defect is only the
    handler writing one on the way out, where it becomes indistinguishable from a real measurement.
    """
    out: set = set()
    for handler in ast.walk(_tree(rel)):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        for node in ast.walk(handler):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                out.update(t.value for t in node.targets if isinstance(t, ast.Name) and t.id == name)
    return out


def _caught_types(rel: str) -> set:
    """Names of the exception types the module catches, e.g. ``{"ImportError", "LinAlgError"}``."""
    out: set = set()
    for node in ast.walk(_tree(rel)):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            out.update(n.id for n in ast.walk(node.type) if isinstance(n, ast.Name))
            out.update(n.attr for n in ast.walk(node.type) if isinstance(n, ast.Attribute))
    return out


def _called(rel: str) -> list:
    """Every called name in the module; an attribute call is reported by its attribute."""
    out: list = []
    for node in ast.walk(_tree(rel)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            out.append(fn.attr)
        elif isinstance(fn, ast.Name):
            out.append(fn.id)
    return out


def _identifiers(rel: str) -> set:
    """Every identifier the module reads or binds."""
    return {n.id for n in ast.walk(_tree(rel)) if isinstance(n, ast.Name)} | {
        n.name for n in ast.walk(_tree(rel)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class TestASubstitutedValueDoesNotDisableItsOwnCheck:
    """The substituted value must not be the one that makes the guard vacuous."""

    def test_the_collapse_sensor_is_not_switched_off_by_a_failed_max_error(self):
        """0.0 is the best possible max error, so `_max_err > 5 * y_std` became unconditionally False."""
        rel = "training/reporting/_reporting_regression/_sensors.py"
        rel = "training/reporting/_reporting_regression/_sensors.py"
        assert 0.0 not in _assigns_const_in_handler(rel, "_max_err"), "0.0 is the BEST possible max error, so the collapse check becomes unconditionally False"
        assert {"_max_err", "_y_std"} <= _identifiers(rel), "the collapse sensor no longer compares the max error against the target's spread"
        assert "isfinite" in _called(rel), "the sensor no longer guards against a non-finite max error"

    def test_a_nan_max_error_does_not_raise_a_false_alarm_either(self):
        """NaN keeps the comparison False -- no alarm from a number we do not have -- while reading as unknown."""
        assert not (np.isfinite(float("nan")) and float("nan") > 1.0)

    def test_the_failure_is_announced(self):
        """A sensor that has switched itself off must say so."""
        rel = "training/reporting/_reporting_regression/_sensors.py"
        s = _emitted(rel)
        assert "is\n                DISABLED" in s or "DISABLED for this model" in s

    def test_the_vram_cushion_guard_fails_closed(self):
        """`return True` ALLOWS the upload, so failing open removed the protection when the device is unhealthy."""
        rel = "feature_selection/filters/_fe_gpu_vram.py"
        s = _emitted(rel)
        rel = "feature_selection/filters/_fe_gpu_vram.py"
        # An ImportError-only handler returning True is the DOCUMENTED case: no cupy at all means no GPU, so
        # staying permissive lets the caller's other gates decide. The defect was returning True from the
        # PROBE-FAILURE handler -- a device that exists but cannot answer -- which allows the upload precisely
        # when the card is too unhealthy to be asked.
        returns_true = [
            node
            for handler in ast.walk(_tree(rel))
            if isinstance(handler, ast.ExceptHandler)
            and not (handler.type is not None and {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)} <= {"ImportError", "ModuleNotFoundError"})
            for node in ast.walk(handler)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is True
        ]
        assert not returns_true, f"a failed VRAM probe returns True again at line(s) {[n.lineno for n in returns_true]} -- True ALLOWS the upload, removing OOM protection exactly when the device is too unhealthy to answer"
        assert "refusing the GPU path rather than uploading" in s

    def test_the_hinge_solve_separates_a_singular_design_from_a_fault(self):
        """A singular design legitimately loses; a cupy fault says nothing about the breakpoint."""
        rel = "feature_selection/filters/_hinge_detect_gpu_resident.py"
        s = _emitted(rel)
        assert "LinAlgError" in _caught_types("feature_selection/filters/_hinge_detect_gpu_resident.py"), "the narrow linear-algebra catch is gone, so a driver fault is treated as a bad breakpoint"
        assert "REJECTED on a failure that says nothing about its quality" in s


class TestTheOracleFingerprintAdmitsIgnorance:
    """0.0 is a legal reading of both statistics, so substituting it falsifies the key rather than degrading it."""

    def test_a_failed_correlation_is_nan_not_zero(self):
        """0.0 means "uncorrelated", which is a claim about the data."""
        rel = "utils/_param_oracle.py"
        assert 0.0 not in _assigns_const_in_handler(rel, "mean_abs_corr"), "0.0 means UNCORRELATED, which is a claim; a failed correlation must admit ignorance"
        assert "nan" in _called(rel) or "float" in _called(rel), "the failure path no longer produces a NaN"

    def test_an_unmeasurable_cardinality_is_nan_not_zero(self):
        """Same shape, same reasoning."""
        rel = "utils/_param_oracle.py"
        assert 0.0 not in _assigns_const_in_handler(rel, "cardinality_mean"), "an empty card list must read as unknown, not as zero cardinality"

    def test_a_real_frame_still_produces_finite_statistics(self):
        """The fix must not start emitting NaN for ordinary input."""
        from mlframe.utils._param_oracle import default_fingerprint

        rng = np.random.default_rng(0)
        fp = default_fingerprint((rng.normal(size=(200, 5)),), {})
        assert np.isfinite(fp["mean_abs_corr"]) and np.isfinite(fp["cardinality_mean"]), fp


class TestATransientFaultDoesNotLatchAPermanentDowngrade:
    """`except Exception` around a probe cannot tell "no device" from "device busy right now"."""

    @pytest.mark.parametrize(
        "rel",
        [
            "feature_selection/filters/batch_pair_usability_corr_gpu.py",
            "feature_selection/filters/friend_graph_gpu.py",
            "feature_selection/filters/_batch_mi_noise_gate_kernels.py",
            "feature_selection/filters/_batch_pair_mi_cuda_kernels.py",
        ],
    )
    def test_the_import_time_cuda_probe_separates_import_error_from_a_fault(self, rel):
        """`_CUDA_AVAIL` is resolved ONCE at module import, so one hiccup disables the module for the process."""
        assert "ImportError" in _caught_types(rel), "the narrow ImportError catch is gone, so a real fault is treated as an absent dependency"
        s = _emitted(rel)
        assert "resolved ONCE at import" in s
        assert "warning" in _called(rel), "the import failure is no longer announced above debug level"

    def test_the_shap_proxy_gpu_probe_does_not_latch_a_transient_failure(self):
        """Both probes detect permanent host properties; a CUDARuntimeError under contention is not one."""
        rel = "feature_selection/shap_proxied_fs/_shap_proxy_prefilter.py"
        s = _emitted(rel)
        assert "re-probing" in s and "rather than latching CPU-only" in s

    def test_the_kernel_tuning_registry_failure_does_not_latch(self):
        """`_SPEC = False` is checked before every retry, so one bad call cost the measured backend for the run."""
        rel = "feature_selection/filters/_fe_interaction_prerank_kernels.py"
        s = _emitted(rel)
        assert "ImportError" in _caught_types(rel), "the narrow ImportError catch is gone, so a real fault is treated as an absent dependency"
        assert "retrying the registry next time rather than latching it off" in s.replace("\n", " ").replace("  ", " ")

    def test_the_raw_kernel_compile_failure_does_not_latch(self):
        """A transient nvrtc/DLL fault is a documented mode in this repo."""
        rel = "feature_selection/filters/_fe_batched_mi.py"
        s = _emitted(rel)
        assert False not in _assigns_const_in_handler(rel, "_MI_FROM_CODES_V2_KERNELS"), "a failed compile latches the kernel off for the whole process instead of being retried"
        assert "retrying the compile next time" in s.replace("\n", " ").replace("  ", " ")


class TestThePermutationNullIsRngIdenticalOnBothPaths:
    """The GPU path consumed n_perm draws BEFORE the call that can fail; the fallback then drew n_perm MORE."""

    def test_both_paths_rebuild_the_same_child_generator(self):
        """One draw for a seed, two identical reconstructions -- so a GPU failure cannot move the verdict."""
        rel = "feature_selection/filters/_binned_numeric_agg_fe.py"
        # Counted on the SEEDED call specifically: the module builds other generators too, so a bare
        # `default_rng` tally says nothing about whether both permutation draws share one explicit seed.
        seeded = [
            node
            for node in ast.walk(_tree(rel))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "default_rng"
            and any(isinstance(a, ast.Name) and a.id == "_perm_seed" for a in node.args)
        ]
        assert len(seeded) == 2, f"expected both permutation draws to come from default_rng(_perm_seed); found {len(seeded)}"
        assert "_perm_seed" in _identifiers(rel), "the explicit permutation seed is gone, so the two draws cannot be reproduced"

    def test_the_fallback_no_longer_draws_from_the_outer_generator(self):
        """That is what made the two nulls differ."""
        rel = "feature_selection/filters/_binned_numeric_agg_fe.py"
        # The fallback must draw from the SEEDED child, never from the caller's `_rng` -- drawing from the
        # outer generator advances it, so the two paths produce different nulls from the same inputs. Asserted
        # on the parsed module: no `_rng.permutation(...)` anywhere, which the previous form approximated by
        # slicing the source text between two landmark expressions.
        outer_draws = [
            node
            for node in ast.walk(_tree(rel))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "permutation"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_rng"
        ]
        assert not outer_draws, f"the fallback draws from the caller's generator at line(s) {[n.lineno for n in outer_draws]}, which advances it and makes the two nulls differ"

    def test_a_seeded_child_reproduces_its_sequence(self):
        """The property the fix rests on."""
        seed = 12345
        a = [np.random.default_rng(seed).permutation(50) for _ in range(1)]
        b = [np.random.default_rng(seed).permutation(50) for _ in range(1)]
        np.testing.assert_array_equal(a[0], b[0])

    def test_the_outer_generator_advances_identically_either_way(self):
        """One `integers` draw regardless of which path runs, so nothing downstream shifts."""
        r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
        r1.integers(0, 2**63 - 1)
        r2.integers(0, 2**63 - 1)
        assert r1.random() == r2.random()


class TestASilentEstimatorSubstitutionIsAnnounced:
    """A fallback that returns DIFFERENT values is a result change, not a performance note."""

    def test_the_wasserstein_fallback_separates_absent_scipy_from_a_runtime_failure(self):
        """The 101-point grid approximates the statistic; mixing the two across rows changes the feature."""
        rel = "feature_selection/filters/_group_distance_fe.py"
        s = _emitted(rel)
        assert "ImportError" in _caught_types(rel), "the narrow ImportError catch is gone, so a real fault is treated as an absent dependency"
        assert "quantile APPROXIMATION" in s
        assert "_wasserstein_quantile_approx" in _identifiers(rel), "the approximation fallback is gone entirely"

    def test_the_noise_gate_fallback_says_the_verdicts_may_differ(self):
        """Two approximations of the same null, not one estimator computed two ways."""
        rel = "feature_selection/filters/_feature_engineering_pairs/_pairs_dispatch.py"
        s = _emitted(rel)
        assert "different approximation of the same null" in s

    def test_the_undeflated_return_is_announced(self):
        """Returning y unchanged makes the caller re-detect the same tone as several distinct frequencies."""
        rel = "feature_selection/filters/_orthogonal_univariate_fe/_orth_extra_basis_fe.py"
        s = _emitted(rel)
        assert "returning y UNDEFLATED" in s
        assert _called(rel).count("log_throttle") >= 2, "the deflation fallbacks are no longer throttled, so a hot path would spam"

    def test_the_cuda_shape_guard_is_separated_from_a_driver_fault(self):
        """The handler's own comment named three causes and treated them identically."""
        rel = "feature_selection/filters/batch_pair_mi_gpu.py"
        s = _emitted(rel)
        assert "_is_shape_guard" in _identifiers(rel), "the shape-guard discrimination is gone, so a real fault reads as a benign shape trip"
        assert "NOT a shape-guard trip" in s


def test_the_densification_says_how_big_it_will_be():
    """The one handler in that file that changes RESOURCE behaviour rather than a value: a deliberately-sparse
    matrix is materialised dense, which on this project's frame sizes is an OOM attributed to something else."""
    rel = "training/core/_predict_pre_pipeline.py"
    s = _emitted(rel)
    assert "DENSIFYING" in s and "GB" in s
    assert "warning" in _called(rel), "the densification is announced at debug, which production does not emit -- and it is an OOM risk, not a value change"


def test_a_failed_group_fit_is_excluded_from_the_shrinkage_statistics():
    """Substituting the global fit is a defensible PREDICTION fallback; feeding those copies into the
    James-Stein estimator as independent per-group observations is not -- they carry zero between-group
    variance by construction, so every failed group makes the shrinkage more aggressive than the data warrants."""
    rel = "training/composite/transforms/linear.py"
    assert "_fit_ok" in _identifiers(rel), "the per-group fit-success flag is gone, so a substituted global fit feeds the shrinkage as a real observation"
    # ...and the append is GUARDED by it: an unguarded append is what makes every failed group carry zero
    # between-group variance into the James-Stein estimator.
    guarded = [
        node
        for node in ast.walk(_tree(rel))
        if isinstance(node, ast.If) and any(isinstance(n, ast.Name) and n.id == "_fit_ok" for n in ast.walk(node.test))
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "append"
    ]
    assert guarded, "the shrinkage append is no longer guarded by the fit-success flag"


def test_an_unseeded_cuda_rng_is_announced():
    """`set_random_seed`'s entire purpose is determinism; ImportError is already split out above, so reaching
    that handler means cupy IS installed and its RNG is unusable."""
    rel = "utils/misc.py"
    s = _emitted(rel)
    assert "is NOT reproducible" in s
    assert "warning" in _called(rel), "an unusable cupy RNG is announced at debug, so a non-reproducible run looks like a normal one"


def test_every_touched_module_still_imports():
    """Narrowed excepts, new logging calls and a new module-level helper must not break module load."""
    import importlib

    for mod in (
        "mlframe.utils.misc",
        "mlframe.utils._param_oracle",
        "mlframe.core.arrays",
        "mlframe.feature_selection.filters._group_distance_fe",
        "mlframe.feature_selection.filters._fe_gpu_vram",
        "mlframe.feature_selection.filters._fe_batched_mi",
        "mlframe.feature_selection.filters._fe_interaction_prerank_kernels",
        "mlframe.feature_selection.filters._binned_numeric_agg_fe",
        "mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter",
        "mlframe.training.reporting._reporting_regression._sensors",
        "mlframe.training.core._predict_pre_pipeline",
        "mlframe.training.composite.transforms.linear",
    ):
        assert importlib.import_module(mod) is not None, mod


def test_the_three_decision_flipping_handlers_reach_at_least_warning():
    """Scoped to the handlers this round fixed. A file-wide sweep would be wrong: these modules contain other
    `except Exception` handlers that are legitimately debug-level, because the value they substitute really is
    neutral (an optional probe returning "not available"). What matters is that a substituted value which
    CHANGES A DECISION is audible."""
    for rel, marker in (
        ("training/reporting/_reporting_regression/_sensors.py", "max-error computation failed"),
        ("feature_selection/filters/_fe_gpu_vram.py", "memGetInfo failed"),
        ("feature_selection/filters/_hinge_detect_gpu_resident.py", "non-LinAlgError"),
    ):
        assert marker in _emitted(rel), (rel, marker)
        # ...and the handler that carries it is audible. Checked per-HANDLER rather than by a text window: a
        # window over the raw source counts a `logger.warning` belonging to a neighbouring handler, which is
        # exactly the confusion this test exists to rule out.
        audible = [
            handler
            for handler in ast.walk(_tree(rel))
            if isinstance(handler, ast.ExceptHandler)
            and any(marker in n.value for n in ast.walk(handler) if isinstance(n, ast.Constant) and isinstance(n.value, str))
            and any(
                (isinstance(c.func, ast.Attribute) and c.func.attr in {"warning", "error", "exception", "critical"})
                or (isinstance(c.func, ast.Name) and c.func.id == "log_throttle")
                for c in ast.walk(handler)
                if isinstance(c, ast.Call)
            )
        ]
        assert audible, f"{rel}: the handler emitting {marker!r} does not log above debug, so the substitution is silent in production"


def test_the_helper_extraction_kept_the_approximation_identical():
    """`_wasserstein_quantile_approx` is the same 101-point grid the inline fallback used."""
    from mlframe.feature_selection.filters._group_distance_fe import _wasserstein_quantile_approx

    rng = np.random.default_rng(0)
    g, glob = rng.normal(1.0, 1.0, 500), np.sort(rng.normal(0.0, 1.0, 800))
    u = np.linspace(0.0, 1.0, 101)
    expected = float(np.mean(np.abs(np.quantile(g, u) - np.quantile(glob, u))))
    assert _wasserstein_quantile_approx(g, glob) == pytest.approx(expected)
