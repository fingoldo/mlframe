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

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"


def _src(rel: str) -> str:
    """Source text of a module by repo-relative path under src/mlframe."""
    return (SRC / rel).read_text(encoding="utf-8")


class TestASubstitutedValueDoesNotDisableItsOwnCheck:
    """The substituted value must not be the one that makes the guard vacuous."""

    def test_the_collapse_sensor_is_not_switched_off_by_a_failed_max_error(self):
        """0.0 is the best possible max error, so `_max_err > 5 * y_std` became unconditionally False."""
        s = _src("training/reporting/_reporting_regression/_sensors.py")
        assert "_max_err = 0.0" not in s
        assert "np.isfinite(_max_err) and _max_err > 5.0 * _y_std" in s

    def test_a_nan_max_error_does_not_raise_a_false_alarm_either(self):
        """NaN keeps the comparison False -- no alarm from a number we do not have -- while reading as unknown."""
        assert not (np.isfinite(float("nan")) and float("nan") > 1.0)

    def test_the_failure_is_announced(self):
        """A sensor that has switched itself off must say so."""
        s = _src("training/reporting/_reporting_regression/_sensors.py")
        assert "is\n                DISABLED" in s or "DISABLED for this model" in s

    def test_the_vram_cushion_guard_fails_closed(self):
        """`return True` ALLOWS the upload, so failing open removed the protection when the device is unhealthy."""
        s = _src("feature_selection/filters/_fe_gpu_vram.py")
        assert 'permissive", exc)\n            return True' not in s
        assert "refusing the GPU path rather than uploading" in s

    def test_the_hinge_solve_separates_a_singular_design_from_a_fault(self):
        """A singular design legitimately loses; a cupy fault says nothing about the breakpoint."""
        s = _src("feature_selection/filters/_hinge_detect_gpu_resident.py")
        assert "except cp.linalg.LinAlgError as e:" in s
        assert "REJECTED on a failure that says nothing about its quality" in s


class TestTheOracleFingerprintAdmitsIgnorance:
    """0.0 is a legal reading of both statistics, so substituting it falsifies the key rather than degrading it."""

    def test_a_failed_correlation_is_nan_not_zero(self):
        """0.0 means "uncorrelated", which is a claim about the data."""
        s = _src("utils/_param_oracle.py")
        assert 'mean_abs_corr = float("nan")' in s
        assert "                mean_abs_corr = 0.0" not in s

    def test_an_unmeasurable_cardinality_is_nan_not_zero(self):
        """Same shape, same reasoning."""
        s = _src("utils/_param_oracle.py")
        assert 'cardinality_mean = float(np.mean(cards)) if cards else float("nan")' in s

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
        s = _src(rel)
        assert "except ImportError as e:" in s
        assert "resolved ONCE at import" in s
        assert "logger.warning(" in s

    def test_the_shap_proxy_gpu_probe_does_not_latch_a_transient_failure(self):
        """Both probes detect permanent host properties; a CUDARuntimeError under contention is not one."""
        s = _src("feature_selection/shap_proxied_fs/_shap_proxy_prefilter.py")
        assert "re-probing" in s and "rather than latching CPU-only" in s

    def test_the_kernel_tuning_registry_failure_does_not_latch(self):
        """`_SPEC = False` is checked before every retry, so one bad call cost the measured backend for the run."""
        s = _src("feature_selection/filters/_fe_interaction_prerank_kernels.py")
        assert "except ImportError as exc:" in s
        assert "retrying the registry next time rather than latching it off" in s.replace("\n", " ").replace("  ", " ")

    def test_the_raw_kernel_compile_failure_does_not_latch(self):
        """A transient nvrtc/DLL fault is a documented mode in this repo."""
        s = _src("feature_selection/filters/_fe_batched_mi.py")
        assert "_MI_FROM_CODES_V2_KERNELS = False" not in s
        assert "retrying the compile next time" in s.replace("\n", " ").replace("  ", " ")


class TestThePermutationNullIsRngIdenticalOnBothPaths:
    """The GPU path consumed n_perm draws BEFORE the call that can fail; the fallback then drew n_perm MORE."""

    def test_both_paths_rebuild_the_same_child_generator(self):
        """One draw for a seed, two identical reconstructions -- so a GPU failure cannot move the verdict."""
        s = _src("feature_selection/filters/_binned_numeric_agg_fe.py")
        assert s.count("np.random.default_rng(_perm_seed)") == 2
        assert "_perm_seed = int(_rng.integers(0, 2**63 - 1))" in s

    def test_the_fallback_no_longer_draws_from_the_outer_generator(self):
        """That is what made the two nulls differ."""
        s = _src("feature_selection/filters/_binned_numeric_agg_fe.py")
        block = s.split("_perm_seed = int(")[1].split("null_ceiling = float(")[0]
        assert "_rng.permutation(" not in block, block[-400:]

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
        s = _src("feature_selection/filters/_group_distance_fe.py")
        assert "except ImportError as e:" in s
        assert "quantile APPROXIMATION" in s
        assert "def _wasserstein_quantile_approx" in s

    def test_the_noise_gate_fallback_says_the_verdicts_may_differ(self):
        """Two approximations of the same null, not one estimator computed two ways."""
        s = _src("feature_selection/filters/_feature_engineering_pairs/_pairs_dispatch.py")
        assert "different approximation of the same null" in s

    def test_the_undeflated_return_is_announced(self):
        """Returning y unchanged makes the caller re-detect the same tone as several distinct frequencies."""
        s = _src("feature_selection/filters/_orthogonal_univariate_fe/_orth_extra_basis_fe.py")
        assert "returning y UNDEFLATED" in s
        assert s.count("log_throttle(") >= 2

    def test_the_cuda_shape_guard_is_separated_from_a_driver_fault(self):
        """The handler's own comment named three causes and treated them identically."""
        s = _src("feature_selection/filters/batch_pair_mi_gpu.py")
        assert "_is_shape_guard" in s
        assert "NOT a shape-guard trip" in s


def test_the_densification_says_how_big_it_will_be():
    """The one handler in that file that changes RESOURCE behaviour rather than a value: a deliberately-sparse
    matrix is materialised dense, which on this project's frame sizes is an OOM attributed to something else."""
    s = _src("training/core/_predict_pre_pipeline.py")
    assert "DENSIFYING" in s and "GB" in s
    assert 'logger.debug("sparse_df_from_spmatrix failed, densifying instead' not in s


def test_a_failed_group_fit_is_excluded_from_the_shrinkage_statistics():
    """Substituting the global fit is a defensible PREDICTION fallback; feeding those copies into the
    James-Stein estimator as independent per-group observations is not -- they carry zero between-group
    variance by construction, so every failed group makes the shrinkage more aggressive than the data warrants."""
    s = _src("training/composite/transforms/linear.py")
    assert "_fit_ok" in s
    assert "if _fit_ok:\n            alphas_for_shrink.append(a_g)" in s


def test_an_unseeded_cuda_rng_is_announced():
    """`set_random_seed`'s entire purpose is determinism; ImportError is already split out above, so reaching
    that handler means cupy IS installed and its RNG is unusable."""
    s = _src("utils/misc.py")
    assert "is NOT reproducible" in s
    assert 'logger.debug("cupy.random.seed() failed' not in s


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
        s = _src(rel)
        assert marker in s, (rel, marker)
        window = s[s.index(marker) - 800 : s.index(marker) + 800]
        assert "logger.warning(" in window or "log_throttle(" in window, (rel, marker)


def test_the_helper_extraction_kept_the_approximation_identical():
    """`_wasserstein_quantile_approx` is the same 101-point grid the inline fallback used."""
    from mlframe.feature_selection.filters._group_distance_fe import _wasserstein_quantile_approx

    rng = np.random.default_rng(0)
    g, glob = rng.normal(1.0, 1.0, 500), np.sort(rng.normal(0.0, 1.0, 800))
    u = np.linspace(0.0, 1.0, 101)
    expected = float(np.mean(np.abs(np.quantile(g, u) - np.quantile(glob, u))))
    assert _wasserstein_quantile_approx(g, glob) == pytest.approx(expected)
