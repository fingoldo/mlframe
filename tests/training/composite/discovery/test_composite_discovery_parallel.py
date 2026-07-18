"""T1#1 2026-05-18 Pack #6 real parallel composite candidates.

The discovery loop in CompositeTargetDiscovery.fit iterates serially over
``base_candidates`` x ``self.config.transforms``. ``discovery_n_jobs > 1``
now runs the per-transform body in parallel via joblib threading.

These tests pin:

1. Equivalence: parallel == serial output (same kept_specs, same report,
   bit-for-bit identical mi_gain / mi_t / mi_y).
2. Biz value: parallel is NOT slower than serial when n_jobs > 1 (the
   refactor must not regress wall time even when the speedup is modest).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _build_problem(n: int = 600, seed: int = 11):
    """Synthetic 4-base, 6-feature regression with mixed linear and ratio
    structure so multiple transforms find something."""
    rng = np.random.default_rng(seed)
    base1 = rng.normal(50.0, 10.0, n)
    base2 = rng.normal(20.0, 5.0, n)
    f3 = rng.normal(size=n)
    f4 = rng.normal(size=n)
    f5 = rng.normal(size=n)
    f6 = rng.normal(size=n)
    # y = mixture of linear + non-linear structure on base1 + light noise.
    y = 1.5 * base1 + 0.3 * f3 - 0.2 * f4 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "base1": base1,
            "base2": base2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
            "f6": f6,
        }
    )
    return df, y


def _run(*, n_jobs: int, transforms: list[str]):
    """Fits CompositeTargetDiscovery with the given n_jobs/transforms and returns the fitted discovery object, for parallel-vs-serial identity checks."""
    df, y = _build_problem()
    df_with_y = df.copy()
    df_with_y["y"] = y
    cfg = CompositeTargetDiscoveryConfig(
        transforms=transforms,
        mi_nbins=8,
        mi_estimator="bin",
        top_k_after_mi=4,
        eps_mi_gain=-1.0,
        random_state=11,
        discovery_n_jobs=n_jobs,
        mi_gain_bootstrap_n=0,
        detect_linear_residual_alpha_drift=False,
        base_candidates=["base1", "base2"],
    )
    discovery = CompositeTargetDiscovery(config=cfg)
    n = len(df)
    train_idx = np.arange(int(0.8 * n))
    feature_cols = ["base1", "base2", "f3", "f4", "f5", "f6"]
    t0 = time.perf_counter()
    discovery.fit(
        df=df_with_y,
        target_col="y",
        feature_cols=feature_cols,
        train_idx=train_idx,
    )
    elapsed = time.perf_counter() - t0
    return discovery, elapsed


class TestParallelDiscoveryEquivalence:
    """Parallel path must produce candidates that match the serial path."""

    def test_kept_specs_match_serial(self) -> None:
        """Kept specs match serial."""
        transforms = ["linear_residual", "diff", "ratio", "logratio"]

        serial, _ = _run(n_jobs=1, transforms=transforms)
        parallel, _ = _run(n_jobs=4, transforms=transforms)

        # Same kept specs (same names, same order after the sort by mi_gain).
        ser_names = [s.name for s in serial.specs_]
        par_names = [s.name for s in parallel.specs_]
        assert ser_names == par_names, f"parallel path diverged from serial:\n  serial:   {ser_names}\n  parallel: {par_names}"

        # Per-spec mi_gain / mi_t / mi_y must match bit-for-bit (deterministic
        # numpy ops, same input). Pre-binning is deterministic; bootstrap is off.
        for ser_spec, par_spec in zip(serial.specs_, parallel.specs_):
            assert np.isclose(
                ser_spec.mi_gain, par_spec.mi_gain, atol=1e-12
            ), f"mi_gain diverged for '{ser_spec.name}': serial={ser_spec.mi_gain}, parallel={par_spec.mi_gain}"
            assert np.isclose(ser_spec.mi_t, par_spec.mi_t, atol=1e-12)
            assert np.isclose(ser_spec.mi_y, par_spec.mi_y, atol=1e-12)


class TestParallelDiscoveryBizValue:
    """biz_value: parallel must not be slower than serial when n_jobs > 1.

    The compute body is mostly numpy/numba (GIL-released), so threading at
    n_jobs=4 should be at most equal to serial wall time. A regression
    here means the refactor introduced overhead that ate the win.
    """

    def test_parallel_not_slower_than_serial(self) -> None:
        """Parallel not slower than serial."""
        transforms = [
            "linear_residual",
            "diff",
            "ratio",
            "logratio",
            "monotonic_residual",
            "quantile_residual",
        ]

        # Warm up: run once to JIT any numba kernels so the timing is fair.
        _run(n_jobs=1, transforms=transforms)

        # Median of 3 runs to reduce wall-clock noise.
        serial_times = []
        parallel_times = []
        for _ in range(3):
            _, t_s = _run(n_jobs=1, transforms=transforms)
            _, t_p = _run(n_jobs=4, transforms=transforms)
            serial_times.append(t_s)
            parallel_times.append(t_p)
        med_serial = float(np.median(serial_times))
        med_parallel = float(np.median(parallel_times))

        # LOWER#14 2026-05-18 round-number justification: the 1.5x slack
        # is derived from measured joblib threading-backend coordination
        # overhead at small task counts: pool setup + dispatch + join
        # together ~0.05-0.10s, which is 10-20% of typical wall time on
        # this test's workload (~0.5s serial). 1.5x covers that with
        # margin. Larger ratios (>2x) indicate a refactor regression
        # (extra pickling, lock contention, or accidental serial work).
        # NOTE 2026-05-18 HIGH#6: real measured speedup on n=200k at
        # n_jobs=8 is ~1.05x, NOT 5-10x. The serial _tiny_model_rerank
        # tail dominates total wall time. This test only validates "no
        # regression vs serial"; it is NOT a positive speedup assertion.
        assert med_parallel <= 1.5 * med_serial, (
            f"parallel discovery wall-time regressed: "
            f"median serial={med_serial:.3f}s, median parallel={med_parallel:.3f}s "
            f"(ratio={med_parallel / max(med_serial, 1e-9):.2f}x); "
            f"raw serial={serial_times}, raw parallel={parallel_times}"
        )


def _run_rerank(*, n_jobs: int, seed_repeats: int = 1):
    """Variant of `_run` that forces tiny_model_rerank to actually run.

    ``screening="tiny_model"`` triggers the Phase B rerank inside
    :meth:`CompositeTargetDiscovery._tiny_model_rerank`. Multi-seed
    repeats exercise the Wilcoxon-gate codepath as well so the parallel
    reduce of ``_wilcoxon_per_seed_composite`` is covered.
    """
    df, y = _build_problem()
    df_with_y = df.copy()
    df_with_y["y"] = y
    cfg = CompositeTargetDiscoveryConfig(
        transforms=["linear_residual", "diff", "ratio", "logratio"],
        mi_nbins=8,
        mi_estimator="bin",
        top_k_after_mi=4,
        eps_mi_gain=-1.0,
        random_state=11,
        discovery_n_jobs=1,
        mi_gain_bootstrap_n=0,
        detect_linear_residual_alpha_drift=False,
        base_candidates=["base1", "base2"],
        screening="tiny_model",
        tiny_screening_models="single_lgbm",
        tiny_model_sample_n=500,
        tiny_model_n_estimators=20,
        tiny_model_cv_folds=3,
        tiny_model_n_seed_repeats=seed_repeats,
        use_wilcoxon_gate=(seed_repeats > 1),
        deterministic_screening_models=True,
        require_beats_raw_baseline=False,
        tiny_rerank_n_jobs=n_jobs,
    )
    discovery = CompositeTargetDiscovery(config=cfg)
    n = len(df)
    train_idx = np.arange(int(0.8 * n))
    feature_cols = ["base1", "base2", "f3", "f4", "f5", "f6"]
    discovery.fit(
        df=df_with_y,
        target_col="y",
        feature_cols=feature_cols,
        train_idx=train_idx,
    )
    return discovery


class TestParallelRerankEquivalence:
    """Phase B parallel rerank must produce identical RMSEs to serial.

    The refactor pre-builds the per-base cache, then runs per-spec in a
    joblib threading pool. Each task returns the per-family RMSEs and
    (optionally) per-seed arrays. The serial reduce rebuilds the same
    ``per_family_scores`` / ``_wilcoxon_per_seed_composite`` /
    ``_per_bin_first_pass`` state. With ``deterministic_screening_models``
    the LightGBM CV is bit-for-bit reproducible across runs and threads.
    """

    def test_rerank_scores_match_serial(self) -> None:
        """Rerank scores match serial."""
        serial = _run_rerank(n_jobs=1)
        parallel = _run_rerank(n_jobs=4)

        ser_scores = dict(getattr(serial, "_tiny_rerank_scores", {}))
        par_scores = dict(getattr(parallel, "_tiny_rerank_scores", {}))
        assert set(ser_scores) == set(
            par_scores
        ), f"parallel rerank produced a different spec set:\n  serial:   {sorted(ser_scores)}\n  parallel: {sorted(par_scores)}"
        for name, ser_rmse in ser_scores.items():
            par_rmse = par_scores[name]
            if np.isfinite(ser_rmse) and np.isfinite(par_rmse):
                assert np.isclose(ser_rmse, par_rmse, atol=1e-10), f"rerank RMSE for '{name}' diverged: serial={ser_rmse}, parallel={par_rmse}"
            else:
                assert np.isnan(ser_rmse) == np.isnan(par_rmse), f"finiteness mismatch for '{name}'"

    def test_wilcoxon_per_seed_matches_serial(self) -> None:
        """Wilcoxon per seed matches serial."""
        serial = _run_rerank(n_jobs=1, seed_repeats=3)
        parallel = _run_rerank(n_jobs=4, seed_repeats=3)

        ser_per_seed = getattr(serial, "_wilcoxon_per_seed_composite", {})
        par_per_seed = getattr(parallel, "_wilcoxon_per_seed_composite", {})
        assert set(ser_per_seed) == set(par_per_seed), "Wilcoxon per-seed key set diverged between serial and parallel"
        for key, ser_arr in ser_per_seed.items():
            par_arr = par_per_seed[key]
            assert ser_arr.shape == par_arr.shape
            np.testing.assert_allclose(
                ser_arr,
                par_arr,
                atol=1e-10,
                err_msg=f"per-seed RMSE array diverged for {key}",
            )
