"""Regression tests for mrmr_audit_2026-07-20 findings B-5/B-6/B-7 (all in ``_confirm_predictor.py``):

* B-5 -- ``confirm_candidate``'s marginal-confirmation ``mi_direct``/``mi_direct_gpu`` calls never
  received a seed, so the CPU branch always drew the identical ``base_seed=0`` permutation stream for
  every candidate regardless of ``ctx.random_seed`` (the GPU branch drew fresh unseeded entropy every
  call, non-reproducible either way). Fixed via a per-candidate ``_marginal_base_seed`` folding both
  ``random_seed`` and ``hash(X)`` into the seed passed to ``mi_direct``.
* B-6 -- ``_fleuret_base_seed`` omitted the candidate's own identity, so every distinct candidate
  confirmed at the same ``selected_vars`` depth within one greedy round drew the IDENTICAL permutation
  stream for the Fleuret conditional recheck. Fixed by folding ``hash(X)`` into the formula.
* B-7 -- ``mi_direct_gpu``/``mi_direct_gpu_batched``/``mi_direct_gpu_batched_streamed`` had no seed
  parameter at all; their permutation shuffle was unconditionally unseeded ``cp.random.default_rng()``.
  Fixed by adding an optional ``base_seed`` that seeds the CuPy Generator when given.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _multi_signal_fixture(seed: int = 0, n: int = 600, n_features: int = 6):
    """Several independently-useful raw features so the confirmation loop runs multiple ``confirm_candidate`` calls within one greedy round."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    return df, pd.Series(y, name="y")


def _mrmr_kw(random_seed, **overrides):
    """Shared MRMR constructor kwargs that reliably exercise the full_npermutations confirm path."""
    kw = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        random_seed=random_seed,
        use_simple_mode=False,
        full_npermutations=8,
        baseline_npermutations=4,
        max_runtime_mins=1.0,
    )
    kw.update(overrides)
    return kw


class TestB5MarginalConfirmSeedVariesByCandidate:
    """The marginal-confirmation base_seed must not be constant across candidates."""

    def test_marginal_base_seed_varies_across_candidates(self, monkeypatch):
        """Spy on ``mi_direct``'s ``base_seed`` kwarg during a real fit; pre-fix it was never passed
        (always the function default 0) for every candidate -- post-fix it must vary."""
        from mlframe.feature_selection.filters import _confirm_predictor as cp
        from mlframe.feature_selection.filters.mrmr import MRMR

        seen_seeds: list = []
        real_mi_direct = cp.mi_direct

        def _spy_mi_direct(*args, **kwargs):
            """Capture the base_seed kwarg on every call, then delegate to the real function."""
            seen_seeds.append(kwargs.get("base_seed"))
            return real_mi_direct(*args, **kwargs)

        monkeypatch.setattr(cp, "mi_direct", _spy_mi_direct)

        X, y = _multi_signal_fixture()
        MRMR(**_mrmr_kw(random_seed=42)).fit(X, y)

        assert seen_seeds, "mi_direct was never called -- fixture did not exercise the confirm path"
        assert all(s is not None for s in seen_seeds), f"base_seed was not threaded through on every call: {seen_seeds}"
        assert len(set(seen_seeds)) > 1, f"base_seed was constant across every candidate (pre-fix behaviour): {seen_seeds}"

    def test_marginal_base_seed_depends_on_random_seed(self, monkeypatch):
        """Two fits differing only in ``random_seed`` must produce a DIFFERENT observed base_seed
        sequence -- pre-fix ``ctx.random_seed`` never reached this call site at all."""
        from mlframe.feature_selection.filters import _confirm_predictor as cp
        from mlframe.feature_selection.filters.mrmr import MRMR

        def _run_and_capture(seed):
            """Run one fit, capturing every base_seed value mi_direct was called with."""
            seen: list = []
            real_mi_direct = cp.mi_direct

            def _spy(*args, **kwargs):
                """Capture the base_seed kwarg on every call, then delegate to the real function."""
                seen.append(kwargs.get("base_seed"))
                return real_mi_direct(*args, **kwargs)

            monkeypatch.setattr(cp, "mi_direct", _spy)
            X, y = _multi_signal_fixture()
            MRMR(**_mrmr_kw(random_seed=seed)).fit(X, y)
            monkeypatch.undo()
            return seen

        seeds_a = _run_and_capture(1)
        seeds_b = _run_and_capture(2)
        assert seeds_a, "fixture did not exercise the confirm path"
        assert seeds_a != seeds_b, f"changing random_seed did not change the observed base_seed sequence: {seeds_a} == {seeds_b}"


class TestB5B6ReproducibilityUnderSameSeed:
    """Two fits with the SAME random_seed must produce the identical selection (the whole point of seeding)."""

    def test_same_random_seed_reproducible_selection(self):
        """Fitting twice with the identical random_seed must select the identical feature set."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _multi_signal_fixture()
        r1 = MRMR(**_mrmr_kw(random_seed=7)).fit(X, y).get_feature_names_out()
        r2 = MRMR(**_mrmr_kw(random_seed=7)).fit(X, y).get_feature_names_out()
        assert list(r1) == list(r2), f"same random_seed produced different selections: {list(r1)} vs {list(r2)}"


class TestB7GpuSeedParameterExists:
    """mi_direct_gpu / mi_direct_gpu_batched / mi_direct_gpu_batched_streamed must accept base_seed."""

    def test_mi_direct_gpu_accepts_base_seed_kwarg(self):
        """mi_direct_gpu's signature must expose an optional base_seed kwarg defaulting to None."""
        import inspect
        from mlframe.feature_selection.filters.gpu import mi_direct_gpu

        sig = inspect.signature(mi_direct_gpu)
        assert "base_seed" in sig.parameters, "mi_direct_gpu must accept a base_seed kwarg (mrmr_audit_2026-07-20 B-7)"
        assert sig.parameters["base_seed"].default is None

    def test_mi_direct_gpu_batched_accepts_base_seed_kwarg(self):
        """Both batched GPU variants' signatures must expose an optional base_seed kwarg defaulting to None."""
        import inspect
        from mlframe.feature_selection.filters._gpu_batched import mi_direct_gpu_batched, mi_direct_gpu_batched_streamed

        for fn in (mi_direct_gpu_batched, mi_direct_gpu_batched_streamed):
            sig = inspect.signature(fn)
            assert "base_seed" in sig.parameters, f"{fn.__name__} must accept a base_seed kwarg (mrmr_audit_2026-07-20 B-7)"
            assert sig.parameters["base_seed"].default is None


@pytest.mark.gpu
class TestB7GpuSeedReproducibility:
    """When cupy/CUDA is available, mi_direct_gpu(base_seed=...) must be reproducible across calls."""

    def test_mi_direct_gpu_reproducible_with_same_base_seed(self):
        """Two mi_direct_gpu calls with the identical base_seed must return the identical result."""
        cp = pytest.importorskip("cupy")
        if not cp.cuda.is_available():
            pytest.skip("CUDA not available on this host")
        from mlframe.feature_selection.filters.gpu import mi_direct_gpu

        rng = np.random.default_rng(0)
        n = 2000
        factors_data = np.column_stack(
            [
                rng.integers(0, 4, n),
                (rng.integers(0, 4, n)),
            ]
        ).astype(np.int32)
        r1 = mi_direct_gpu(factors_data, x=(0,), y=(1,), factors_nbins=np.array([4, 4], dtype=np.int64), npermutations=16, base_seed=123)
        r2 = mi_direct_gpu(factors_data, x=(0,), y=(1,), factors_nbins=np.array([4, 4], dtype=np.int64), npermutations=16, base_seed=123)
        assert r1 == r2, f"mi_direct_gpu(base_seed=123) was not reproducible across calls: {r1} != {r2}"
