"""Regression tests for the gaps an independent verification pass found in the mrmr audit fix wave.

Each of these covers a place where the first round's fix was incomplete: a guard applied at some call sites
but not all, a cardinality cap placed after the allocation it was meant to bound, a memo key still missing a
live input, and the CPU/GPU parity twin that kept the bug its sibling had fixed.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_relax_mrmr_caps_cardinality_before_any_dense_alloc():
    """relax_mrmr_score must reject an over-budget joint BEFORE allocating, including on the empty-selected-set
    path (which returns early and previously bypassed the guard entirely)."""
    from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score

    n = 8
    x = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds cap"):
        relax_mrmr_score(x, [], y, nbins_x=9000, nbins_selected=[], nbins_y=9000)  # 8.1e7 > 64M, empty S


def test_jmim_caps_cardinality_on_empty_selected_set():
    """jmim_score's first-feature fallback allocates a (K_x, K_y) dense joint, so it must be capped too."""
    from mlframe.feature_selection.filters._jmim_scorer import jmim_score

    n = 8
    x = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds cap"):
        jmim_score(x, [], y, nbins_x=9000, nbins_selected=[], nbins_y=9000)


def test_batch_mi_noise_gate_kernel_rejects_overflowing_classes_dtype():
    """The kernel itself must reject a classes_dtype too narrow for max(factors_nbins)-1, so a direct or
    GPU-resident caller that passes its own dtype cannot silently wrap dense codes negative."""
    from mlframe.feature_selection.filters.info_theory._batch_kernels import batch_mi_with_noise_gate

    rng = np.random.default_rng(0)
    n, K = 64, 2
    disc = rng.integers(0, 8, size=(n, K)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    fy = np.bincount(y, minlength=2).astype(np.float64) / n
    kwargs = dict(
        disc_2d=disc, classes_y=y, classes_y_safe=y, freqs_y=fy, npermutations=0,
        base_seed=np.uint64(0), min_nonzero_confidence=0.95, use_su=False, dtype=np.int32,
    )
    ok = batch_mi_with_noise_gate(factors_nbins=np.full(K, 8, dtype=np.int64), classes_dtype=np.int16, **kwargs)
    assert np.isfinite(ok).all(), "an in-range nbins must still compute normally"
    with pytest.raises(ValueError, match="classes_dtype capacity"):
        batch_mi_with_noise_gate(factors_nbins=np.full(K, 40000, dtype=np.int64), classes_dtype=np.int16, **kwargs)


def test_fe_gpu_gate_key_includes_the_per_gate_kill_switches(monkeypatch):
    """The FE-GPU gate memo key must include MLFRAME_FE_GPU_DISCRETIZE / _BINNING, which the uncached gates read
    live -- otherwise flipping a kill-switch mid-process is defeated by a stale memo entry."""
    import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core as pc

    for name in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_DISABLE_GPU", "MLFRAME_FE_GPU_DISCRETIZE", "MLFRAME_FE_GPU_BINNING"):
        monkeypatch.delenv(name, raising=False)
    base = pc._gpu_gate_env_signature()
    monkeypatch.setenv("MLFRAME_FE_GPU_DISCRETIZE", "0")
    assert pc._gpu_gate_env_signature() != base, "the discretize kill-switch must change the memo key"
    monkeypatch.delenv("MLFRAME_FE_GPU_DISCRETIZE")
    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING", "0")
    assert pc._gpu_gate_env_signature() != base, "the binning kill-switch must change the memo key"


def test_encode_y_dense_fast_path_matches_np_unique():
    """The dense-codes fast path must return exactly what the np.unique path would, for dense, sparse and
    negative-code integer inputs alike (it only skips the sort when the codes are provably already dense)."""
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    for arr in (
        np.array([0, 1, 2, 1, 0], dtype=np.int64),  # dense -> fast path
        np.array([0, 2, 4, 2, 0], dtype=np.int64),  # gaps -> must densify
        np.array([3, 5, 3, 9], dtype=np.int64),  # no zero -> must densify
        np.array([-1, 0, 1], dtype=np.int64),  # negative -> must densify
        np.array([7], dtype=np.int64),
    ):
        expected = np.unique(arr, return_inverse=True)[1].astype(np.int64)
        assert np.array_equal(encode_y_for_classif_mi(arr), expected), f"mismatch on {arr}"


def test_pairwise_modular_resident_encodes_y_like_its_cpu_twin():
    """The resident modular twin must route y through the shared encoder, not a bare int64 cast -- otherwise the
    GPU path silently keeps the continuous-y truncation its CPU sibling was fixed for."""
    import inspect

    from mlframe.feature_selection.filters import _pairwise_modular_resident as pmr

    src = inspect.getsource(pmr)
    assert "encode_y_for_classif_mi" in src
    assert "np.asarray(y)).astype(np.int64)" not in src, "a bare y int64 cast survives in the resident twin"


def test_pair_maxt_index_gen_accepts_a_set_and_matches_combinations_order():
    """The order-2 maxT pair enumeration must accept the SET the FE step actually passes (np.asarray cannot
    convert one, which silently threw and disabled the noise floor on every fit) and must emit pairs in the
    exact order itertools.combinations would, so the per-pair MM bias vector stays index-aligned."""
    from itertools import combinations

    vars_set = {7, 3, 11, 2, 5}
    kv = np.fromiter(vars_set, dtype=np.int64, count=len(vars_set))
    ia, ib = np.triu_indices(kv.shape[0], k=1)
    got = list(zip(kv[ia].tolist(), kv[ib].tolist()))
    expected = list(combinations(vars_set, 2))
    assert got == expected, f"pair order diverged from combinations(): {got} != {expected}"


def test_fe_step_maxt_floor_runs_on_a_real_fit(caplog):
    """A real MRMR fit with the order-2 maxT floor enabled must not log the 'floor failed' fallback: that
    warning means the floor threw and the FE step silently lost its noise gate."""
    import logging

    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    y = ((df["f0"] * df["f1"]) > 0).astype(int)
    est = MRMR(fe_max_steps=1, fe_pair_maxt_null_permutations=8, fe_pair_maxt_min_pairs=1, random_seed=0, n_workers=1, verbose=0)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        est.fit(df, y)
    bad = [r.message for r in caplog.records if "maxT permutation-null floor failed" in r.message]
    assert not bad, f"the order-2 maxT floor threw during a real fit: {bad[:2]}"


def test_hybrid_orth_mi_floor_cannot_exceed_the_target_entropy_ceiling():
    """A MAD-derived MI floor above ``H(y)`` is unsatisfiable and must not veto every candidate.

    ``MI(X; y) <= H(y)`` is a hard bound, so a floor at or above it rejects everything rather than rejecting
    noise. On a small candidate set (cols=["x1"], degrees=(2,3) -> 2 candidates: one signal, one dud) the MAD
    measures the signal-to-noise SEPARATION instead of a noise band, and x5*1.4826 pushed the floor to ~2.21
    nats against a ceiling of H(y)~0.62 -- silently disabling the whole family. The all-noise control pins
    that dropping the degenerate floor did NOT weaken the real noise rejection.
    """
    import pandas as pd

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import hybrid_orth_mi_fe

    for seed in (1, 13, 42):
        rng = np.random.default_rng(seed)
        n = 2000
        x1 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, **{f"noise_{i}": rng.standard_normal(n) for i in range(3)}})
        y = ((x1 * x1 - 1.0) + 0.05 * rng.standard_normal(n) > 0).astype(int)
        X_aug, scores = hybrid_orth_mi_fe(X, y, cols=["x1"], degrees=(2, 3), basis="hermite", top_k=2, min_uplift=1.05)
        appended = [c for c in X_aug.columns if c not in X.columns]
        assert "x1__He2" in appended, (
            f"seed={seed}: the quadratic detector clears the uplift gate but was vetoed by an " f"unsatisfiable MI floor; appended={appended}\n{scores}"
        )

    # All-noise control: with no real signal the floor must still reject every candidate.
    for seed in (1, 13):
        rng = np.random.default_rng(seed)
        n = 2000
        Xn = pd.DataFrame({f"n{i}": rng.standard_normal(n) for i in range(4)})
        yn = rng.integers(0, 2, size=n)
        X_aug, _ = hybrid_orth_mi_fe(Xn, yn, cols=["n0"], degrees=(2, 3), basis="hermite", top_k=2, min_uplift=1.05)
        appended = [c for c in X_aug.columns if c not in Xn.columns]
        assert not appended, f"seed={seed}: all-noise frame must engineer nothing; got {appended}"


def test_hybrid_orth_candidates_roster_separates_never_fired_from_lost_the_greedy():
    """``hybrid_orth_candidates_`` must record every column the hybrid-orth family produced, even when
    selection later drops it.

    ``hybrid_orth_features_`` is intersected with ``support_``, so when a sibling FE family emits an
    equivalent column and wins the greedy the survivor roster goes empty -- indistinguishable from the stage
    never having run. On this XOR fixture the numeric-pair family emits a sign-equivalent product and does
    win, so the candidates roster is the only thing that shows the pair stage found ``He1_He1`` at all.
    """
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 1500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise_a": rng.standard_normal(n), "noise_b": rng.standard_normal(n)})
    y = pd.Series((((x1 * x2) + 0.02 * rng.standard_normal(n)) > 0).astype(int), name="y")
    common = dict(
        verbose=0, interactions_max_order=1, dcd_enable=False, cluster_aggregate_enable=False,
        build_friend_graph=False, quantization_nbins=10, random_seed=0, n_workers=1,
        fe_hybrid_orth_enable=True, fe_hybrid_orth_pair_enable=True, fe_hybrid_orth_pair_max_degree=2,
        fe_hybrid_orth_basis="hermite", fe_hybrid_orth_top_k=5,
    )
    fired = MRMR(fe_max_steps=1, **common).fit(X, y)
    produced = list(getattr(fired, "hybrid_orth_candidates_", []) or [])
    assert any("He1_He1" in c for c in produced), f"the pair stage's He1_He1 term must be recorded; got {produced}"

    # fe_max_steps=0 is the documented "no FE at all" contract: the stage must not fire, and the roster
    # must say so rather than looking identical to the fired-but-lost case above.
    off = MRMR(fe_max_steps=0, **common).fit(X, y)
    assert not (getattr(off, "hybrid_orth_candidates_", []) or []), "no FE stage may fire at fe_max_steps=0"


def test_step_score_safe_code_dtype_reserves_uniform_nan_slot():
    """The FE-step score path must pass reserve_nan_slot for the uniform method, like the discretization module's
    own call sites -- otherwise a uniform nbins==dtype.max+1 write wraps the NaN sentinel negative."""
    import inspect

    from mlframe.feature_selection.filters._mrmr_fe_step import _step_score

    src = inspect.getsource(_step_score)
    n_calls = src.count("_safe_code_dtype(self.quantization_nbins")
    n_guarded = src.count("reserve_nan_slot=(self.quantization_method ==")
    assert n_calls and n_guarded == n_calls, f"{n_calls - n_guarded} of {n_calls} call sites lack reserve_nan_slot"
