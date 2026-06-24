"""``check_prospective_fe_pairs`` -- the FE pair-search core (candidate generation
via unary+binary ops, batched MI + permutation noise-gate, prewarp / median-gate
pseudo-unaries, chunked materialise, kernel-tuning dispatch).

This is the irreducible single-function body of the ``_feature_engineering_pairs``
subpackage; the supporting kernels / gates / dispatch live in sibling submodules
and are re-exported from the package ``__init__``.
"""
from __future__ import annotations

import logging
import os
from itertools import combinations

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype

from pyutilz.system import tqdmu

from ._pairs_chunks import _FE_CHUNK_MAX_COLS_HARD_CAP, _plan_fe_chunks
from ._pairs_gates import (
    _FE_REJECTION_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _GATE_MED_UNARY,
    _PREWARP_SPECS_RESULT_KEY, _PREWARP_UNARY,
)
from ._pairs_score import _score_one_pair
from ._pairs_setup import _build_operand_table, _fit_prewarp_and_gate_med

def _short_fe_name(name, maxlen: int = 30) -> str:
    """Truncate a (possibly long engineered) feature expression for live progress-bar
    display, keeping head + tail so both operator and operand stay legible
    (``mul(log(c),sin(d))`` -> ``mul(log(c..sin(d))``). Robust to non-str input."""
    try:
        s = str(name)
    except Exception:
        return "?"
    if len(s) <= maxlen:
        return s
    head = (maxlen - 2) // 2
    tail = maxlen - 2 - head
    return s[:head] + ".." + s[-tail:]


# Subsample default for the FE pair-search entry point. UNIFIED (2026-06-25) onto the single
# ``feature_engineering.UNIFIED_FE_SUBSAMPLE_N`` source of truth (30k) -- see the full rationale there.
# This duplicate constant used to hold an independent 200_000; it now aliases the unified knob so the FE
# block has ONE subsample value. (Lazy local import to avoid a circular import at module load: this
# package is imported by feature_engineering's transitive deps; the value is a plain int, read at def time.)
def _unified_fe_subsample_n() -> int:
    from ..feature_engineering import UNIFIED_FE_SUBSAMPLE_N

    return int(UNIFIED_FE_SUBSAMPLE_N)


try:
    FE_DEFAULT_SUBSAMPLE_N: int = _unified_fe_subsample_n()
except Exception:
    FE_DEFAULT_SUBSAMPLE_N = 30_000  # bootstrap fallback if feature_engineering not yet importable

logger = logging.getLogger(__name__)


def _fe_gpu_discretize_enabled(n_rows: int, n_cands: int) -> bool:
    """Whether to run the per-pair candidate MI (binning + observed-MI) on the GPU. The GPU path is
    BIT-IDENTICAL to the CPU analytic dispatch (verified maxdiff 0 on binning + observed MI), so the FE
    selection is unchanged either way -- this only chooses the faster backend for the size.

    ``MLFRAME_FE_GPU_DISCRETIZE`` tri-state: ``0/false`` forces CPU; ``1/true`` forces GPU when CUDA is
    present; unset/``auto`` (the default) routes per-host via kernel_tuning_cache -- GPU only above the
    measured n*K crossover, CPU below -- so a small fit is never regressed and a slow-H2D host that loses
    on GPU is routed to CPU. Requires CUDA; any GPU failure falls back to the CPU dispatcher downstream."""
    _env = os.environ.get("MLFRAME_FE_GPU_DISCRETIZE", "auto").strip().lower()
    if _env in ("0", "false", "no", "off"):
        return False
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return False
    except Exception:
        return False
    if _env in ("1", "true", "yes", "on"):
        return True
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, diagnostic, default OFF): force GPU past the KTC crossover.
    # The GPU pair-MI path is bit-identical to the CPU analytic dispatch (verified maxdiff 0) -> selection-equivalent.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            return True
    except Exception:
        pass
    try:  # auto: per-host crossover from kernel_tuning_cache (measurement-backed fallback)
        from .._gpu_resident_fe import fe_gpu_pairs_mi_backend_choice
        return fe_gpu_pairs_mi_backend_choice(int(n_rows), int(n_cands)) == "gpu"
    except Exception:
        return False


def _fe_gpu_binning_enabled(n_rows: int, n_cands: int) -> bool:
    """Whether to run the FE candidate BINNING (``discretize_2d_quantile_batch``) on the GPU.

    DECOUPLED (2026-06-23) from ``_fe_gpu_discretize_enabled`` / the full ``fe_gpu_pairs_mi`` analytic
    path. The binning alone (``gpu_discretize_codes_host``) is BIT-IDENTICAL to the CPU njit binning
    (verified maxdiff 0) and 17-24x faster at n=100k -- but the FULL pair-MI sweep cached "cpu" for the
    n_rows<=100000 region (its extra GPU MI/chi2 overhead lost a noisy A/B there), which wrongly forced
    the cheap binning back onto the CPU njit path (the #1 wall hotspot: 116s of a 228s GPU-mode F2 100k
    fit). The binning is a strictly simpler op with its own crossover, so it routes through its own
    dedicated KTC backend choice here. Selection is unchanged either way (bit-identical codes).

    ``MLFRAME_FE_GPU_BINNING`` tri-state: ``0/false`` forces CPU; ``1/true`` forces GPU when CUDA is
    present; unset/``auto`` (default) routes per-host via kernel_tuning_cache. ``MLFRAME_FE_GPU_DISCRETIZE=0``
    still forces CPU for BOTH (a global FE-GPU kill switch), so existing CPU-only configs are unchanged."""
    _env_all = os.environ.get("MLFRAME_FE_GPU_DISCRETIZE", "auto").strip().lower()
    if _env_all in ("0", "false", "no", "off"):
        return False  # global FE-GPU kill switch also disables binning
    _env = os.environ.get("MLFRAME_FE_GPU_BINNING", "auto").strip().lower()
    if _env in ("0", "false", "no", "off"):
        return False
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return False
    except Exception:
        return False
    if _env in ("1", "true", "yes", "on"):
        return True
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, diagnostic, default OFF): force GPU binning past the KTC
    # crossover. The GPU binning is bit-identical to the CPU njit binning (verified maxdiff 0) -> selection-equivalent.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            return True
    except Exception:
        pass
    try:  # auto: per-host binning crossover from kernel_tuning_cache (measurement-backed fallback)
        from .._gpu_resident_fe import fe_gpu_binning_backend_choice
        return fe_gpu_binning_backend_choice(int(n_rows), int(n_cands)) == "gpu"
    except Exception:
        return False


def check_prospective_fe_pairs(
    prospective_pairs,
    X,
    unary_transformations,
    binary_transformations,
    classes_y,
    classes_y_safe,
    freqs_y,
    num_fs_steps,
    cols,
    original_cols,
    fe_max_steps,
    fe_npermutations,
    fe_max_pair_features,
    fe_print_best_mis_only,
    fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    numeric_vars_to_consider,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    times_spent,
    verbose,
    # CRITICAL #2 follow-up (2026-05-21): subsample rows for the MI sweep so the
    # transformed_vars + shared_buffer allocations scale with subsample_n rather
    # than the (possibly multi-million-row) full X. Survivor columns are still
    # produced at full n via _rebuild_full_survivor_col so the caller contract
    # is preserved. Default 200_000 matches polynom_pair_fe's existing knob
    # (FE_DEFAULT_SUBSAMPLE_N); the standalone accuracy bench in
    # bench_fe_pair_subsample_accuracy.py shows jaccard=1.0 vs full at n_eff
    # >= 50k on synthetic 3-pair-competition data. 0 = use full data (legacy).
    subsample_n: int = FE_DEFAULT_SUBSAMPLE_N,
    subsample_seed: int = 42,
    # STRATIFIED SUBSAMPLE (R1, 2026-06-18). When True the MI-sweep row subsample below draws a
    # TARGET-STRATIFIED set of rows (per-class proportional for classification -- guaranteeing the
    # rare class survives; y-quantile-bin proportional for regression -- preserving the tails)
    # instead of the plain uniform ``rng.choice``. False (default) keeps the byte-identical legacy
    # uniform draw. The caller (``_mrmr_fe_step``) resolves the MRMR ``fe_subsample_stratify``
    # tri-state knob (None=auto) to a concrete bool via ``_resolve_fe_subsample_stratify``.
    fe_subsample_stratify: bool = False,
    # PER-OPERAND PRE-WARP (2026-06-02). When ``prewarp_enable`` is True the
    # unary/binary search gains, per raw operand, one extra "pseudo-unary"
    # ``prewarp(x)`` -- a learned 1-D orthogonal-polynomial warp fit JOINTLY
    # across the pair via the rank-1 ALS sweep (``hermite_fe.fit_pair_prewarp_als``,
    # which reuses the orthogonal-poly path's ``warm_start_als_seed``; an
    # INDEPENDENT 1-D fit cannot recover the b-side of a product target whose
    # b-marginal is ~0). This lets the elementary unary/binary path represent a
    # within-operand non-monotone distortion such as ``a**3 - 2a`` that no single
    # library unary can express, so a target ``F3(F1(a), F2(b))`` with a
    # non-monotone inner ``F1`` becomes recoverable.
    # ``prewarp_y`` is the SAME discretised target codes (``classes_y``) the MI
    # sweep already scores against; the fit is supervised but the produced column
    # is a closed-form function of ``x`` alone, so replay is leak-safe (the
    # fitted coeffs are stored in the recipe by ``_mrmr_fe_step``). Default OFF
    # so behaviour on data that does not need it is byte-identical.
    prewarp_enable: bool = False,
    prewarp_y: np.ndarray | None = None,
    # PREWARP ALS RECONSTRUCTION TARGET (2026-06-11). The rank-1 ALS warp fit /
    # held-out validation / winning-spec reconstruction is a least-squares solve
    # of ``y ~ f(a)*g(b)``; its fidelity depends on the RESOLUTION of the target it
    # reconstructs. The 2026-06-10 target-rebin guard coarsens ``classes_y`` to the
    # 10-bin equal-frequency screening codes (correctly -- the MI screen/gates need
    # the faithful coarse codes), but feeding those binned codes to the ALS dropped
    # the F-POLY non-monotone product reconstruction |corr| 0.97 -> 0.88. When the
    # CONTINUOUS target is threaded here it drives the ALS fit/validate/score ONLY;
    # the MI screen + every gate keep using ``classes_y`` codes. None -> legacy
    # behaviour (ALS reconstructs against ``classes_y``).
    prewarp_y_continuous: np.ndarray | None = None,
    # LINEAR-USABILITY GUARD TARGET (2026-06-17). The leader-equivalence tie-break and the
    # noise-wrap |corr| guard must score against the CONTINUOUS regression target, NOT the
    # binned ``classes_y`` codes: on a heavy-tailed target the Pearson |corr| of a magnitude-
    # carrying form (e.g. ``a**2/b``) with the quantile-RANK codes COLLAPSES (~0.05) while a
    # bounded monotone warp (e.g. ``a/sqrt(b)``) scores higher (~0.4) -- the EXACT INVERSE of
    # linear usability. The prewarp path already threaded continuous y, so the exhaustive
    # (prewarp-on) path was correct; the fast (prewarp-off) path fell back to ``classes_y`` and
    # picked the linearly-useless leg (biz_value_mrmr_fast_search MAE 0.05 -> 37). Thread the
    # continuous y here UNCONDITIONALLY (independent of prewarp) so both paths agree. None ->
    # legacy ``classes_y`` fallback (correct for classification / non-numeric y).
    usability_y_continuous: np.ndarray | None = None,
    prewarp_basis: str = "chebyshev",
    prewarp_max_degree: int = 4,
    # Minimum ratio (best-prewarp-MI / best-nonprewarp-MI) for the alternative
    # acceptance path. 1.20 = the prewarp must beat the elementary library by
    # >= 20% engineered MI to be admitted past the joint-prevalence gate. Tuned
    # to fire on the F-POLY non-monotone inner (measured uplift ~1.42x) while
    # staying silent on linear/monotone/noise (uplift ~1.0x there).
    prewarp_uplift_threshold: float = 1.20,
    # Held-out floor for the out-of-sample prewarp validation: a warp fit on a
    # train slice is kept only if its rank-1 reconstruction f(a)*g(b) tracks y on
    # a held-out slice with |corr| >= this. Rejects supervised overfitting on noise
    # operands at small n; 0.0 disables (legacy in-sample-only fit).
    prewarp_min_val_corr: float = 0.08,
    prewarp_specs_out: dict | None = None,
    # PER-OPERAND MEDIAN GATE (2026-06-04). When ``fe_gate_med_enable`` is True the
    # unary/binary search gains, per raw operand, one extra "pseudo-unary"
    # ``gate_med(x) = (x > train_median_x).astype(float)``. Combined with the
    # existing ``mul`` binary this lets the elementary path represent the
    # median-gated operators ``(a > median_a) * b`` (``mul(gate_med(a), b)``) and
    # the conjunction ``(a > median_a) & (b > median_b)``
    # (``mul(gate_med(a), gate_med(b))``) that a bilinear product cannot -- the
    # signal is non-product / conditional. Unlike a fixed threshold-0 gate, the
    # median ADAPTS the split to each operand's distribution, so it recovers the
    # gate on shifted / skewed operands where threshold-0 is useless (measured
    # skew-bench: gated_med +0.0355 / thr_and_med +0.0435 downstream-AUC d_mean
    # vs raw, beating products +0.022/+0.020 and threshold-0 +0.009/+0.0001).
    # The fitted state is ONE float per operand (the TRAIN median), stored in the
    # recipe by ``_mrmr_fe_step`` for leak-safe closed-form replay (no y, no
    # test-time recompute). The median does not overfit, so -- unlike prewarp --
    # NO held-out validation is needed; the gate is still subject to the same
    # MI-prevalence / external-validation acceptance gates every engineered
    # feature passes (it competes on equal footing in the per-pair MI sweep and
    # wins via ``best_mi`` only where the conditional form genuinely beats the
    # library). Default OFF so behaviour is byte-identical when not requested.
    fe_gate_med_enable: bool = False,
    gate_med_specs_out: dict | None = None,
    # OPT-A (2026-06-07): True when the caller (``_mrmr_fe_step``) dispatches this on the
    # SERIAL MAIN THREAD with NO joblib threading nest (the ``len(X) < 50000`` /
    # ``len(prospective_pairs) < 2`` branch). On that path the FE materialise / searchsorted
    # kernels may use their ``parallel=True`` column-prange twins (a numba prange is safe --
    # nothing nests it). On the joblib ``backend="threading"`` path this stays False so the
    # serial ``nogil`` kernels are used (a nested prange deadlocks the threading layer).
    # Byte-identical either way -- only thread-count of the embarrassingly-parallel per-column
    # work changes. Default False = the always-safe serial path.
    serial_main_thread: bool = False,
    # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08). When True (default), an operand var
    # that is NOT a raw ``feature_names_in_`` column (i.e. it is an engineered column
    # appended by a prior FE step, hence absent from ``original_cols``) is resolved by
    # NAME from the augmented frame ``X`` rather than skipped. This lets the step-k>1
    # pair search build COMPOSITES of two engineered features -- e.g. the additive
    # ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))`` that captures ~the entire
    # deterministic signal of ``y = a**2/b + log(c)*sin(d)``. The number of engineered
    # operands fed back is capped UPSTREAM (``fe_max_engineered_operands`` in
    # ``_mrmr_fe_step``) so the O(k^2) pair count stays bounded. Set False to restore
    # the legacy raw-only operand pool (no engineered x engineered composites).
    allow_engineered_operands: bool = True,
    # ENGINEERED-OPERAND CONTINUOUS VALUES (2026-06-08). ``{engineered_col_name ->
    # full-n float64 ndarray}`` of the CONTINUOUS engineered values produced by prior
    # FE steps. ``_extval_raw_col`` reads this for engineered operands so the pair
    # search combines the CONTINUOUS values rather than the augmented frame's
    # DISCRETISED bin codes (combining codes is severely lossy -- it sinks the
    # additive composite below the engineered-MI gate). ``None`` (default) -> fall
    # back to the by-name frame extract (bin codes), preserving the legacy behaviour.
    engineered_operand_values: dict | None = None,
    # MULTI-CANDIDATE DIVERSE EMISSION (2026-06-12). Per raw pair, the search picks the
    # single MAX-target-MI engineered form and discards every other. But MI is a RANK
    # statistic blind to LINEAR usability: on F2 ``log(2c)*sin(d/3)`` the MI-winner
    # ``sub(exp(c),cbrt(d))`` (MI 0.288) helps a LINEAR downstream by ~0 (MAE 0.092->0.093)
    # while the lower-MI ``mul(log(c),sin(d))`` (MI 0.264, INSIDE the 0.85 leaders band) is
    # the linearly-aligned additive term and cuts MAE 0.092->0.063. Emitting only one loses
    # whichever the actual model needs. With ``fe_multi_emit_max_per_pair > 1`` the search
    # additionally emits the next DISTINCT forms (greedy by target MI, skipping any whose
    # continuous values correlate above ``fe_multi_emit_diversity_corr`` with an
    # already-emitted column, down to ``fe_multi_emit_mi_floor`` x best_mi) -- a tree-friendly
    # AND a linear-friendly form both survive, and the downstream MRMR redundancy gate prunes
    # any residual overlap. The cap + the diversity filter keep it from flooding near-duplicates.
    fe_multi_emit_max_per_pair: int = 1,
    fe_multi_emit_mi_floor: float = 0.5,
    fe_multi_emit_diversity_corr: float = 0.90,
    # LARGE-N PEAK-MEMORY FIX (2026-06-08). Number of ``check_prospective_fe_pairs`` calls
    # that may run CONCURRENTLY in this process. On the serial-main-thread path this is 1; on
    # the joblib ``backend="threading"`` path it is ``n_jobs`` (each thread allocates its OWN
    # candidate / chunk / disc / MI buffers in the SHARED address space, so the per-call RAM
    # budget must be divided by the worker count or N threads collectively OOM). Used only to
    # SIZE the candidate buffers (chunk width + shared-buffer fit check) -- never changes which
    # candidates are produced or their MI, so selection stays byte-identical.
    concurrent_workers: int = 1,
    # MILLER-MADOW DEBIAS of the joint-prevalence RATIO gate (2026-06-09, backlog #1 + #4).
    # The gate ``best_mi / pair_mi > fe_min_engineered_mi_prevalence`` compares a 1-D
    # engineered MI (over ~``quantization_nbins`` bins) against a 2-D joint MI (over
    # ~``nbins^2`` bins). Both are RAW plug-in MIs whose positive finite-sample bias is
    # ``(k_x-1)(k_y-1)/2n``; the JOINT denominator's term is ~``nbins``x larger, so the raw
    # ratio is structurally depressed below 1.0 at small/moderate n. When True we subtract the
    # Miller-Madow MI bias term (OCCUPIED bin counts, #4) from BOTH sides before the ratio,
    # with a denominator-positivity guard that falls back to the raw ratio when the joint bias
    # swamps the finite-sample joint MI; the order-2 maxT floor is MM-debiased CONSISTENTLY
    # upstream (IRON RULE). bench-rejected as DEFAULT (2026-06-09): the isolated ratio fix is
    # real but it adds 0 end-to-end recovery on clean synergy (the marginal-uplift fallback
    # already recovers it) and ADMITS cross-mix noise on the weak F2 (cross-mix 3/10 -> 9/10
    # seeds, genuine_ab 10/10 -> 8/10). Default False (opt-in); see the full numbers + rationale
    # on ``MRMR.fe_mm_debias_prevalence``. False == legacy raw-plug-in ratio (byte-reproduction).
    fe_mm_debias_prevalence: bool = False,
    # REJECTION-LEDGER out-param (additive, 2026-06-11). When a list is passed, every pair
    # the per-pair acceptance gate REJECTS (joint-prevalence floor AND the marginal-uplift /
    # joint-recovery fallback both declined) appends one record dict carrying the operands,
    # the winning binary operator, the observed best_mi/pair_mi ratio, the prevalence
    # threshold, and the marginal-uplift diagnostics -- all values the gate ALREADY computed,
    # no recompute. The caller (``_mrmr_fe_step``) drains it into ``self``'s fe rejection
    # ledger. ``None`` (default) = legacy behaviour, no records captured.
    rejection_ledger_out: list | None = None,
):
    # Starting from the most heavily connected pairs, create a big pool of original features + their unary transforms. Individual vars referenced more than once go
    # to the global pool, the rest to the local (not stored)?

    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from ..feature_engineering import _FE_BUFFER_RAM_BUDGET_RATIO, _can_hoist_shared_buffer, _estimate_fe_shared_buffer_bytes, _fe_effective_buffer_budget_bytes, _rebuild_full_survivor_col, discretize_array, discretize_2d_quantile_batch, get_new_feature_name, gpu_compatible_unary_names, logger, mi_direct
    # 2026-06-05: batched FE-candidate MI + permutation noise-gate (bit-identical to the
    # per-candidate mi_direct on the default outer/n_workers=1 path -- see kernel docstring).
    from ..info_theory import batch_mi_with_noise_gate, use_su_normalization
    # P-SEAM (matrix-native FE replatform): the SINGLE integration point for the framework-agnostic
    # matrix path. GATED behind MLFRAME_FE_MATRIX_P0 -- default OFF, so this is a pure no-op and X is
    # byte-untouched (the legacy pandas path runs unchanged). When enabled, X is routed through the
    # single-copy float32 matrix adapter (a round-trip here today; on-device kernels in later phases),
    # so the SAME numba/cupy path can serve pandas and polars. The float32 cast is the intended P0
    # behaviour change. Wrapped so the experimental path can never break the production FE pipeline.
    from .._fe_matrix_io import fe_matrix_p0_enabled
    if fe_matrix_p0_enabled():
        try:
            from .._fe_matrix_io import from_feature_matrix, to_feature_matrix
            X = from_feature_matrix(to_feature_matrix(X))
        except Exception:
            logger.warning("FE matrix P-seam round-trip failed; using X unchanged.", exc_info=True)
    res = {}
    # REJECTION-LEDGER local accumulator (additive, 2026-06-11): per-pair acceptance-gate
    # drops collect here, then are exported via BOTH the ``rejection_ledger_out`` side channel
    # (serial / threading path) AND the reserved ``res`` key (survives the loky-parallel path,
    # where the caller's list cannot be mutated cross-process). Records carry only values the
    # gate already computed -- no recompute.
    _rejection_records: list = []

    # Seeded RNG for the external-validation factor subsample below. Pre-fix code used the
    # process-global ``np.random.choice`` there, which (a) made the choice depend on whatever
    # had consumed the global numpy RNG earlier in the process (so two fits of the SAME (X, y)
    # in one session could pick DIFFERENT validation factors -> a different tie-break -> a
    # different engineered-recipe SET / column NAMES), and (b) raced under the joblib
    # ``backend="threading"`` chunked path (N workers sharing one global RNG). Derive a local
    # Generator from ``subsample_seed`` (instance-controlled) so the factor pick is reproducible
    # from the MRMR seed and thread-safe, mirroring the seeded ``_rng_sub`` and the fleuret LCG fix.
    _rng_extval = np.random.default_rng(int(subsample_seed))

    # SUBSAMPLE-SETUP: when caller asks for subsample_n > 0 AND len(X) exceeds it,
    # build subsampled views of X / classes_y / classes_y_safe / freqs_y. The MI
    # sweep operates on these views; survivor packing always rebuilds at full n.
    # When subsample_n is 0 / negative / >= len(X) the legacy full-data path runs
    # unchanged (everything below uses ``X`` / ``classes_y`` / ... directly).
    _X_full = X
    _full_n_rows = len(_X_full)
    _use_subsample = isinstance(subsample_n, int) and 0 < subsample_n < _full_n_rows
    _sample_idx = None  # set below when subsampling; threaded into _fit_prewarp_and_gate_med
    if _use_subsample:
        _rng_sub = np.random.default_rng(int(subsample_seed))
        if fe_subsample_stratify:
            # Stratify on the discretised class codes (classification: balances classes; this path's
            # ``classes_y`` is the discrete target the MI sweep scores against). is_clf=True is correct
            # because ``classes_y`` is always discrete codes here (a continuous y is binned upstream).
            from .._fe_subsample import stratified_subsample_idx
            _sample_idx = stratified_subsample_idx(
                _rng_sub, np.asarray(classes_y), int(subsample_n), is_clf=True
            )
        else:
            _sample_idx = np.sort(_rng_sub.choice(_full_n_rows, size=int(subsample_n), replace=False))
        if isinstance(_X_full, pd.DataFrame):
            X = _X_full.iloc[_sample_idx].reset_index(drop=True)
        else:
            # Polars path -- row indexing returns a fresh frame; preserves zero-copy where possible.
            X = _X_full[_sample_idx]
        # Realign per-row target encodings; recompute freqs from the subsampled
        # class labels so MI estimates use the actual subsample distribution
        # rather than the full-n freq table (which would bias the MI estimator
        # toward classes that shrank under the random subset).
        _cy = np.asarray(classes_y)
        _cy_safe = np.asarray(classes_y_safe)
        classes_y = _cy[_sample_idx]
        classes_y_safe = _cy_safe[_sample_idx]
        # Recompute freqs from subsampled class labels. merge_vars returns
        # freqs_y as a FLOAT proportions array (sum=1.0), not raw counts; the
        # subsample needs the same shape. bincount gives counts -> divide by
        # total to get proportions matching the caller's expectation.
        if classes_y.size > 0 and classes_y.dtype.kind in ("i", "u"):
            _counts = np.bincount(classes_y.astype(np.int64))
            _total = _counts.sum()
            if _total > 0:
                freqs_y = (_counts.astype(np.float64) / float(_total))
        # else: leave the caller-supplied freqs_y; mi_direct handles its own
        # validation and would crash anyway on a non-integer class table.
        if verbose:
            logger.info(
                "check_prospective_fe_pairs: subsample_n=%d active (full_n=%d, %.1f%% sample); "
                "MI sweep runs on the subsample, survivor columns rebuilt at full n.",
                int(subsample_n), _full_n_rows, 100.0 * subsample_n / _full_n_rows,
            )

    # EXTERNAL-VALIDATION raw-column EXTRACTION MEMO (2026-06-07, LEVER 1).
    # The lazy external-validation tie-break (below) extracts the RAW values of
    # every external factor via ``X.iloc[:, original_cols[ext]].values`` (or the
    # polars ``.to_numpy()``) once per tied-leader config. The external-factor set
    # is ``numeric_vars_to_consider`` minus the 2 pair operands, so the SAME raw
    # column is re-extracted across every config AND every raw pair: on the wide
    # scene bed (2407x299) that is ~276k pandas ``.iloc`` calls / ~43s of pure
    # pandas-indexing (call-site cProfile). The extraction is DETERMINISTIC -- a
    # var-index maps to a FIXED column-values ndarray for the lifetime of this
    # call (``X`` is fixed after the subsample reassignment above) -- so memoising
    # it by the var key (the ``external_factor`` index into ``original_cols``)
    # yields BYTE-IDENTICAL values while collapsing the per-config re-extraction to
    # one extraction per distinct external factor. Keyed by the var id only, never
    # by array contents, so it cannot collide across distinct columns. The cache is
    # a plain local dict scoped to this call -> never pickled (no __getstate__
    # concern; the MRMR estimator never holds a reference to it).
    _extval_raw_col_cache: dict = {}

    def _densify_nullable(_arr):
        """Cast a pandas nullable extension array (Int64/Float64/boolean + pd.NA) to a plain
        float64 ndarray (pd.NA -> np.nan) so the numba/numpy unary-transform kernels can type it.
        No-op for ordinary numpy float64/int64 input -- only ExtensionArrays are converted."""
        if isinstance(getattr(_arr, "dtype", None), ExtensionDtype):
            return _arr.to_numpy(dtype=np.float64, na_value=np.nan)
        return _arr

    def _extval_raw_col(_var):
        """Memoised operand-values ndarray for var ``_var`` (cols-space index).

        For a RAW operand (``_var in original_cols``) returns ``X``'s column at the
        ``original_cols[_var]`` position (the RAW position into ``feature_names_in_``),
        bit-identical to the legacy ``X.iloc[...].values`` / ``.to_numpy()`` extract.

        ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): at FE step k>1 the operand pool
        also carries the engineered columns appended by the prior step(s)
        (``selected_vars`` includes their cols-space indices, so the pair-MI sweep
        surfaces ``(eng_i, eng_j)`` pairs -- e.g. the additive composite of the two
        real step-1 features that captures ~the entire deterministic signal). Those
        columns are NOT in ``original_cols`` (which holds raw ``feature_names_in_``
        positions only), but they ARE present in the AUGMENTED frame ``X`` under their
        ``cols[_var]`` name (``_mrmr_fe_step`` appends each engineered column to BOTH
        ``cols`` and ``X`` in lockstep). When ``allow_engineered_operands`` is on we
        fetch them by NAME so ``(eng_i, eng_j)`` can produce a real composite candidate.
        Returns ``None`` only when the var is neither a raw position nor a resolvable
        augmented-frame column (the caller then skips it, exactly as before)."""
        if _var in _extval_raw_col_cache:
            return _extval_raw_col_cache[_var]
        if _var in original_cols:
            if isinstance(X, pd.DataFrame):
                _vals = _densify_nullable(X.iloc[:, original_cols[_var]].values)
            else:
                _vals = X[:, original_cols[_var]].to_numpy()
            _extval_raw_col_cache[_var] = _vals
            return _vals
        # Engineered operand: resolve by name. PREFER the CONTINUOUS engineered values
        # (``engineered_operand_values[name]``) over the augmented frame's column, which
        # holds the DISCRETISED bin codes -- combining bin codes (e.g. ``add(codes_a,
        # codes_b)``) is severely lossy and sinks the composite below the engineered-MI
        # gate (measured: 0.88 from codes vs 1.81 -- the full signal -- from continuous
        # values). Fall back to the by-name frame extract when no continuous value is
        # stored (e.g. an engineered column produced by a stage that did not register one).
        if allow_engineered_operands and 0 <= _var < len(cols):
            _name = cols[_var]
            _vals = None
            if engineered_operand_values is not None:
                _cv = engineered_operand_values.get(_name)
                if _cv is not None:
                    _cv = np.asarray(_cv)
                    # The continuous store is full-n; align to the (possibly subsampled) X.
                    if _cv.shape[0] == len(X):
                        _vals = _cv
                    elif _use_subsample and _cv.shape[0] == _full_n_rows:
                        _vals = _cv[_sample_idx]
            if _vals is None:
                try:
                    if isinstance(X, pd.DataFrame):
                        _vals = _densify_nullable(X[_name]) if isinstance(X[_name].dtype, ExtensionDtype) else (
                            X[_name].to_numpy() if hasattr(X[_name], "to_numpy") else X[_name].values
                        )
                    elif hasattr(X, "columns") and _name in getattr(X, "columns", []):
                        _vals = X[_name].to_numpy()  # polars
                    else:
                        _vals = None
                except Exception:
                    _vals = None
            if _vals is not None:
                _vals = np.asarray(_vals)
                _extval_raw_col_cache[_var] = _vals
                return _vals
        _extval_raw_col_cache[_var] = None
        return None

    # Per-operand learned pre-warp + median-gate fit (carved to _pairs_setup.py).
    (_prewarp_active, _prewarp_spec_by_var, _gate_med_active, _gate_med_median_by_var) = _fit_prewarp_and_gate_med(
        prospective_pairs=prospective_pairs,
        prewarp_enable=prewarp_enable,
        prewarp_y=prewarp_y,
        prewarp_y_continuous=prewarp_y_continuous,
        prewarp_basis=prewarp_basis,
        prewarp_max_degree=prewarp_max_degree,
        prewarp_min_val_corr=prewarp_min_val_corr,
        fe_gate_med_enable=fe_gate_med_enable,
        original_cols=original_cols,
        _use_subsample=_use_subsample,
        _full_n_rows=_full_n_rows,
        _sample_idx=_sample_idx,
        _extval_raw_col=_extval_raw_col,
    )

    # Effective unary name list: the real registry plus the pre-warp pseudo-unary
    # when active. Used everywhere a per-pair combination over unary names is
    # built so the pseudo-unary participates exactly like a real one.
    _unary_names_eff = list(unary_transformations.keys())
    if _prewarp_active:
        _unary_names_eff = _unary_names_eff + [_PREWARP_UNARY]
    if _gate_med_active:
        _unary_names_eff = _unary_names_eff + [_GATE_MED_UNARY]

    # Exact preallocation. ``n_pairs * n_unary * 2`` over-counts because (var, tr_name) keys are de-duplicated in ``vars_transformations``; the unique-key set is the
    # true upper bound.
    unique_keys: set = set()
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        for var in raw_vars_pair:
            for tr_name in _unary_names_eff:
                unique_keys.add((var, tr_name))

    if verbose >= 2:
        logger.info(
            "Creating a pool of %d unary transformations for feature engineering "
            "(legacy upper bound was %d).",
            len(unique_keys),
            len(prospective_pairs) * len(unary_transformations) * 2,
        )

    transformed_vars = np.empty(shape=(len(X), len(unique_keys)), dtype=np.float32)

    # Hoist ``final_transformed_vals`` outside the per-pair loop: precompute each pair's ``combs``, find the max length, allocate one shared buffer. Each pair writes
    # then reads the same ``[:, i]`` slice so stale tail data is never observed.
    #
    # OPT-C structural-candidate-dedup analysed + REJECTED (2026-06-07, 0% yield): a proposal to
    # dedup PROVABLY-EQUAL candidate columns (symmetric binaries on equal operand-key sets) before
    # discretize+MI was measured to remove NOTHING -- this generation ALREADY eliminates every
    # provable structural duplicate:
    #   (1) ``combinations`` below emits each UNORDERED operand pair ONCE -> no symmetric-op order
    #       dups within a pair (``mul((a,u1),(b,u2))`` and ``mul((b,u2),(a,u1))`` never both appear);
    #   (2) ``tp[0][0] != tp[1][0]`` keeps operands from DISTINCT raw vars;
    #   (3) ``prospective_pairs`` keys are unique raw {a,b} pairs -> no two candidate columns across
    #       pairs share the same (op, operand-key set);
    #   (4) the shared UNARY operand columns (``sqr(a)`` reused across pairs (a,b),(a,c)) are already
    #       materialised ONCE via the ``vars_transformations`` {(var,unary):col} dict below.
    # Empirical: a canonical-(op, frozenset-of-operand-keys) scan over a 40-raw-pair x 13-unary x
    # 9-op pool = 60840 candidate columns (33800 symmetric-op) found ZERO provable duplicates
    # (0.00%). Any remaining "duplicate" would be VALUE-equal (not structurally provable) operands,
    # which the proposal explicitly excludes. So a structural-dedup pass is pure complexity for no
    # win; do NOT re-implement without a NEW source of structural collisions.
    pair_combs: dict = {}
    max_n_combs = 0
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        combs = list(
            combinations(
                [(raw_vars_pair[0], k) for k in _unary_names_eff]
                + [(raw_vars_pair[1], k) for k in _unary_names_eff],
                2,
            )
        )
        combs = [tp for tp in combs if tp[0][0] != tp[1][0]]
        pair_combs[raw_vars_pair] = combs
        if len(combs) > max_n_combs:
            max_n_combs = len(combs)

    # CRITICAL #2 (2026-05-21): memory-aware dispatch. The full hoisted buffer is the fast path
    # but on n=4M with medium preset this lands at ~17.6 GiB and crashes the suite. Estimate the
    # required buffer, check psutil.virtual_memory().available, and either keep the buffer (fast)
    # or set it to None and switch to recompute-from-metadata in the inner loop and survivor
    # rebuild stages (memory-safe; ~1% extra bin_func calls per pair).
    _n_binary = len(binary_transformations)
    final_transformed_vals_shared = None
    # CROSS-PAIR chunk budget (cols). Default 0 == not chunkable (per-pair buffer).
    # Filled below when the single-pair buffer fits RAM: the chunk buffer reuses the
    # SAME available-RAM budget but may hold MANY pairs (each pair packed whole).
    _fe_chunk_max_cols = 0
    # LARGE-N PEAK-MEMORY FIX (2026-06-08): the candidate buffers (chunk float32 + disc
    # int8 codes + batch-MI working set + the held-alive single-pair buffer) coexist while a
    # chunk is scored, and on the joblib ``backend="threading"`` path ``concurrent_workers``
    # copies of them are alive at once. So BOTH the single-pair fit check and the chunk-width
    # cap use the overhead+worker-aware envelope (raw 0.4*available / _FE_PEAK_OVERHEAD_FACTOR
    # / concurrent_workers) instead of the raw 0.4*available that under-counted the siblings
    # and ignored the thread multiplication. Byte-identical selection (width only).
    _n_workers = max(1, int(concurrent_workers))
    if max_n_combs > 0:
        _buf_bytes = _estimate_fe_shared_buffer_bytes(len(X), max_n_combs, _n_binary)
        _can_hoist, _bb, _avail = _can_hoist_shared_buffer(_buf_bytes, n_workers=_n_workers)
        if _can_hoist:
            try:
                final_transformed_vals_shared = np.empty(
                    shape=(len(X), max_n_combs * _n_binary),
                    dtype=np.float32,
                )
                # Cross-pair chunk width cap: the largest column count whose
                # ``n_rows * cols * 4`` float32 buffer stays inside the overhead+worker-aware
                # envelope (the SUM of the coexisting per-worker buffers fits the SAME 0.4
                # global budget). One pair (``max_n_combs * _n_binary`` cols) already passed,
                # so this is always >= one pair; we pack whole pairs up to it.
                _per_row_bytes = max(1, int(len(X)) * 4)
                _eff_budget_bytes = _fe_effective_buffer_budget_bytes(_avail, n_workers=_n_workers)
                if _eff_budget_bytes >= 0:
                    _fe_chunk_max_cols = int(_eff_budget_bytes // _per_row_bytes)
                else:
                    # No psutil reading -> bound the chunk to the hard cap only.
                    _fe_chunk_max_cols = _FE_CHUNK_MAX_COLS_HARD_CAP
                _fe_chunk_max_cols = max(
                    max_n_combs * _n_binary,
                    min(_fe_chunk_max_cols, _FE_CHUNK_MAX_COLS_HARD_CAP),
                )
            except MemoryError:
                # psutil over-reported available; falling back stays safe.
                final_transformed_vals_shared = None
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: shared buffer (%.1f GiB) allocation raised "
                        "MemoryError despite passing the available-RAM check (%.1f GiB available, "
                        "%.0f%% budget); switching to recompute-from-metadata fallback (~1%% extra "
                        "bin_func calls per pair).",
                        _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                        _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    )
        else:
            if verbose:
                logger.warning(
                    "check_prospective_fe_pairs: shared buffer would need %.1f GiB but only %.1f GiB "
                    "RAM is available (%.0f%% budget = %.1f GiB cap); using recompute-from-metadata "
                    "fallback path (~1%% extra bin_func calls per pair, identical survivors). To force "
                    "the fast path either free RAM or raise _FE_BUFFER_RAM_BUDGET_RATIO; to bound "
                    "compute, pass subsample_n>0 from the MRMR config.",
                    _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                    _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    (_avail * _FE_BUFFER_RAM_BUDGET_RATIO) / 2**30 if _avail >= 0 else float("nan"),
                )
    # bench-attempt-rejected (2026-06-17): BLOCK-STREAMING the candidate buffer when the full
    # per-pair buffer does not hoist (alloc a narrow N-col block buffer, flush materialise ->
    # discretize_2d_quantile_batch -> batched-MI per block; route the 7 downstream
    # ``final_transformed_vals[:, cfg]`` reads -- usability-corr, winner occupied-K, multi-emit,
    # ext-val, survivor -- through a recompute helper). Implemented + VALIDATED bit-identical
    # (block-on selection == per-column selection on the n=100k user fixture), but MEASURED SLOWER
    # in every config: n=100k 4-core box, per-column fallback 103s vs block 119s (joblib default) /
    # 192s (n_jobs=1). The per-block discretise+MI dispatch + the recompute of the downstream
    # winner/leader columns outweigh the parallel-kernel gain, and the per-column fallback ALREADY
    # RAM-bounds (one column at a time via ``_col_buf_1d``), so block-streaming adds no OOM benefit
    # either. The genuine speed lever is the FULL single-buffer batch path (n=100k 54s when it hoists)
    # firing more often, NOT blocking it. Do not re-attempt block-streaming for speed; if 1M-on-16GB
    # ever needs it for RAM, gate it on n and re-bench against the per-column serial baseline first.
    # In the recompute-fallback path we need to look up the
    # ``(transformations_pair, bin_func_name)`` for any index ``i`` that
    # was assigned in the inner loop, so the validation + survivor-packing
    # phases can rebuild the column on demand. The shared-buffer path
    # ignores this dict; either way it is bounded by ``max_n_combs *
    # _n_binary`` lightweight tuples per pair.
    _need_recompute_map = final_transformed_vals_shared is None

    vars_transformations = _build_operand_table(
        prospective_pairs=prospective_pairs,
        transformed_vars=transformed_vars,
        _unary_names_eff=_unary_names_eff,
        unary_transformations=unary_transformations,
        _extval_raw_col=_extval_raw_col,
        _prewarp_active=_prewarp_active,
        _prewarp_spec_by_var=_prewarp_spec_by_var,
        _gate_med_median_by_var=_gate_med_median_by_var,
        cols=cols,
        verbose=verbose,
        logger=logger,
        gpu_compatible_unary_names=gpu_compatible_unary_names,
    )

    if verbose >= 2:
        logger.info("Created. For every pair from the pool, trying all known functions...")

    # Per-operand marginal MI cache for the MARGINAL-UPLIFT alternative acceptance gate.
    # The marginal MI is the discretised RAW operand (its ``identity`` unary, already
    # materialised in ``transformed_vars``) scored against the target with the SAME
    # estimator the engineered columns use, so the uplift ratio is apples-to-apples.
    # Operands recur across pairs, so memoise by cols-space var index.
    _operand_marginal_mi_cache: dict = {}

    # NOISE-WRAP CORR-COLLAPSE GUARD (2026-06-15). A subsample-aligned CONTINUOUS target for a cheap |corr|
    # discriminator that catches the failure mode where the per-pair search WRAPS a strong, clean operand
    # (e.g. a univariate-basis column ``a__T2`` ~ a**2) with a PURE-NOISE operand (``e``) via an extreme
    # heavy-tailed transform (``sub(log(e),invqubed(a__T2))``). That composite's binned-MI ``best_mi/pair_mi``
    # ratio CLEARS the joint-prevalence gate (the extreme transform inflates BOTH MIs), yet its |corr| with the
    # target COLLAPSES to ~0 while the clean operand's |corr| is ~1.0 -- so the artefact displaces the clean
    # basis from the support and recovery dies. Genuine synergy pairs (a*b, log(c)*sin(d)) do NOT collapse this
    # way: the engineered column still tracks y monotonically. Prefer the continuous y; fall back to the binned
    # ``classes_y`` codes (still a usable monotone proxy) when no continuous target was threaded.
    _corr_y_cont = None
    _corr_y_cont_finite = None  # cached np.isfinite(_corr_y_cont); _corr_y_cont is never mutated after assignment
    try:
        _cyc_src = (
            usability_y_continuous if usability_y_continuous is not None
            else (prewarp_y_continuous if prewarp_y_continuous is not None else classes_y)
        )
        _cyc = np.asarray(_cyc_src, dtype=np.float64).ravel()
        if _use_subsample and _cyc.shape[0] == _full_n_rows:
            _cyc = _cyc[_sample_idx]
        if _cyc.shape[0] == len(classes_y) and np.isfinite(_cyc).any() and float(np.nanstd(_cyc)) > 1e-12:
            _corr_y_cont = _cyc
            _corr_y_cont_finite = np.isfinite(_corr_y_cont)
    except Exception:
        _corr_y_cont = None
        _corr_y_cont_finite = None

    def _safe_abs_corr(_v) -> float:
        """|Pearson corr| of a column with the (subsample-aligned) target over their jointly-finite rows;
        0.0 when the guard target is unavailable or either side is degenerate. Cheap (one corrcoef)."""
        if _corr_y_cont is None:
            return 0.0
        try:
            _a = np.asarray(_v, dtype=np.float64).ravel()
            if _a.shape[0] != _corr_y_cont.shape[0]:
                return 0.0
            _m = np.isfinite(_a) & _corr_y_cont_finite
            if int(_m.sum()) < 8:
                return 0.0
            _av, _yv = _a[_m], _corr_y_cont[_m]
            if float(np.std(_av)) <= 1e-12 or float(np.std(_yv)) <= 1e-12:
                return 0.0
            return abs(float(np.corrcoef(_av, _yv)[0, 1]))
        except Exception:
            return 0.0

    # The winning composite is condemned as a noise-wrap when its |corr| with the target is a small FRACTION
    # of the best single operand's |corr| AND that operand is genuinely strong on its own. Calibrated wide:
    # the artefact collapses to ~0.02 vs the clean operand's ~0.99 (a >40x collapse), while a genuine synergy
    # pair's engineered column tracks y at least comparably to its strongest operand. ``0.5`` keeps a 2x margin
    # so a real synergy that modestly trades linear |corr| for a higher-order MI gain is never condemned.
    _NOISE_WRAP_CORR_COLLAPSE_FRAC: float = 0.5
    _NOISE_WRAP_MIN_OPERAND_CORR: float = 0.30

    # bench-attempt-rejected (2026-06-07): BATCH all distinct operands' marginal MI through
    # one discretize_2d_quantile_batch + one _dispatch_batch_mi_with_noise_gate (Q9), instead
    # of this per-var single-column discretize_array + mi_direct. It WOULD be bit-identical
    # (the batch discretise + batch gate are already proven equal to the per-column path), but
    # it is not worth the invasive control-flow change + the risk to the load-bearing uplift
    # gate comparison below: _operand_marginal_mi is already memoised (each distinct operand
    # computed at most once) AND it only fires on the marginal-uplift FALLBACK gate (pairs that
    # miss the joint + prewarp gates), so on the scene scene-profile the ENTIRE mi_direct family
    # is only 0.1-0.3% of fit wall -- batching it saves a sub-noise fraction that is dwarfed by
    # the GPU-clock variance between runs. Re-evaluate only if a workload makes this gate hot.
    def _operand_marginal_mi(_var) -> float:
        if _var in _operand_marginal_mi_cache:
            return _operand_marginal_mi_cache[_var]
        _mi_val = 0.0
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _disc = discretize_array(
                    arr=transformed_vars[:, _idx], n_bins=quantization_nbins,
                    method=quantization_method, dtype=quantization_dtype,
                )
                _m, _ = mi_direct(
                    _disc.reshape(-1, 1),
                    x=np.array([0], dtype=np.int64), y=None,
                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                    classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence, npermutations=fe_npermutations,
                )
                _mi_val = float(_m)
            except Exception as _mm_exc:
                # FAIL-CLOSED (audit A3, 2026-06-13): the previous ``0.0`` was FAIL-OPEN -- it fed
                # the marginal-uplift gate's ``max(operand marginals)``, so a FAILED marginal on the
                # operand that actually has the LARGER marginal would shrink that max and LOOSEN the
                # admission bar (``best_nonprewarp_mi >= max_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO``),
                # wrongly admitting a feature whose uplift was never validated. Return +inf instead so
                # an UNKNOWN marginal can only TIGHTEN the gate (the pair fails the uplift fallback and
                # is dropped-on-uncertainty); it can still be admitted by the joint/prewarp gates, which
                # do not use this marginal. The whole-pair both-operand-fail case was already
                # fail-closed via the ``_max_operand_marginal > 0.0`` guard.
                if verbose:
                    logger.warning(
                        "MRMR FE: operand %s marginal-MI computation failed (%s); failing the "
                        "marginal-uplift gate CLOSED (+inf) so it cannot loosen admission.",
                        _var, type(_mm_exc).__name__,
                    )
                _mi_val = float("inf")
        _operand_marginal_mi_cache[_var] = _mi_val
        return _mi_val

    # MM-DEBIAS (2026-06-09, backlog #1 + #4): per-operand DISCRETISED codes for the
    # occupied-joint-K of a raw pair. Memoised + bit-identical to the discretise the
    # gate's ``pair_mi`` was computed over (same ``quantization_nbins`` / method /
    # ``transformed_vars`` identity column ``_operand_marginal_mi`` uses). Returns the
    # int code array, or None when the operand has no identity transform.
    _operand_disc_cache: dict = {}

    def _operand_discretized(_var):
        if _var in _operand_disc_cache:
            return _operand_disc_cache[_var]
        _codes = None
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _codes = discretize_array(
                    arr=transformed_vars[:, _idx], n_bins=quantization_nbins,
                    method=quantization_method, dtype=quantization_dtype,
                )
            except Exception:
                _codes = None
        _operand_disc_cache[_var] = _codes
        return _codes

    # CROSS-PAIR (CHUNK) BATCHING precompute (2026-06-06). Only on the hoist+quantile
    # path (the per-pair 3-phase batch path). We partition the prospective pairs into
    # CHUNKS whose total candidate-column count fits the RAM-budgeted chunk buffer,
    # then -- per chunk -- materialise ALL the chunk's candidate columns into ONE wide
    # buffer, run ONE discretize_2d + ONE batch_mi over the whole chunk, and stash the
    # per-pair results (ordered candidate list + per-candidate MI + buffer columns) in
    # ``_chunk_mi_cache``. The per-pair loop below consumes the cache (reads MI + the
    # chunk buffer columns) instead of doing its own per-pair Phase 1/2/3 -- enlarging
    # the njit-prange + GPU-dispatch batch from one pair's K to the whole chunk's
    # K_chunk, so the kernels saturate the cores / cross the GPU threshold. Bit-identical
    # (each column is scored independently; shuffle seeded by (0, perm_index) only).
    #
    # ``_chunk_mi_cache[raw_vars_pair]`` -> (ordered list of
    # (transformations_pair, bin_func_name, buf_col, uses_pw), fe_mi_array_aligned,
    # local_times_for_pair). ``_chunk_buffer`` is the wide buffer the buf_col indices
    # point into; it is held alive for the whole pair loop so survivor packing can read
    # ``_chunk_buffer[:, buf_col]`` exactly as the per-pair path read final_transformed_vals.
    _chunk_global_batch = (
        (final_transformed_vals_shared is not None)
        and (quantization_method == "quantile")
        and (_fe_chunk_max_cols > max_n_combs * _n_binary)  # chunk holds > 1 pair's worth
        and (len(prospective_pairs) > 1)
    )
    # RESIDENCY DEFERRAL gate (default OFF). When ON, the chunk's GPU FUSED codes path skips the (n,K)
    # float D2H (out_cand=None) and the few intermediate buffer reads below RE-MATERIALISE their column on
    # the GPU via ``_fe_materialise_block_gpu`` -- the SAME kernel that filled the buffer, so the bytes are
    # BIT-IDENTICAL (no cupy-vs-numpy ULP shift; the prior numpy-recompute scaffold flipped the clean-form
    # demotion). The operand table ``transformed_vars`` is uploaded ONCE per deferred chunk and cached;
    # per-buf_col columns are cached too (a column may be read several times). Only the GPU-fused path is
    # eligible (else the CPU binning still needs the host buffer); CPU/no-CUDA path is unchanged.
    _fe_defer_float = os.environ.get("MLFRAME_FE_GPU_DEFER_FLOAT", "1").strip().lower() in ("1", "true", "on", "yes")
    # Cross-pair chunk-materialise state, threaded through ``_score_one_pair`` as ONE mutable dict so the
    # lazy per-chunk load / reset semantics persist across pairs exactly as the in-loop locals did:
    #   loaded_idx     : index of the chunk currently materialised in ``_chunk_buffer`` (-1 = none).
    #   mi_cache       : ``_compute_one_fe_chunk`` result for the loaded chunk (per-pair MI + buf cols).
    #   float_deferred : the chunk's ``__float_deferred__`` signal (GPU buffer-D2H deferral).
    #   defer_meta     : (a_cols, b_cols, ops) int arrays for on-demand GPU re-materialise, per chunk.
    #   tv_gpu         : cupy upload of ``transformed_vars`` (the operand table), per chunk.
    #   resolved_cols  : buf_col -> host float32 column (re-materialised), per chunk.
    _chunk_state: dict = {
        "loaded_idx": -1,
        "mi_cache": {},
        "float_deferred": False,
        "defer_meta": None,
        "tv_gpu": None,
        "resolved_cols": {},
    }
    _chunk_buffer = None
    _pair_to_chunk: dict = {}   # raw_vars_pair -> chunk index
    _fe_chunks: list = []
    _pair_valid_combs: dict = {}
    if _chunk_global_batch:
        _fe_chunks, _pair_valid_combs, _chunk_buf_width = _plan_fe_chunks(
            prospective_pairs=prospective_pairs,
            pair_combs=pair_combs,
            vars_transformations=vars_transformations,
            n_binary=_n_binary,
            chunk_max_cols=_fe_chunk_max_cols,
        )
        # Only worth chunking when at least one chunk groups MORE than one pair.
        if _fe_chunks and max(len(c) for c in _fe_chunks) > 1 and _chunk_buf_width > 0:
            try:
                _chunk_buffer = np.empty((len(X), _chunk_buf_width), dtype=np.float32)
                for _ci_chunk, _chunk in enumerate(_fe_chunks):
                    for _p in _chunk:
                        _pair_to_chunk[_p] = _ci_chunk
                if verbose:
                    logger.info(
                        "check_prospective_fe_pairs: cross-pair chunk batching active "
                        "(%d pairs -> %d chunks, buffer %d cols, widest chunk %d pairs).",
                        len(prospective_pairs), len(_fe_chunks), _chunk_buf_width,
                        max(len(c) for c in _fe_chunks),
                    )
            except MemoryError:
                _chunk_buffer = None
                _pair_to_chunk = {}
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: cross-pair chunk buffer (%d x %d) raised "
                        "MemoryError; using per-pair batching.", len(X), _chunk_buf_width,
                    )
        else:
            _chunk_global_batch = False

    # Sweep-wide leader for the live "pair" bar postfix: the best engineered MI found
    # so far across ALL pairs in this FE step + the feature that produced it. Both are
    # already computed per pair (``best_mi`` / ``best_config``) -- display-only, no extra
    # MI work. Starts blank so the no-candidate / NaN edge case renders gracefully.
    _sweep_best_mi = -1.0
    _sweep_best_name = None

    # For every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM). Record best pairs.
    for (
        raw_vars_pair,
        pair_mi,
    ), _uplift in (pair_pbar := tqdmu(
        prospective_pairs.items(), desc="pair", leave=False, disable=not verbose
    )):  # better to start considering form the most prospective pairs with highest mis ratio!

        (_pair_res_entry, best_config, best_mi) = _score_one_pair(
            raw_vars_pair=raw_vars_pair,
            pair_mi=pair_mi,
            chunk_state=_chunk_state,
            rejection_records=_rejection_records,
            rejection_ledger_out=rejection_ledger_out,
            X=X,
            transformed_vars=transformed_vars,
            vars_transformations=vars_transformations,
            binary_transformations=binary_transformations,
            unary_transformations=unary_transformations,
            pair_combs=pair_combs,
            final_transformed_vals_shared=final_transformed_vals_shared,
            _need_recompute_map=_need_recompute_map,
            _chunk_global_batch=_chunk_global_batch,
            _chunk_buffer=_chunk_buffer,
            _pair_to_chunk=_pair_to_chunk,
            _fe_chunks=_fe_chunks,
            _pair_valid_combs=_pair_valid_combs,
            _fe_defer_float=_fe_defer_float,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            fe_npermutations=fe_npermutations,
            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
            quantization_nbins=quantization_nbins,
            quantization_method=quantization_method,
            quantization_dtype=quantization_dtype,
            num_fs_steps=num_fs_steps,
            fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
            fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
            fe_max_external_validation_factors=fe_max_external_validation_factors,
            numeric_vars_to_consider=numeric_vars_to_consider,
            fe_max_steps=fe_max_steps,
            fe_print_best_mis_only=fe_print_best_mis_only,
            fe_mm_debias_prevalence=fe_mm_debias_prevalence,
            _prewarp_active=_prewarp_active,
            prewarp_uplift_threshold=prewarp_uplift_threshold,
            _PREWARP_UNARY=_PREWARP_UNARY,
            _corr_y_cont=_corr_y_cont,
            _corr_y_cont_finite=_corr_y_cont_finite,
            _NOISE_WRAP_CORR_COLLAPSE_FRAC=_NOISE_WRAP_CORR_COLLAPSE_FRAC,
            _NOISE_WRAP_MIN_OPERAND_CORR=_NOISE_WRAP_MIN_OPERAND_CORR,
            fe_multi_emit_max_per_pair=fe_multi_emit_max_per_pair,
            fe_multi_emit_mi_floor=fe_multi_emit_mi_floor,
            fe_multi_emit_diversity_corr=fe_multi_emit_diversity_corr,
            cols=cols,
            original_cols=original_cols,
            _use_subsample=_use_subsample,
            _X_full=_X_full,
            _full_n_rows=_full_n_rows,
            _prewarp_spec_by_var=_prewarp_spec_by_var,
            _gate_med_median_by_var=_gate_med_median_by_var,
            engineered_operand_values=engineered_operand_values,
            _rng_extval=_rng_extval,
            _n_workers=_n_workers,
            times_spent=times_spent,
            verbose=verbose,
            serial_main_thread=serial_main_thread,
            _extval_raw_col=_extval_raw_col,
            _safe_abs_corr=_safe_abs_corr,
            _operand_marginal_mi=_operand_marginal_mi,
            _operand_discretized=_operand_discretized,
            batch_mi_with_noise_gate=batch_mi_with_noise_gate,
            use_su_normalization=use_su_normalization,
            discretize_array=discretize_array,
            discretize_2d_quantile_batch=discretize_2d_quantile_batch,
            mi_direct=mi_direct,
            get_new_feature_name=get_new_feature_name,
            _rebuild_full_survivor_col=_rebuild_full_survivor_col,
            _can_hoist_shared_buffer=_can_hoist_shared_buffer,
            _fe_gpu_discretize_enabled=_fe_gpu_discretize_enabled,
        )
        if _pair_res_entry is not None:
            res[raw_vars_pair] = _pair_res_entry

        # Live progress: surface the best engineered feature found so far in this sweep
        # (its MI with y) plus the pair just evaluated, on the "pair" bar. ``best_mi`` /
        # ``best_config`` are already computed for this pair -- no extra MI compute. Robust
        # to the no-config / NaN edge cases (we only adopt a finite, improving best_mi).
        if verbose:
            try:
                _bm = float(best_mi)
                if best_config is not None and np.isfinite(_bm) and _bm > _sweep_best_mi:
                    _sweep_best_mi = _bm
                    _sweep_best_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
                _cur_pair = f"{cols[raw_vars_pair[0]]},{cols[raw_vars_pair[1]]}"
                _pf = {"pair": _short_fe_name(_cur_pair, 22)}
                if _sweep_best_name is not None:
                    _pf["best"] = f"{_short_fe_name(_sweep_best_name)}={_sweep_best_mi:.4f}"
                pair_pbar.set_postfix(_pf, refresh=False)
            except (TypeError, ValueError, IndexError):
                pass

    # Surface the fitted per-operand pre-warp specs (keyed by cols-space var
    # index) so the caller (``_mrmr_fe_step``) can persist them in each survivor
    # recipe for leak-safe replay. Only the non-None specs that were actually
    # fitted are exported. We populate BOTH the optional ``prewarp_specs_out``
    # side-channel (works for the in-process serial path) AND a reserved key in
    # the returned ``res`` (survives the loky-parallel path where the side
    # channel dict cannot be mutated cross-process; the caller merges per-chunk
    # results). The reserved key is a private 3-tuple that can never collide with
    # a real ``raw_vars_pair`` (which is always length 2).
    _fitted_specs = {_v: _s for _v, _s in _prewarp_spec_by_var.items() if _s is not None}
    if _fitted_specs:
        if prewarp_specs_out is not None:
            prewarp_specs_out.update(_fitted_specs)
        res[_PREWARP_SPECS_RESULT_KEY] = _fitted_specs

    # Same dual-channel export for the fitted per-operand TRAIN medians so the
    # caller can persist them in each survivor recipe for leak-safe replay. The
    # value is a single float per cols-space var index.
    _fitted_medians = {_v: float(_m) for _v, _m in _gate_med_median_by_var.items()}
    if _fitted_medians:
        if gate_med_specs_out is not None:
            gate_med_specs_out.update(_fitted_medians)
        res[_GATE_MED_SPECS_RESULT_KEY] = _fitted_medians

    # REJECTION LEDGER export via the reserved result key (survives the loky-parallel path).
    if _rejection_records:
        res[_FE_REJECTION_RESULT_KEY] = _rejection_records

    return res
