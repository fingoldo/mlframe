"""Second funnel stage for the wide-frame synergy sweep: a FORCE/OPT-IN GPU-exhaustive
joint-MI pass over ALL raw numeric columns (bypassing the ``fe_synergy_screen_max_features``
pre-rank cap) for the IRREDUCIBLE balanced-interaction case.

Background (the funnel)
-----------------------
The first stage (``_fe_interaction_prerank.top_k_by_interaction_propensity``, shipped
2026-06-19) ranks the raw numeric columns by ``|corr(x^2,y)|+|corr(x,y^2)|`` and keeps the
top ``cap`` when the frame is wider than the cap, so the existing O(p^2) joint-MI synergy
sweep stays bounded. That pre-rank PROVABLY cannot recover a *perfectly balanced* (L=0)
interaction: a pair ``(a,b)`` whose every univariate higher moment vs y is zero (e.g. a
balanced XOR / sign product) carries ZERO signal in any O(p) per-column score, so neither
operand enters the kept cap. Only the EXHAUSTIVE C(p,2) joint-MI sweep -- which the measured
CUDA kernel ``batch_pair_mi_cuda`` runs at ~5e4 pairs/s, fitting 4 GB at p<=10k -- recovers
such a pair (it ranks a planted balanced XOR pair #0 of 50M with joint MI = ln2).

This module is the SECOND funnel stage: when correctness outranks wall-time, run the full
exhaustive sweep over every raw numeric column instead of the capped pre-rank path.

Decision policy (``decide_exhaustive_sweep``)
---------------------------------------------
* ``fe_synergy_exhaustive == "never"``: always take the pre-rank + capped sweep path; the
  exhaustive path does NOT fire (this is what isolates ``fe_synergy_prerank``).
* ``fe_synergy_exhaustive == "auto"`` (default): ESCALATE to the full exhaustive C(p,2) sweep
  whenever it is affordable -- i.e. unless a time budget is set (``max_runtime_mins`` /
  ``fe_synergy_exhaustive_max_seconds``) AND the PREDICTED wall-time exceeds it; with no budget set,
  auto always sweeps. The prediction uses the available backend (CUDA where present, else the CPU
  njit-prange fallback) so the affordable-or-not verdict is hardware-independent. Falls back to the
  pre-rank path only when the predicted sweep would blow the budget.
* ``fe_synergy_exhaustive in {"force", True}``: run the FULL exhaustive C(p,2) sweep over ALL raw
  numeric columns whenever a backend exists, regardless of the budget.

The throughput (pairs/s) is NEVER hardcoded into the decision: it is measured-and-cached
per host + (n, p) region via ``pyutilz.performance.kernel_tuning`` (mirroring the
``batch_pair_mi`` dispatch + ``joint_hist_batched``). The ~5e4 pairs/s figure is only the
source-code FALLBACK applied on a cold cache / when pyutilz is unavailable.

This module owns NO kernel: the exhaustive sweep reuses the existing
``batch_pair_mi_cuda`` via ``dispatch_batch_pair_mi(force_backend="cuda")`` in the caller.
"""
from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Source-code FALLBACK throughput (pairs/s) for the measured CUDA kernel on the bench GTX
# 1050 Ti (cc 6.1): RESULTS.md reported ~5e4 pairs/s (p=2000 -> 38 s, p=5000 -> 241 s,
# p=10000 -> 1004 s). Used ONLY when the kernel_tuning_cache has no entry for the live HW.
# Per `feedback_use_kernel_tuning_cache_for_gpu` this is NOT the dispatch number -- the live
# path measures-and-caches per (n, p) below.
_EXHAUSTIVE_FALLBACK_PAIRS_PER_SEC = 5.0e4

# CPU (njit-prange) fallback throughput (pairs/s). The exhaustive sweep has a CPU backend
# (batch_pair_mi_njit_prange); on a GPU-less host the exhaustive decision must be made on the CPU cost so
# that feature EXISTENCE (recovering a balanced L=0 interaction) does NOT depend on whether a CUDA device
# is present -- only on whether the sweep is affordable on the available hardware. Conservative: the prange
# kernel is ~1-2 orders slower than the CUDA one, so default to ~2e3 pairs/s (a GPU-less host then runs
# exhaustive only at small p / generous budget, and force-mode runs it regardless).
_EXHAUSTIVE_CPU_FALLBACK_PAIRS_PER_SEC = 2.0e3

# Tuner sweep grid for the throughput model. Sparse on purpose -- one cheap CUDA timing per
# region; the cache interpolates the nearest region for the (n, p) at decision time.
_EXH_TPUT_SWEEP_N_SAMPLES = [50_000, 200_000, 1_000_000]
_EXH_TPUT_SWEEP_N_PAIRS = [4096, 32768, 131072]
_EXH_TPUT_SALT = 1


def _measure_exhaustive_throughput(dims: dict) -> float:
    """Time ``batch_pair_mi_cuda`` on a synthetic (n_samples, n_pairs) cell and return the
    achieved pairs/second. Used as the kernel_tuning tuner body; on any failure raises so the
    orchestrator records the fallback region instead."""
    from .batch_pair_mi_gpu import batch_pair_mi_cuda

    n_samples = int(dims["n_samples"])
    n_pairs = int(dims["n_pairs"])
    rng = np.random.default_rng(0)
    nbins_val = 8
    n_features = 64
    factors_data = rng.integers(0, nbins_val, size=(n_samples, n_features)).astype(np.int32)
    nbins = np.full(n_features, nbins_val, dtype=np.int32)
    pair_a = rng.integers(0, n_features, size=n_pairs).astype(np.int64)
    pair_b = (pair_a + 1 + rng.integers(0, n_features - 1, size=n_pairs)) % n_features
    pair_b = pair_b.astype(np.int64)
    classes_y = rng.integers(0, 4, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=4).astype(np.float64) / max(1, n_samples)

    # Warm-up (JIT compile) not timed.
    batch_pair_mi_cuda(factors_data, pair_a[:64], pair_b[:64], nbins, classes_y, freqs_y)
    t0 = time.perf_counter()
    batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
    dt = time.perf_counter() - t0
    if dt <= 0.0:
        return float(_EXHAUSTIVE_FALLBACK_PAIRS_PER_SEC)
    return float(n_pairs / dt)


def measured_pairs_per_second(n_samples: int, n_pairs: int) -> tuple[float, str]:
    """Per-host measured CUDA pair-MI throughput (pairs/s) for this (n_samples, n_pairs),
    looked up from the kernel_tuning_cache (measured on first miss). Returns
    ``(pairs_per_second, source)`` where source is "cache" | "fallback".

    NEVER hardcodes the throughput in the dispatch decision: the ~5e4 figure is only the
    cold-cache fallback (mirrors ``batch_pair_mi`` + ``joint_hist_batched``)."""
    try:
        from ._kernel_tuning import get_kernel_tuning_cache

        cache = get_kernel_tuning_cache()
        if cache is not None:
            region = cache.lookup("batch_pair_mi_exhaustive_throughput", n_samples=int(n_samples), n_pairs=int(n_pairs))
            if region is not None:
                # The tuner stores the throughput under the region's "value" / "choice".
                val = region.get("value", region.get("choice"))
                if val is not None:
                    try:
                        pps = float(val)
                        if pps > 0:
                            return pps, "cache"
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:  # cache miss / pyutilz unavailable -> fallback
        logger.debug("exhaustive-throughput cache lookup failed (%s: %s); using fallback", type(exc).__name__, exc)
    return float(_EXHAUSTIVE_FALLBACK_PAIRS_PER_SEC), "fallback"


def warm_exhaustive_throughput_cache() -> None:
    """Populate the per-host throughput cache via ``get_or_tune`` (one-time, async-safe). Best
    effort: callers may skip this and rely on ``measured_pairs_per_second``'s fallback."""
    try:
        from ._kernel_tuning import get_kernel_tuning_cache
        from .batch_pair_mi_gpu import _CUDA_AVAIL

        if not _CUDA_AVAIL:
            return
        cache = get_kernel_tuning_cache()
        if cache is None:
            return

        def _tuner() -> list:
            regions = []
            for n in _EXH_TPUT_SWEEP_N_SAMPLES:
                for npairs in _EXH_TPUT_SWEEP_N_PAIRS:
                    try:
                        pps = _measure_exhaustive_throughput({"n_samples": n, "n_pairs": npairs})
                    except Exception:
                        pps = float(_EXHAUSTIVE_FALLBACK_PAIRS_PER_SEC)
                    regions.append({"n_samples": n, "n_pairs": npairs, "value": pps})
            return regions

        cache.get_or_tune(
            "batch_pair_mi_exhaustive_throughput",
            dims={"n_samples": _EXH_TPUT_SWEEP_N_SAMPLES[0], "n_pairs": _EXH_TPUT_SWEEP_N_PAIRS[0]},
            tuner=_tuner,
            axes=["n_samples", "n_pairs"],
            fallback=lambda n_samples, n_pairs: _EXHAUSTIVE_FALLBACK_PAIRS_PER_SEC,
            salt=_EXH_TPUT_SALT,
            once_per_process=True,
        )
    except Exception as exc:
        logger.debug("exhaustive-throughput cache warm failed (%s: %s)", type(exc).__name__, exc)


def predict_exhaustive_seconds(n_samples: int, n_raw: int) -> tuple[float, float, str]:
    """Predicted wall-time (seconds) for the full exhaustive C(n_raw, 2) joint-MI sweep at
    ``n_samples`` rows. Returns ``(predicted_seconds, pairs_per_second, throughput_source)``."""
    n_pairs = (int(n_raw) * (int(n_raw) - 1)) // 2
    pps, source = measured_pairs_per_second(n_samples, n_pairs)
    if pps <= 0:
        pps = float(_EXHAUSTIVE_FALLBACK_PAIRS_PER_SEC)
    return float(n_pairs) / pps, pps, source


def _normalise_mode(raw) -> str:
    """Map the ``fe_synergy_exhaustive`` knob to {"never", "auto", "force"}.

    * False / "never"/"off"/"prerank" -> "never": always the O(p) pre-rank + capped sweep (guaranteed fast,
      never pays for the GPU sweep -- the legacy behaviour).
    * "auto" (default; also None) -> "auto": run the exhaustive C(p,2) sweep WHEN it is affordable (a CUDA GPU
      is available AND the predicted wall-time <= fe_synergy_exhaustive_max_seconds), else fall back to the
      pre-rank. So at small / moderate p the default gets the COMPLETE result -- including the balanced (L=0)
      interactions the O(p) pre-rank provably cannot reach -- essentially for free, and only wide frames where
      exhaustive would blow the budget fall back to the pre-rank.
    * True / "force"/"1"/"yes"/"on" -> "force": run the exhaustive sweep whenever a GPU is available, IGNORING
      the time budget (the user explicitly wants completeness and accepts the wall-time)."""
    if raw is True:
        return "force"
    if raw is False:
        return "never"
    if raw is None:
        return "auto"
    s = str(raw).strip().lower()
    if s in ("force", "true", "1", "yes", "on"):
        return "force"
    if s in ("never", "off", "no", "none", "prerank", "pre-rank", "0"):
        return "never"
    return "auto"


def _resolve_exhaustive_budget_seconds(self):
    """Resolve the wall-time budget (seconds) for the auto-escalation, from MRMR's OWN time budget.

    Priority: an explicit ``fe_synergy_exhaustive_max_seconds`` override (if set) wins; otherwise
    ``max_runtime_mins`` * 60 (MRMR's fit-wide budget). When NEITHER is set the budget is ``None`` =
    UNLIMITED -- auto runs the exhaustive sweep regardless of p (the user did not ask to bound wall-time)."""
    override = getattr(self, "fe_synergy_exhaustive_max_seconds", None)
    if override is not None:
        try:
            ov = float(override)
            if ov > 0:
                return ov
        except (TypeError, ValueError):
            pass
    mins = getattr(self, "max_runtime_mins", None)
    if mins is not None:
        try:
            m = float(mins)
            if m > 0:
                return m * 60.0
        except (TypeError, ValueError):
            pass
    return None  # no budget set anywhere -> do not bound p


def decide_exhaustive_sweep(
    self,
    *,
    n_samples: int,
    n_raw: int,
    verbose,
) -> tuple[bool, str]:
    """Decide whether the GPU-exhaustive synergy sweep should run over ALL ``n_raw`` raw numeric columns
    (bypassing the pre-rank cap), recovering balanced (L=0) interactions the O(p) pre-rank cannot.

    Returns ``(use_exhaustive, reason)``. The decision depends on ``fe_synergy_exhaustive``:
      * "never"  -> always False (pre-rank).
      * "auto"   -> True iff a CUDA GPU is available AND predicted wall-time <= the budget (affordable);
                    else False (pre-rank). This makes the DEFAULT exhaustive-when-cheap, pre-rank-when-not.
      * "force"  -> True whenever a CUDA GPU is available, regardless of the budget.
    The reason string is logged either way.
    """
    mode = _normalise_mode(getattr(self, "fe_synergy_exhaustive", "auto"))
    if mode == "never":
        return False, "never (pre-rank + capped sweep; fe_synergy_exhaustive='auto' escalates to exhaustive when affordable)"

    if n_raw < 2:
        return False, f"declined: only {n_raw} raw numeric column(s); nothing to sweep"

    try:
        from .batch_pair_mi_gpu import _CUDA_AVAIL
    except Exception:
        _CUDA_AVAIL = False

    budget = _resolve_exhaustive_budget_seconds(self)   # MRMR's own max_runtime_mins; None => unlimited
    n_pairs = (n_raw * (n_raw - 1)) // 2
    if _CUDA_AVAIL:
        # Opportunistically warm the per-host throughput cache so the prediction is measured, not the cold
        # fallback (no-op if already warm).
        warm_exhaustive_throughput_cache()
        predicted, pps, source = predict_exhaustive_seconds(n_samples, n_raw)
        backend = "CUDA"
    else:
        # No GPU: predict on the CPU njit-prange backend (which DOES exist) rather than hard-declining.
        # The exhaustive sweep recovers a balanced (L=0) interaction the O(p) pre-rank provably cannot, so
        # gating its EXISTENCE on GPU presence makes a feature appear on a CUDA host and vanish on a CPU one.
        # Decide on the CPU cost instead -> the decision is hardware-independent (affordable-or-not), not
        # device-gated; the sweep then runs on the CPU backend (see _step_core force_backend selection).
        pps = float(_EXHAUSTIVE_CPU_FALLBACK_PAIRS_PER_SEC)
        predicted = float(n_pairs) / pps
        source = "cpu-njit-prange-fallback"
        backend = "CPU"
    if mode == "force":
        return True, (
            f"running FULL exhaustive C({n_raw},2)={n_pairs}-pair joint-MI sweep over ALL raw numeric columns "
            f"on the {backend} backend (force; predicted {predicted:.1f}s @ {pps:.0f} pairs/s [{source}], "
            f"budget ignored) -- recovers balanced (L=0) interactions the O(p) pre-rank cannot."
        )
    # AUTO: escalate to exhaustive unless a budget is set AND the predicted sweep would exceed it. With NO
    # budget set (max_runtime_mins is None and no explicit override) p is NOT limited -- auto always sweeps.
    # The prediction uses the AVAILABLE backend's throughput, so the affordability verdict is consistent
    # with what will actually run (CUDA where present, CPU prange otherwise).
    if budget is not None and predicted > budget:
        return False, (
            f"auto -> pre-rank: predicted exhaustive wall-time {predicted:.1f}s on the {backend} backend "
            f"(C({n_raw},2)={n_pairs} pairs @ {pps:.0f} pairs/s [{source}]) exceeds the MRMR time budget "
            f"{budget:.0f}s (from max_runtime_mins / fe_synergy_exhaustive_max_seconds); the O(p) pre-rank "
            f"recovers leaky interactions cheaply (only a perfectly-balanced L=0 interaction is missed). "
            f"Raise max_runtime_mins or set fe_synergy_exhaustive='force'."
        )
    _bud = "unlimited (no MRMR time budget set)" if budget is None else f"<= budget {budget:.0f}s"
    return True, (
        f"auto -> exhaustive: FULL C({n_raw},2)={n_pairs}-pair joint-MI sweep over ALL raw numeric columns "
        f"on the {backend} backend (predicted {predicted:.1f}s @ {pps:.0f} pairs/s [{source}], {_bud}) -- the "
        f"complete result, recovering balanced (L=0) interactions the pre-rank cannot."
    )


__all__ = [
    "decide_exhaustive_sweep",
    "predict_exhaustive_seconds",
    "measured_pairs_per_second",
    "warm_exhaustive_throughput_cache",
]
