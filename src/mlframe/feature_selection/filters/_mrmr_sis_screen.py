"""Sure-Independence-Screening (SIS) front gate for MRMR at very wide p (~100k).

Design source: ``tests/feature_selection/MRMR_100K_SCALING_DESIGN.md`` (measured cost map + cascade).

MRMR's full path (relevance MI + Fleuret conditional-MI redundancy + FE/synergy sweep) is super-linear in
the candidate-pool width and becomes infeasible at p ~ 100k. This module is GATE A of the cascade: a single
O(p*n) pass over ALL columns that scores each by

    fused_score_j = z(marginal_MI_j)   "max-rank"   z(second_moment_propensity_j)

and cuts 100k -> a few thousand survivors. Only the survivors then enter full MRMR (Gates B/C), which are
left completely unchanged.

WHY BOTH STATISTICS, FUSED
--------------------------
* marginal MI (``_mi_classif_batch``) catches MAIN-EFFECT features but ranks a pure-interaction operand
  (a*b with ~0 marginal MI by construction) at the noise floor;
* second-moment propensity (``second_moment_propensity`` = ``|corr(x^2,y)|+|corr(x,y^2)|``) catches exactly
  those zero-marginal interaction operands via higher-moment leakage.
Fusing = z-score each ranking and take the BEST-OF-EITHER rank, so an operand surviving on EITHER signal is
kept. Neither class of signal is lost. (Irreducible floor: a perfectly balanced XOR with zero higher-moment
leakage is invisible to ANY O(p) score -- out of scope for the screen, recoverable only by the O(cap^2)
sweep itself.)

MEMORY / I/O (load-bearing, measured -- see the design doc)
-----------------------------------------------------------
At p=100k the (n,p) frame is an 8 GB on-disk float32 memmap and is NEVER fully resident. We read it in
COLUMN blocks of width ``chunk_width`` (the survivor count is O(p) accumulators only). The production
``second_moment_propensity`` upcasts a block to float64 and squares it -> transiently ~3x a block's float32
footprint, so ``chunk_width`` is chosen from FREE RAM via the kernel_tuning_cache (never hardcoded).

HOTSPOT (cProfile, p=20000 n=4000, 2026-06-19)
----------------------------------------------
Screen wall ~6.8 s. The cost is entirely in the two REUSED sibling kernels (this module is glue):
``second_moment_propensity`` ~3.4 s (its column-standardize + corr matmul) and ``_mi_classif_batch`` njit
~2.5 s. Both are already numba/vectorised and one (the propensity kernel) is under active optimization by a
sibling change -- so the screen inherits those speedups for free; there is nothing to optimize in the glue.

DETERMINISM / PICKLE
--------------------
No global RNG; column-block order is ascending index; survivor-cut ties broken by ascending index. Identical
survivors across runs. This module stores no live numba/cuda kernel objects -- it only CALLS module-level
cached kernels in sibling modules -- so it is pickle-clean by construction.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------
# chunk-width selection from FREE RAM (kernel_tuning_cache, mirroring batch_pair_mi_gpu's dispatch pattern)
# ----------------------------------------------------------------------------------------------------------
def _free_ram_bytes() -> int:
    """Best-effort free physical RAM in bytes; conservative fallback if psutil is missing."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 2 * 1024**3  # 2 GB conservative fallback


def _ram_bucket(free_bytes: int) -> int:
    """Coarse log2 bucket so the cache key is stable across small RAM jitter."""
    gb = max(1, int(free_bytes // (1024**3)))
    return int(gb.bit_length())  # 1->1, 2-3->2, 4-7->3, 8-15->4, ...


def _fallback_chunk_width(n_rows: int, free_bytes: int) -> int:
    """Measurement-backed default chunk width.

    The second-moment kernel transiently needs ~3x a block's float64 footprint (upcast + square): a block is
    ``n_rows * chunk_width * 8`` float64 bytes, and we budget ~3x of that plus headroom. Use at most ~1/8 of
    free RAM for the transient so a sibling agent's large memmap is not starved. Clamp to a sane [256, 8192].
    """
    budget = max(1, free_bytes // 8)  # use at most 1/8 of free RAM for the transient
    # transient ~= 3 * n_rows * w * 8 bytes  ->  w = budget / (24 * n_rows)
    w = int(budget // (24 * max(1, n_rows)))
    return int(np.clip(w, 256, 8192))


def _choose_chunk_width(n_rows: int, p: int, free_bytes: int) -> int:
    """Look the chunk width up in the kernel_tuning_cache keyed on (n-bucket, free-RAM-bucket); fall back to
    the measured analytic default. Never hardcodes a single constant; deterministic for a given host+RAM."""
    n_bucket = int(max(1, n_rows).bit_length())
    ram_bucket = _ram_bucket(free_bytes)
    fb = _fallback_chunk_width(n_rows, free_bytes)
    fb = int(min(fb, max(1, p)))  # no point chunking wider than the frame
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        ktc = KernelTuningCache.load_or_create()
        hit = ktc.lookup("mrmr_sis_chunk_width", n_bucket=n_bucket, ram_bucket=ram_bucket)
        if hit and "chunk_width" in hit:
            w = int(hit["chunk_width"])
            return int(np.clip(min(w, max(1, p)), 1, max(1, p)))
        # Persist the analytic choice so a later micro-bench / retune can refine it; best-effort.
        try:
            ktc.update(
                "mrmr_sis_chunk_width",
                axes=["n_bucket", "ram_bucket"],
                regions=[{"n_bucket": n_bucket, "ram_bucket": ram_bucket, "chunk_width": fb}],
            )
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _mrmr_sis_screen.py:109: %s", e)
            pass
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _mrmr_sis_screen.py:111: %s", e)
        pass
    return fb


# ----------------------------------------------------------------------------------------------------------
# fusion + survivor-count rule
# ----------------------------------------------------------------------------------------------------------
def _zscore(a: np.ndarray) -> np.ndarray:
    """Standardize ``a`` to zero mean / unit std; returns all-zeros (not NaN) when the array has zero spread, so a constant score column contributes nothing to ``fuse_scores`` instead of poisoning it."""
    a = np.asarray(a, dtype=np.float64)
    mu = float(a.mean()) if a.size else 0.0
    sd = float(a.std())
    if sd <= 0.0:
        return np.zeros_like(a)
    return (a - mu) / sd


def fuse_scores(mi: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """Best-of-either fused score: per-column MAX of the z-scored MI and z-scored 2nd-moment propensity.

    Max (not sum) so a feature strong on EITHER signal survives -- a pure-interaction operand (MI~0,
    high propensity) is not diluted by its low MI, and a main effect (high MI, modest propensity) is not
    diluted by its propensity. Both classes surface."""
    return np.asarray(np.maximum(_zscore(mi), _zscore(prop)))


def survivor_count(
    fused: np.ndarray,
    *,
    k_target: Optional[int] = None,
    mad_c: float = 3.0,
    ram_cap: Optional[int] = None,
) -> int:
    """DATA-DERIVED survivor count ``m`` (NOT a hardcoded constant), per the design:

    m = max(
          information-knee: count of features above ``median + mad_c * MAD`` of the fused score,
          floor: max(20 * k_target, 1000)   so the downstream MRMR/synergy pool is never starved,
        )
    clamped to ``ram_cap`` (Gate-B discretized-pool RAM budget) and to ``p`` itself.

    Robust (median/MAD) so it adapts to how concentrated the signal is; deterministic (no RNG)."""
    p = int(fused.size)
    if p == 0:
        return 0
    med = float(np.median(fused))
    mad = float(np.median(np.abs(fused - med)))
    # 1.4826 scales MAD to a normal-consistent sigma estimate.
    thresh = med + mad_c * 1.4826 * mad
    knee = int(np.count_nonzero(fused > thresh))
    floor = max(20 * int(k_target or 0), 1000)
    m = max(knee, floor)
    if ram_cap is not None:
        # RAM cap SUPERSEDES the floor (a hard memory budget wins over the "never starve downstream" intent).
        # Warn when it does so the under-starved survivor set is not silent (2026-06-19, critique Low-5).
        if int(ram_cap) < floor and int(ram_cap) < min(knee if knee else floor, p):
            logger.warning(
                "sis_screen: RAM cap (%d) is below the survivor floor (%d); returning %d survivors -- the "
                "downstream MRMR pool may be starved. Free RAM or lower k_target to raise the cap.",
                int(ram_cap), floor, min(m, int(ram_cap)),
            )
        m = min(m, int(ram_cap))
    m = int(np.clip(m, 1, p))
    return m


def _ram_cap_survivors(n_rows: int, free_bytes: int) -> int:
    """Upper cap on survivors from the Gate-B discretized-pool RAM budget: an int16 (n, m) matrix is
    ``n * m * 2`` bytes; budget at most ~1/4 of free RAM for it."""
    budget = max(1, free_bytes // 4)
    return max(1000, int(budget // (2 * max(1, n_rows))))


# ----------------------------------------------------------------------------------------------------------
# the chunked screen
# ----------------------------------------------------------------------------------------------------------
def sis_screen(
    X: Any,
    y: Any,
    *,
    target_survivors: Optional[int] = None,
    k_target: Optional[int] = None,
    chunk_width: Optional[int] = None,
    nbins: int = 10,
    mad_c: float = 3.0,
    dedup_corr_thr: float = 0.92,
    return_scores: bool = False,
):
    """Score every column of a wide ``(n, p)`` matrix by fused (marginal-MI + 2nd-moment) signal in COLUMN
    BLOCKS (the matrix is never fully resident -- a memmap is read one block at a time) and return the
    survivor column indices.

    Parameters
    ----------
    X : np.ndarray | np.memmap, shape (n, p)
        Numeric feature matrix. May be an on-disk memmap; only ``chunk_width`` columns are resident at once.
    y : array-like, shape (n,)
        Target (any type; ``second_moment_propensity`` factorises non-numeric / discrete labels).
    target_survivors : int, optional
        If given, exactly this many top-fused survivors are returned (clamped to p). Otherwise the
        data-derived ``survivor_count`` rule decides.
    k_target : int, optional
        Requested number of finally-selected features (feeds the ``20*k_target`` survivor floor).
    chunk_width : int, optional
        Column block width. Default: chosen from free RAM via the kernel_tuning_cache.
    nbins : int
        Quantile bins for the marginal-MI estimator.
    return_scores : bool
        Also return the fused/MI/propensity score arrays.

    Returns
    -------
    survivors : np.ndarray[int]  (ascending column indices)
    (optional) dict of score arrays when ``return_scores``.
    """
    Xarr = X
    if hasattr(Xarr, "to_numpy"):  # pandas / polars -> ndarray (caller normally passes ndarray/memmap)
        Xarr = Xarr.to_numpy()
    if Xarr.ndim != 2:
        raise ValueError(f"sis_screen expects a 2-D (n, p) matrix; got shape {getattr(Xarr, 'shape', None)}")
    n, p = int(Xarr.shape[0]), int(Xarr.shape[1])

    y_arr = np.asarray(y.to_numpy()).ravel() if hasattr(y, "to_numpy") else np.asarray(y).ravel()
    if y_arr.shape[0] != n:
        raise ValueError(f"y length {y_arr.shape[0]} != n_rows {n}")

    free_bytes = _free_ram_bytes()
    if chunk_width is None:
        chunk_width = _choose_chunk_width(n, p, free_bytes)
    chunk_width = int(max(1, min(chunk_width, p)))

    from ._fe_interaction_prerank import second_moment_propensity
    from ._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch

    # MARGINAL-MI TARGET ENCODING (2026-06-19, critique P0-1). ``_mi_classif_batch`` int64-casts y, so a
    # CONTINUOUS regression target collapses to a single class -> MI==0 for EVERY column and the whole
    # marginal/main-effect half of the gate goes dead. Encode y to discrete classes first: factorise a
    # non-numeric target, and quantile-bin a continuous one (> nbins distinct numeric values) into nbins
    # codes -- mirroring second_moment_propensity's discrete/continuous switch. (second_moment_propensity
    # does its OWN y-encoding internally, so it keeps the raw y_arr.)
    y_mi = np.asarray(y_arr)
    if y_mi.dtype.kind in "USO" or y_mi.dtype == bool:
        _, y_mi = np.unique(y_mi, return_inverse=True)  # nominal labels -> codes
    elif y_mi.dtype.kind in "fc" or np.unique(y_mi).size > max(nbins, 2):
        yf = np.nan_to_num(y_mi.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        edges = np.quantile(yf, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        y_mi = np.searchsorted(edges, yf).astype(np.int64)  # continuous/high-card -> quantile bins
    y_mi = np.ascontiguousarray(y_mi)
    # CAVEAT (continuous target): the MI channel scores against this quantile-binned ``y_mi`` while ``second_moment_propensity`` keeps the RAW continuous ``y_arr`` and takes
    # the moment path (|corr(x^2,y)|+|corr(x,y^2)|). The two channels therefore use slightly DIFFERENT y representations -- they are not on a strictly comparable y-grid. This
    # is tolerable because ``fuse_scores`` is a MAX over the per-channel z-scores (robust to scale differences) AND this gate only decides which survivors enter full MRMR; it
    # never sets the final selection. Aligning the two channels onto one y-grid would be over-engineering for a coarse pre-rank.

    mi = np.zeros(p, dtype=np.float64)
    prop = np.zeros(p, dtype=np.float64)

    # REUSE-AUDIT RU-4 disposition (2026-06-19): aligning this screen's binning with categorize's content-hash
    # cache to avoid "re-binning survivors" was evaluated and REJECTED -- there is no reusable double-work. The
    # screen bins RAW columns with a fast fixed quantile (nbins=10) to SCORE/rank all p before selection;
    # categorize later bins only the ~2000 survivors with the DEFAULT supervised MDLP recipe for MRMR. Different
    # recipes, different purposes, both necessary -- the screen cannot use MDLP codes (MDLP needs the expensive
    # supervised pass the gate exists to avoid), and categorize cannot reuse the quantile codes. The MDLP bin on
    # survivors (~10.5s/2k cols, measured) is unavoidable for MRMR regardless of the screen.
    # Single ascending sweep over contiguous COLUMN blocks (deterministic order). Each block is materialized
    # as a small float32 buffer; the MI estimator and 2nd-moment kernel both consume it, then it is dropped.
    for j0 in range(0, p, chunk_width):
        j1 = min(j0 + chunk_width, p)
        block = np.ascontiguousarray(Xarr[:, j0:j1], dtype=np.float32)
        # marginal MI (full n -- a subsample collapses recall, see design). y_mi is class-encoded above.
        try:
            mi[j0:j1] = _mi_classif_batch(block, y_mi, nbins=nbins)
        except Exception as exc:  # never let one block kill the whole screen
            logger.warning("sis_screen: MI block [%d:%d] failed (%s); scored 0", j0, j1, exc)
        # second-moment interaction propensity (reuse the sibling kernel as-is)
        try:
            prop[j0:j1] = second_moment_propensity(block, y_arr)
        except Exception as exc:
            logger.warning("sis_screen: propensity block [%d:%d] failed (%s); scored 0", j0, j1, exc)
        del block

    fused = fuse_scores(mi, prop)

    if target_survivors is not None:
        m = int(np.clip(int(target_survivors), 1, p))
    else:
        ram_cap = _ram_cap_survivors(n, free_bytes)
        m = survivor_count(fused, k_target=k_target, mad_c=mad_c, ram_cap=ram_cap)

    # Top-m by fused score; ties broken by ascending index (lexsort: primary -fused, secondary +index).
    order = np.lexsort((np.arange(p), -fused))[:m]
    survivors = np.sort(order).astype(np.int64)

    # REDUNDANCY DEDUP (2026-06-19, reuse audit RU-1): collapse near-duplicate survivors BEFORE the downstream
    # O(k*p*n) Fleuret CMI loop chews on them. Reuses hybrid_selector.corr_clusters (the blocked O(n*p) greedy)
    # at |Pearson|>=dedup_corr_thr and keeps, per cluster, the HIGHEST-FUSED representative. Selection-neutral:
    # MRMR's CMI redundancy gate would reject these copies anyway (only the rep is selectable), so dropping them
    # upfront is pure speedup. The win is DATA-DEPENDENT -- ~1% on a mostly-independent survivor set, up to the
    # redundant fraction on a frame with real correlated families (measured: an 8-copy cluster collapses to 1,
    # ~63ms for 600 survivors). At thr>=0.92 only genuine near-linear duplicates merge; pure-interaction operands
    # (statistically independent) are never merged. Set dedup_corr_thr<=0 to disable.
    n_pre_dedup = int(survivors.size)
    n_clusters_collapsed = 0
    if dedup_corr_thr and 0.0 < float(dedup_corr_thr) <= 1.0 and survivors.size > 2:
        try:
            import pandas as pd
            from ..hybrid_selector import corr_clusters  # lazy: avoids any import cycle at module load

            # The float64 upcast is load-bearing, NOT a removable copy: ``Xarr`` is a float32 memmap at
            # wide p, and ``corr_clusters`` computes Pearson |corr| at the ``dedup_corr_thr`` boundary --
            # float32 accumulation would shift a correlation across the threshold and flip a cluster
            # membership (a selection change). The gather is over the SMALL (n x m) survivor sub-matrix
            # (m = post-screen survivors, a few thousand), never the full p-wide frame, and only when
            # ``dedup_corr_thr`` is set; ``corr_clusters`` requires a DataFrame (it reads ``.columns``).
            surv_df = pd.DataFrame(np.asarray(Xarr[:, survivors], dtype=np.float64), columns=[str(int(s)) for s in survivors])
            _, members = corr_clusters(surv_df, thr=float(dedup_corr_thr))
            pos = {str(int(s)): i for i, s in enumerate(survivors)}
            # representative = the cluster member with the highest fused screen score (signal-preserving).
            keep = [int(max(mem, key=lambda nm: fused[survivors[pos[nm]]])) for mem in members.values()]
            deduped = np.sort(np.asarray(keep, dtype=np.int64))
            n_clusters_collapsed = n_pre_dedup - int(deduped.size)
            if n_clusters_collapsed > 0:
                logger.info("sis_screen: redundancy dedup collapsed %d near-duplicate survivor(s) (%d -> %d) "
                            "at |corr|>=%.2f before the CMI loop.", n_clusters_collapsed, n_pre_dedup,
                            int(deduped.size), float(dedup_corr_thr))
                survivors = deduped
        except Exception as exc:  # correctness over the optimisation -- keep the full survivor set on any failure
            logger.warning("sis_screen: redundancy dedup skipped (%s: %s); keeping all %d survivors", type(exc).__name__, exc, n_pre_dedup)

    if return_scores:
        return survivors, {"fused": fused, "mi": mi, "propensity": prop, "chunk_width": chunk_width}
    return survivors


__all__ = ["sis_screen", "fuse_scores", "survivor_count"]
