"""Entry point for the separate KTC-free GPU-resident FE step (``MLFRAME_FE_GPU_STRICT`` +
``MLFRAME_FE_GPU_STRICT_RESIDENT``).

Phase 0: this is a SCAFFOLD. ``run_fe_step_gpu_strict`` raises ``NotImplementedError`` so the caller's
try/except falls back to the existing per-family FE step -> zero behavior change until a later phase implements
the resident pipeline and the resident flag is turned on. The branch in ``_run_fe_step`` is gated behind the
default-OFF resident flag, so STRICT itself is unaffected too."""
from __future__ import annotations

import os


def fe_gpu_strict_resident_enabled() -> bool:
    """Whether the resident GPU-strict FE stages are active. DEFAULT ON under ``MLFRAME_FE_GPU_STRICT``.

    The resident GPU stages (recipe replay, fourier detection, prewarp ALS, usability pool, pure-form
    retention, the CMI perm-null) are selection-equivalent to the CPU path and are now the DEFAULT behaviour
    of STRICT: whenever ``MLFRAME_FE_GPU_STRICT`` is on they engage. ``MLFRAME_FE_GPU_STRICT_RESIDENT=0`` is
    the explicit OPT-OUT (kept so the resident path can still be disabled per-fit for diagnosis / rollback
    without touching the byte-identical DEFAULT non-strict path). STRICT itself is a selection-equivalent
    force-GPU mode (the CPU/CUDA backends agree to ~1e-9); the byte-identical contract lives on the non-strict
    default path, which this never touches."""
    if os.environ.get("MLFRAME_FE_GPU_STRICT_RESIDENT", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled())
    except Exception:
        return False


def fe_gpu_strict_bytematch_enabled() -> bool:
    """Whether the STRICT-resident gate MI uses the RANK binner for a byte-match with the CPU rank MI.

    DEFAULT OFF. Requires the resident path (``fe_gpu_strict_resident_enabled``) AND the opt-in
    ``MLFRAME_FE_GPU_STRICT_BYTEMATCH=1``. With it OFF the resident gate MI uses the FAST percentile-edge
    binner (selection-equivalent to CPU on F2 -- the gate's edge-vs-rank difference does not flip the F2
    selection, only the gate's lift MAGNITUDE on heavily-tied operator outputs). With it ON the gate MI bins by
    argsort equi-frequency RANK so it byte-matches the CPU njit rank MI, at the cost of an irreducible per-gate
    argsort (~1s on a full fit, GTX 1050 Ti). Read live (no frozen cache) so it tracks the env per call."""
    if os.environ.get("MLFRAME_FE_GPU_STRICT_BYTEMATCH", "").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def fe_gpu_device_born_gate_enabled() -> bool:
    """Whether the conditional-gate / unified-gate tau-grid candidates are built DEVICE-BORN (cupy elementwise
    from resident operand columns) and scored by the resident plug-in MI, instead of host-materialised + uploaded
    at ``_orth_mi_backends.py:311``.

    DEFAULT ON under STRICT-residency (``fe_gpu_strict_resident_enabled``). Collapses the dominant H2D of a
    GPU-strict FE fit (the host gate-grid matrix upload, ~2.8 GB / 65% on a 300k fit) by keeping the candidate
    matrix on the device and uploading only the small operand columns once per fit. The resident batch is
    per-column bit-identical to the host estimator (each column binned independently) AND threads the SAME
    ``rank_binning`` flag the per-triple / host path would have used, so the binning estimator never switches
    (no EDGE<->RANK shift). ``MLFRAME_FE_GPU_DEVICE_BORN_GATE=0`` is the explicit OPT-OUT for diagnosis /
    rollback. The non-strict DEFAULT path is untouched (the host ``_gate_grid_mi`` runs) -> byte-identical."""
    if os.environ.get("MLFRAME_FE_GPU_DEVICE_BORN_GATE", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def fe_gpu_device_born_binagg_enabled() -> bool:
    """Whether the binned-numeric-aggregate FE family's Tier-1 ``local_mi_gate`` OOF candidate matrix is built
    DEVICE-BORN (cupy ``bincount`` raw-moment OOF reconstruction from resident operand columns) and scored by the
    resident plug-in MI, instead of host-materialised in ``fit_binned_numeric_agg`` and uploaded at
    ``_binned_numeric_agg_fe.py:360``.

    DEFAULT ON under STRICT-residency (``fe_gpu_strict_resident_enabled``). Collapses the ~192 MB host OOF matrix
    upload (the #2 single-site H2D of a 300k GPU-strict F2 fit) by rebuilding the (n, K) candidate matrix on the
    device from only the small operand columns (the two raw columns per pair + the host-generated fold-id vector +
    the stored quantile edges), uploaded once per fit via the operand cache. The OOF fold-id / quantile-code /
    per-fold-gather / global-fallback STRUCTURE is bit-identical to the host (only the per-cell moment values
    differ at ULP -- the approved selection-equivalent trade), and the MI is scored with the SAME percentile-edge
    estimator the host STRICT path uses (no EDGE<->RANK switch). ``MLFRAME_FE_GPU_DEVICE_BORN_BINAGG=0`` is the
    explicit OPT-OUT for diagnosis / rollback. The non-strict DEFAULT path is untouched (the host
    ``local_mi_gate`` runs over the host-built matrix) -> byte-identical."""
    if os.environ.get("MLFRAME_FE_GPU_DEVICE_BORN_BINAGG", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def fe_gpu_device_born_dispersion_enabled() -> bool:
    """Whether the conditional-dispersion FE family's Tier-1 ``local_mi_gate`` candidate matrix is built
    DEVICE-BORN (cupy bin-code gather + z-score + |z|/z**2 fold from resident operand columns) and scored by the
    resident plug-in MI, instead of host-materialised in ``generate_conditional_dispersion_features`` and
    uploaded at ``_extra_fe_families_dispersion.py:563``.

    DEFAULT ON under STRICT-residency (``fe_gpu_strict_resident_enabled``). Collapses the ~288 MB host
    conditional-dispersion matrix upload (the dominant single-site H2D of the dispersion gate on a 300k
    GPU-strict F2 fit) by rebuilding the (n, K) candidate matrix on the device from only the small operand
    columns (the two raw columns per pair + the stored x_j quantile edges + the per-bin (mu_hat, sigma_hat)),
    uploaded once per fit via the operand cache. The transform is PURE-X / Y-INDEPENDENT (no OOF / fold / target
    -> no leak surface), so the bin-code / sigma-floor / NaN-fold / emission-fold STRUCTURE is bit-identical to
    the host and only the per-row f64 divide differs at ~1e-10 ULP (the per-bin moments are the SAME host-stored
    recipe constants, NOT recomputed on the device). MI is scored with the SAME percentile-edge estimator the
    host STRICT path uses (no EDGE<->RANK switch). ``MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION=0`` is the explicit
    OPT-OUT for diagnosis / rollback. The non-strict DEFAULT path is untouched (the host ``local_mi_gate`` runs
    over the host-built matrix) -> byte-identical."""
    if os.environ.get("MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def fe_gpu_device_born_dual_uplift_enabled() -> bool:
    """Whether the conditional-dispersion FE family's DUAL-UPLIFT filter scores its Family-B mean-residual
    SIBLING matrix (``|x_i - E[x_i|bin(x_j)]|``) DEVICE-BORN (cupy bin-code gather + subtract + abs from
    resident operand columns) instead of host-materialising ``sib_abs`` in ``_dual_uplift_filter`` and
    uploading it at ``_extra_fe_families_dispersion.py:489``.

    DEFAULT ON under STRICT-residency (``fe_gpu_strict_resident_enabled``). Collapses the ~120 MB host
    Family-B sibling matrix upload (the dual-uplift residual on a 300k GPU-strict F2 fit) by rebuilding the
    (n, K) sibling matrix on the device from only the small operand columns (the two raw columns per pair +
    the stored x_j quantile edges + the per-bin mean of x_i), uploaded once per fit via the operand cache.
    The Family-B residual ``x_i - E[x_i|bin(x_j)]`` is EXACTLY the conditional-dispersion z-score NUMERATOR
    before the sigma-divide; ``|residual|`` is therefore reconstructed by the SAME bin-code / per-bin-mean
    gather the device dispersion builder already uses, only without the ``/sigma`` step (subtract + abs). The
    transform is PURE-X / Y-INDEPENDENT (no OOF / fold / target -> no leak surface) and there is NO divide,
    so the bin-code / NaN-fold STRUCTURE is bit-identical to the host and the emitted values agree to the last
    ULP (subtract + abs, per-element independent). The dual-uplift comparison is an ADDITIVE uplift on the
    SAME estimator (under STRICT all three of cand / raw / sibling MI route through the resident plug-in MI),
    so there is NO uplift-RATIO / baseline-mismatch flip surface. MI is scored with the SAME percentile-edge
    estimator the host STRICT ``_mi_classif_batch`` uses (no EDGE<->RANK switch).
    ``MLFRAME_FE_GPU_DEVICE_BORN_DUAL_UPLIFT=0`` is the explicit OPT-OUT for diagnosis / rollback. The
    non-strict DEFAULT path is untouched (the host ``_mi_classif_batch`` runs over the host-built ``sib_abs``)
    -> byte-identical."""
    if os.environ.get("MLFRAME_FE_GPU_DEVICE_BORN_DUAL_UPLIFT", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def fe_gpu_device_born_wavelet_enabled() -> bool:
    """Whether the BATCHED wavelet leg-rank MI builds its dyadic-Haar leg code matrix DEVICE-BORN (cupy Haar
    indicator + dense-code stack from the resident z-column) and passes the resident cupy code matrix to
    ``binned_mi_from_codes_gpu`` (``isinstance cp.ndarray`` -> no upload), instead of host-stacking
    ``tr_mat`` / ``va_mat`` and ``cp.asarray``-uploading them at ``_fe_batched_mi.py:394``.

    DEFAULT ON under STRICT-residency (``fe_gpu_strict_resident_enabled``). Collapses the ~180 MB host wavelet
    code-matrix upload (the #2 single-site H2D of a 300k GPU-strict F2 fit) by building the (n, K) leg code
    matrix on the device from the single resident z-column (``z = clip((x-lo)/span, 0, 1)``) and the host leg
    metas, uploaded once per fit via the operand cache. The dyadic-Haar leg is a DETERMINISTIC interval
    indicator (``{-1, 0, +1}`` by the dyadic sub-interval of z), and ``_dense_leg_codes`` maps ``leg -> leg+1``
    (cardinality 3) -- both are selection-equivalent partition labels (MI is partition-based), so the device
    twin is bit-identical in PARTITION to the host (pinned by ``test_wavelet_batched_mi_parity``). The leg gate
    is a held-out MI gate (``_WAVELET_MIN_HELDOUT_MI``), NOT an uplift-RATIO, so there is no baseline-mismatch
    flip surface. MI is scored with the SAME plug-in estimator the host batched path uses (no estimator
    switch). ``MLFRAME_FE_GPU_DEVICE_BORN_WAVELET=0`` is the explicit OPT-OUT for diagnosis / rollback. The
    non-strict DEFAULT path is untouched (host ``np.stack`` + ``cp.asarray`` upload) -> byte-identical."""
    if os.environ.get("MLFRAME_FE_GPU_DEVICE_BORN_WAVELET", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def run_fe_step_gpu_strict(self, **kwargs):
    """One FE step, fully GPU-resident, multi-GPU + hw-spec aware. Returns the SAME contract as
    ``_run_fe_step`` (``data, cols, nbins, X, selected_vars, n_recommended_features`` + mutated
    ``engineered_recipes``).

    PHASE 0 STUB: not yet implemented. Raises ``NotImplementedError`` so ``_run_fe_step`` falls back to the
    existing per-family path (no behavior change). Implemented incrementally in Phases 1-3."""
    raise NotImplementedError("run_fe_step_gpu_strict: resident GPU-strict FE step not yet implemented (Phase 0 scaffold)")
