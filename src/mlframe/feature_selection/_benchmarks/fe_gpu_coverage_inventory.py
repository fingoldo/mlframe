"""Inventory of the FE modules the MRMR layer imports, with static markers of a GPU path and of a dispatcher.

Static only: it reads source text, so it says which modules mention a GPU kernel or a dispatcher, not whether the path is fast or correct. The
profiled / optimized columns of the audit table are filled by hand from measurements.

Usage: python fe_gpu_coverage_inventory.py [OUTPUT.md]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILTERS = Path(__file__).resolve().parents[1] / "filters"
MRMR_LAYER = ("_mrmr_fit_impl", "mrmr", "_mrmr_fe_step")
GPU_MARKERS = re.compile(r"\bcupy\b|\bcp\.|numba\.cuda|cuda\.jit|RawKernel|ElementwiseKernel|_cuda\b|_gpu\b|gpu_[a-z]|CUDA_AVAIL|_CUPY_AVAIL")
DISPATCH_MARKERS = re.compile(r"kernel_tuner|kernel_tuning_cache|_backend_choice|dispatch_[a-z_]*\(|force_backend|gpu_globally_disabled")
IMPORT = re.compile(r"from\s+(?:mlframe\.feature_selection\.filters|\.+)\s*\.?(_[A-Za-z0-9_]+|[a-z][A-Za-z0-9_]*)\s+import|from\s+\.+(_[A-Za-z0-9_]+)\s+import")
FE_NAME = re.compile(
    r"_fe_|_fe\b|_fe\.|engineer|_basis|_wavelet|_fourier|_orth|_lattice|_modular|_gate|_argmax|_encoding|_target_enc|_synergy|_binned|_cell2d|_row_stat|_count_freq|_pair|_mi_greedy|_missingness|_ratio|_polynom|_hermite"
)


def _direct_imports(files: list[Path]) -> set[str]:
    """Names imported by ``from mlframe.feature_selection.filters[.x] import ...`` or ``from .x import ...`` in ``files``, read as text (no execution)."""
    found: set[str] = set()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in IMPORT.finditer(text):
            name = m.group(1) or m.group(2)
            if name:
                found.add(name)
    return found


def imported_modules() -> set[str]:
    """Names of the filters-level modules reachable from the MRMR layer, by TRANSITIVE closure of their imports.

    A one-hop read (only the files named in ``MRMR_LAYER``) missed modules imported from inside a function body of an already-reached module - e.g. a batched probe imported lazily by a
    gate stage, itself importing a RawKernel module one hop further in. The closure walks every newly reached module's own source (one file each; a package module's ``__init__.py``) until
    no new name appears, so a module several imports deep from the three entry points is still found. Static only, as the one-hop version was: it reads import statements, including ones
    inside a function body, not which ones actually execute.
    """
    frontier: set[str] = set()
    for layer in MRMR_LAYER:
        base = FILTERS / layer
        files = [base] if base.is_file() else list(base.rglob("*.py")) if base.is_dir() else []
        files += [FILTERS / f"{layer}.py"] if (FILTERS / f"{layer}.py").is_file() else []
        frontier |= _direct_imports(files)
    found = set(frontier)
    seen_layers = {"mrmr", "_mrmr_fit_impl", "_mrmr_fe_step"}
    while frontier:
        next_frontier: set[str] = set()
        for name in frontier:
            if name in seen_layers:
                continue
            seen_layers.add(name)
            f = module_file(name)
            if f is None:
                continue
            files = [f] if f.name != "__init__.py" else list(f.parent.rglob("*.py"))
            next_frontier |= _direct_imports(files) - found
        found |= next_frontier
        frontier = next_frontier
    return found


def module_file(name: str) -> Path | None:
    """Source file of a filters-level module or package ``__init__``.

    Checks directly under ``filters/`` first, then falls back to a one-level-deeper search (``filters/*/name.py``)
    for a module living inside a subpackage (e.g. ``_feature_engineering_pairs/_pairs_analytic_gpu.py``,
    ``_orthogonal_univariate_fe/_orth_extra_basis_fe.py``) - these are real, imported modules that previously
    read as "module not found" and were silently excluded from every contract check."""
    for cand in (FILTERS / f"{name}.py", FILTERS / name / "__init__.py"):
        if cand.is_file():
            return cand
    for cand in FILTERS.glob(f"*/{name}.py"):
        if cand.is_file():
            return cand
    for cand in FILTERS.glob(f"*/{name}/__init__.py"):
        if cand.is_file():
            return cand
    # last: the shared kernel-tuning-cache package, whose sweep modules are the dispatchers of some families (``KTC_DISPATCHERS``)
    cand = FILTERS.parent / "_benchmarks" / "kernel_tuning_cache" / f"{name}.py"
    if cand.is_file():
        return cand
    return None


def row(name: str) -> tuple[str, str, int, int, int]:
    """(module, kind, GPU-marker count, dispatcher-marker count, lines) for one module; the whole package for a directory module."""
    f = module_file(name)
    if f is None:
        return name, "missing", 0, 0, 0
    files = list(f.parent.rglob("*.py")) if f.name == "__init__.py" else [f]
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    return name, "package" if f.name == "__init__.py" else "module", len(GPU_MARKERS.findall(text)), len(DISPATCH_MARKERS.findall(text)), text.count("\n") + 1


REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILES_DIR = REPO_ROOT / "audits" / "2026-10-22" / "profiles"
CPU_KERNEL = re.compile(r"@njit|numba\.njit|\bnjit\(")
#: A fused GPU kernel is a RawKernel / RawModule / ElementwiseKernel / numba.cuda kernel, or a cuBLAS GEMM called explicitly on device arrays (``cp.matmul`` / ``cp.dot`` / ``cp.tensordot`` /
#: ``cp.einsum``): a GEMM is the fused kernel of a family whose work is a matrix product (the interaction pre-rank), written as a library call because no hand kernel beats cuBLAS there.
GPU_KERNEL = re.compile(r"RawKernel|RawModule|cuda\.jit|ElementwiseKernel|cp\.(?:matmul|dot|tensordot|einsum)\(")
KTC_REGISTRATION = re.compile(r"\bkernel_tuner\(")

#: Backend modules whose per-host choice is made by a tuned DISPATCHER module: backend -> (dispatcher, why the backend has no tuner entry of its own). The kernel-tuning point is met when the
#: dispatcher registers a ``kernel_tuner(`` entry AND imports the backend (``ktc_dispatch_problems`` checks both, so a stale mapping fails the contract test). A backend with a choice of its own
#: (its own thresholds, variants, block size) registers itself instead.
KTC_DISPATCHERS: dict[str, tuple[str, str]] = {
    "_fe_cpu_batch": (
        "_fe_batch_dispatch_ktc",
        "the host or device choice of the FE batcher is the kernel_tuner entry fe_batch_backend_choice of _fe_batch_dispatch_ktc (cpu_fe_batch_mi timed against multi_gpu_fe_batch_mi over the candidate work n x K at identical MI vectors, the work mark as fallback, MLFRAME_FE_BATCH_DEVICE=0|1 forces a side)",
    ),
    "_fe_gpu_batch": (
        "_fe_batch_dispatch_ktc",
        "the host or device choice of the FE batcher is the kernel_tuner entry fe_batch_backend_choice of _fe_batch_dispatch_ktc (cpu_fe_batch_mi timed against multi_gpu_fe_batch_mi over the candidate work n x K at identical MI vectors, the work mark as fallback, MLFRAME_FE_BATCH_DEVICE=0|1 forces a side)",
    ),
    "_fe_additive_fusion_gpu_resident": (
        "_fe_additive_fusion_ktc",
        "the host or device choice of the additive-fusion proposal is fe_additive_fusion_backend of _fe_additive_fusion_ktc (sweep over the row count of the host proposal against this module at identical admitted sets, the measured row window as fallback, MLFRAME_FE_FUSION_DEVICE=0|1 forces a side)",
    ),
    "_orthogonal_adaptive_degree_fe": (
        "_basis_cells_ktc",
        (
            "the device cell enumeration is asked for through use_device_cells of _basis_cells_ktc (kernel_tuner fe_basis_cells_device: a sweep over the row count of the host cells against the device cells at 1e-6 "
            "equivalence, the measured row mark as fallback, MLFRAME_BASIS_CELLS_DEVICE=0|1 forces a side); the adaptive-degree family is one of its two consumers"
        ),
    ),
    "_cupy_polynom_optimizer": (
        "_polynom_search_backend",
        "the polynom_generation_backend sweep times one generation of 100 candidates on the host (_eval_coef_pair_batch) and on the device (_score_generation_gpu) per row count and requires the same top candidates; the pair search asks it before taking the cupy_kernel route",
    ),
    "_offset_product_fe": (
        "_auto_tune_sweeps_c",
        "the njit/prange scan_offset_products and the fused CUDA scan (scan_offset_products_gpu) are timed per (scan rows, pool columns) cell by the offset_product_scan sweep, which _offset_product_fe._scan_all reads through lookup_offset_scan_backend",
    ),
    "_gpu_resident_fe": (
        "_gpu_resident_k_chunk_ktc",
        "the candidate chunk width (K) of the resident generation + scoring loop is the kernel_tuner entry gpu_k_chunk of _gpu_resident_k_chunk_ktc, swept per host over the row count",
    ),
    "_hermite_fe_mi": (
        "_resident_candidate_mi_ktc",
        "its resident plug-in MI (_plugin_mi_classif_batch_cuda_resident) is timed against the host njit batch MI it replaces by fe_gate_resident_candidate_mi_crossover (kernel_tuner over (n, k)); the resident path engages only where measured faster",
    ),
    "_fe_batched_mi": (
        "_mi_greedy_cmi_fe_ktc",
        "the batched device CMI (batched_cmi_gpu) is chosen against the host loop by cmi_candidate_batch_backend (host_device_sweep over the candidate-rows); the kernel variant inside the family is chosen by cmi_fused_entropy_backend of _fe_cmi_fused_backend",
    ),
    "_batch_pair_mi_cuda_fm": (
        "batch_pair_mi_gpu",
        "chosen among the cpu / serial / parallel / cuda backends by batch_pair_mi_gpu._BPMI_SPEC; its own launch (256 threads) measured at the floor, nothing left to sweep",
    ),
    "_batch_pair_mi_cuda_shared_fused": (
        "batch_pair_mi_gpu",
        "chosen among the same backends by batch_pair_mi_gpu._BPMI_SPEC; its block size follows the occupancy rule (monotone, nothing to sweep)",
    ),
    "_wavelet_legs_fused_gpu": (
        "_wavelet_legs_tuner",
        "its leg code matrices serve the device-born route that fe_wavelet_legs_backend (_wavelet_legs_tuner, host_device_sweep over the row count) chooses against the host body; its block size follows the occupancy rule",
    ),
    "_gpu_pairs": (
        "_cat_pair_search_tuner",
        "the categorical pair search is chosen against the njit prange kernel by cat_pair_search_backend (host_device_sweep over the row count, 60 columns = 1770 pairs of 12-bin codes, same joint MI); the histogram launch follows the joint_hist_multi_pair block-size entry",
    ),
    "_fourier_batch_gpu": (
        "_fourier_detect_batch",
        "chosen against the host loop by fe_fourier_batch_backend (host_device_sweep over the row count) in _fourier_detect_batch; its block size follows the occupancy rule",
    ),
}

#: Device modules whose host twin (the numba kernels the same decision runs when the device is not taken) lives in another module: backend -> (twin module, why). The CPU-kernel point is met
#: when the twin module carries an ``@njit`` kernel (``cpu_twin_problems`` checks it, so a stale mapping fails the contract test).
#: Modules whose device path is a COMPOSITION of the fused kernels of another module that itself meets the whole contract: module -> (kernel module, why the module has no kernel of its own). The point is
#: met when the kernel module exists, the module's text names it, and ``contract_violations(kernel module)`` is empty (``gpu_backend_problems`` checks all three, so a stale mapping, or a backend that
#: loses a contract point, fails the contract test). This is for glue that routes, enumerates and scores cells through kernels owned elsewhere; a module that does array work of its own needs its own kernel.
GPU_BACKENDS: dict[str, tuple[tuple[str, ...], str, bool]] = {
    "_fe_synergy_exhaustive": (
        ("batch_pair_mi_gpu",),
        "the sweep itself is batch_pair_mi_cuda of batch_pair_mi_gpu (one launch per pair block, tuned backend choice); this module decides whether the full sweep is affordable from per-host measured pairs per second (get_or_tune keys batch_pair_mi_exhaustive_throughput and batch_pair_mi_exhaustive_cpu_throughput)",
        True,
    ),
    "_fe_cpu_batch": (
        ("_fe_gpu_batch",),
        "the device half of the FE batcher is _fe_gpu_batch (resident edge-binned plug-in MI, VRAM column chunks, multi-GPU packing); the host or device choice is fe_batch_backend_choice of _fe_batch_dispatch_ktc",
        True,
    ),
    "_fe_auto_escalation": (
        ("_fe_cmi_redundancy_gate", "_mi_greedy_cmi_fe"),
        "the escalation proposes on a deterministic row subsample and lets the existing gates decide: the CMI redundancy gate (_cmi_gate_accepted -> apply_cmi_redundancy_gate of _fe_cmi_redundancy_gate, device partition counting and null, tuned choice) and the batched CMI of _mi_greedy_cmi_fe score the candidates",
        True,
    ),
    "_fe_stability_vote": (
        ("_mi_greedy_cmi_fe",),
        "the per-fold MI statistics are _cmi_from_binned of _mi_greedy_cmi_fe (device partition counting, own tuning entries); under STRICT-resident the recipe is replayed and binned on the device (apply_unary_binary_gpu_resident, the sync-free binner of _mi_greedy_cmi_fe_binning)",
        True,
    ),
    "_mrmr_fe_step_helpers": (
        ("_pair_mi_perm_cuda",),
        "the FE-step blocks carved here are orchestration; the order-2 maxT floor runs the one-launch pair_mi_perm kernel of _pair_mi_perm_cuda (tuned host-or-device choice floor_backend_choice, shape rules, declined -> cupy twin -> njit floor)",
        True,
    ),
    "polynom_pair_fe": (
        ("_hermite_fe_optimise",),
        "the per-pair search is run by optimise_hermite_pair of _hermite_fe_optimise (host numba optimizers, or fe_optimizer=cupy_kernel through _cupy_polynom_optimizer, tuned generation scoring); this module only dispatches pairs, subsamples rows and commits survivors",
        True,
    ),
    "_ratio_delta_fe": (
        ("_ratio_delta_screen_gpu",),
        "the device form of the ratio / log-ratio redundancy screen is _ratio_delta_screen_gpu (one block per ordered pair, row-streamed); the host or device choice is ratio_pair_screen_backend; the MI gate that follows runs through local_mi_gate on the batched MI backends",
        True,
    ),
    "_fe_additive_fusion": (
        ("_fe_additive_fusion_gpu_resident",),
        "the device form of the proposal is _fe_additive_fusion_gpu_resident (resident binning, batched CMI, one float32 Gram matrix for the OLS separability test); the host or device choice is fe_additive_fusion_backend",
        True,
    ),
    "_orth_extra_basis_fe": (
        ("_orth_extra_basis_fe_generate",),
        (
            "the extra-basis generator (spline, Fourier with the adaptive and chirp grids, wavelet) is _orth_extra_basis_fe_generate, which builds the columns on the host from this module's fits or, device-born, through "
            "_extra_basis_resident (periodogram_power, the leg and fourier code kernels, the resident MI); the fits, the njit coarse-basis kernels and the carved helpers are this module"
        ),
        True,
    ),
    "_fe_pure_form_retention": (
        ("_fe_pure_form_retention_gpu_resident",),
        (
            "the device form of the pure-form relevance gate and the resident usability scoring is _fe_pure_form_retention_gpu_resident (njit twin _pure_form_gate_njit, tuned host/device choice, column-block streaming); "
            "the pool-wide usability greedy has its own resident scorer (_usability_greedy_gpu_resident), reached from retain_usable_pure_forms"
        ),
        True,
    ),
    "_fe_retention_subsumption": (
        ("_fe_cmi_redundancy_gate",),
        (
            "the subsumption guard is the CMI redundancy gate applied to a re-attached form: _conditional_perm_null and _cmi_from_binned of _fe_cmi_redundancy_gate score it "
            "(device partition counting, the device or host-table null, tuned choice)"
        ),
        True,
    ),
    "_fe_raw_redundancy_drop": (
        ("_fe_cmi_redundancy_gate",),
        (
            "the observed CMI and the conditional permutation null of each raw operand are scored by _cmi_from_binned and _conditional_perm_null of _fe_cmi_redundancy_gate (compliant: device partition counting, the "
            "device null and the host table null, the host-versus-device choice of cmi_perm_null_backend); the pass itself touches only the final selection"
        ),
        True,
    ),
    "_fe_cmi_redundancy_gate": (
        ("_fe_cmi_perm_null_gpu", "_fe_batched_mi"),
        (
            "the gate scores a round of candidates with batched_cmi_gpu of _fe_batched_mi (the device partition counting, own tuning entries and column-block streaming) and its conditional permutation null runs on the device "
            "through conditional_perm_null_gpu of _fe_cmi_perm_null_gpu, with the host-versus-device choice of the cmi_perm_null_backend entry (_swept_device_null); STRICT-GPU routes the whole gate to the device"
        ),
        True,
    ),
    "_conditional_gate_fe": (
        ("_row_argmax_gpu", "_gate_rank_divergence"),
        (
            "the row-argmax candidate is built by the fused row_argmax kernel of _row_argmax_gpu (numba twin, per-host choice) and the gate columns are ranked by the one-walk divergence of _gate_rank_divergence "
            "(numba and cupy forms, per-host choice); the tau grids go through gate_grid_mi_resident / the host scan under the fe_gate_grid_backend entry of _resident_candidate_mi_ktc"
        ),
        True,
    ),
    "_hermite_fe_optimise": (
        ("_cupy_polynom_optimizer",),
        (
            "the pair-polynomial optimizer dispatcher: optimise_hermite_pair (carved into _hermite_fe_optimise_pair) runs the host optimizers (random_batch / cma_batch through _eval_coef_pair_batch, the njit batch "
            "evaluators) or, as the default fe_optimizer=cupy_kernel, run_cupy_kernel_search of _cupy_polynom_optimizer (cuBLAS GEMMs, bf_apply, the radix-edge binner, mi_from_codes_tile); the host-versus-device "
            "scoring of a generation is the polynom_generation_backend entry of _polynom_search_backend"
        ),
        True,
    ),
    "_numba_polynom_optimizer": (
        ("_cupy_polynom_optimizer",),
        (
            "the pair-polynomial search kernel on the host; its generation-batched device twin run_cupy_kernel_search (two cuBLAS GEMMs per generation, bf_apply, the radix-edge binner, mi_from_codes_tile) "
            "is the default optimizer (fe_optimizer=cupy_kernel) and the host-versus-device choice is the polynom_generation_backend entry of _polynom_search_backend"
        ),
        True,
    ),
    "_orth_mi_backends": (
        ("_hermite_fe_mi",),
        (
            "the resident device path of the batched MI is _plugin_mi_classif_batch_cuda_resident of _hermite_fe_mi (radix-select edges, mi_hist_split, mi_from_counts_col), reached through _strict_resident_mi; "
            "the streamed path (_streamed_gpu_mi) runs the same MI in column blocks sized to free VRAM, and the choice between the paths is made per call from free VRAM and RAM"
        ),
        True,
    ),
    "_class_encoding": (
        ("_fe_batched_mi",),
        (
            "the numba forms of the class merge and the 2-variable joint counts; the device form of the same partition counting for a block of candidate columns is _fe_batched_mi (batched_joint_hist2_i32 and its "
            "shared-memory / global-slice twins, the rows-entropy kernels), which has its own kernel-tuning entries and column-block streaming"
        ),
        True,
    ),
    "_orthogonal_adaptive_degree_fe": (
        ("_gpu_resident_basis",),
        (
            "the device path (_orthogonal_adaptive_degree_device, carved into this family) routes the basis per column with the batched router, builds every (column, degree) cell with the fused Clenshaw "
            "kernel clenshaw_basis and the non-polynomial cell kernel nonpoly_cells, checks the cells with the fused validity pass and scores them with the resident plug-in MI; those kernels are owned by _gpu_resident_basis"
        ),
        False,
    ),
    "_orth_auto_scorer_fe": (
        (
            "_orthogonal_ksg_mi_fe",
            "_orthogonal_copula_mi_fe",
            "_orthogonal_xi_fe",
            "_orthogonal_dcor_fe",
            "_orthogonal_hsic_fe",
            "_orthogonal_tail_dependence_fe",
            "_orthogonal_univariate_fe",
        ),
        (
            "the auto scorer draws the bootstrap resamples, builds the (n, k) block and its transposed device copy once per resample and hands them to the seven batched scorers (KSG, copula MI, xi, dCor, HSIC, "
            "tail dependence, plug-in MI); each scorer's numba and CUDA kernels and its tuner entry (the per-host device choice of the resample block comes from their use_device) are owned by the modules named here, "
            "every one of which meets the whole contract itself"
        ),
        True,
    ),
}

CPU_TWINS: dict[str, tuple[str, str]] = {
    "_fe_synergy_exhaustive": (
        "batch_pair_mi_gpu",
        "the host backend of the exhaustive C(p,2) joint-MI sweep is batch_pair_mi_njit_prange of batch_pair_mi_gpu (parallel numba over pairs); the module times it for the affordability prediction",
    ),
    "_fe_cpu_batch": (
        "_fe_edge_mi",
        "the host scorer is the parallel njit edge-binned plug-in MI kernel plugin_mi_classif_batch_edge_njit of _fe_edge_mi (parallel over columns), column-chunked to the RAM budget",
    ),
    "_fe_gpu_batch": (
        "_fe_edge_mi",
        "the host twin of the resident edge-binned plug-in MI is the njit kernel plugin_mi_classif_batch_edge_njit of _fe_edge_mi (identical MI vectors to 1e-15, test_fe_batch_parity)",
    ),
    "_fe_auto_escalation": (
        "_mi_greedy_cmi_fe",
        "the host form of the escalation's decision statistics is the njit partition kernels of _mi_greedy_cmi_fe (_cmi_from_binned on _quantile_bin codes) and the host contingency-table null of the CMI redundancy gate; the proposers are numpy / njit prewarp fits on a row subsample",
    ),
    "_fe_stability_vote": (
        "_mi_greedy_cmi_fe",
        "the per-fold gate statistic is the plug-in MI / CMI of _mi_greedy_cmi_fe (_cmi_from_binned, njit partition kernels) on equi-frequency codes of _quantile_bin; the replayed recipe is the host apply_recipe",
    ),
    "_mrmr_fe_step_helpers": (
        "_permutation_null",
        "the host form of the order-2 maxT floor is the permutation-batched njit kernel behind pooled_pair_permutation_null_joint_mi_floor of _permutation_null (same host shuffles as the device kernel)",
    ),
    "polynom_pair_fe": (
        "_numba_polynom_optimizer",
        "the host form of the per-pair polynomial search is the njit batch evaluators and the CMA / random-batch optimizers of _numba_polynom_optimizer (reached through optimise_hermite_pair)",
    ),
    "_ratio_delta_fe": (
        "_ratio_delta_screen",
        "the host redundancy screen of the ratio / log-ratio pools is the parallel numba kernel of _ratio_delta_screen (one candidate buffer per task, only accepted columns materialised); the grouped-delta statistics are a gather of per-group values by once-factorised row codes",
    ),
    "_ratio_delta_screen_gpu": (
        "_ratio_delta_screen",
        "the host twin is the parallel numba screen _screen_njit of _ratio_delta_screen (same per-pair decisions, accept matrices identical in the sweep)",
    ),
    "_fe_additive_fusion_gpu_resident": (
        "_usability_njit_pool",
        "the host twin is _fe_additive_fusion.propose_additive_fusions (njit quantile binning of _usability_njit_pool, njit partition MI kernels, numpy OLS solve); the two are timed against each other by fe_additive_fusion_backend",
    ),
    "_fe_additive_fusion": (
        "_usability_njit_pool",
        "the host proposal bins every candidate half with the njit quantile binner of _usability_njit_pool (_qbin_into) and scores the binned MI with the njit partition kernels of _mi_greedy_cmi_fe; the OLS separability test is a numpy normal-equation solve on the scoring subsample",
    ),
    "_fe_pure_form_retention": (
        "_pure_form_gate_njit",
        "the host form of the pure-form relevance gate is the njit kernel family of _pure_form_gate_njit (parallel over candidate forms), the usability pool is built by the njit pair kernels (_pair_combo_mi_njit_table_parallel, _combine_factorize_njit)",
    ),
    "_fe_retention_subsumption": (
        "_cmi_strata_tables_njit",
        "the subsumption guard conditions a re-attached form on the admitted engineered survivors through the CMI redundancy gate, whose host null is the numba contingency-table draw (StratifiedNull) and whose observed CMI is the njit partition kernels",
    ),
    "_fe_raw_redundancy_drop": (
        "_cmi_strata_tables_njit",
        (
            "the pass is a decision layer over a handful of raw operands: its conditional permutation null on the host is the numba contingency-table draw (StratifiedNull, through _conditional_perm_null of "
            "_fe_cmi_redundancy_gate) and its observed CMI the njit partition kernels of _mi_greedy_cmi_fe"
        ),
    ),
    "_fe_cmi_redundancy_gate": (
        "_cmi_strata_tables_njit",
        (
            "the conditional permutation null of the gate on the host is the numba contingency-table draw StratifiedNull(debias=True) (prange over permutations, one multivariate-hypergeometric table per non-trivial stratum, the "
            "Miller-Madow CMI), reached through _fe_cmi_redundancy_null._conditional_table_null; the joint entropies of the observed CMI are the njit partition kernels of _mi_greedy_cmi_fe_steps"
        ),
    ),
    "_hermite_fe_optimise": (
        "_numba_polynom_optimizer",
        "the host optimizers of the pair search: _eval_coef_pair_batch scores a generation with the njit batch evaluators (_fill_bf_batch_njit of _numba_polynom_optimizer, the plug-in MI row kernels), and optimizer=numba_kernel is the all-njit search",
    ),
    "_orth_mi_backends": (
        "_hermite_fe_mi",
        "the batched plug-in MI the orthogonal stage scores with is routed here to the numba plug-in kernel (plugin_mi_classif_batch_njit through plugin_mi_classif_batch_dispatch, parallel over columns) on the host path",
    ),
    "_fe_cmi_perm_null_gpu": (
        "_cmi_strata_tables_njit",
        (
            "the host form of the conditional permutation null is the numba contingency-table draw StratifiedNull(debias=True) (prange over permutations, one multivariate-hypergeometric table per stratum, the same "
            "Miller-Madow CMI): the null distribution of the device module's shuffle at O(strata x levels) per permutation; it is the path _conditional_cpu_null takes and the reference the device null is swept against"
        ),
    ),
    "_orth_auto_scorer_fe": (
        "_ksg_cd_fast",
        (
            "the host form of the scorers the auto scorer batches: the k-nearest-neighbour MI of the dominant scorer is the numba kernel _mi_cd_columns of _ksg_cd_fast (parallel over columns), the plug-in, "
            "xi, dCor and HSIC batches are numba / numpy kernels of their own modules; it is the path every device scorer falls back to and the reference its values are compared with"
        ),
    ),
    "_orthogonal_adaptive_degree_fe": (
        "hermite_fe",
        (
            "the host path builds each (column, degree) cell through _evaluate_basis_column, the numba polynomial evaluators of the hermite_fe package, and scores the stacked matrix with the batched MI of the "
            "orthogonal layer; it is the path the device twin falls back to and the reference its cells are compared with"
        ),
    ),
    "_extra_basis_resident": (
        "_fourier_block_host",
        "fourier_block_host (numba, prange over the rows of every name) is the host evaluation of the Fourier / chirp columns of the pool the device block replaces, and the host generators of the family (generate_extra_basis_features) build the spline and wavelet columns the device columns are checked against; the host matrix is what the resident scorer falls back to",
    ),
    "_gpu_resident_cross_basis": (
        "hermite_fe",
        "polyeval_dispatch of the hermite_fe package (the numba polynomial evaluators) builds the same per-leg columns on the host and the family generators multiply them into the product matrix that mi_classif_batch_chunked scores; it is the path the resident scorer falls back to and the baseline the device products and MI are checked against",
    ),
    "_extra_fe_families_dispersion_resident": (
        "_extra_fe_families_dispersion",
        "generate_conditional_dispersion_features builds the same candidate matrix on the host (the per-bin z-score is the numba kernel _zscore_from_bins_njit) and mi_classif_batch_chunked scores it; it is the path the resident gate falls back to and the reference the device columns are compared with",
    ),
    "_cupy_polynom_optimizer": (
        "_numba_polynom_optimizer",
        "run_numba_kernel_search (one @njit(parallel=True) kernel inlining the polynomial evaluation, the six binary functions and the plug-in MI) is the all-host twin of the same search, with the same return contract and the same candidate scoring",
    ),
    "_offset_product_fe": (
        "_numba_utils_offset",
        "scan_offset_products (njit, prange over the tasks) is the host scan _offset_product_fe._scan_all runs when the device is not chosen, and the reference the fused CUDA scan is timed against",
    ),
    "_gpu_resident_fe": (
        "hermite_fe",
        "_gpu_resident_shared.cpu_pair_candidate_mi builds the same candidate grid with numpy and scores it with the production njit batch MI (plugin_mi_classif_batch_njit of the hermite_fe package); it is the baseline the resident path is checked against",
    ),
    "batch_pair_mi_gpu": (
        "info_theory",
        "the host backends of the dispatcher are the njit batch_pair_mi_serial / batch_pair_mi_prange kernels of the info_theory package, the numerical reference every device backend is checked against",
    ),
    "_fe_batched_mi": (
        "_mi_greedy_cmi_fe",
        "the host twin of the batched device CMI is the njit per-candidate loop of _mi_greedy_cmi_fe (_cmi_from_binned with the hoisted (Y,Z) terms); the two are timed against each other by cmi_candidate_batch_backend",
    ),
    "_batch_mi_noise_gate_bits": (
        "batch_mi_noise_gate_gpu",
        "the host noise gate of the same decision is batch_mi_noise_gate_gpu's njit permutation gate; the bit-set gate and the histogram pair are the two device backends chosen by the noise_gate_bits tuner entry",
    ),
    "_fe_batched_qbin": (
        "_mi_greedy_cmi_fe",
        "the host twin of the device equi-frequency coder is quantile_bin_columns, which bins the finite columns in parallel with the njit quantile_bin_block kernel of _numba_utils (equal codes, checked by test_qbin_column_blocks)",
    ),
    "_basis_fused_gpu": (
        "hermite_fe",
        "the host design is hermite_fe's build_basis_matrix (njit recurrence); the device build is only taken for a column that is already resident",
    ),
    "_modular_codes_gpu": (
        "_periodic_fe",
        "the host pool of the same decision is _modular_all_ops_into (njit) plus the njit quantile coder, built by modular_candidate_block",
    ),
    "_wavelet_legs_fused_gpu": (
        "_wavelet_basis_fe",
        "the host side of the same decision is _select_wavelet_legs, the fused parallel njit body of _wavelet_basis_fe (the leg codes and their MI in one pass)",
    ),
    "_gpu_pairs": (
        "cat_interactions",
        "_pair_search_kernel_njit (njit, prange over the pairs) is the host pair search the cat interaction step runs when the device is not chosen, and the reference the device joint MI is checked against",
    ),
    "_fourier_batch_gpu": (
        "_orth_extra_basis_fe",
        "the host side of the Fourier batch is _detect_fourier_freqs_host, whose coarse basis and power reductions are njit(parallel) kernels",
    ),
}


def gpu_backend_problems(name: str) -> list[str]:
    """Why ``name``'s ``GPU_BACKENDS`` entry does not hold (empty when it holds or there is none): every kernel module must exist, be named by ``name``'s text, and meet the whole contract itself."""
    if name not in GPU_BACKENDS:
        return []
    text = module_text(name)
    problems = []
    for backend in GPU_BACKENDS[name][0]:
        if module_text(backend) is None:
            problems.append(f"kernel module {backend} not found")
            continue
        if text is None or not re.search(rf"\b{re.escape(backend)}\b", text):
            problems.append(f"{name} does not name its kernel module {backend}")
        inner = contract_violations(backend)
        if inner:
            problems.append(f"kernel module {backend} does not meet the contract: {inner}")
    return problems


def cpu_twin_problems(name: str) -> list[str]:
    """Why ``name``'s ``CPU_TWINS`` entry does not hold (empty when it holds or there is none): the twin module must exist and carry a numba kernel."""
    if name not in CPU_TWINS:
        return []
    twin = CPU_TWINS[name][0]
    text = module_text(twin)
    if text is None:
        return [f"host twin {twin} not found"]
    return [] if CPU_KERNEL.search(text) else [f"host twin {twin} has no numba kernel"]


def ktc_dispatch_problems(name: str) -> list[str]:
    """Why ``name``'s ``KTC_DISPATCHERS`` entry does not hold (empty when it holds or there is none): the dispatcher must exist, register a ``kernel_tuner(`` entry and import ``name``."""
    if name not in KTC_DISPATCHERS:
        return []
    dispatcher = KTC_DISPATCHERS[name][0]
    text = module_text(dispatcher)
    if text is None:
        return [f"dispatcher {dispatcher} not found"]
    problems = []
    if not KTC_REGISTRATION.search(text):
        problems.append(f"dispatcher {dispatcher} registers no kernel_tuner entry")
    if not re.search(rf"\b{re.escape(name)}\b", text):
        problems.append(f"dispatcher {dispatcher} does not import {name}")
    return problems


CRIT_DTYPE = re.compile(r"_crit_np_dtype|crit_float_dtype|Intentional float64")
#: Device memory bounded by blocks: column blocks, or row blocks for a kernel whose operand is one small column set and many rows (a row-streamed scorer such as the LOF transform).
BLOCK_STREAMING = re.compile(r"block_cols|BLOCK_COLS|column block|row block|best_pair_streamed")
HOST_PROFILE = re.compile(r"cProfile")
GPU_PROFILE = re.compile(r"\bnsys\b|\bncu\b|\bnvprof\b")
GPU_OCCUPANCY = re.compile(r"occupancy", re.IGNORECASE)
LAUNCHES_RECORDED = re.compile(r"launches?\s+per", re.IGNORECASE)
KERNEL_STRING = re.compile(r'(?:r)?"""(.*?)"""', re.DOTALL)
_TYPE_PLACEHOLDERS = ("{REAL}", "{ACC}", "{DT}", "{XT}")
BARE_FP_TYPE = re.compile(r"\b(?:double|float)\b")
GPU_THROUGHPUT = re.compile(r"throughput|utilization|bandwidth", re.IGNORECASE)


#: Modules carved out of a parent module to keep it under the file-size ceiling: parent -> its carved children. The contract is judged on the family as one unit (the parent's text plus
#: the children's, one profile record under the parent's name), and a child is covered by the parent's verdict as long as the parent still imports it (``carved_family_problems``).
#: This is not a way to skip the contract: the parent has to meet every point with its children's code counted, and one record covers the family, where one record per carved file
#: would repeat the same measurements.
CARVED_FAMILIES: dict[str, tuple[str, ...]] = {
    "_fe_auto_escalation": ("_fe_auto_escalation_proposers",),
    "_ratio_delta_fe": ("_ratio_delta_screen",),
    "_orth_extra_basis_fe": ("_orth_extra_basis_fe_helpers",),
    "_orth_extra_basis_fe_generate": ("_orth_extra_basis_fe_generate_doc",),
    "_fe_pure_form_retention": ("_fe_pure_form_sides",),
    "_fe_raw_redundancy_drop": ("_fe_raw_redundancy_anchors", "_fe_raw_redundancy_helpers"),
    "_fe_cmi_redundancy_gate": ("_fe_cmi_redundancy_null",),
    "_fe_batched_mi_cmi": ("_fe_cmi_fused_backend",),
    "_conditional_gate_fe": ("_conditional_gate_hybrid", "_conditional_gate_kernels"),
    "_hermite_fe_optimise": ("_hermite_fe_optimise_pair", "_hermite_fe_optimise_parts"),
    "_target_encoding_order_stats": ("_order_stats_gpu",),
    "_temporal_agg_fe_rolling": ("_rolling_window_gpu",),
    "_periodic_fe": ("_modular_codes_gpu",),
    "batch_pair_usability_corr_gpu": ("_pair_usability_table",),
    "_orthogonal_univariate_fe": ("_orth_scoring_memo", "_orth_basis_memo", "_orth_gpu_resident",),
    "hermite_fe": ("_hermite_basis_eval", "_hermite_oracle", "_hermite_prewarp", "_hermite_prewarp_gpu_resident", "_hermite_robust"),
    "_extra_basis_resident": ("_orth_extra_basis_fe_generate",),
    "_gpu_resident_fe": ("_gpu_resident_basis",),
    "_gpu_resident_basis": ("_gpu_resident_basis_factories",),
    "_wavelet_basis_fe": ("_wavelet_basis_fe_batched",),
    "_fe_pure_form_retention_gpu_resident": ("_pure_form_gate_njit",),
    "_cupy_polynom_optimizer": ("_polynom_search_backend",),
    "_relaxmrmr_pair_loop": ("_relaxmrmr_pair_gpu",),
    "_hermite_fe_mi": ("_fe_edge_mi",),
    "_orthogonal_triplet_fe": ("_triple_mi_gpu", "_orthogonal_triplet_recipes",),
    "_offset_product_fe": ("_offset_product_gpu",),
    "_parity_fe": ("_parity_gpu",),
    "_orthogonal_dcor_fe": ("_dcor_gpu",),
    "_family_maxt_gate": ("_family_maxt_cuda_bits", "_family_maxt_cuda", "_family_maxt_stack", "_family_maxt_stream", "_family_maxt_null_v2_njit"),
    "_orthogonal_hsic_fe": ("_hsic_gpu",),
    "_pair_residual_screen": ("_pair_residual_gpu",),
    "_fe_synergy_screen": ("_synergy_combo_gpu",),
    "_cat_target_encoding_and_weighted": ("_cat_weighted_pair_gpu",),
    "_fourier_batch_gpu": ("_fourier_detect_batch", "_fourier_detect_gpu_resident", "_fourier_fused_gpu"),
    "_extra_basis_fourier_cuda": ("_fourier_block_host",),
    "_dcd_pair_su_batch": ("_dcd_pair_su_gpu",),
    "_added_information_gate": ("_added_information_gpu",),
    "_cat_pair_joint_njit": ("_cat_pair_joint_gpu",),
    "_numeric_decompose_fe": ("_numeric_decompose_gpu",),
    "_extra_fe_families": ("_extra_fe_families_dispersion", "_conditional_bin_moments"),
    "_fe_interaction_prerank": ("_fe_interaction_prerank_kernels",),
    "_residual_rescue_fe": ("_residual_rescue_gpu", "_residual_rescue_ktc"),
    "batch_pair_mi_gpu": ("_batch_pair_mi_cuda_kernels",),
    "_orthogonal_xi_fe": ("_orthogonal_xi_gpu",),
    "_orthogonal_adaptive_degree_fe": ("_orthogonal_adaptive_degree_device",),
    "_orthogonal_cmim_fe": ("_cmim_min_gpu",),
    "_orthogonal_meta_scorer_fe": ("_meta_fingerprint_gpu",),
    "_orthogonal_ksg_mi_fe": ("_ksg_cd_fast", "_ksg_cd_gpu"),
    "_grouped_quantile_fe": ("_grouped_quantile_ranks", "_grouped_quantile_gpu"),
    "_orthogonal_bootstrap_mi_fe": ("_bootstrap_rank_mi_njit", "_bootstrap_rank_mi_gpu", "_bootstrap_rank_mi_tied_gpu"),
    "_hinge_basis_fe": ("_hinge_precheck_gpu", "_hinge_detect_gpu_resident", "_hinge_detect_gpu_resident_batch"),
    "_fe_sufficient_summary": ("_fe_sufficient_summary_pairs",),
    "_group_distance_fe": ("_group_distance_gpu",),
    "_orthogonal_copula_mi_fe": ("_orthogonal_copula_gpu",),
    "_orthogonal_tail_dependence_fe": ("_tail_dependence_gpu",),
    "_lof_fe": ("_lof_gpu",),
    "_pairwise_modular_fe": ("_pairwise_modular_rank_gpu", "_pairwise_modular_resident"),
    "_fe_batched_mi": (
        "_fe_batched_mi_blocks",
        "_fe_batched_mi_cmi",
        "_fe_batched_mi_cmi_hist",
        "_fe_batched_mi_cmi_lut",
        "_fe_batched_mi_guards",
        "_fe_batched_mi_mm_cm",
        "_fe_batched_mi_mm_values",
        "_fe_batched_qbin",
    ),
    "_feature_engineering_pairs": (
        "_pairs_abs_corr_gpu",
        "_pairs_analytic_gpu",
        "_pairs_chunk_codes",
        "_pairs_chunks",
        "_pairs_code_bins",
        "_pairs_common",
        "_pairs_core",
        "_pairs_core_helpers",
        "_pairs_core_parts",
        "_pairs_core_steps",
        "_pairs_dispatch",
        "_pairs_emit",
        "_pairs_emit_extval",
        "_pairs_gate_dedupe",
        "_pairs_gates",
        "_pairs_gpu_gates",
        "_pairs_materialise",
        "_pairs_operand_floor",
        "_pairs_score",
        "_pairs_score_helpers",
        "_pairs_score_recompute",
        "_pairs_score_step4",
        "_pairs_score_steps",
        "_pairs_setup",
    ),
    "_binned_numeric_agg_fe": (
        "_binned_agg_cheap_mi",
        "_binned_agg_linear_usability",
        "_binned_numeric_agg_batches",
        "_binned_numeric_agg_cands",
        "_binned_numeric_agg_redundancy",
        "_binned_numeric_agg_resident",
    ),
    "_fe_accuracy_gate_batched": (
        "_fe_accuracy_gate_kernel",
        "_fe_accuracy_gate_newton",
    ),
    "_mi_greedy_cmi_fe": (
        "_mi_greedy_cmi_fe_binning",
        "_mi_greedy_cmi_fe_gpu",
        "_mi_greedy_cmi_fe_ktc",
        "_mi_greedy_cmi_fe_steps",
        "_mi_greedy_fe",
    ),
    "_mi_greedy_fe": ("_mi_greedy_device",),
    "batch_mi_noise_gate_gpu": (
        "_batch_mi_noise_gate_fused",
        "_batch_mi_noise_gate_tuning",
        "_batch_mi_noise_gate_column_tiles",
        "_batch_mi_noise_gate_gpu_helpers",
        "_batch_mi_noise_gate_screen_v1",
        "_batch_mi_noise_gate_kernels",
    ),
}
_CARVED_PARENT: dict[str, str] = {child: parent for parent, children in CARVED_FAMILIES.items() for child in children}


def carved_family_problems(name: str) -> list[str]:
    """Why ``name``'s ``CARVED_FAMILIES`` membership does not hold (empty when it holds or there is none): a child must exist and be imported by its parent module, or, for a parent that
    is a package, live inside the package directory."""
    parent = _CARVED_PARENT.get(name)
    if parent is None:
        return []
    f = module_file(parent)
    if f is None:
        return [f"parent {parent} not found"]
    child_file = module_file(name)
    if child_file is None:
        return [f"carved module {name} not found"]
    if f.name == "__init__.py":
        return [] if f.parent in child_file.parents else [f"{name} is not inside the package {parent}"]
    parent_text = f.read_text(encoding="utf-8", errors="ignore")
    return [] if re.search(rf"\b{re.escape(name)}\b", parent_text) else [f"parent {parent} does not import {name}"]


def _own_text(name: str) -> str | None:
    """Source text of one filters-level module, the whole package for a directory module; ``None`` when the module does not exist."""
    f = module_file(name)
    if f is None:
        return None
    files = list(f.parent.rglob("*.py")) if f.name == "__init__.py" else [f]
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)


def module_text(name: str) -> str | None:
    """Source text of a filters-level module, the whole package for a directory module, the parent plus its carved children for a family parent; ``None`` when the module does not exist."""
    own = _own_text(name)
    if own is None or name not in CARVED_FAMILIES:
        return own
    return "\n".join([own] + [_own_text(child) or "" for child in CARVED_FAMILIES[name]])


def contract_violations(name: str) -> list[str]:
    """The points of the FE GPU / streaming / profiling contract a module does not meet (empty when it meets all of them).

    Static: it reads the source text and the profile record, so it pins that each requirement is present and recorded, and the behaviour tests of the
    module pin that it works (equivalence of the kernels, the memory bound).
    """
    if name in _CARVED_PARENT:
        problems = carved_family_problems(name)
        if problems:
            return problems
        parent_problems = contract_violations(_CARVED_PARENT[name])
        return [f"carved from {_CARVED_PARENT[name]}, which does not meet the contract: {parent_problems}"] if parent_problems else []
    text = module_text(name)
    if text is None:
        return ["module not found"]
    found = []
    ktc_via_dispatcher = name in KTC_DISPATCHERS and not ktc_dispatch_problems(name)
    cpu_via_twin = name in CPU_TWINS and not cpu_twin_problems(name)
    gpu_via_backend = name in GPU_BACKENDS and not gpu_backend_problems(name)
    ktc_via_backends = gpu_via_backend and GPU_BACKENDS[name][2]
    for pattern, message in (
        (re.compile(r".") if cpu_via_twin else CPU_KERNEL, "no numba CPU kernel"),
        (re.compile(r".") if gpu_via_backend else GPU_KERNEL, "no fused GPU kernel"),
        (re.compile(r".") if (ktc_via_dispatcher or ktc_via_backends) else KTC_REGISTRATION, "not registered with the kernel-tuning cache"),
        (CRIT_DTYPE, "no crit dtype use"),
        (BLOCK_STREAMING, "no column-block streaming"),
    ):
        if not pattern.search(text):
            found.append(message)
    kernel_sources = [m.group(1) for m in KERNEL_STRING.finditer(text) if "__global__" in m.group(1)]
    if kernel_sources:
        joined = "\n".join(kernel_sources)
        # {REAL} types the values a kernel reads, {ACC} its accumulators. A kernel over integer bin codes and class labels has no real-valued operand, so it is type-generic through {ACC} alone;
        # either placeholder meets the point, and the bare-type check below still rejects a literal double / float in either kind. A kernel over integer codes of a caller-chosen width
        # (the bit-set packers, the entropy kernels) is typed by {DT} / {XT}, the code type, and meets the point the same way.
        if not any(ph in joined for ph in _TYPE_PLACEHOLDERS):
            found.append("GPU kernel source has none of the type placeholders {REAL} (values), {ACC} (accumulators), {DT} / {XT} (integer code type)")
        if BARE_FP_TYPE.search(joined):
            found.append("GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator)")
    record = PROFILES_DIR / f"{name}.md"
    if not record.is_file():
        found.append("no profile record")
    else:
        recorded = record.read_text(encoding="utf-8", errors="ignore")
        if not HOST_PROFILE.search(recorded):
            found.append("profile record has no cProfile")
        if not GPU_PROFILE.search(recorded):
            found.append("profile record has no GPU profiler (nsys / ncu / nvprof)")
        if not LAUNCHES_RECORDED.search(recorded):
            found.append("profile record does not state the kernel launches per call (fusion)")
        if not GPU_OCCUPANCY.search(recorded):
            found.append("profile record has no occupancy")
        if not GPU_THROUGHPUT.search(recorded):
            found.append("profile record has no whole-GPU throughput / utilization figures")
    return found


def fe_modules() -> list[str]:
    """The FE modules the MRMR layer imports."""
    return sorted(n for n in imported_modules() if FE_NAME.search(n))


def main() -> None:
    """Print the coverage table of every FE module (static verdicts) as markdown."""
    names = fe_modules()
    rows = [row(n) for n in names]
    lines = ["| module | kind | GPU markers | dispatcher markers | lines | static verdict |", "|---|---|---:|---:|---:|---|"]
    for name, kind, gpu, disp, loc in rows:
        verdict = "no GPU marker" if gpu == 0 else ("GPU code, no dispatcher" if disp == 0 else "GPU code + dispatcher")
        lines.append(f"| `{name}` | {kind} | {gpu} | {disp} | {loc} | {verdict} |")
    counts = {
        v: sum(1 for r in rows if (("no GPU marker" if r[2] == 0 else ("GPU code, no dispatcher" if r[3] == 0 else "GPU code + dispatcher")) == v))
        for v in ("no GPU marker", "GPU code, no dispatcher", "GPU code + dispatcher")
    }
    out = "\n".join(lines) + "\n\n" + f"{len(rows)} modules: " + ", ".join(f"{v}={c}" for v, c in counts.items()) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
