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
GPU_KERNEL = re.compile(r"RawKernel|RawModule|cuda\.jit|ElementwiseKernel")
KTC_REGISTRATION = re.compile(r"\bkernel_tuner\(")

#: Backend modules whose per-host choice is made by a tuned DISPATCHER module: backend -> (dispatcher, why the backend has no tuner entry of its own). The kernel-tuning point is met when the
#: dispatcher registers a ``kernel_tuner(`` entry AND imports the backend (``ktc_dispatch_problems`` checks both, so a stale mapping fails the contract test). A backend with a choice of its own
#: (its own thresholds, variants, block size) registers itself instead.
KTC_DISPATCHERS: dict[str, tuple[str, str]] = {
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
    "_fourier_batch_gpu": (
        "_fourier_detect_batch",
        "chosen against the host loop by fe_fourier_batch_backend (host_device_sweep over the row count) in _fourier_detect_batch; its block size follows the occupancy rule",
    ),
}

#: Device modules whose host twin (the numba kernels the same decision runs when the device is not taken) lives in another module: backend -> (twin module, why). The CPU-kernel point is met
#: when the twin module carries an ``@njit`` kernel (``cpu_twin_problems`` checks it, so a stale mapping fails the contract test).
CPU_TWINS: dict[str, tuple[str, str]] = {
    "_batch_mi_noise_gate_bits": (
        "batch_mi_noise_gate_gpu",
        "the host noise gate of the same decision is batch_mi_noise_gate_gpu's njit permutation gate; the bit-set gate and the histogram pair are the two device backends chosen by the noise_gate_bits tuner entry",
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
    "_fourier_batch_gpu": (
        "_orth_extra_basis_fe",
        "the host side of the Fourier batch is _detect_fourier_freqs_host, whose coarse basis and power reductions are njit(parallel) kernels",
    ),
}


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
BLOCK_STREAMING = re.compile(r"block_cols|BLOCK_COLS|column block|best_pair_streamed")
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
    for pattern, message in (
        (re.compile(r".") if cpu_via_twin else CPU_KERNEL, "no numba CPU kernel"),
        (GPU_KERNEL, "no fused GPU kernel"),
        (re.compile(r".") if ktc_via_dispatcher else KTC_REGISTRATION, "not registered with the kernel-tuning cache"),
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
