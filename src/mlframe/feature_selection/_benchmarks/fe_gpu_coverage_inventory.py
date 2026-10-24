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
FE_NAME = re.compile(r"_fe_|_fe\b|_fe\.|engineer|_basis|_wavelet|_fourier|_orth|_lattice|_modular|_gate|_argmax|_encoding|_target_enc|_synergy|_binned|_cell2d|_row_stat|_count_freq|_pair|_mi_greedy|_missingness|_ratio|_polynom|_hermite")


def imported_modules() -> set[str]:
    """Names of the filters-level modules imported from the MRMR layer."""
    found: set[str] = set()
    for layer in MRMR_LAYER:
        base = FILTERS / layer
        files = [base] if base.is_file() else list(base.rglob("*.py")) if base.is_dir() else []
        files += [FILTERS / f"{layer}.py"] if (FILTERS / f"{layer}.py").is_file() else []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for m in IMPORT.finditer(text):
                name = m.group(1) or m.group(2)
                if name:
                    found.add(name)
    return found


def module_file(name: str) -> Path | None:
    """Source file of a filters-level module or package ``__init__``."""
    for cand in (FILTERS / f"{name}.py", FILTERS / name / "__init__.py"):
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
GPU_KERNEL = re.compile(r"RawKernel|cuda\.jit|ElementwiseKernel")
KTC_REGISTRATION = re.compile(r"\bkernel_tuner\(")
CRIT_DTYPE = re.compile(r"_crit_np_dtype|crit_float_dtype|Intentional float64")
BLOCK_STREAMING = re.compile(r"block_cols|BLOCK_COLS|column block|best_pair_streamed")
HOST_PROFILE = re.compile(r"cProfile")
GPU_PROFILE = re.compile(r"\bnsys\b|\bncu\b|\bnvprof\b")
GPU_OCCUPANCY = re.compile(r"occupancy", re.IGNORECASE)
LAUNCHES_RECORDED = re.compile(r"launches?\s+per", re.IGNORECASE)
KERNEL_STRING = re.compile(r'(?:r)?"""(.*?)"""', re.DOTALL)
BARE_FP_TYPE = re.compile(r"\b(?:double|float)\b")
GPU_THROUGHPUT = re.compile(r"throughput|utilization|bandwidth", re.IGNORECASE)


def module_text(name: str) -> str | None:
    """Source text of a filters-level module, the whole package for a directory module; ``None`` when the module does not exist."""
    f = module_file(name)
    if f is None:
        return None
    files = list(f.parent.rglob("*.py")) if f.name == "__init__.py" else [f]
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)


def contract_violations(name: str) -> list[str]:
    """The points of the FE GPU / streaming / profiling contract a module does not meet (empty when it meets all of them).

    Static: it reads the source text and the profile record, so it pins that each requirement is present and recorded, and the behaviour tests of the
    module pin that it works (equivalence of the kernels, the memory bound).
    """
    text = module_text(name)
    if text is None:
        return ["module not found"]
    found = []
    for pattern, message in (
        (CPU_KERNEL, "no numba CPU kernel"),
        (GPU_KERNEL, "no fused GPU kernel"),
        (KTC_REGISTRATION, "not registered with the kernel-tuning cache"),
        (CRIT_DTYPE, "no crit dtype use"),
        (BLOCK_STREAMING, "no column-block streaming"),
    ):
        if not pattern.search(text):
            found.append(message)
    kernel_sources = [m.group(1) for m in KERNEL_STRING.finditer(text) if "__global__" in m.group(1)]
    if kernel_sources:
        joined = "\n".join(kernel_sources)
        if "{REAL}" not in joined:
            found.append("GPU kernel source has no {REAL} placeholder")
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
    names = fe_modules()
    rows = [row(n) for n in names]
    lines = ["| module | kind | GPU markers | dispatcher markers | lines | static verdict |", "|---|---|---:|---:|---:|---|"]
    for name, kind, gpu, disp, loc in rows:
        verdict = "no GPU marker" if gpu == 0 else ("GPU code, no dispatcher" if disp == 0 else "GPU code + dispatcher")
        lines.append(f"| `{name}` | {kind} | {gpu} | {disp} | {loc} | {verdict} |")
    counts = {v: sum(1 for r in rows if (("no GPU marker" if r[2] == 0 else ("GPU code, no dispatcher" if r[3] == 0 else "GPU code + dispatcher")) == v)) for v in ("no GPU marker", "GPU code, no dispatcher", "GPU code + dispatcher")}
    out = "\n".join(lines) + "\n\n" + f"{len(rows)} modules: " + ", ".join(f"{v}={c}" for v, c in counts.items()) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
