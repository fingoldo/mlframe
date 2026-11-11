"""Offline per-architecture checks of every production CUDA kernel: NVRTC to PTX, then ``ptxas -v``, no GPU needed.

Plan and critique: audits/2026-11-09/offline_ptxas_static_checks_plan.md (tracker GPU11-32). For each kernel variant (a source with the placeholders its
call site fills) and each compute capability this asserts: the source has no unresolved placeholder, NVRTC and ptxas accept it (static shared memory
over the limit is a ptxas error, so it fails here), every ``.entry`` has a resource block, none has a stack frame or spill, and ptxas prints no warning
outside the allowlist.

What this does NOT prove: production compiles through cupy's NVRTC and the driver JIT, while this uses the toolkit's ptxas on the PTX NVRTC emits (its
``.version`` line is lowered when NVRTC is newer than ptxas). It is a PTX-validity and resource check under the toolkit's ptxas, not a statement about the
exact binary a user's driver builds.

Architectures: ``MLFRAME_OFFLINE_ARCHS`` (comma list, or ``all``); the default is sm_61 and sm_120 so a local run stays short, CI sets ``all``.
sm_52 and sm_61 are deprecated by nvcc 12.8 ("may be removed in a future release"): a toolchain that drops them skips those columns through the
TOOLCHAIN note below, never silently.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _offline_kernel_sources import KernelVariant, all_variants, leftover_placeholders

_ALL_ARCHS = (61, 70, 75, 86, 89, 90, 120)  # sm_52 is out: double atomicAdd needs sm_60+, and its Maxwell cards are legacy
_DEFAULT_ARCHS = (61, 120)

# variant id -> reason a spill or warning is accepted for it, with the number that justifies it. Empty is the default: fix the kernel instead.
_SPILL_EXEMPT: dict = {
    "mlframe/feature_selection/filters/_honest_null_accept_bits.py::_SRC": (
        "root_split_accept at (512, MLF_THREADS_PER_SM / 512) spills 16 to 48 bytes/thread on sm_61 to sm_120 (32 to 42 register cap) and none on sm_75 (64 registers). Measured on the GTX 1050 Ti, 7 repeats, min "
        "(2026-11-12, bench of the three minBlocks values; gains bit-identical): minBlocks 4 = 10.05/12.81/60.21 ms, minBlocks 3 (40 regs, 16 B local) = 10.60/12.88/64.33 ms, minBlocks 2 (48 regs, no spill) "
        "= 10.72/13.01/62.55 ms at n = 20000/64000/100003. Removing the spill costs 0 to 7 percent, so the small L1-resident local spill is cheaper than the lost occupancy (CUDA.md section 3)."
    ),
    "mlframe/feature_selection/filters/_pairwise_modular_rank_gpu.py::_KERNEL_V4_TEMPLATE": (
        "Version 4 of residue_rank spills 8 bytes/thread on sm_70 only, at the 32-register cap of (128, MLF_THREADS_PER_SM / 128). It is a kept experimental version (the default is 8, "
        "MLFRAME_MODULAR_RANK_KERNEL selects others); loosening its cap would change what that version measures."
    ),
}
# ptxas warning substrings that are accepted everywhere, each with the reason.
_WARNING_ALLOWLIST = {
    "Architectures prior to": "NVRTC deprecation notice for sm_52/sm_61, not a defect of the kernel (TOOLCHAIN disposition in the plan)",
}


def _archs() -> tuple:
    """Compute capabilities to check: ``MLFRAME_OFFLINE_ARCHS`` (comma list or ``all``), else the short default."""
    raw = os.environ.get("MLFRAME_OFFLINE_ARCHS", "").strip().lower()
    if not raw:
        return _DEFAULT_ARCHS
    if raw == "all":
        return _ALL_ARCHS
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _ptxas() -> str:
    """Path of the toolkit's ptxas (CUDA_PATH, then PATH), or skip the test: the check needs the assembler, not a device."""
    exe = "ptxas.exe" if os.name == "nt" else "ptxas"
    candidates = []
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]) / "bin" / exe)
    for base in os.environ.get("PATH", "").split(os.pathsep):
        candidates.append(Path(base) / exe)
    for path in candidates:
        if path.exists():
            return str(path)
    pytest.skip("no ptxas on PATH or under CUDA_PATH")


def _variant_id(v: KernelVariant) -> str:
    """The id of one kernel build in a failure message and in the exemption table (source key plus label)."""
    return f"{v.key}[{v.label}]"


_VARIANTS = all_variants()


def test_every_spill_exemption_names_a_real_kernel_and_a_reason():
    """An exemption must match an extracted kernel source and carry a reason; a stale or empty one fails."""
    keys = {v.key for v in _VARIANTS}
    for key, reason in _SPILL_EXEMPT.items():
        assert key in keys, f"exemption for a kernel that is not extracted: {key}"
        assert reason.strip(), f"empty reason for {key}"


def test_every_file_with_a_kernel_yields_a_variant():
    """Coverage diff: a production file that contains ``__global__`` but yields no variant is a silently dropped kernel (an extractor gap), not a pass."""
    from _offline_kernel_sources import ROOT

    have = {v.key.split("::")[0].split("mlframe/", 1)[1] for v in _VARIANTS}
    missing = []
    for path in ROOT.rglob("*.py"):
        if any(part in path.parts for part in ("_benchmarks", "kernel_prototypes")):
            continue
        if "__global__" in path.read_text(encoding="utf-8", errors="replace") and path.relative_to(ROOT).as_posix() not in have:
            missing.append(path.relative_to(ROOT).as_posix())
    assert not missing, f"files with a __global__ kernel but no extracted variant: {missing}"


def test_every_production_kernel_source_is_covered():
    """The extraction finds production kernels and every variant resolved all its placeholders (a table gap fails here, it is not skipped)."""
    assert len(_VARIANTS) >= 100, f"only {len(_VARIANTS)} kernel variants found"
    unresolved = {_variant_id(v): leftover_placeholders(v) for v in _VARIANTS if leftover_placeholders(v)}
    assert not unresolved, f"placeholders without a fill in tests/test_meta/_offline_kernel_sources.py: {unresolved}"


@pytest.mark.parametrize("arch", _archs())
@pytest.mark.parametrize("variant", _VARIANTS, ids=_variant_id)
def test_kernel_assembles_without_spill_on_arch(variant: KernelVariant, arch: int, tmp_path):
    """NVRTC then ptxas -v: compiles, one resource block per entry, no stack frame or spill, no unlisted warning."""
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    ptxas = _ptxas()
    if leftover_placeholders(variant):
        pytest.fail(f"unresolved placeholders {leftover_placeholders(variant)}")
    prog = nvrtc.createProgram(variant.source, "k.cu", [], [])
    try:
        nvrtc.compileProgram(prog, ["-std=c++14", f"-arch=compute_{arch}", *variant.options])
        ptx = nvrtc.getPTX(prog)
    except Exception as exc:
        log = nvrtc.getProgramLog(prog)
        errors = [ln for ln in log.splitlines() if "error" in ln.lower()]
        pytest.fail(f"NVRTC failed on sm_{arch}: {'; '.join(errors[:3]) or exc}")
    finally:
        nvrtc.destroyProgram(prog)
    ptx = re.sub(rb"\.version \d+\.\d+", b".version 8.7", ptx)
    path = tmp_path / "k.ptx"
    path.write_bytes(ptx)
    res = subprocess.run([ptxas, "-v", f"-arch=sm_{arch}", "-o", str(tmp_path / "k.cubin"), str(path)], capture_output=True, text=True)
    report = res.stdout + res.stderr
    assert res.returncode == 0, f"ptxas failed on sm_{arch}: {report[-500:]}"

    entry_names = [n.decode() for n in re.findall(rb"\.entry\s+(\w+)", ptx)]
    assert entry_names, "no .entry in the PTX"
    # ptxas prints a "Function properties for NAME" block per function (device helpers too): match them to the entries by name.
    block = r"Function properties for (\w+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
    props = {m[0]: m[1:] for m in re.findall(block, report)}
    missing = [n for n in entry_names if n not in props]
    assert not missing, f"entries without a ptxas resource block: {missing}"
    spills = {n: props[n] for n in entry_names if props[n][1] != "0" or props[n][2] != "0"}
    if variant.key not in _SPILL_EXEMPT:
        assert not spills, f"sm_{arch}: register spill (stack, spill st, spill ld) per entry: {spills}"

    warnings = [ln for ln in report.splitlines() if "warning" in ln.lower() and not any(k in ln for k in _WARNING_ALLOWLIST)]
    assert not warnings, f"sm_{arch}: unlisted ptxas warnings: {warnings[:3]}"


_LITERAL_MIN_BLOCKS = re.compile(r"__launch_bounds__\(\s*\w+\s*,\s*\d+\s*\)")  # second argument a bare number


def _fits_every_arch(text: str) -> bool:
    """Both numbers literal and threads x blocks within 1024, the smallest threads-per-SM of any architecture here (sm_75)."""
    nums = re.findall(r"\d+", text)
    return len(nums) == 2 and int(nums[0]) * int(nums[1]) <= 1024


def test_no_kernel_hardcodes_the_min_blocks_of_launch_bounds():
    """``__launch_bounds__(T, B)`` with a literal ``B`` is a number tuned on one card: threads per SM are 2048, 1536 or 1024 by architecture, and a hint above the limit is
    ignored with a ptxas warning. ``B`` must come from a macro derived from ``__CUDA_ARCH__`` (the host-side launch-geometry gate does not see kernel source)."""
    root = Path(__file__).resolve().parents[2] / "src" / "mlframe"
    hits = []
    for path in root.rglob("*.py"):
        if any(part in path.parts for part in ("_benchmarks", "kernel_prototypes")):
            continue
        for no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = _LITERAL_MIN_BLOCKS.search(line)
            if "__launch_bounds__" in line and m and not line.lstrip().startswith("#") and not _fits_every_arch(m.group(0)):
                hits.append(f"{path.relative_to(root).as_posix()}:{no}: {line.strip()[:90]}")
    assert not hits, "literal minBlocks in __launch_bounds__: " + "; ".join(hits)
