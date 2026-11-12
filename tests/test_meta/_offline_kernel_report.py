"""Static per-kernel, per-architecture report from NVRTC PTX and ``ptxas -v``: registers and theoretical occupancy, the FP64 share of arithmetic, and the architecture an intrinsic requires.

These are REPORTS, not gates, by the review of audits/2026-11-09/offline_ptxas_static_checks_plan.md: occupancy is theoretical (the block size is chosen at launch from the device, so it is
shown at 128, 256 and 512 threads), the FP64 share is an unweighted static count of arithmetic opcodes (loop trip counts are unknown), and an intrinsic's minimum architecture is a text scan.
The architecture limits below come from the CUDA C++ Programming Guide compute-capability table; only the local device's can be read back at run time (``local_device_limits``), so a mismatch
there is reported by the test and the other rows stay a documented table, not a measurement.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# arch -> (max threads per SM, max resident blocks per SM, registers per SM, max shared memory per SM in bytes)
REGISTER_FILE = 65536  # 32-bit registers per SM, the same on every listed architecture
ARCH_LIMITS = {
    61: (2048, 32, 65536, 96 * 1024),
    70: (2048, 32, 65536, 96 * 1024),
    75: (1024, 16, 65536, 64 * 1024),
    86: (1536, 16, 65536, 100 * 1024),
    89: (1536, 24, 65536, 100 * 1024),
    90: (2048, 32, 65536, 228 * 1024),
    120: (1536, 24, 65536, 100 * 1024),
}
_REG_ALLOC_UNIT = 256  # registers are allocated per warp in units of 256
_BLOCK_SIZES = (128, 256, 512)

# intrinsic regex -> architecture that first has it (CUDA C++ Programming Guide)
_INTRINSIC_MIN_ARCH = {
    r"__dp4a|__dp2a": 61,
    r"\bwmma::|mma\.sync|\bmma_sync": 70,
    r"__nanosleep|__match_(any|all)_sync": 70,
    r"cooperative_groups|this_grid\(\)|grid\.sync": 60,
    r"cp\.async|__pipeline_|memcpy_async": 80,
    r"__reduce_(add|min|max|and|or|xor)_sync": 80,
    r"__nv_bfloat16|__bfloat16|__float2bfloat16": 80,
    r"__nv_fp8|__nv_fp4": 89,
    r"cluster\.|__cluster_|barrier\.cluster|cuda::barrier": 90,
}
_ARITH = r"(?:add|sub|mul|fma\.rn|div\.rn|div\.full|div\.approx|rcp\.rn|rcp\.approx|sqrt\.rn|sqrt\.approx|min|max|abs|neg|ex2\.approx|lg2\.approx|sin\.approx|cos\.approx)"


@dataclass(frozen=True)
class EntryReport:
    """Static facts of one kernel entry on one architecture."""

    variant: str
    arch: int
    entry: str
    registers: int
    static_shared: int
    stack: int
    spill: int
    fp64_ops: int
    fp32_ops: int

    @property
    def fp64_share(self) -> float:
        """Share of f64 among f32 and f64 arithmetic opcodes (0 when there is none)."""
        total = self.fp64_ops + self.fp32_ops
        return self.fp64_ops / total if total else 0.0


def blocks_per_sm(arch: int, registers: int, static_shared: int, threads: int) -> tuple:
    """``(resident blocks per SM, limiting factor)`` for a block of ``threads``: the minimum of the thread, block-count, register-file and shared-memory limits."""
    max_threads, max_blocks, regs_sm, smem_sm = ARCH_LIMITS[arch]
    warps = threads // 32
    per_warp = -(-registers * 32 // _REG_ALLOC_UNIT) * _REG_ALLOC_UNIT
    limits = {
        "threads": max_threads // threads,
        "blocks": max_blocks,
        "registers": regs_sm // (per_warp * warps) if per_warp else max_blocks,
        "shared": smem_sm // static_shared if static_shared else max_blocks,
    }
    factor = min(limits, key=lambda k: limits[k])
    return limits[factor], factor


def occupancy(arch: int, registers: int, static_shared: int, threads: int) -> tuple:
    """``(theoretical occupancy as a fraction of the SM's thread slots, limiting factor)``."""
    blocks, factor = blocks_per_sm(arch, registers, static_shared, threads)
    return min(1.0, blocks * threads / ARCH_LIMITS[arch][0]), factor


def required_arch(source: str) -> int:
    """The highest first-supporting architecture of any intrinsic found in ``source`` (0 when none); a text scan, so a guarded use (``__CUDA_ARCH__``) still counts."""
    found = [arch for pat, arch in _INTRINSIC_MIN_ARCH.items() if re.search(pat, source)]
    return max(found) if found else 0


def _split_entries(ptx: str) -> dict:
    """``{entry name: its PTX text}``."""
    out, name, buf = {}, None, []
    for line in ptx.splitlines():
        m = re.match(r"\.(?:visible\s+)?\.entry\s+(\w+)", line)
        if m:
            name, buf = m.group(1), []
        if name:
            buf.append(line)
            if line.startswith("}"):
                out[name] = "\n".join(buf)
                name = None
    return out


def find_ptxas() -> str | None:
    """The toolkit's ptxas under ``CUDA_PATH`` or on ``PATH``, else ``None``."""
    exe = "ptxas.exe" if os.name == "nt" else "ptxas"
    candidates = [Path(os.environ["CUDA_PATH"]) / "bin" / exe] if os.environ.get("CUDA_PATH") else []
    candidates += [Path(base) / exe for base in os.environ.get("PATH", "").split(os.pathsep)]
    return next((str(p) for p in candidates if p.exists()), None)


def build_ptx_and_report(source: str, options: tuple, arch: int, ptxas: str, workdir: Path) -> tuple:
    """``(ptx text, ptxas -v report)`` for one build. NVRTC may emit a newer PTX ISA than the toolkit's ptxas accepts, so the ``.version`` line is lowered: this checks the PTX under
    the toolkit's ptxas, not the binary a user's driver builds."""
    from cupy_backends.cuda.libs import nvrtc

    prog = nvrtc.createProgram(source, "k.cu", [], [])
    try:
        nvrtc.compileProgram(prog, ["-std=c++14", f"-arch=compute_{arch}", *options])
        ptx = nvrtc.getPTX(prog)
    except Exception as exc:
        log = nvrtc.getProgramLog(prog)
        errors = [ln for ln in log.splitlines() if "error" in ln.lower()]
        raise RuntimeError(f"NVRTC failed on sm_{arch}: {'; '.join(errors[:3]) or exc}") from exc
    finally:
        nvrtc.destroyProgram(prog)
    ptx = re.sub(rb"\.version \d+\.\d+", b".version 8.7", ptx)
    path = workdir / "k.ptx"
    path.write_bytes(ptx)
    res = subprocess.run([ptxas, "-v", f"-arch=sm_{arch}", "-o", str(workdir / "k.cubin"), str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ptxas failed on sm_{arch}: {(res.stdout + res.stderr)[-500:]}")
    return ptx.decode("ascii", errors="replace"), res.stdout + res.stderr


def entry_reports(variant_id: str, arch: int, ptx: str, report: str) -> list:
    """One ``EntryReport`` per ``.entry`` of the build, joining the ptxas resource lines with the PTX opcode counts by entry name."""
    props = {m[0]: (int(m[1]), int(m[2]) + int(m[3])) for m in re.findall(r"Function properties for (\w+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", report)}
    regs = {m[0]: (int(m[1]), int(m[2] or 0)) for m in re.findall(r"Compiling entry function '(\w+)' for 'sm_\d+'.*?Used (\d+) registers(?:, (\d+) bytes smem)?", report, re.S)}
    out = []
    for name, text in _split_entries(ptx).items():
        stack, spill = props.get(name, (0, 0))
        r, smem = regs.get(name, (0, 0))
        f64 = len(re.findall(rf"^\s*{_ARITH}(?:\.\w+)*\.f64\b", text, re.M))
        f32 = len(re.findall(rf"^\s*{_ARITH}(?:\.\w+)*\.f32\b", text, re.M))
        out.append(EntryReport(variant_id, arch, name, r, smem, stack, spill, f64, f32))
    return out


def local_device_limits() -> tuple | None:
    """``(arch, max threads per SM, max blocks per SM)`` read from the local device, or ``None`` without one; used to check this module's table against hardware."""
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        arch = int(dev.compute_capability)
        return arch, int(dev.attributes["MaxThreadsPerMultiProcessor"]), int(dev.attributes["MaxBlocksPerMultiprocessor"])
    except Exception:
        return None


def markdown(rows: list, intrinsics: dict) -> str:
    """The report as markdown: occupancy where registers or shared memory limit it, FP64-heavy entries, and intrinsic requirements."""
    lines = ["# Offline kernel report (generated by tests/test_meta/test_cuda_kernels_offline_per_arch.py)", ""]
    lines += ["Theoretical occupancy from ptxas registers and static shared memory; block size is chosen at launch, so 128/256/512 threads are shown. Report, not a gate.", ""]
    worst = {}
    for r in rows:
        occs = {t: occupancy(r.arch, r.registers, r.static_shared, t) for t in _BLOCK_SIZES}
        key = (r.variant, r.entry)
        score = min(o for o, _ in occs.values())
        if key not in worst or score < worst[key][0]:
            worst[key] = (score, r, occs)
    lines += ["## Register- or shared-memory-limited entries (best block size below 50% occupancy)", "", "| kernel | entry | arch | regs | static smem | best occupancy (block, limiter) |", "|---|---|---|---|---|---|"]
    limited = []
    for (variant, entry), (_, r, occs) in worst.items():
        best_t = max(occs, key=lambda t: occs[t][0])
        if occs[best_t][0] < 0.5 and occs[best_t][1] in ("registers", "shared"):
            limited.append((occs[best_t][0], variant, entry, r, best_t, occs[best_t]))
    for _score, variant, entry, r, t, (o, why) in sorted(limited, key=lambda x: x[0])[:60]:
        lines.append(f"| {variant} | {entry} | sm_{r.arch} | {r.registers} | {r.static_shared} | {o:.0%} ({t}, {why}) |")
    lines += ["", "## FP64 share of arithmetic opcodes (static count, entries with at least 20 arithmetic ops, share >= 50%)", "", "| kernel | entry | fp64 ops | fp32 ops | share |", "|---|---|---|---|---|"]
    seen = set()
    for r in sorted(rows, key=lambda x: -x.fp64_share):
        if (r.variant, r.entry) in seen or r.fp64_ops + r.fp32_ops < 20 or r.fp64_share < 0.5:
            continue
        seen.add((r.variant, r.entry))
        lines.append(f"| {r.variant} | {r.entry} | {r.fp64_ops} | {r.fp32_ops} | {r.fp64_share:.0%} |")
    lines += ["", "## Intrinsics that need a newer architecture than sm_61", "", "| kernel | needs |", "|---|---|"]
    lines += [f"| {k} | sm_{v} |" for k, v in sorted(intrinsics.items()) if v > 61]
    return "\n".join(lines) + "\n"


def launch_bounds_of(ptx: str) -> dict:
    """``{entry name: (max threads, min blocks per SM or None)}`` from the ``.maxntid`` / ``.minnctapersm`` directives NVRTC writes for ``__launch_bounds__``."""
    out = {}
    for name, text in _split_entries(ptx).items():
        header = text.split("{", 1)[0]
        ntid = re.search(r"\.maxntid\s+(\d+)", header)
        nctas = re.search(r"\.minnctapersm\s+(\d+)", header)
        if ntid:
            out[name] = (int(ntid.group(1)), int(nctas.group(1)) if nctas else None)
    return out


def launch_bounds_violations(ptx: str, report: str) -> list:
    """``(entry, registers, cap)`` for every entry whose ptxas register count exceeds ``REGISTER_FILE // (max threads * min blocks)``.

    ``__launch_bounds__(T, B)`` makes ptxas cap registers so that B blocks of T threads fit the register file; a count above that cap means the bound was ignored.
    """
    regs = {m[0]: int(m[1]) for m in re.findall(r"Compiling entry function '(\w+)' for 'sm_\d+'.*?Used (\d+) registers", report, re.S)}
    bad = []
    for name, (threads, blocks) in launch_bounds_of(ptx).items():
        if blocks is None or name not in regs:
            continue
        cap = REGISTER_FILE // (threads * blocks)
        if regs[name] > cap:
            bad.append((name, regs[name], cap))
    return bad
