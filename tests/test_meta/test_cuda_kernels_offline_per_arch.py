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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _offline_kernel_report import ARCH_LIMITS, build_ptx_and_report, entry_reports, launch_bounds_violations, local_device_limits, markdown, occupancy, required_arch
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
    "mlframe/feature_selection/filters/_pairwise_modular_rank_gpu.py::_KERNEL_TEMPLATE[bounds=__launch_bounds__(128, MLF_THREADS_PER_SM / 128) int reduce_v1]": (
        "Versions 1 to 3 of residue_rank are built from ``_KERNEL_TEMPLATE`` with their real reduction fragments (the recipe compiled the template with an empty fragment until 2026-11-13). The capped build "
        "(``__launch_bounds__(128, MLF_THREADS_PER_SM / 128)``, version 3: version 2 at 32 registers) spills 8 to 24 bytes/thread on sm_61 only. The file itself records why version 4 exists: its leaner body is the one "
        "that meets the 32-register cap without spilling. Versions 1 and 2 (no cap) and every other arch assemble clean. Version 3 is a kept reference version (the default is 8, MLFRAME_MODULAR_RANK_KERNEL "
        "selects others); loosening its cap would change what that version measures."
    ),
    "mlframe/feature_selection/filters/_pairwise_modular_rank_gpu.py::_KERNEL_TEMPLATE[bounds=__launch_bounds__(128, MLF_THREADS_PER_SM / 128) int reduce_v2]": (
        "Versions 1 to 3 of residue_rank are built from ``_KERNEL_TEMPLATE`` with their real reduction fragments (the recipe compiled the template with an empty fragment until 2026-11-13). The capped build "
        "(``__launch_bounds__(128, MLF_THREADS_PER_SM / 128)``, version 3: version 2 at 32 registers) spills 8 to 24 bytes/thread on sm_61 only. The file itself records why version 4 exists: its leaner body is the one "
        "that meets the 32-register cap without spilling. Versions 1 and 2 (no cap) and every other arch assemble clean. Version 3 is a kept reference version (the default is 8, MLFRAME_MODULAR_RANK_KERNEL "
        "selects others); loosening its cap would change what that version measures."
    ),
    "mlframe/feature_selection/filters/_pairwise_modular_rank_gpu.py::_KERNEL_TEMPLATE[bounds=__launch_bounds__(128, MLF_THREADS_PER_SM / 128) long long reduce_v1]": (
        "Versions 1 to 3 of residue_rank are built from ``_KERNEL_TEMPLATE`` with their real reduction fragments (the recipe compiled the template with an empty fragment until 2026-11-13). The capped build "
        "(``__launch_bounds__(128, MLF_THREADS_PER_SM / 128)``, version 3: version 2 at 32 registers) spills 8 to 24 bytes/thread on sm_61 only. The file itself records why version 4 exists: its leaner body is the one "
        "that meets the 32-register cap without spilling. Versions 1 and 2 (no cap) and every other arch assemble clean. Version 3 is a kept reference version (the default is 8, MLFRAME_MODULAR_RANK_KERNEL "
        "selects others); loosening its cap would change what that version measures."
    ),
    "mlframe/feature_selection/filters/_pairwise_modular_rank_gpu.py::_KERNEL_TEMPLATE[bounds=__launch_bounds__(128, MLF_THREADS_PER_SM / 128) long long reduce_v2]": (
        "Versions 1 to 3 of residue_rank are built from ``_KERNEL_TEMPLATE`` with their real reduction fragments (the recipe compiled the template with an empty fragment until 2026-11-13). The capped build "
        "(``__launch_bounds__(128, MLF_THREADS_PER_SM / 128)``, version 3: version 2 at 32 registers) spills 8 to 24 bytes/thread on sm_61 only. The file itself records why version 4 exists: its leaner body is the one "
        "that meets the 32-register cap without spilling. Versions 1 and 2 (no cap) and every other arch assemble clean. Version 3 is a kept reference version (the default is 8, MLFRAME_MODULAR_RANK_KERNEL "
        "selects others); loosening its cap would change what that version measures."
    ),
    "mlframe/feature_selection/filters/_pairwise_modular_rank_gpu.py::_KERNEL_V4_TEMPLATE": (
        "Version 4 of residue_rank spills 8 bytes/thread on sm_70 only, at the 32-register cap of (128, MLF_THREADS_PER_SM / 128). It is a kept experimental version (the default is 8, "
        "MLFRAME_MODULAR_RANK_KERNEL selects others); loosening its cap would change what that version measures."
    ),
}
# NVRTC and ptxas warning substrings that are accepted everywhere, each with the reason.
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


def _skip_or_fail(reason: str):
    """Skip when the toolchain is absent, or FAIL when ``MLFRAME_REQUIRE_OFFLINE=1`` (set it in the CI image that has the CUDA toolkit: a missing ptxas must not turn the whole suite into silent skips)."""
    if os.environ.get("MLFRAME_REQUIRE_OFFLINE", "").strip() == "1":
        pytest.fail(f"MLFRAME_REQUIRE_OFFLINE=1 but the offline toolchain is missing: {reason}")
    pytest.skip(reason)


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
    _skip_or_fail("no ptxas on PATH or under CUDA_PATH")


def _nvrtc_or_skip() -> None:
    """Import cupy's NVRTC binding, or skip (fail under ``MLFRAME_REQUIRE_OFFLINE=1``)."""
    try:
        import cupy_backends.cuda.libs.nvrtc  # noqa: F401
    except ImportError as exc:
        _skip_or_fail(f"cupy's NVRTC binding is not importable: {exc}")


def _variant_id(v: KernelVariant) -> str:
    """The id of one kernel build in a failure message and in the exemption table (source key plus label)."""
    return f"{v.key}[{v.label}]"


_VARIANTS = all_variants()
_REPORT_ROWS: list = []  # filled by the per-arch test as it runs; the report test below reads it (a partial list under xdist only shortens the report)


@pytest.mark.parametrize("key,reason", sorted(_SPILL_EXEMPT.items()) or [pytest.param(None, None, marks=pytest.mark.skip(reason="no spill exemptions registered"))])
def test_every_spill_exemption_names_a_real_kernel_and_a_reason(key, reason):
    """An exemption must match an extracted kernel source (or one variant of it, by its ``key[label]`` id) and carry a reason; a stale or empty one fails (an empty registry is a skip, not a silent pass)."""
    assert key in {v.key for v in _VARIANTS} or key in {_variant_id(v) for v in _VARIANTS}, f"exemption for a kernel or variant that is not extracted: {key}"
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


_FAILURE_ROWS: list = []  # (variant id, arch, first line of the failure), the per-architecture failure table of the report


@pytest.mark.parametrize("arch", _archs())
@pytest.mark.parametrize("variant", _VARIANTS, ids=_variant_id)
def test_kernel_assembles_without_spill_on_arch(variant: KernelVariant, arch: int, tmp_path):
    """NVRTC then ptxas -v: compiles, one resource block per entry, no stack frame or spill, no unlisted warning."""
    try:
        _check_variant(variant, arch, tmp_path)
    except BaseException as exc:  # pytest.fail raises a BaseException subclass; recorded, then re-raised unchanged
        _FAILURE_ROWS.append((_variant_id(variant), arch, (str(exc).splitlines() or [type(exc).__name__])[0][:200]))
        raise


def _check_variant(variant: KernelVariant, arch: int, tmp_path) -> None:
    """The body of the per-arch check (kept separate so the wrapper above can record its failure)."""
    _nvrtc_or_skip()
    ptxas = _ptxas()
    if leftover_placeholders(variant):
        pytest.fail(f"unresolved placeholders {leftover_placeholders(variant)}")
    try:
        ptx_text, report = build_ptx_and_report(variant.source, variant.options, arch, ptxas, tmp_path)
    except RuntimeError as exc:
        pytest.fail(str(exc))
    ptx = ptx_text.encode()
    _REPORT_ROWS.extend(entry_reports(_variant_id(variant), arch, ptx_text, report))

    entry_names = [n.decode() for n in re.findall(rb"\.entry\s+(\w+)", ptx)]
    assert entry_names, "no .entry in the PTX"
    # ptxas prints a "Function properties for NAME" block per function (device helpers too): match them to the entries by name.
    block = r"Function properties for (\w+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
    props = {m[0]: m[1:] for m in re.findall(block, report)}
    missing = [n for n in entry_names if n not in props]
    assert not missing, f"entries without a ptxas resource block: {missing}"
    spills = {n: props[n] for n in entry_names if props[n][1] != "0" or props[n][2] != "0"}
    # a ``__noinline__`` helper has its own resource block: a spill there costs every call, and a check on the entries alone never sees it
    helper_spills = {n: v for n, v in props.items() if n not in entry_names and (v[1] != "0" or v[2] != "0")}
    if variant.key not in _SPILL_EXEMPT and _variant_id(variant) not in _SPILL_EXEMPT:
        assert not spills, f"sm_{arch}: register spill (stack, spill st, spill ld) per entry: {spills}"
        assert not helper_spills, f"sm_{arch}: register spill (stack, spill st, spill ld) in a device helper: {helper_spills}"

    warnings = [ln for ln in report.splitlines() if "warning" in ln.lower() and not any(k in ln for k in _WARNING_ALLOWLIST)]
    assert not warnings, f"sm_{arch}: unlisted NVRTC or ptxas warnings: {warnings[:3]}"
    bounds_bad = launch_bounds_violations(ptx_text, report)
    assert not bounds_bad, f"sm_{arch}: __launch_bounds__ not honoured (entry, registers, cap): {bounds_bad}"


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


def test_occupancy_arithmetic_on_known_cases():
    """The occupancy model on hand-computed cases: thread-limited, register-limited, shared-memory-limited and block-count-limited."""
    assert occupancy(61, 32, 0, 128) == (1.0, "threads")  # 16 blocks of 128 = 2048 threads
    occ, why = occupancy(61, 255, 0, 256)  # 255 regs -> 8192 per warp-group x 8 warps = 65536: one block, 256 of 2048 threads
    assert (occ, why) == (0.125, "registers")
    occ, why = occupancy(75, 32, 48 * 1024, 128)  # 64 KB per SM / 48 KB: one block of 128 on a 1024-thread SM
    assert (occ, why) == (0.125, "shared")
    occ, why = occupancy(61, 16, 0, 32)  # 32-thread blocks: the 32-block cap, 1024 of 2048 threads
    assert (occ, why) == (0.5, "blocks")


def test_architecture_limit_table_matches_the_local_device():
    """The threads-per-SM and blocks-per-SM rows of ARCH_LIMITS agree with what the local device reports (the other rows are the documented table, not a measurement)."""
    local = local_device_limits()
    if local is None:
        pytest.skip("no CUDA device to read limits from")
    arch, threads, blocks = local
    if arch not in ARCH_LIMITS:
        pytest.skip(f"sm_{arch} is not in the table")
    assert (threads, blocks) == ARCH_LIMITS[arch][:2], f"sm_{arch}: device says {(threads, blocks)}, table says {ARCH_LIMITS[arch][:2]}"


# kernel key -> reason a kernel that names an intrinsic needing an architecture newer than sm_61 is accepted (a text hit that is guarded by __CUDA_ARCH__, or a gate that skips the card).
_NEWER_ARCH_OK: dict = {}


def test_kernels_using_a_newer_architecture_intrinsic_are_listed():
    """Every kernel whose source names an intrinsic that needs more than sm_61 is accepted explicitly: the matrix compiles for sm_61 too, so an unguarded one would already fail there; a
    guarded one needs its reason recorded."""
    needs = {v.key: required_arch(v.source) for v in _VARIANTS}
    unlisted = {k: a for k, a in needs.items() if a > 61 and k not in _NEWER_ARCH_OK}
    assert not unlisted, f"kernels naming an intrinsic newer than sm_61 without a recorded reason: {unlisted}"


def test_report_builds_and_can_be_written():
    """The occupancy / FP64 / intrinsic report builds from the collected rows (every row sane); ``MLFRAME_OFFLINE_REPORT=<path>`` writes it (CI keeps it as an artifact)."""
    if not _REPORT_ROWS:
        _skip_or_fail("the per-arch test did not run in this process")
    if os.environ.get("MLFRAME_REQUIRE_OFFLINE", "").strip() == "1":
        built = {(r.variant, r.arch) for r in _REPORT_ROWS}
        failed = {(vid, arch) for vid, arch, _msg in _FAILURE_ROWS}
        expected = {(_variant_id(v), a) for v in _VARIANTS for a in _archs()}
        assert not (expected - built - failed), f"variants never built in this run: {sorted(expected - built - failed)[:5]}"
    for r in _REPORT_ROWS:
        assert r.registers > 0 and r.registers <= 255, r
        occ, _ = occupancy(r.arch, r.registers, r.static_shared, 256)
        assert 0.0 <= occ <= 1.0
    text = markdown(_REPORT_ROWS, {v.key: required_arch(v.source) for v in _VARIANTS}, failures=_FAILURE_ROWS)
    assert "Offline kernel report" in text
    assert "Per-architecture failure table" in text
    out = os.environ.get("MLFRAME_OFFLINE_REPORT")
    if out:
        Path(out).write_text(text, encoding="utf-8")


def test_launch_bounds_violation_is_detected_on_synthetic_resource_report():
    """A register count above REGISTER_FILE // (threads * blocks) is reported; one at or under the cap is not (mutant check for the launch-bounds gate)."""
    from _offline_kernel_report import REGISTER_FILE

    ptx = ".visible .entry k(\n)\n.maxntid 128, 1, 1\n.minnctapersm 16\n{\n}\n"
    cap = REGISTER_FILE // (128 * 16)
    over = f"Compiling entry function 'k' for 'sm_61'\nUsed {cap + 8} registers"
    under = f"Compiling entry function 'k' for 'sm_61'\nUsed {cap} registers"
    assert launch_bounds_violations(ptx, over) == [("k", cap + 8, cap)]
    assert launch_bounds_violations(ptx, under) == []


def test_failure_table_lists_each_failing_variant_per_arch():
    """The report's failure table names every recorded failure with its arch, and says so when there is none."""
    text = markdown([], {}, failures=[("mlframe/x.py::_SRC[float]", 61, "NVRTC failed on sm_61")])
    assert "| mlframe/x.py::_SRC[float] | sm_61 | NVRTC failed on sm_61 |" in text
    assert "none: every checked variant assembled" in markdown([], {}, failures=[])


def test_an_nvrtc_warning_of_a_successful_build_reaches_the_report(tmp_path):
    """A kernel with an unused variable builds, and its NVRTC warning is in the report the warning check reads (the log of a successful build used to be dropped)."""
    _nvrtc_or_skip()
    src = 'extern "C" __global__ void k(float* o) { int unused = 3; o[threadIdx.x] = 1.0f; }'
    _ptx, report = build_ptx_and_report(src, (), 75, _ptxas(), tmp_path)
    assert any("nvrtc:" in ln and "warning" in ln.lower() and "unused" in ln for ln in report.splitlines()), report[:400]


_KERNEL_NAME_RE = re.compile(r"__global__\s+(?:void\s+)?(?:__launch_bounds__\([^)]*\)\s*)?(?:void\s+)?(\w+)\s*\(")


def _template_name_prefix(name: str) -> str:
    """The fixed part of a kernel name spelled with a template token (``bin_codes_TYPENAME`` -> ``bin_codes_``); the name itself when it has none."""
    return re.split(r"[A-Z]{4,}", name)[0] if re.search(r"[A-Z]{4,}", name) else name


def test_every_global_kernel_name_written_in_a_file_is_compiled():
    """Name-level coverage diff: every ``__global__`` kernel name in a production file appears in the text of a compiled variant of that file (a template spelling like ``bin_codes_TYPENAME`` matches the
    concrete names its recipe builds). The file-level test above passes when only one kernel of a file is extracted; this one does not."""
    from _offline_kernel_sources import ROOT, _EXCLUDED_PARTS

    compiled: dict = {}
    for v in _VARIANTS:
        compiled.setdefault(v.key.split("::")[0], set()).update(_KERNEL_NAME_RE.findall(v.source))
    gaps = {}
    for path in ROOT.rglob("*.py"):
        if any(part in path.parts for part in _EXCLUDED_PARTS):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "__global__" not in text:
            continue
        rel = "mlframe/" + path.relative_to(ROOT).as_posix()
        have = compiled.get(rel, set())
        missing = []
        for name in sorted(set(_KERNEL_NAME_RE.findall(text)) - {"void"}):
            prefix = _template_name_prefix(name)
            if name in have or (prefix != name and len(prefix) >= 5 and any(h.startswith(prefix) for h in have)):
                continue
            missing.append(name)
        if missing:
            gaps[rel] = missing
    assert not gaps, f"kernel names written in a file but never compiled offline (extractor gap or a recipe that drops them): {gaps}"


def test_the_template_name_rule_matches_only_a_fixed_prefix():
    """``bin_codes_TYPENAME`` is matched by its prefix; a plain name, or a short or empty prefix, is never matched by prefix."""
    assert _template_name_prefix("bin_codes_TYPENAME") == "bin_codes_"
    assert _template_name_prefix("legval_kernel") == "legval_kernel"
    assert _template_name_prefix("TYPENAME_k") == ""
