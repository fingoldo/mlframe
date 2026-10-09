"""``mlframe-tune-kernels ensure``: tune only the kernels whose cached tuning is missing or stale on THIS machine.

A tuning is valid for one hardware fingerprint (CPU model, GPU model, driver, CUDA runtime, numba/cupy versions) and one ``code_version`` of the kernel's source. A new
machine, a driver or library upgrade, or an edit of a kernel therefore leaves the entry missing (cold) or stale. ``ensure`` finds exactly those and sweeps only them, so
running it after an install, an upgrade or a kernel edit costs nothing when everything is current and a few minutes when something moved - unlike ``refresh-all``, which
re-sweeps every kernel unconditionally.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess  # nosec B404 - runs this same interpreter on a fixed module path, never a shell or user-supplied command
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version

__all__ = ["cmd_ensure", "spec_code_version", "spec_status", "survey"]

# Exit codes of ``ensure``: 0 current (or advisory), 1 work pending under --check, 2 a sweep failed, 3 the time budget ran out before everything was tuned.
EXIT_SWEEP_FAILED = 2
EXIT_BUDGET_SPENT = 3  # also: a sweep exceeded its per-kernel limit (the kernel stays untuned; nothing is broken)
DEFAULT_PER_KERNEL_MINUTES = 15.0

logger = logging.getLogger(__name__)

FRESH = "fresh"
STALE = "stale"
MISSING = "missing"


def spec_code_version(spec: Any) -> str:
    """The live ``code_version`` of a spec: its variant and extra functions plus its salt."""
    return compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)


def spec_status(spec: Any, cache: Any, code_version: Optional[str] = None) -> str:
    """``"fresh"`` when the kernel is tuned at its current code version on this hardware, ``"stale"`` when the stored tuning was made for other source, ``"missing"`` when none exists.

    The test is the one ``tune_spec(skip_existing=True)`` applies, so ``ensure --check`` and the sweep it triggers can never disagree.
    """
    if not cache.has(spec.kernel_name):
        return MISSING
    cv = code_version if code_version is not None else spec_code_version(spec)
    return STALE if cache.code_version_stale(spec.kernel_name, cv) else FRESH


def _cuda_present() -> bool:
    """Whether a CUDA device is usable here (GPU specs cannot be tuned, or be cold in any useful sense, without one)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return bool(is_cuda_available())
    except Exception as exc:  # no pyutilz helper / no driver: treat as no CUDA, but say why at debug level
        logger.debug("CUDA availability probe failed, assuming no CUDA: %s", exc)
        return False


def _selected(specs: dict, only: str, cuda: bool) -> list:
    """Specs to consider: all / gpu / cpu, and never a GPU spec on a host without CUDA."""
    out = []
    for name in sorted(specs):
        spec = specs[name]
        if spec.gpu_capable and not cuda:
            continue
        if only == "gpu" and not spec.gpu_capable:
            continue
        if only == "cpu" and spec.gpu_capable:
            continue
        out.append(spec)
    return out


def survey(specs: dict, only: str = "all", cache: Any = None, cuda: Optional[bool] = None) -> dict:
    """``{kernel_name: status}`` for the selected specs on this hardware."""
    cache = cache if cache is not None else KernelTuningCache()
    cuda = _cuda_present() if cuda is None else cuda
    return {spec.kernel_name: spec_status(spec, cache) for spec in _selected(specs, only, cuda)}


def _print_survey(status: dict) -> None:
    """One line per kernel that needs work, then the totals."""
    for name, st in sorted(status.items()):
        if st != FRESH:
            print(f"  {st:7} {name}", file=sys.stdout)
    n_bad = sum(1 for st in status.values() if st != FRESH)
    print(f"{len(status) - n_bad} of {len(status)} kernel tuning(s) current on this machine; {n_bad} need tuning.", file=sys.stdout)


def _lock_path() -> Path:
    """The lock file of the tuning run on this machine (next to the cache, so two checkouts of the project share it)."""
    override = os.environ.get("PYUTILZ_KERNEL_CACHE_DIR", "").strip()
    base = Path(override) if override else Path.home() / ".pyutilz" / "kernel_tuning"
    base.mkdir(parents=True, exist_ok=True)
    return base / ".ensure.lock"


def _pid_alive(pid: int) -> bool:
    """Whether a process with this id is running."""
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception as exc:  # cannot tell: assume the holder is gone so a dead lock cannot block tuning forever
        logger.warning("could not check whether process %s is alive (%s: %s); treating it as gone", pid, type(exc).__name__, exc)
        return False


def _lock_holder(path: Path) -> int:
    """The pid recorded in the lock file, or 0 (logged) when the file cannot be read - an unreadable lock is treated as stale."""
    try:
        return int(path.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError) as exc:
        logger.warning("tuning lock %s is unreadable (%s: %s); treating it as stale", path, type(exc).__name__, exc)
        return 0


@contextlib.contextmanager
def single_flight() -> Iterator[bool]:
    """Yield True when this process holds the machine-wide tuning lock, False when a live process already does.

    Two sweeps at once measure each other (and the fits around them), so a second ``ensure`` must not start: it reports that a run is in progress and leaves. A lock left by a
    process that died is taken over.
    """
    path = _lock_path()
    for _ in range(2):
        try:
            with open(path, "x", encoding="utf-8") as lock:  # "x": create exclusively, so exactly one process wins
                lock.write(str(os.getpid()))
        except FileExistsError:
            holder = _lock_holder(path)
            if holder and holder != os.getpid() and _pid_alive(holder):
                yield False
                return
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("could not remove the stale tuning lock %s (%s: %s); not starting a run", path, type(exc).__name__, exc)
                yield False
                return
            continue
        try:
            yield True
        finally:
            with contextlib.suppress(OSError):
                path.unlink()
        return
    yield False


def _run_sweep(spec: Any, timeout_s: float) -> tuple:
    """Tune one kernel in a CHILD process so a hung or crashing sweep (a CUDA fault, a runaway grid) can be stopped and cannot take the other kernels down with it.

    Returns ``("ok", "")``, ``("timeout", detail)`` or ``("failed", detail)``. The child is ``mlframe-tune-kernels refresh <kernel>`` and persists its own result.
    """
    cmd = [sys.executable, "-m", "mlframe.system.kernel_tuning_cache", "refresh", spec.kernel_name]
    try:
        proc = subprocess.run(cmd, timeout=timeout_s, check=False)  # nosec B603 - fixed argv: this interpreter, this package, a registered kernel name
    except subprocess.TimeoutExpired:
        return "timeout", f"exceeded the {timeout_s / 60:.0f} min per-kernel limit"
    if proc.returncode != 0:
        return "failed", f"the sweep process exited with code {proc.returncode}"
    return "ok", ""


def cmd_ensure(
    specs: dict,
    *,
    check: bool = False,
    only: str = "all",
    if_cuda: bool = False,
    max_minutes: Optional[float] = None,
    advisory: bool = False,
    per_kernel_minutes: float = DEFAULT_PER_KERNEL_MINUTES,
) -> int:
    """Tune the missing/stale kernels; with ``check`` only report them.

    ``if_cuda``: exit 0 silently on a host without CUDA (the form for hooks and deploy scripts that run everywhere). ``max_minutes``: stop starting new sweeps once the
    budget is spent (the rest stay missing and are reported). ``per_kernel_minutes``: each kernel is swept in its own process and stopped after this long. ``advisory``: never fail - report what is out of date and exit 0.

    Exit code: 0 when everything is current afterwards (or ``advisory``), 1 when ``check`` finds work, 2 when a sweep failed, 3 when the budget ran out first or a sweep hit its per-kernel limit. Only one
    tuning run per machine at a time: a second one reports that a run is in progress and exits 0.
    """
    cuda = _cuda_present()
    if if_cuda and not cuda:
        return 0
    cache = KernelTuningCache()
    status = survey(specs, only, cache, cuda)
    todo = [spec for spec in _selected(specs, only, cuda) if status[spec.kernel_name] != FRESH]
    if check or not todo:
        _print_survey(status)
        return 0 if (advisory or not todo) else 1
    with single_flight() as mine:
        if not mine:
            print("another kernel-tuning run is already in progress on this machine; not starting a second one.", flush=True)
            return 0
        return _tune_all(todo, status, max_minutes, advisory, per_kernel_minutes)


def _tune_all(todo: list, status: dict, max_minutes: Optional[float], advisory: bool, per_kernel_minutes: float) -> int:
    """Sweep the given specs one by one within the budget; per-kernel failures are reported and do not stop the rest."""
    deadline = None if max_minutes is None else time.monotonic() + float(max_minutes) * 60.0
    failed: list = []
    skipped: list = []
    timed_out: list = []
    for spec in todo:
        if deadline is not None and time.monotonic() > deadline:
            skipped.append(spec.kernel_name)
            continue
        t0 = time.monotonic()
        print(f"tuning {spec.kernel_name} ({status[spec.kernel_name]}) ...", flush=True)
        limit_s = float(per_kernel_minutes) * 60.0
        if deadline is not None:
            limit_s = max(1.0, min(limit_s, deadline - time.monotonic()))  # never let one sweep run past the overall budget
        outcome, detail = _run_sweep(spec, limit_s)
        if outcome == "ok":
            n = len(KernelTuningCache().get_regions(spec.kernel_name) or [])
            print(f"  {spec.kernel_name}: {n} region(s), {time.monotonic() - t0:.0f}s", flush=True)
        elif outcome == "timeout":
            timed_out.append(spec.kernel_name)
            print(f"  {spec.kernel_name}: STOPPED ({detail})", file=sys.stderr, flush=True)
        else:
            failed.append(spec.kernel_name)
            print(f"  {spec.kernel_name}: FAILED ({detail})", file=sys.stderr, flush=True)
    if timed_out:
        print(f"stopped at the time limit (left untuned): {', '.join(timed_out)}", file=sys.stderr)
    if skipped:
        print(f"time budget of {max_minutes} min spent; not tuned: {', '.join(skipped)}", file=sys.stderr)
    if failed:
        print(f"sweeps failed: {', '.join(failed)}", file=sys.stderr)
    if advisory or not (failed or skipped or timed_out):
        return 0
    return EXIT_SWEEP_FAILED if failed else EXIT_BUDGET_SPENT
