"""The pre-sweep fallback shared by the row-count-gated device kernels: take the device from a measured row count, else stay on the host."""

from __future__ import annotations

import sys

RESIDENCIES = ("host", "device")  # the two placements of an operand or a result; a sweep scenario is an (in, out) pair of them


def residency_entry_name(kernel_name: str, residency_in: str, residency_out: str) -> str:
    """The tuner entry of one scenario: ``<name>__<in>_<out>`` (``fe_modular_codes_backend__host_host``)."""
    if residency_in not in RESIDENCIES or residency_out not in RESIDENCIES:
        raise ValueError(f"residency must be one of {RESIDENCIES}, got ({residency_in!r}, {residency_out!r})")
    return f"{kernel_name}__{residency_in}_{residency_out}"


def detect_residency(*operands) -> str:
    """``"device"`` when every array operand is a cupy array, ``"host"`` when none is; a mix is ``"host"`` (the device path pays the upload of the host operand, which is the host-in cost).

    Non-array operands (scalars, strings, None) are ignored."""
    flags = []
    for op in operands:
        if op is None or isinstance(op, (str, bytes, int, float, bool)):
            continue
        mod = type(op).__module__
        if mod.startswith("cupy"):
            flags.append(True)
        elif hasattr(op, "shape"):
            flags.append(False)
    return "device" if flags and all(flags) else "host"


def device_from_rows(n_samples: int, min_rows: int) -> str:
    """``"device"`` when ``n_samples`` reaches ``min_rows`` (the measured crossover of the kernel), else ``"host"``."""
    return "device" if int(n_samples) >= int(min_rows) else "host"


def rows_fallback(owner: str, name: str = "DEVICE_FALLBACK_ROWS"):
    """The pre-sweep fallback of a row-count-gated kernel: ``device_from_rows`` with the threshold read from the module named ``owner`` (pass ``__name__``) at call
    time, so a test or an operator lowering the module constant takes effect."""

    def _fallback_choice(n_samples: int) -> str:
        """Pre-sweep fallback: the device from the owner module's row threshold."""
        return device_from_rows(n_samples, getattr(sys.modules[owner], name))

    return _fallback_choice


def host_device_sweep(owner: str, host, device, *, repeats: int = 3, rtol: float = 1e-9, atol: float | None = None):
    """The kernel-tuning sweep of a host-versus-device kernel: time both per row count and require the device to agree with the host.

    ``host`` and ``device`` are the two callables; ``owner`` is the module's ``__name__``, in which ``_SWEEP_ROWS`` and ``_sweep_inputs`` are looked up at call time, so the module
    can define them after the sweep is built. ``atol`` defaults to ``rtol``."""

    def _run_sweep() -> list:
        """Time the host and the device kernel per row count; the device must agree with the host."""
        from pyutilz.dev.benchmarking import sweep_backend_grid

        module = sys.modules[owner]
        return list(
            sweep_backend_grid(
                {"host": host, "device": device},
                {"n_samples": module._SWEEP_ROWS},
                module._sweep_inputs,
                reference="host",
                repeats=repeats,
                equiv_rtol=rtol,
                equiv_atol=rtol if atol is None else atol,
            )
        )

    return _run_sweep


def residency_report(owner: str, *, repeats: int = 3) -> list:
    """Per swept row count and declared scenario, the seconds of the host and of the device callable, and the transfer share of the device time.

    ``owner`` (pass ``__name__``) defines ``_SWEEP_ROWS``, ``_sweep_inputs``, ``_sweep_host``, ``_sweep_device`` and, for the scenarios beyond host-in / host-out, ``_SWEEP_RESIDENCIES``
    (a list of ``(in, out)`` pairs) and ``_sweep_place(inputs, residency)`` which puts the inputs where the scenario says (outside the timing). The transfer share is
    ``1 - device_time(device, device) / device_time(host, host)`` when both scenarios are declared, else ``None``. Rows are dicts: ``n_samples``, ``scenario``, ``host_s``, ``device_s``, ``transfer_share``."""
    import time

    module = sys.modules[owner]
    scenarios = list(getattr(module, "_SWEEP_RESIDENCIES", [("host", "host")]))
    place = getattr(module, "_sweep_place", lambda inputs, residency: inputs)

    def _sync() -> None:
        """Wait for the queued device work (nothing to do without a device)."""
        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
        except Exception:  # nosec B110 - no device: nothing to synchronise
            pass

    def _best(fn, args) -> float:
        """Best of ``repeats`` synchronised timings of ``fn(*args)`` after one warm-up call."""
        fn(*args)
        _sync()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args)
            _sync()
            times.append(time.perf_counter() - t0)
        return min(times)

    rows: list = []
    for n in module._SWEEP_ROWS:
        inputs = module._sweep_inputs({"n_samples": n})
        timings = {}
        for res_in, res_out in scenarios:
            placed = place(inputs, res_in)
            timings[(res_in, res_out)] = (_best(module._sweep_host, placed) if res_in == "host" else None, _best(module._sweep_device, placed))
        base = timings.get(("host", "host"))
        resident = timings.get(("device", "device"))
        share = None if base is None or resident is None else 1.0 - resident[1] / base[1]
        for scenario, (host_s, device_s) in timings.items():
            rows.append({"n_samples": n, "scenario": scenario, "host_s": host_s, "device_s": device_s, "transfer_share": share})
    return rows
