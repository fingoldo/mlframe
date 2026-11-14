"""The pre-sweep fallback shared by the row-count-gated device kernels: take the device from a measured row count, else stay on the host."""

from __future__ import annotations

import sys

import numpy as np

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


def host_device_sweep(owner: str, host, device, *, repeats: int = 3, rtol: float = 1e-9, atol: float | None = None, residency: bool = False):
    """The kernel-tuning sweep of a host-versus-device kernel: time both per row count and require the device to agree with the host.

    ``host`` and ``device`` are the two callables; ``owner`` is the module's ``__name__``, in which ``_SWEEP_ROWS`` and ``_sweep_inputs`` are looked up at call time, so the module
    can define them after the sweep is built. ``atol`` defaults to ``rtol``.

    ``residency=True`` makes the sweep scenario-aware (for an entry declared with ``residencies=``): the returned tuner takes ``data_in_residency`` and ``data_out_residency``, the
    inputs are placed by the module's ``_sweep_place(inputs, residency)`` outside the timing, and both callables receive ``data_out_residency=`` and return their result in that
    placement as a host-comparable value (a vector reduced on the side the result lives, so the comparison adds one small copy to both backends alike)."""

    def _run_sweep(data_in_residency: str = "host", data_out_residency: str = "host") -> list:
        """Time the host and the device kernel per row count in the given data placement; the device must agree with the host."""
        from pyutilz.dev.benchmarking import sweep_backend_grid

        module = sys.modules[owner]
        make_inputs = module._sweep_inputs
        host_fn, device_fn = host, device
        if residency:
            place = vars(module).get("_sweep_place", lambda inputs, res: inputs)  # optional module hook, see the docstring
            make_inputs = lambda dims: place(module._sweep_inputs(dims), data_in_residency)  # noqa: E731 - small adapter
            host_fn = lambda *a: host(*a, data_out_residency=data_out_residency)  # noqa: E731
            device_fn = lambda *a: device(*a, data_out_residency=data_out_residency)  # noqa: E731
        return list(
            sweep_backend_grid(
                {"host": host_fn, "device": device_fn},
                {"n_samples": module._SWEEP_ROWS},
                make_inputs,
                reference="host",
                repeats=repeats,
                equiv_rtol=rtol,
                equiv_atol=rtol if atol is None else atol,
            )
        )

    return _run_sweep


def selection_view(scores: object, top: int = 10, tie_band: float = 1e-4, width: int = 1024) -> np.ndarray:
    """What a selection step reads from a scoring pass, as a fixed-length vector: how many candidates were scored, and which of them are among the best ``top`` (those at or within
    the relative ``tie_band`` of the ``top``-th best).

    Two scorers that bin or round differently differ in the last digits of the scores (a few rows across a bin edge) while ranking the same candidates at the top; comparing this view
    lets a sweep hold both to the strict tolerance instead of a slack one on the score values."""
    scores = np.asarray(scores, dtype=np.float64)  # Intentional float64: comparison vector
    out = np.zeros(1 + int(width), dtype=np.float64)  # Intentional float64: comparison vector
    out[0] = scores.size
    if scores.size:
        kth = np.sort(scores)[-min(int(top), scores.size)]
        out[1 : 1 + min(scores.size, int(width))] = (scores >= kth - tie_band * abs(kth)).astype(np.float64)[: int(width)]
    return out


def residency_report(owner: str, *, repeats: int = 3) -> list:
    """Per swept row count and declared scenario, the seconds of the host and of the device callable, and the transfer share of the device time.

    ``owner`` (pass ``__name__``) defines ``_SWEEP_ROWS``, ``_sweep_inputs``, ``_sweep_host``, ``_sweep_device`` and, for the scenarios beyond host-in / host-out, ``_SWEEP_RESIDENCIES``
    (a list of ``(in, out)`` pairs) and ``_sweep_place(inputs, residency)`` which puts the inputs where the scenario says (outside the timing). The transfer share is
    ``1 - device_time(device, device) / device_time(host, host)`` when both scenarios are declared, else ``None``. Rows are dicts: ``n_samples``, ``scenario``, ``host_s``, ``device_s``, ``transfer_share``."""
    import time

    module = sys.modules[owner]
    scenarios = list(vars(module).get("_SWEEP_RESIDENCIES", [("host", "host")]))
    place = vars(module).get("_sweep_place", lambda inputs, residency: inputs)

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
