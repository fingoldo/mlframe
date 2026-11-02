"""Report how the launch geometry of the new device kernels sits on THIS GPU: device limits, registers and shared memory of each kernel, the constant block width it uses today, the width ``_gpu_hw_launch`` derives and the resident threads per SM at every width.

Run it on a new card before trusting the launch constants: ``python -m mlframe.feature_selection._benchmarks.profiling.launch_report``. It compiles each kernel (a few seconds), launches nothing and times nothing.
"""

from __future__ import annotations

import sys

WIDTHS = (64, 128, 256, 512, 1024)
ROW_COUNTS = (20_000, 100_000, 1_000_000)


def _kernels(cp):
    """``(label, compiled function, current constant width, shared-bytes rule)`` of the kernels whose launch geometry depends on the device."""
    from mlframe.feature_selection.filters import _honest_null_accept_bits as accept
    from mlframe.feature_selection.filters import _parity_gpu as parity
    from mlframe.feature_selection.filters import _quantile_codes_gpu as qcodes
    from mlframe.feature_selection.filters import _triple_mi_gpu as triple

    out = [
        ("honest_null root_split_accept", accept._MODULE.get(cp).get_function("root_split_accept"), accept.THREADS, lambda n: 2 * ((n + 31) // 32) * 4),
        ("triple_mi (base 5, 2 classes)", triple._MODULE.get(cp).get_function("triple_mi"), triple.THREADS, lambda n: 125 * 2 * 4),
        ("parity_mi_grid", parity._MODULE.get(cp).get_function("parity_mi_grid"), parity.THREADS, lambda n: 0),
    ]
    radix = qcodes._RADIX.get(cp)
    for name in ("radix_hist1", "radix_hist2", "radix_hist3"):
        out.append((f"quantile {name}", radix.get_function(name), qcodes.THREADS, lambda n: 0))
    return out


def main() -> int:
    """Print the device limits and the per-kernel table; returns a process exit code."""
    try:
        import cupy as cp
    except Exception as exc:
        print(f"no cupy / CUDA: {exc}")
        return 1
    from mlframe.feature_selection.filters import _gpu_hw_launch as hw

    props = hw.device_props()
    if not props:
        print("no device properties")
        return 1
    dev = cp.cuda.Device()
    print(f"device {dev.id}: cc {dev.compute_capability}, {props['sm_count']} SMs")
    for key in ("max_threads_per_block", "max_threads_per_sm", "max_blocks_per_sm", "warp", "shared_per_block", "shared_per_sm", "reserved_shared_per_block", "regs_per_sm", "regs_per_block", "reg_alloc_unit", "l2_bytes"):
        print(f"  {key:28s} {props.get(key)}")
    for label, fn, constant, shared_rule in _kernels(cp):
        regs, static = int(fn.num_regs), int(fn.shared_size_bytes)
        print(f"\n{label}: {regs} registers, {static} B static shared, constant width {constant}")
        for n in ROW_COUNTS:
            dyn = int(shared_rule(n))
            picked = hw.fixed_shared_block_size(dyn, regs_per_thread=regs, static_smem=static, min_width=64, max_width=1024, work_items=n)
            cells = []
            for w in WIDTHS:
                blocks = hw._max_active_blocks_per_sm(w, regs, static, dyn, props)
                cells.append(f"{w}:{blocks * w / props['max_threads_per_sm']:.0%}")
            print(f"  n={n:>9,d} dyn shared {dyn:>7,d} B  picked {picked or '-':>5}  resident-thread share by width: " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
