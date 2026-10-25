"""Print the key metrics of every kernel in one or more Nsight Compute reports (`*.ncu-rep`).

On this Windows host the text pages of `ncu --import` fail with "bad conversion" (see CUDA.md section 12), so the report is read through
the `ncu_report` Python module that ships with Nsight Compute instead. It needs no GPU and no admin rights.

    python scripts/ncu_key_metrics.py report.ncu-rep [other.ncu-rep ...]

A metric the report does not hold prints as "(not collected)": collect with `--set full`, or add `--metrics` for a specific one.
"""

from __future__ import annotations

import glob
import os
import sys

# (label, metric name, format). Names follow Nsight Compute 2025.2; check `ncu --query-metrics` before adding one.
KEY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("duration (us)", "gpu__time_duration.sum", "{:.1f}"),
    ("grid size", "launch__grid_size", "{:.0f}"),
    ("block size", "launch__block_size", "{:.0f}"),
    ("waves per SM", "launch__waves_per_multiprocessor", "{:.2f}"),
    ("registers per thread", "launch__registers_per_thread", "{:.0f}"),
    ("shared memory per block (B)", "launch__shared_mem_per_block_dynamic", "{:.0f}"),
    ("SM throughput (% of peak)", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "{:.1f}"),
    ("DRAM throughput (% of peak)", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "{:.1f}"),
    ("achieved occupancy (%)", "sm__warps_active.avg.pct_of_peak_sustained_active", "{:.1f}"),
    ("theoretical occupancy (%)", "sm__maximum_warps_per_active_cycle_pct", "{:.1f}"),
    ("occupancy limit: registers", "launch__occupancy_limit_registers", "{:.0f}"),
    ("occupancy limit: shared memory", "launch__occupancy_limit_shared_mem", "{:.0f}"),
    ("FP64 pipe active (%)", "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "{:.2f}"),
    ("local loads (spill)", "smsp__sass_inst_executed_op_local_ld.sum", "{:.0f}"),
    ("local stores (spill)", "smsp__sass_inst_executed_op_local_st.sum", "{:.0f}"),
)

_NCU_PYTHON_GLOB = r"C:\Program Files\NVIDIA Corporation\Nsight Compute*\extras\python"


def _load_ncu_report():
    """Import `ncu_report` from the newest Nsight Compute install, or from wherever `PYTHONPATH` already finds it."""
    try:
        import ncu_report  # type: ignore[import-not-found]

        return ncu_report
    except ImportError:
        pass
    for folder in sorted(glob.glob(_NCU_PYTHON_GLOB), reverse=True):
        sys.path.insert(0, folder)
        try:
            import ncu_report  # type: ignore[import-not-found]

            return ncu_report
        except ImportError:
            sys.path.pop(0)
    raise SystemExit("ncu_report not found: install Nsight Compute or put its extras/python folder on PYTHONPATH")


def _value(action, name: str, fmt: str) -> str:
    """One metric of one kernel launch, formatted, or the marker for a metric the report does not hold."""
    metric = action.metric_by_name(name)
    if metric is None:
        return "(not collected)"
    value = metric.as_double()
    # gpu__time_duration is reported in ns: show microseconds, the unit every other timing in this repo uses.
    return fmt.format(value / 1000.0 if name == "gpu__time_duration.sum" else value)


def main(paths: list[str]) -> int:
    """Print the key metrics of each kernel launch in each report; return the number of launches printed."""
    ncu_report = _load_ncu_report()
    printed = 0
    for path in paths:
        context = ncu_report.load_report(path)
        print(f"== {os.path.basename(path)}")
        for range_index in range(context.num_ranges()):
            profile_range = context.range_by_idx(range_index)
            for action_index in range(profile_range.num_actions()):
                action = profile_range.action_by_idx(action_index)
                print(f"-- {action.name()}")
                for label, name, fmt in KEY_METRICS:
                    print(f"   {label:34s} {_value(action, name, fmt)}")
                printed += 1
    return printed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
