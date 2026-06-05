"""Kernel tuning registry CLI for mlframe.

Commands:
  mlframe-tune-kernels list             - List all discovered specs
  mlframe-tune-kernels show <kernel>    - Show spec details
  mlframe-tune-kernels explain <kernel> - Show cached regions + decisions
  mlframe-tune-kernels refresh <kernel> - Tune one spec (cold cache)
  mlframe-tune-kernels refresh-all      - Tune all specs (force=True)
  mlframe-tune-kernels clear <kernel>   - Evict cache for one spec
"""

import argparse
import json
import sys

from pyutilz.system.kernel_tuner import discover_tuners, get_registry, retune_all, tune_spec
from pyutilz.system.kernel_tuning_cache import KernelTuningCache

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for mlframe-tune-kernels."""
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="mlframe-tune-kernels",
        description="Kernel tuning registry CLI for mlframe.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # list: show all specs
    subparsers.add_parser("list", help="List all discovered tuner specs")

    # show: show one spec
    sp_show = subparsers.add_parser("show", help="Show spec details")
    sp_show.add_argument("kernel", help="Kernel name (e.g. joint_hist_2d)")

    # explain: show cached regions
    sp_explain = subparsers.add_parser(
        "explain", help="Show cached regions and decisions"
    )
    sp_explain.add_argument("kernel", help="Kernel name")
    sp_explain.add_argument(
        "--dims", help="Limit output to specific dims (e.g. n=1000)"
    )

    # refresh: tune one spec
    sp_refresh = subparsers.add_parser("refresh", help="Tune one spec")
    sp_refresh.add_argument("kernel", help="Kernel name")

    # refresh-all: tune all specs
    subparsers.add_parser("refresh-all", help="Tune all specs (force=True)")

    # clear: evict cache
    sp_clear = subparsers.add_parser("clear", help="Clear cache for one spec")
    sp_clear.add_argument("kernel", help="Kernel name")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Discover specs.
    specs = discover_tuners(package="mlframe")
    if not specs:
        print("No specs discovered in mlframe.", file=sys.stderr)
        return 1

    # Dispatch to command handlers.
    if args.command == "list":
        return cmd_list(specs)
    elif args.command == "show":
        return cmd_show(specs, args.kernel)
    elif args.command == "explain":
        return cmd_explain(specs, args.kernel, args.dims if hasattr(args, "dims") else None)
    elif args.command == "refresh":
        return cmd_refresh(specs, args.kernel)
    elif args.command == "refresh-all":
        return cmd_refresh_all(specs)
    elif args.command == "clear":
        return cmd_clear(specs, args.kernel)
    else:
        parser.print_help()
        return 1


def _find_spec(specs: dict, kernel: str):
    """Resolve a kernel_name (or cli_label) to its spec; None if absent."""
    if kernel in specs:
        return specs[kernel]
    return next((s for s in specs.values() if s.cli_label == kernel), None)


def _parse_dims(dims: str | None) -> dict:
    """Parse 'a=1,b=2' -> {'a': 1, 'b': 2} (ints where possible, else str)."""
    out = {}
    for pair in (dims or "").split(","):
        if "=" not in pair:
            continue
        k, v = (p.strip() for p in pair.split("=", 1))
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def cmd_list(specs: dict) -> int:
    """List all registered tuner specs."""
    if not specs:
        print("No specs registered.")
        return 0
    print()
    print(f"Discovered {len(specs)} tuner specs:")
    print()
    for kernel_name, spec in sorted(specs.items()):
        gpu_tag = "[GPU]" if spec.gpu_capable else "[CPU]"
        print(f"  {gpu_tag} {kernel_name}")
    print()
    return 0


def cmd_show(specs: dict, kernel: str) -> int:
    """Show one spec's details."""
    spec = _find_spec(specs, kernel)
    if spec is None:
        print(f"Kernel not found: {kernel}", file=sys.stderr)
        return 1
    print()
    print(f"Spec: {spec.kernel_name}")
    print()
    print(f"  CLI label:    {spec.cli_label or spec.kernel_name}")
    print(f"  GPU capable:  {spec.gpu_capable}")
    print(f"  Axes:         {spec.axes}")
    print(f"  Extra FNs:    {len(spec.extra_fns)} functions")
    print(f"  Salt:         {spec.salt}")
    print(f"  Env key:      {spec.env_key}")
    print(f"  Equiv tol:    {spec.equiv_tol}")
    print()
    return 0


def cmd_explain(specs: dict, kernel: str, dims: str | None) -> int:
    """Show the cache's lookup decision for a kernel at the given dims."""
    spec = _find_spec(specs, kernel)
    name = spec.kernel_name if spec is not None else kernel
    print(json.dumps(KernelTuningCache().lookup_explain(name, **_parse_dims(dims)), indent=2, default=str))
    return 0


def cmd_refresh(specs: dict, kernel: str) -> int:
    """Force a fresh sweep for one spec and persist the regions."""
    spec = _find_spec(specs, kernel)
    if spec is None:
        print(f"Kernel not found: {kernel}", file=sys.stderr)
        return 1
    print(f"Refreshing {spec.kernel_name} (force=True)...")
    n = tune_spec(spec, force=True)
    print(f"  {spec.kernel_name}: {n} region(s) persisted")
    return 0


def cmd_refresh_all(specs: dict) -> int:
    """Tune all discovered specs (force=True) across unique GPU models."""
    print(f"Refreshing all {len(specs)} spec(s) (force=True)...")
    for kernel_name, n in sorted(retune_all(package="mlframe", force=True).items()):
        print(f"  {kernel_name}: {n} region(s)")
    return 0


def cmd_clear(specs: dict, kernel: str) -> int:
    """Evict the cache entry for one spec."""
    spec = _find_spec(specs, kernel)
    name = spec.kernel_name if spec is not None else kernel
    removed = KernelTuningCache().evict(name)
    print(f"Cleared {name}: {'evicted' if removed else 'no entry'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
