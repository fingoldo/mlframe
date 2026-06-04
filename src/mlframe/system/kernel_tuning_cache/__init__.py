"""Kernel tuning registry CLI for mlframe.

Commands:
  mlframe-tune-kernels list             - List all discovered specs
  mlframe-tune-kernels show <kernel>    - Show spec details
  mlframe-tune-kernels explain <kernel> - Show cached regions + decisions
  mlframe-tune-kernels refresh <kernel> - Tune one spec (cold cache)
  mlframe-tune-kernels refresh-all      - Tune all specs (force=True)
  mlframe-tune-kernels clear <kernel>   - Evict cache for one spec
"""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for mlframe-tune-kernels."""
    import argparse
    import sys

    from pyutilz.system.kernel_tuner import discover_tuners, get_registry

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


def cmd_list(specs: dict) -> int:
    """List all specs."""
    if not specs:
        print("No specs registered.")
        return 0
    print(f"\nDiscovered {len(specs)} tuner specs:\n")
    for (mod_name, kernel_name), spec in sorted(specs.items()):
        label = spec.cli_label or kernel_name
        gpu_tag = "[GPU]" if spec.gpu_capable else "[CPU]"
        print(f"  {gpu_tag} {mod_name}:{label}")
    print()
    return 0


def cmd_show(specs: dict, kernel: str) -> int:
    """Show spec details."""
    key = None
    for (mod_name, kname), spec in specs.items():
        if kname == kernel or (spec.cli_label and spec.cli_label == kernel):
            key = (mod_name, kname)
            break
    if not key:
        print(f"Kernel not found: {kernel}", file=sys.stderr)
        return 1
    mod_name, kname = key
    spec = specs[key]
    print(f"\nSpec: {mod_name}:{kname}\n")
    print(f"  CLI label:    {spec.cli_label or kname}")
    print(f"  GPU capable:  {spec.gpu_capable}")
    print(f"  Axes:         {spec.axes}")
    print(f"  Extra FNs:    {len(spec.extra_fns)} functions")
    print(f"  Salt:         {spec.salt}")
    print(f"  Env key:      {spec.env_key}")
    print(f"  Equiv tol:    {spec.equiv_tol}")
    print()
    return 0


def cmd_explain(specs: dict, kernel: str, dims: str | None) -> int:
    """Show cached regions + decisions (placeholder)."""
    print(f"Explain {kernel}: regions + decisions from cache")
    print("(Not implemented yet.)")
    return 0


def cmd_refresh(specs: dict, kernel: str) -> int:
    """Tune one spec (placeholder)."""
    print(f"Refresh {kernel}: running tuning sweep")
    print("(Not implemented yet.)")
    return 0


def cmd_refresh_all(specs: dict) -> int:
    """Tune all specs (placeholder)."""
    print(f"Refresh all {len(specs)} specs (force=True)")
    print("(Not implemented yet.)")
    return 0


def cmd_clear(specs: dict, kernel: str) -> int:
    """Clear cache for one spec (placeholder)."""
    print(f"Clear {kernel}: evicting cache")
    print("(Not implemented yet.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
