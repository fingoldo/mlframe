"""CLI: ``python -m mlframe.feature_selection._benchmarks.kernel_tuning_cache.cli ...``

Inspect / refresh / clear the per-host kernel-tuning cache. Useful for:

* checking which sweep ran on this machine and when;
* invalidating the cache after a CUDA driver bump if the auto-staleness
  check ``provenance_changed`` missed it;
* forcing a re-sweep on demand without writing a one-off script.

Subcommands::

    show           dump the live cache as JSON to stdout
    where          print the on-disk path
    clear          delete the live host's cache file
    refresh        force-rerun the joint_hist auto-tune sweep (~30s)
    refresh-mi     force-rerun the plugin_mi_classif_dispatch sweep (~30s)
    refresh-all    force-rerun every registered kernel sweep

The cache lives at ``pyutilz.system.kernel_tuning_cache.cache_path()``
(``~/.pyutilz/kernel_tuning/{hw_fingerprint}.json`` by default; override
with ``$PYUTILZ_KERNEL_CACHE_DIR``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _cmd_show(args) -> int:
    from pyutilz.system.kernel_tuning_cache import cache_path
    path = cache_path()
    if not os.path.isfile(path):
        print(f"# no cache at {path}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    json.dump(data, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_where(args) -> int:
    from pyutilz.system.kernel_tuning_cache import (
        cache_dir, cache_path, hw_fingerprint,
    )
    print(f"hw_fingerprint: {hw_fingerprint()}")
    print(f"cache_dir:      {cache_dir()}")
    print(f"cache_path:     {cache_path()}")
    print(f"exists:         {os.path.isfile(cache_path())}")
    return 0


def _cmd_clear(args) -> int:
    from pyutilz.system.kernel_tuning_cache import cache_path
    path = cache_path()
    if not os.path.isfile(path):
        print(f"# no cache to clear at {path}", file=sys.stderr)
        return 1
    if not args.yes:
        # E1 fix (Critic 2): refuse to block on non-TTY stdin (piped
        # invocation in a CI). Force the operator to pass --yes for
        # non-interactive runs.
        if not sys.stdin.isatty():
            print(
                "# refusing: non-TTY stdin, pass --yes to confirm",
                file=sys.stderr,
            )
            return 1
        ans = input(f"Delete {path}? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("aborted")
            return 1
    os.remove(path)
    print(f"deleted {path}")
    return 0


def _cmd_refresh(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_tuning,
    )
    regions = ensure_joint_hist_tuning(force=True)
    if not regions:
        print("# joint_hist auto-tune failed; cache unchanged", file=sys.stderr)
        return 1
    print(f"# joint_hist_batched: re-tuned, {len(regions)} regions saved")
    return 0


def _cmd_refresh_mi(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_mi_classif_dispatch_tuning,
    )
    regions = ensure_mi_classif_dispatch_tuning(force=True)
    if not regions:
        print(
            "# plugin_mi_classif_dispatch auto-tune failed; cache unchanged",
            file=sys.stderr,
        )
        return 1
    print(
        f"# plugin_mi_classif_dispatch: re-tuned, {len(regions)} regions saved"
    )
    return 0


def _cmd_refresh_all(args) -> int:
    """Re-tune every registered kernel sweep. Saves ~10-30s per kernel.

    Each sweep's exit code is folded into a single return value: 0 if all
    succeeded, 1 if any failed (the others still ran + persisted on their
    own merit, so partial success is not a no-op).
    """
    rc1 = _cmd_refresh(args)
    rc2 = _cmd_refresh_mi(args)
    return 0 if (rc1 == 0 and rc2 == 0) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlframe.kernel_tuning_cache.cli",
        description="Inspect / refresh / clear the per-host kernel-tuning cache.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show",  help="dump the live cache as JSON to stdout")
    sub.add_parser("where", help="print the on-disk path + hw_fingerprint")
    p_clear = sub.add_parser("clear", help="delete the live host's cache file")
    p_clear.add_argument("--yes", "-y", action="store_true",
                          help="skip the y/N prompt")
    sub.add_parser(
        "refresh",
        help="force-rerun the joint_hist_batched auto-tune sweep (~30s)",
    )
    sub.add_parser(
        "refresh-mi",
        help="force-rerun the plugin_mi_classif_dispatch sweep (~30s)",
    )
    sub.add_parser(
        "refresh-all",
        help="force-rerun every registered kernel sweep",
    )
    args = parser.parse_args(argv)

    return {
        "show": _cmd_show, "where": _cmd_where,
        "clear": _cmd_clear, "refresh": _cmd_refresh,
        "refresh-mi": _cmd_refresh_mi, "refresh-all": _cmd_refresh_all,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
