"""CLI: ``python -m mlframe.feature_selection._benchmarks.kernel_tuning_cache.cli ...``

Inspect / refresh / clear the per-host kernel-tuning cache. Useful for:

* checking which sweep ran on this machine and when;
* invalidating the cache after a CUDA driver bump if the auto-staleness
  check ``provenance_changed`` missed it;
* forcing a re-sweep on demand without writing a one-off script.

Subcommands::

    show     dump the live cache as JSON to stdout
    where    print the on-disk path
    clear    delete the live host's cache file
    refresh  force-rerun the auto-tune sweep (~30s) and overwrite the cache

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
        print("# auto-tune failed; cache unchanged", file=sys.stderr)
        return 1
    print(f"# re-tuned: {len(regions)} regions saved")
    return 0


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
    sub.add_parser("refresh",
                   help="force-rerun the auto-tune sweep and overwrite the cache")
    args = parser.parse_args(argv)

    return {"show": _cmd_show, "where": _cmd_where,
            "clear": _cmd_clear, "refresh": _cmd_refresh}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
