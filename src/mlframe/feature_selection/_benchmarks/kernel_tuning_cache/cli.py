"""CLI: ``python -m mlframe.feature_selection._benchmarks.kernel_tuning_cache.cli ...``

Inspect / refresh / clear the per-host kernel-tuning cache. Useful for:

* checking which sweep ran on this machine and when;
* invalidating the cache after a CUDA driver bump if the auto-staleness
  check ``provenance_changed`` missed it;
* forcing a re-sweep on demand without writing a one-off script.

Subcommands::

    show                            dump the live cache as JSON to stdout
    where                           print the on-disk path
    clear                           delete the live host's cache file
    refresh                         tune joint_hist_batched (skip if cached; --force to re-run)
    refresh-mi                      tune plugin_mi_classif_dispatch (skip if cached)
    refresh-polyeval                tune the polyeval sweep (~30s)
    refresh-joint-hist-single-perm  tune joint_hist_single_perm (skip if cached)
    refresh-joint-hist-multi-pair   tune joint_hist_multi_pair (skip if cached)
    refresh-batch-pair-mi           tune batch_pair_mi (skip if cached)
    refresh-cat-fe-perm-kernel      tune cat_fe_perm_kernel (skip if cached)
    refresh-rmse-partial-sum        tune rmse_partial_sum (skip if cached)
    refresh-unary-elementwise       tune unary_elementwise (skip if cached)
    refresh-rff-matmul              tune rff_matmul (skip if cached)
    refresh-knn-hnsw-crossover      tune knn_hnsw_crossover (CPU; skip if cached)
    refresh-discretize-2d-array     tune discretize_2d_array (skip if cached)
    refresh-all                     tune every registered kernel sweep (skip those already cached; --force to re-run)

All refresh-* skip a kernel already validly cached for this host; pass --force to re-benchmark.

The cache lives at ``pyutilz.performance.kernel_tuning.cache.cache_path()``
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
    from pyutilz.performance.kernel_tuning.cache import cache_path
    path = cache_path()
    # Wave 48 (2026-05-20): drop the redundant isfile precheck; just try-open.
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"# no cache at {path}", file=sys.stderr)
        return 1
    json.dump(data, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_where(args) -> int:
    from pyutilz.performance.kernel_tuning.cache import (
        cache_dir, cache_path, hw_fingerprint,
    )
    print(f"hw_fingerprint: {hw_fingerprint()}")
    print(f"cache_dir:      {cache_dir()}")
    print(f"cache_path:     {cache_path()}")
    print(f"exists:         {os.path.isfile(cache_path())}")
    return 0


def _cmd_clear(args) -> int:
    from pyutilz.performance.kernel_tuning.cache import cache_path
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
    # Wave 48 (2026-05-20): tolerate race with concurrent _cmd_clear / external cleanup.
    try:
        os.remove(path)
    except FileNotFoundError:
        print(f"# already removed: {path}", file=sys.stderr)
        return 0
    print(f"deleted {path}")
    return 0


def _refresh_generic(kernel_label: str, ensure_fn, force: bool = False) -> int:
    """Shared wrapper used by every refresh-X subcommand. Returns 0 when
    ``ensure_fn(force=force)`` returns at least one region; 1 otherwise.

    ``force=False`` (the default): the sweep is SKIPPED when a valid result is
    already cached for this host -- ``ensure_fn`` returns the cached regions
    without re-benchmarking. ``force=True`` re-runs the sweep unconditionally.
    The Wave 24 sweeps that genuinely can't run on the live HW (e.g.
    hnswlib not installed) return [] -- that's a successful no-op for
    the API surface but a non-success exit for the operator, so the CLI
    still reports rc=1. ``refresh-all`` folds these into its rollup."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    regions = ensure_fn(force=force)
    if not regions:
        print(
            f"# {kernel_label} auto-tune produced 0 regions (skipped or failed)",
            file=sys.stderr,
        )
        return 1
    print(f"# {kernel_label}: {'re-tuned' if force else 'present (re-used cache or tuned if missing)'}, "
          f"{len(regions)} regions")
    return 0


def _refresh_via_new_registry(kernel_label: str, force: bool = False) -> int:
    """A kernel that has been MIGRATED to ``pyutilz.performance.kernel_tuning``:
    tune it through the new registry (writes the correct backend_choice schema +
    code_version) instead of the superseded legacy sweep, so the two writers do
    not collide on the same cache key (the legacy sweep wrote regions without a
    backend_choice / code_version, which silently shadowed the new dispatcher).

    ``force=False`` (default) passes ``skip_existing=True`` to ``tune_spec`` so a
    kernel whose CURRENT code_version is already cached for this host is NOT
    re-swept; ``force=True`` evicts and re-sweeps unconditionally."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from pyutilz.performance.kernel_tuning import discover_tuners, get_registry, tune_spec

    discover_tuners(package="mlframe")
    spec = get_registry().get(kernel_label)
    if spec is None:
        print(f"# {kernel_label}: not found in the new registry", file=sys.stderr)
        return 1
    n = tune_spec(spec, force=force, skip_existing=True)
    print(f"# {kernel_label}: {'re-tuned' if force else 'ensured (skipped if code_version already cached)'} "
          f"via pyutilz.performance.kernel_tuning, {n} regions")
    return 0


def _cmd_refresh(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_tuning,
    )
    return _refresh_generic("joint_hist_batched", ensure_joint_hist_tuning, force=args.force)


def _cmd_refresh_mi(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_mi_classif_dispatch_tuning,
    )
    return _refresh_generic(
        "plugin_mi_classif_dispatch", ensure_mi_classif_dispatch_tuning, force=args.force,
    )


def _cmd_refresh_polyeval(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_polyeval_tuning,
    )
    return _refresh_generic("polyeval", ensure_polyeval_tuning, force=args.force)


def _cmd_refresh_joint_hist_single_perm(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_single_perm_tuning,
    )
    return _refresh_generic(
        "joint_hist_single_perm", ensure_joint_hist_single_perm_tuning, force=args.force,
    )


def _cmd_refresh_joint_hist_multi_pair(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_multi_pair_tuning,
    )
    return _refresh_generic(
        "joint_hist_multi_pair", ensure_joint_hist_multi_pair_tuning, force=args.force,
    )


def _cmd_refresh_batch_pair_mi(args) -> int:
    # Migrated to pyutilz.performance.kernel_tuning -> tune via the new registry.
    return _refresh_via_new_registry("batch_pair_mi", force=args.force)


def _cmd_refresh_cat_fe_perm_kernel(args) -> int:
    return _refresh_via_new_registry("cat_fe_perm_kernel", force=args.force)


def _cmd_refresh_rmse_partial_sum(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_rmse_partial_sum_tuning,
    )
    return _refresh_generic("rmse_partial_sum", ensure_rmse_partial_sum_tuning, force=args.force)


def _cmd_refresh_unary_elementwise(args) -> int:
    return _refresh_via_new_registry("unary_elementwise", force=args.force)


def _cmd_refresh_rff_matmul(args) -> int:
    return _refresh_via_new_registry("rff_matmul", force=args.force)


def _cmd_refresh_knn_hnsw_crossover(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_knn_hnsw_crossover_tuning,
    )
    return _refresh_generic(
        "knn_hnsw_crossover", ensure_knn_hnsw_crossover_tuning, force=args.force,
    )


def _cmd_refresh_discretize_2d_array(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_discretize_2d_array_tuning,
    )
    return _refresh_generic(
        "discretize_2d_array", ensure_discretize_2d_array_tuning, force=args.force,
    )


def _cmd_refresh_batch_mi_noise_gate(args) -> int:
    # The FE pair-search noise-gate CPU-vs-GPU sweep. Previously absent from the CLI (+ refresh-all),
    # so it only ever tuned ASYNC during the first fit (a multi-minute GPU-thrashing sweep mid-MRMR).
    from mlframe.feature_selection.filters.batch_mi_noise_gate_gpu import (
        ensure_batch_mi_noise_gate_tuning,
    )
    return _refresh_generic(
        "batch_mi_noise_gate", ensure_batch_mi_noise_gate_tuning, force=args.force,
    )


def _cmd_refresh_all(args) -> int:
    """Re-tune every registered kernel sweep. Saves ~10-30s per kernel.

    Each sweep's exit code is folded into a single return value: 0 if all
    succeeded, 1 if any failed (the others still ran + persisted on their
    own merit, so partial success is not a no-op).
    """
    rcs = [
        _cmd_refresh(args),
        _cmd_refresh_mi(args),
        _cmd_refresh_polyeval(args),
        _cmd_refresh_joint_hist_single_perm(args),
        _cmd_refresh_joint_hist_multi_pair(args),
        _cmd_refresh_batch_pair_mi(args),
        _cmd_refresh_cat_fe_perm_kernel(args),
        _cmd_refresh_rmse_partial_sum(args),
        _cmd_refresh_unary_elementwise(args),
        _cmd_refresh_rff_matmul(args),
        _cmd_refresh_knn_hnsw_crossover(args),
        _cmd_refresh_discretize_2d_array(args),
        _cmd_refresh_batch_mi_noise_gate(args),
    ]
    return 0 if all(rc == 0 for rc in rcs) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlframe.kernel_tuning_cache.cli",
        description="Inspect / refresh / clear the per-host kernel-tuning cache.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    # Shared --force flag for every refresh-* command. DEFAULT False: a sweep is skipped when a valid
    # result is already cached for this host (kernels migrated to the new registry skip by code_version;
    # legacy ensure_* skip when regions are present). --force re-benchmarks unconditionally.
    _force = argparse.ArgumentParser(add_help=False)
    _force.add_argument(
        "--force", action="store_true",
        help="re-run sweeps even if a valid result is already cached for this host (default: skip cached)",
    )
    sub.add_parser("show",  help="dump the live cache as JSON to stdout")
    sub.add_parser("where", help="print the on-disk path + hw_fingerprint")
    p_clear = sub.add_parser("clear", help="delete the live host's cache file")
    p_clear.add_argument("--yes", "-y", action="store_true",
                          help="skip the y/N prompt")
    sub.add_parser(
        "refresh", parents=[_force],
        help="tune the joint_hist_batched auto-tune sweep (~30s)",
    )
    sub.add_parser(
        "refresh-mi", parents=[_force],
        help="tune the plugin_mi_classif_dispatch sweep (~30s)",
    )
    sub.add_parser(
        "refresh-polyeval", parents=[_force],
        help="tune the polyeval (Hermite/Legendre/...) sweep (~30s)",
    )
    sub.add_parser(
        "refresh-joint-hist-single-perm", parents=[_force],
        help="tune the joint_hist_single_perm sweep",
    )
    sub.add_parser(
        "refresh-joint-hist-multi-pair", parents=[_force],
        help="tune the joint_hist_multi_pair sweep",
    )
    sub.add_parser(
        "refresh-batch-pair-mi", parents=[_force],
        help="tune the batch_pair_mi backend-choice sweep",
    )
    sub.add_parser(
        "refresh-cat-fe-perm-kernel", parents=[_force],
        help="tune the cat_fe_perm_kernel crossover sweep",
    )
    sub.add_parser(
        "refresh-rmse-partial-sum", parents=[_force],
        help="tune the rmse_partial_sum block_n sweep",
    )
    sub.add_parser(
        "refresh-unary-elementwise", parents=[_force],
        help="tune the unary_elementwise (cupy vs numpy) sweep",
    )
    sub.add_parser(
        "refresh-rff-matmul", parents=[_force],
        help="tune the rff_matmul (cupy vs numpy) sweep",
    )
    sub.add_parser(
        "refresh-knn-hnsw-crossover", parents=[_force],
        help="tune the knn_hnsw_crossover (hnswlib vs sklearn) sweep",
    )
    sub.add_parser(
        "refresh-discretize-2d-array", parents=[_force],
        help="tune the discretize_2d_array crossover sweep",
    )
    sub.add_parser(
        "refresh-batch-mi-noise-gate", parents=[_force],
        help="tune the batch_mi_noise_gate (FE pair-search noise-gate) CPU-vs-GPU sweep",
    )
    sub.add_parser(
        "refresh-all", parents=[_force],
        help="tune every registered kernel sweep (skip those already cached; --force to re-run)",
    )
    args = parser.parse_args(argv)

    return {
        "show": _cmd_show,
        "where": _cmd_where,
        "clear": _cmd_clear,
        "refresh": _cmd_refresh,
        "refresh-mi": _cmd_refresh_mi,
        "refresh-polyeval": _cmd_refresh_polyeval,
        "refresh-joint-hist-single-perm": _cmd_refresh_joint_hist_single_perm,
        "refresh-joint-hist-multi-pair": _cmd_refresh_joint_hist_multi_pair,
        "refresh-batch-pair-mi": _cmd_refresh_batch_pair_mi,
        "refresh-cat-fe-perm-kernel": _cmd_refresh_cat_fe_perm_kernel,
        "refresh-rmse-partial-sum": _cmd_refresh_rmse_partial_sum,
        "refresh-unary-elementwise": _cmd_refresh_unary_elementwise,
        "refresh-rff-matmul": _cmd_refresh_rff_matmul,
        "refresh-knn-hnsw-crossover": _cmd_refresh_knn_hnsw_crossover,
        "refresh-discretize-2d-array": _cmd_refresh_discretize_2d_array,
        "refresh-batch-mi-noise-gate": _cmd_refresh_batch_mi_noise_gate,
        "refresh-all": _cmd_refresh_all,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
