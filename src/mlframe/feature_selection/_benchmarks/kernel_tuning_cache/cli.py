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
    refresh                         force-rerun joint_hist_batched auto-tune (~30s)
    refresh-mi                      force-rerun plugin_mi_classif_dispatch (~30s)
    refresh-polyeval                force-rerun the polyeval sweep (~30s)
    refresh-joint-hist-single-perm  force-rerun joint_hist_single_perm
    refresh-joint-hist-multi-pair   force-rerun joint_hist_multi_pair
    refresh-batch-pair-mi           force-rerun batch_pair_mi
    refresh-cat-fe-perm-kernel      force-rerun cat_fe_perm_kernel
    refresh-rmse-partial-sum        force-rerun rmse_partial_sum
    refresh-unary-elementwise       force-rerun unary_elementwise
    refresh-rff-matmul              force-rerun rff_matmul
    refresh-knn-hnsw-crossover      force-rerun knn_hnsw_crossover (CPU)
    refresh-discretize-2d-array     force-rerun discretize_2d_array
    refresh-all                     force-rerun every registered kernel sweep

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


def _refresh_generic(kernel_label: str, ensure_fn) -> int:
    """Shared wrapper used by every refresh-X subcommand. Returns 0 when
    ``ensure_fn(force=True)`` returns at least one region; 1 otherwise.
    The Wave 24 sweeps that genuinely can't run on the live HW (e.g.
    hnswlib not installed) return [] -- that's a successful no-op for
    the API surface but a non-success exit for the operator, so the CLI
    still reports rc=1. ``refresh-all`` folds these into its rollup."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    regions = ensure_fn(force=True)
    if not regions:
        print(
            f"# {kernel_label} auto-tune produced 0 regions (skipped or failed)",
            file=sys.stderr,
        )
        return 1
    print(f"# {kernel_label}: re-tuned, {len(regions)} regions saved")
    return 0


def _refresh_via_new_registry(kernel_label: str) -> int:
    """A kernel that has been MIGRATED to ``pyutilz.performance.kernel_tuning``:
    tune it through the new registry (writes the correct backend_choice schema +
    code_version) instead of the superseded legacy sweep, so the two writers do
    not collide on the same cache key (the legacy sweep wrote regions without a
    backend_choice / code_version, which silently shadowed the new dispatcher)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from pyutilz.performance.kernel_tuning import discover_tuners, get_registry, tune_spec

    discover_tuners(package="mlframe")
    spec = get_registry().get(kernel_label)
    if spec is None:
        print(f"# {kernel_label}: not found in the new registry", file=sys.stderr)
        return 1
    n = tune_spec(spec, force=True)
    print(f"# {kernel_label}: re-tuned via pyutilz.performance.kernel_tuning, {n} regions saved")
    return 0


def _cmd_refresh(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_tuning,
    )
    return _refresh_generic("joint_hist_batched", ensure_joint_hist_tuning)


def _cmd_refresh_mi(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_mi_classif_dispatch_tuning,
    )
    return _refresh_generic(
        "plugin_mi_classif_dispatch", ensure_mi_classif_dispatch_tuning,
    )


def _cmd_refresh_polyeval(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_polyeval_tuning,
    )
    return _refresh_generic("polyeval", ensure_polyeval_tuning)


def _cmd_refresh_joint_hist_single_perm(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_single_perm_tuning,
    )
    return _refresh_generic(
        "joint_hist_single_perm", ensure_joint_hist_single_perm_tuning,
    )


def _cmd_refresh_joint_hist_multi_pair(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_joint_hist_multi_pair_tuning,
    )
    return _refresh_generic(
        "joint_hist_multi_pair", ensure_joint_hist_multi_pair_tuning,
    )


def _cmd_refresh_batch_pair_mi(args) -> int:
    # Migrated to pyutilz.performance.kernel_tuning -> tune via the new registry.
    return _refresh_via_new_registry("batch_pair_mi")


def _cmd_refresh_cat_fe_perm_kernel(args) -> int:
    return _refresh_via_new_registry("cat_fe_perm_kernel")


def _cmd_refresh_rmse_partial_sum(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_rmse_partial_sum_tuning,
    )
    return _refresh_generic("rmse_partial_sum", ensure_rmse_partial_sum_tuning)


def _cmd_refresh_unary_elementwise(args) -> int:
    return _refresh_via_new_registry("unary_elementwise")


def _cmd_refresh_rff_matmul(args) -> int:
    return _refresh_via_new_registry("rff_matmul")


def _cmd_refresh_knn_hnsw_crossover(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_knn_hnsw_crossover_tuning,
    )
    return _refresh_generic(
        "knn_hnsw_crossover", ensure_knn_hnsw_crossover_tuning,
    )


def _cmd_refresh_discretize_2d_array(args) -> int:
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
        ensure_discretize_2d_array_tuning,
    )
    return _refresh_generic(
        "discretize_2d_array", ensure_discretize_2d_array_tuning,
    )


def _cmd_refresh_batch_mi_noise_gate(args) -> int:
    # The FE pair-search noise-gate CPU-vs-GPU sweep. Previously absent from the CLI (+ refresh-all),
    # so it only ever tuned ASYNC during the first fit (a multi-minute GPU-thrashing sweep mid-MRMR).
    from mlframe.feature_selection.filters.batch_mi_noise_gate_gpu import (
        ensure_batch_mi_noise_gate_tuning,
    )
    return _refresh_generic(
        "batch_mi_noise_gate", ensure_batch_mi_noise_gate_tuning,
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
        "refresh-polyeval",
        help="force-rerun the polyeval (Hermite/Legendre/...) sweep (~30s)",
    )
    sub.add_parser(
        "refresh-joint-hist-single-perm",
        help="force-rerun the joint_hist_single_perm sweep",
    )
    sub.add_parser(
        "refresh-joint-hist-multi-pair",
        help="force-rerun the joint_hist_multi_pair sweep",
    )
    sub.add_parser(
        "refresh-batch-pair-mi",
        help="force-rerun the batch_pair_mi backend-choice sweep",
    )
    sub.add_parser(
        "refresh-cat-fe-perm-kernel",
        help="force-rerun the cat_fe_perm_kernel crossover sweep",
    )
    sub.add_parser(
        "refresh-rmse-partial-sum",
        help="force-rerun the rmse_partial_sum block_n sweep",
    )
    sub.add_parser(
        "refresh-unary-elementwise",
        help="force-rerun the unary_elementwise (cupy vs numpy) sweep",
    )
    sub.add_parser(
        "refresh-rff-matmul",
        help="force-rerun the rff_matmul (cupy vs numpy) sweep",
    )
    sub.add_parser(
        "refresh-knn-hnsw-crossover",
        help="force-rerun the knn_hnsw_crossover (hnswlib vs sklearn) sweep",
    )
    sub.add_parser(
        "refresh-discretize-2d-array",
        help="force-rerun the discretize_2d_array crossover sweep",
    )
    sub.add_parser(
        "refresh-batch-mi-noise-gate",
        help="force-rerun the batch_mi_noise_gate (FE pair-search noise-gate) CPU-vs-GPU sweep",
    )
    sub.add_parser(
        "refresh-all",
        help="force-rerun every registered kernel sweep",
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
