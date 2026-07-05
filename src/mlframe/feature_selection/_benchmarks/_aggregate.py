"""Diff two benchmark JSON captures (before/after).

Output: markdown table to stdout + structured diff to ``--out diff.json``.
Exit code 1 if median speedup < 1.0 on any scenario (regression). Warns when
captures span mismatched library versions per ``MANIFEST.json`` rotation policy.

Usage
-----
    python -m mlframe.feature_selection._benchmarks._aggregate \\
        --before _results/pre-refactor_<sha>.json \\
        --after  _results/final_<sha>.json \\
        [--out  _results/diff.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD = 1.0  # speedup factor; <1 = regression
MAX_VERSION_AGE_DAYS = 90


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_versions(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Return list of warning messages for mismatched versions / stale captures."""
    warnings: list[str] = []
    manifest_path = Path(__file__).parent / "_results" / "MANIFEST.json"
    if not manifest_path.exists():
        warnings.append("MANIFEST.json missing; cannot validate version rotation policy")
        return warnings
    manifest = _load(manifest_path)
    captures = {c["tag"] + "_" + c["git_sha"]: c for c in manifest.get("captures", [])}
    before_key = f"{before['tag']}_{before['git_sha']}"
    after_key = f"{after['tag']}_{after['git_sha']}"
    bef = captures.get(before_key)
    aft = captures.get(after_key)
    if bef and aft:
        for lib in ("numpy", "numba", "cupy", "sklearn"):
            if bef["versions"].get(lib) != aft["versions"].get(lib):
                warnings.append(f"version mismatch on '{lib}': before={bef['versions'].get(lib)} " f"after={aft['versions'].get(lib)}")
        if bef.get("cpu_model") != aft.get("cpu_model"):
            warnings.append(f"CPU model differs: {bef['cpu_model']} vs {aft['cpu_model']}")
    return warnings


def _diff_one(b: dict[str, Any], a: dict[str, Any]) -> dict[str, Any]:
    if b.get("skipped") or a.get("skipped"):
        return {"skipped": b.get("skipped") or a.get("skipped")}
    bw = b["wall_time"]["median"]
    aw = a["wall_time"]["median"]
    speedup = bw / aw if aw > 0 else float("inf")
    bm = b["mem_peak_mb"]["median"]
    am = a["mem_peak_mb"]["median"]
    mem_ratio = am / bm if bm > 0 else float("inf")
    stages_diff: dict[str, dict[str, float]] = {}
    for sname in set(b.get("stages", {})) & set(a.get("stages", {})):
        bs = b["stages"][sname]["wall_s"]
        as_ = a["stages"][sname]["wall_s"]
        stages_diff[sname] = {
            "before_s": bs,
            "after_s": as_,
            "speedup": (bs / as_) if as_ > 0 else float("inf"),
        }
    return {
        "wall_speedup": speedup,
        "before_wall_s": bw,
        "after_wall_s": aw,
        "mem_ratio": mem_ratio,
        "before_mem_mb": bm,
        "after_mem_mb": am,
        "stages": stages_diff,
        "support_changed": (b.get("support_set") != a.get("support_set")),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True, type=Path)
    p.add_argument("--after", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--regression-threshold", type=float, default=REGRESSION_THRESHOLD)
    args = p.parse_args()

    before = _load(args.before)
    after = _load(args.after)

    warnings = _check_versions(before, after)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    diff: dict[str, dict[str, Any]] = {}
    for name in sorted(set(before["results"]) | set(after["results"])):
        b = before["results"].get(name, {})
        a = after["results"].get(name, {})
        if not b or not a:
            diff[name] = {"missing_in": "before" if not b else "after"}
            continue
        diff[name] = _diff_one(b, a)

    # Markdown summary table.
    print()
    print(f"# Benchmark diff: {before['tag']} -> {after['tag']}")
    print()
    print("| Scenario | before (s) | after (s) | speedup x | mem ratio | support changed? |")
    print("|---|---:|---:|---:|---:|:---:|")
    has_regression = False
    for name, d in diff.items():
        if "skipped" in d or "missing_in" in d:
            print(f"| {name} | -- | -- | -- | -- | (skipped/missing) |")
            continue
        sp_marker = "[!]" if d["wall_speedup"] < args.regression_threshold else " "
        if d["wall_speedup"] < args.regression_threshold:
            has_regression = True
        sup_marker = "yes" if d["support_changed"] else "no"
        print(f"| {name} | {d['before_wall_s']:.3f} | {d['after_wall_s']:.3f} | " f"{sp_marker}{d['wall_speedup']:.2f} | {d['mem_ratio']:.2f} | {sup_marker} |")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "before_tag": before["tag"],
                    "after_tag": after["tag"],
                    "before_sha": before["git_sha"],
                    "after_sha": after["git_sha"],
                    "warnings": warnings,
                    "scenarios": diff,
                    "has_regression": has_regression,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    if has_regression:
        print(f"\nFAIL: regression detected (median speedup < {args.regression_threshold} on at least one scenario)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
