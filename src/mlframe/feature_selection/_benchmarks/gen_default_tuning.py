"""Generate the anonymized, repo-committed *default* kernel-tuning JSON.

Why this exists
---------------
``pyutilz``'s kernel-tuning cache (v3) stores each measured tuning as an
**immutable per-(host, kernel, code_version) file** under the user's home dir,
keyed by a *hardware fingerprint*. That cache is per-host and never committed.

A brand-new host (CI runner, a colleague's laptop, a fresh container) therefore
starts cold: every dispatcher falls back to its hand-written heuristic until its
own background sweep finishes. ``register_default_cache(path)`` closes that gap
-- it loads a small, **anonymized** (hw-agnostic) defaults file that the cache
consults on a local MISS, *before* the hand fallback. So a fresh host gets a
near-optimal, measurement-derived dispatch immediately while its own sweep runs.

This module is the *producer* of that defaults file. The runtime *consumer* is
``mlframe.feature_selection.filters._kernel_tuning`` (which calls
``register_default_cache`` once at import).

What it does
------------
1. ``discover_tuners("mlframe")`` -- import every mlframe module so each
   ``kernel_tuner(...)`` registration fires, yielding the registry of
   :class:`~pyutilz.performance.kernel_tuning.registry.TunerSpec`.
2. For every registered kernel whose CURRENT ``code_version`` is not already in
   the committed defaults file (``skip_existing=True``, the default), ensure a
   measured tuning exists for THIS host: read the v3 per-host cache; if it is
   missing (or stale at the live code_version), run the kernel's sweep
   synchronously via ``tune_spec(..., skip_existing=True)``. ``--force``
   re-sweeps every kernel unconditionally.
3. ANONYMIZE + MERGE the measured per-host regions into the defaults document:
   * drop the ``hw_fingerprint`` (the file is hw-agnostic);
   * keep ONLY the current ``code_version`` per kernel (stale ones pruned);
   * for every region, record an abstract ``device`` profile (``"cpu"`` /
     ``"gpu"``) derived from its ``backend_choice``, so a future multi-host
     merge can *average / vote* per region without re-deriving it. With a single
     host's data we keep that host's regions verbatim (local-wins is the runtime
     rule; this file is only the cross-host seed).
4. Write ``default_kernel_tuning.json`` next to this module, fully sorted (keys
   and kernels) for clean, reviewable diffs.

CLI
---
``mlframe-gen-default-tuning`` (console entry) or
``python -m mlframe.feature_selection._benchmarks.gen_default_tuning``.

  --force          re-sweep every kernel even if already tuned / present
  --no-skip-existing   re-derive the defaults entry for kernels already present
  --check          do NOT write; exit non-zero if the file is out of sync
                   (the pre-commit gate; pairs with ``--skip-existing``)
  --output PATH    override the defaults file location (tests)
  --package NAME   discover from a different package (default: ``mlframe``)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# The committed defaults file lives next to this module so it ships inside the
# wheel (``package-data`` picks up the json) and the consumer can resolve it by
# ``importlib.resources`` / a relative path.
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_kernel_tuning.json")

# Schema version of the defaults document. Must match what
# ``register_default_cache`` expects (pyutilz cache SCHEMA_VERSION == 3).
DEFAULTS_SCHEMA_VERSION = 3

# ``backend_choice`` tokens -> abstract device profile. The defaults file records
# this so a multi-host merge can vote per region without re-classifying. GPU
# tokens are the only ones that route to device; everything else (njit / numpy /
# serial / parallel / sklearn / hnsw / cpu / a bare threshold) is host compute.
_GPU_BACKEND_TOKENS = frozenset({"cuda", "cupy", "gpu", "device"})


def classify_device(backend_choice: Optional[str]) -> str:
    """Map a region's ``backend_choice`` token to an abstract ``"cpu"``/``"gpu"``
    device profile. Unknown / missing tokens default to ``"cpu"`` (the safe
    host-compute profile -- a default that wrongly routed to an absent GPU would
    be worse than one that under-uses a present one, and the local sweep corrects
    it anyway)."""
    if not backend_choice:
        return "cpu"
    token = str(backend_choice).strip().lower()
    # A composite token like "cupy" or "njit_parallel"; match on any GPU word.
    if token in _GPU_BACKEND_TOKENS or any(t in _GPU_BACKEND_TOKENS for t in token.replace("-", "_").split("_")):
        return "gpu"
    return "cpu"


def _region_decision(region: dict) -> Optional[str]:
    """Extract the decision string from a region: prefer ``backend_choice``, else
    the first non-constraint scalar value (a bare-threshold kernel)."""
    if "backend_choice" in region:
        return region.get("backend_choice")
    for k, v in region.items():
        if k.endswith(("_max", "_min", "_eq")):
            continue
        if isinstance(v, str):
            return v
    return None


def anonymize_regions(regions: list) -> list:
    """Strip nothing structural (the axis caps drive ``lookup``) but ADD an
    abstract ``device`` profile to each region and DROP per-host noise fields
    (``wall_ms`` -- a measured latency that means nothing on another host). The
    ``max_abs_diff`` equiv-tol field is KEPT (it is a correctness property of the
    kernel, hw-invariant, and the consumer's equiv gate may re-check it).

    Returns NEW region dicts with deterministically-ordered keys; the input is
    not mutated."""
    out = []
    for r in regions:
        new_r = {}
        for k, v in r.items():
            if k == "wall_ms":
                continue  # per-host latency -- meaningless as a cross-host default
            new_r[k] = v
        new_r["device"] = classify_device(_region_decision(r))
        out.append(_sort_region(new_r))
    return out


def _sort_region(region: dict) -> dict:
    """Return the region with keys in a stable, diff-friendly order: declared
    constraint keys first (sorted), then ``backend_choice``, ``device``, then any
    remaining payload (sorted)."""
    constraint = {k: v for k, v in region.items() if k.endswith(("_max", "_min", "_eq"))}
    rest = {k: v for k, v in region.items() if not k.endswith(("_max", "_min", "_eq"))}
    ordered: dict = {}
    for k in sorted(constraint):
        ordered[k] = constraint[k]
    for k in ("backend_choice", "device"):
        if k in rest:
            ordered[k] = rest.pop(k)
    for k in sorted(rest):
        ordered[k] = rest[k]
    return ordered


def build_kernel_entry(spec, regions: list, code_version: Optional[str]) -> dict:
    """Build one kernel's defaults entry: ``{axes, code_version, regions}`` (+
    ``salt`` when set). ``code_version`` is REQUIRED in the entry so the consumer's
    ``_code_version_stale`` check passes (otherwise the default is ignored as
    stale). ``axes`` mirrors the spec's declared axis order."""
    entry: dict = {
        "axes": list(spec.axes.keys()),
        "code_version": code_version,
        "regions": anonymize_regions(regions),
    }
    if spec.salt:
        entry["salt"] = int(spec.salt)
    return entry


def load_existing_defaults(path: str) -> dict:
    """Load the committed defaults document, or an empty skeleton if absent /
    unreadable. Never raises -- a corrupt file just regenerates from scratch."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("kernels"), dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"schema_version": DEFAULTS_SCHEMA_VERSION, "kernels": {}}


def _kernel_already_current(existing: dict, kernel_name: str, code_version: Optional[str]) -> bool:
    """True iff the committed defaults already carry this kernel at the live
    code_version (so ``skip_existing`` may skip both the sweep and the rewrite)."""
    entry = existing.get("kernels", {}).get(kernel_name)
    if not entry:
        return False
    # code_version None on both sides means an un-versioned kernel -> treat
    # presence as current. Otherwise require an exact match.
    return entry.get("code_version") == code_version


def generate_defaults(
    *,
    package: str = "mlframe",
    output_path: str = DEFAULT_OUTPUT_PATH,
    force: bool = False,
    skip_existing: bool = True,
    discover_fn=None,
    tune_fn=None,
    cache_cls=None,
) -> dict:
    """Discover, ensure-tuned, anonymize, and assemble the defaults document
    (does NOT write -- the caller decides). Returns the full document dict.

    ``skip_existing`` (default True): kernels already present at the live
    code_version are carried over untouched (no sweep, no re-derive).
    ``force=True`` re-sweeps every kernel AND re-derives every entry.

    ``discover_fn`` / ``tune_fn`` / ``cache_cls`` are SEAMS for tests: when None
    they resolve to the real pyutilz ``discover_tuners`` / ``tune_spec`` /
    ``KernelTuningCache`` (lazy-imported so ``import gen_default_tuning`` stays
    cheap). A test injects fakes to exercise the assembly + anonymization +
    skip_existing logic with NO real discovery walk, sweep, or per-host cache."""
    if discover_fn is None or tune_fn is None or cache_cls is None:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache as _Cache
        from pyutilz.performance.kernel_tuning.registry import discover_tuners as _disc, tune_spec as _tune
        discover_fn = discover_fn or _disc
        tune_fn = tune_fn or _tune
        cache_cls = cache_cls or _Cache

    specs = discover_fn(package=package)
    if not specs:
        logger.warning("gen_default_tuning: no tuner specs discovered in %r", package)

    existing = load_existing_defaults(output_path)
    cache = cache_cls()
    new_kernels: dict = {}

    # discover_tuners returns names in registration order; sort for a deterministic
    # sweep order + stable logs.
    for kernel_name in sorted(specs):
        spec = specs[kernel_name]
        code_version = spec.code_version()

        # skip_existing: the live code_version is already committed -> carry the
        # existing entry over verbatim (prune happens implicitly: only live
        # kernels are re-emitted; stale code_versions never re-enter).
        if skip_existing and not force and _kernel_already_current(existing, kernel_name, code_version):
            new_kernels[kernel_name] = existing["kernels"][kernel_name]
            logger.info("gen_default_tuning: %s already current (code_version=%s); kept", kernel_name, code_version)
            continue

        # Ensure a measured tuning exists for THIS host at the live code_version.
        # tune_spec(skip_existing=...) is a no-op if already tuned (unless force);
        # otherwise it runs the sweep SYNCHRONOUSLY and persists to the v3 cache.
        try:
            n = tune_fn(spec, force=force, skip_existing=not force)
            logger.info("gen_default_tuning: %s tuned -> %d region(s) on this host", kernel_name, n)
        except Exception as e:
            logger.warning("gen_default_tuning: sweep for %s failed (%s); skipping kernel", kernel_name, e)

        # Read back the measured per-host regions (None if the sweep produced
        # nothing -- e.g. a GPU spec on a CPU-only host where cupy is absent).
        regions = cache.get_regions(kernel_name)
        meta = cache.get_metadata(kernel_name)
        measured_cv = (meta or {}).get("code_version", code_version)

        if not regions:
            # No measured data for this host. If the committed file already had a
            # (possibly older-host) entry, KEEP it rather than drop the kernel --
            # losing a default is strictly worse than carrying a slightly stale
            # one. If there was nothing, the kernel is simply absent (the runtime
            # falls back to the hand heuristic, as before).
            prior = existing.get("kernels", {}).get(kernel_name)
            if prior is not None:
                new_kernels[kernel_name] = prior
                logger.info("gen_default_tuning: %s not measurable here; kept prior entry", kernel_name)
            else:
                logger.info("gen_default_tuning: %s has no regions on this host and no prior entry; omitting", kernel_name)
            continue

        new_kernels[kernel_name] = build_kernel_entry(spec, regions, measured_cv)

    document = {
        "schema_version": DEFAULTS_SCHEMA_VERSION,
        # Provenance ONLY: which package + when. Deliberately NO hw_fingerprint --
        # the file is hw-agnostic. ``hosts`` lists the count of hosts whose data
        # is folded in (1 today; a multi-host merge increments it), so a future
        # voting merge has a place to record provenance.
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "package": package,
        # 0 when nothing was measurable on this host (an honest empty seed); 1 once
        # this host's data is folded in. A future multi-host merge increments it.
        "hosts": 1 if new_kernels else 0,
        "kernels": {k: new_kernels[k] for k in sorted(new_kernels)},
    }
    return document


def write_defaults(document: dict, output_path: str = DEFAULT_OUTPUT_PATH) -> None:
    """Write the defaults document fully sorted (kernels + nested keys) with a
    trailing newline, for clean reviewable diffs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, sort_keys=True)
        f.write("\n")


def _documents_equivalent(a: dict, b: dict) -> bool:
    """Compare two defaults documents IGNORING the volatile ``generated_utc``
    timestamp -- used by ``--check`` so a no-op regeneration doesn't report drift
    purely because the clock advanced."""
    def _strip(d: dict) -> str:
        c = dict(d)
        c.pop("generated_utc", None)
        return json.dumps(c, sort_keys=True)
    return _strip(a) == _strip(b)


def main(argv: Optional[list] = None) -> int:
    """CLI entry point for ``mlframe-gen-default-tuning``."""
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="mlframe-gen-default-tuning",
        description="Generate the anonymized default kernel-tuning JSON for mlframe.",
    )
    parser.add_argument("--package", default="mlframe", help="Package to discover specs from (default: mlframe)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Defaults JSON path (default: alongside this module)")
    parser.add_argument("--force", action="store_true", help="Re-sweep every kernel even if already tuned/present")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-derive defaults for kernels already present (still respects the per-host cache)")
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 if the committed file is out of sync (pre-commit gate)")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO-level logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    document = generate_defaults(
        package=args.package, output_path=args.output,
        force=args.force, skip_existing=args.skip_existing,
    )

    if args.check:
        existing = load_existing_defaults(args.output)
        if _documents_equivalent(existing, document):
            print(f"default_kernel_tuning.json is up to date ({len(document['kernels'])} kernels).")
            return 0
        print(
            "default_kernel_tuning.json is OUT OF SYNC.\n"
            "  Run: mlframe-gen-default-tuning   (then `git add` the file)\n"
            "  Or skip this commit with: git commit --no-verify",
            file=sys.stderr,
        )
        return 1

    write_defaults(document, args.output)
    print(f"Wrote {args.output} ({len(document['kernels'])} kernels).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
