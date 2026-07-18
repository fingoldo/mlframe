"""Wave-18 sensors: hash() randomization vs cross-process reproducibility.

Python's builtin ``hash(str)`` / ``hash(tuple_of_strings)`` is salted per
process when ``PYTHONHASHSEED`` is unset (the default). Any cache key /
seed / signature built on it that PROMISES cross-process or cross-run
stability silently breaks: two separate Python processes get different
``hash(target_name)`` values.

Two fix sites this wave:

#1 ``_dummy_baseline_compute._per_target_seed(base, name)`` -- the
   docstring claims "keeps reproducibility across runs (same target ->
   same seed)" but the pre-fix `hash(target_name) & 0xFFFF` form was
   process-randomised. Stochastic baselines downstream
   (``random_quantile``, ``_pick_per_group_categorical``) silently
   produced different predictions and metric values run-to-run, then
   those values were persisted into ``BaselineReport``s.

#2 ``RFECV.fit`` random_state=None branch -- comment promises "derive a
   stable seed from the signature so re-fits on the SAME data are
   deterministic"; pre-fix used `hash(signature)` where signature is a
   tuple of strings (column names + hex digests). Across worker spawns
   the same data produced different ``_seed`` -> different
   stability-selection / fold randomisation -> different ``support_``.

Both fixes route through ``hashlib.blake2b(name.encode(), digest_size=4)``
which is bit-stable across processes.

The sensor pattern is process-isolation: launch a fresh ``subprocess.run``
that imports the helper, computes the seed for a fixed name, prints it;
do this twice and assert equality. The previous (broken) form would
print different values because PYTHONHASHSEED is fresh per subprocess.
"""

from __future__ import annotations

import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys


def _run_once(target_name: str) -> int:
    """Launch a fresh Python subprocess, compute ``_per_target_seed(42, name)``,
    return the integer. Each call uses a DIFFERENT PYTHONHASHSEED (the
    default), so two calls of this with the same name would have returned
    different values under the pre-fix code.

    Resolve the mlframe package via the parent's import so the subprocess
    finds it regardless of how pytest set up sys.path (editable install,
    src-layout, etc.).
    """
    import mlframe as _mlframe
    import pathlib

    # ``mlframe.__file__`` -> ``.../src/mlframe/__init__.py``; the package
    # root (``src/``) is two parents up.
    _src_root = pathlib.Path(_mlframe.__file__).resolve().parent.parent
    code = (
        "import sys\n"
        f"sys.path.insert(0, r'{_src_root}')\n"
        "from mlframe.training.baselines._dummy_baseline_compute import _per_target_seed\n"
        f"print(_per_target_seed(42, {target_name!r}))\n"
    )
    proc = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [sys.executable, "-c", code],
        # Each fresh subprocess cold-imports mlframe (heavy numba-JIT chain via _dummy_baseline_compute);
        # 30s is too tight under suite contention (-n workers + parallel sessions) where the cold import alone
        # can exceed it, while the test runs comfortably (~2min total) in isolation. Generous budget so the
        # sensor measures seed STABILITY, not import wall-clock.
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"subprocess failed: stderr={proc.stderr}"
    return int(proc.stdout.strip())


def test_per_target_seed_stable_across_subprocesses():
    """Two fresh subprocesses computing the seed for the same target name
    MUST agree. Pre-fix used builtin hash() which is per-process salted."""
    seed1 = _run_once("churn")
    seed2 = _run_once("churn")
    assert seed1 == seed2, (
        f"_per_target_seed is NOT stable across processes: {seed1} vs {seed2}. "
        "This is wave-18 P0 regression -- the docstring promises 'same target "
        "-> same seed' but builtin hash() was per-process salted."
    )
    # Different targets in fresh subprocesses still differ.
    seed_a = _run_once("target_alpha")
    seed_b = _run_once("target_beta")
    assert seed_a != seed_b, f"_per_target_seed collides across distinct targets: {seed_a} == {seed_b}. The blake2b digest should still be distinguishable."


def test_per_target_seed_no_hash_call_in_source():
    """Source-level guard: the function MUST NOT use builtin hash() any
    more. Future refactors that re-introduce it silently break the
    docstring promise."""
    import pathlib
    import mlframe as _mlframe

    src = (pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "baselines" / "_dummy_baseline_compute.py").read_text(encoding="utf-8")
    # AST-walk so docstrings don't show up as code. Walk the function body's
    # statements and search for any `Name(id="hash")` Call (the builtin).
    import ast

    tree = ast.parse(src)
    func_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_per_target_seed"),
        None,
    )
    assert func_node is not None, "_per_target_seed missing from module"
    builtin_hash_calls = [n for n in ast.walk(func_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash"]
    assert not builtin_hash_calls, (
        f"Wave 18 P0 regression: _per_target_seed re-introduced builtin "
        f"hash() at line(s) {[c.lineno for c in builtin_hash_calls]}. "
        f"Use hashlib.blake2b(target_name.encode(...)) instead -- the "
        f"builtin is per-process salted under default PYTHONHASHSEED."
    )
    blake2b_calls = [n for n in ast.walk(func_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "blake2b"]
    assert blake2b_calls, "Post-fix marker missing -- _per_target_seed should derive its offset from hashlib.blake2b for cross-process stability."


def test_rfecv_random_state_none_uses_hashlib_not_builtin_hash():
    """Source-level guard for the RFECV._seed derivation. Same rationale:
    builtin hash() of the signature tuple (which contains strings) is
    per-process salted and silently breaks the 'same data -> same
    support_' guarantee across worker spawns."""
    import pathlib
    import mlframe as _mlframe

    # The fit body and its submodule helpers all live under wrappers/rfecv/;
    # concat every submodule so the source-grep sensor catches the pattern
    # regardless of which one owns the relocated code.
    _rfecv = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection" / "wrappers" / "rfecv"
    src = "\n".join(p.read_text(encoding="utf-8") for p in _rfecv.glob("*.py"))
    # Pre-fix shape must be gone:
    assert "_seed = abs(hash(signature)) % (2 ** 32)" not in src, (
        "Wave 18 P1 regression: RFECV _seed derivation re-introduced "
        "abs(hash(signature)). Use hashlib.blake2b for cross-process "
        "stability (the 'same data -> same support_' guarantee in the "
        "in-line comment depends on it)."
    )
    # Post-fix marker:
    assert "_hashlib.blake2b(_sig_bytes, digest_size=4).digest()" in src
