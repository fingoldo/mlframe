"""One-off migration: rewrites the word float64 to double-precision inside docstrings of the files listed in the file given as
the first argument, and leaves lines marked as intentional float64 alone.

The rewrite has already been applied to the source tree; keep this script as the record of that migration and review the file list before any rerun.
"""
import ast, sys
from pathlib import Path
M = "Intentional float64"
files = [line.split()[0] for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
tot = 0
for f in files:
    p = Path(f); src = p.read_text(encoding="utf-8"); L = src.split("\n")
    tree = ast.parse(src)
    rng = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            e = n.body[0]
            if isinstance(e, ast.Expr) and isinstance(e.value, ast.Constant) and isinstance(e.value.value, str):
                rng.update(range(e.lineno, e.end_lineno + 1))
    n_changed = 0
    for i in sorted(rng):
        line = L[i - 1]
        if "float64" in line and M not in line and not (i >= 2 and M in L[i - 2]) and not line.strip().startswith("#"):
            L[i - 1] = line.replace("float64", "double-precision")
            n_changed += 1
    if n_changed:
        p.write_text("\n".join(L), encoding="utf-8")
        sys.stdout.write(f"{f} {n_changed}\n")
    tot += n_changed
sys.stdout.write(f"TOTAL {tot}\n")
