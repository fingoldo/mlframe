"""Every production CUDA kernel source of ``src/mlframe`` as concrete, compilable text, for the offline NVRTC/ptxas checks.

Kernel sources live in Python string constants with template placeholders that each module fills in its own way. The sources are found by AST
(no import side effects), then each placeholder is filled with the value the production call site uses. ``_benchmarks/`` and kernel prototypes are
out of scope by decision (audits/2026-11-09/offline_ptxas_static_checks_plan.md). A source that still carries a placeholder after filling is an
error in this table, never a silent skip: ``leftover_placeholders`` reports it and the test fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from py_ci_shared.cuda_kernel_sources import find_cuda_kernel_sources, module_str_constant

ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_EXCLUDED_PARTS = ("_benchmarks", "kernel_prototypes")
_REAL_TYPES = ("float", "double")
_PLACEHOLDER_RE = re.compile(r"\{[A-Z_0-9]+\}|%[A-Z_0-9]+%|__[A-Z_0-9]+__")


@dataclass(frozen=True)
class KernelVariant:
    """One compilable build of a kernel source: its text, the ``-D`` options production passes, and the label that names it in a failure."""

    key: str
    label: str
    source: str
    options: tuple = ()


def _kernel_constants() -> dict:
    """``{"mlframe/path.py::NAME": source}`` for every assignment whose value holds a ``__global__`` kernel string (see ``py_ci_shared.cuda_kernel_sources``).

    The string may be a plain constant, wrapped in ``render(...)``/``.replace(...)``, concatenated, or an f-string; the first kernel-bearing string literal of the value is taken (its remaining
    placeholders are filled by the recipe table) and further literals of one assignment get ``NAME#2``, ``NAME#3``. Assignments inside functions are included, since several modules build their
    source at call time.
    """
    return find_cuda_kernel_sources(ROOT, exclude=_EXCLUDED_PARTS, key_base=ROOT.parent, min_files=500)


def _str_constant(rel_path: str, name: str) -> str:
    """The module-level string constant ``name`` of ``src/mlframe/<rel_path>``, read by AST (a ``__device__`` helper that production compiles in front of a kernel)."""
    return module_str_constant(ROOT / rel_path, name)


def _fill(src: str, **names: str) -> str:
    """``src`` with ``{REAL}``/``{ACC}`` and every ``{NAME}``/``__NAME__`` given in ``names`` replaced."""
    for k, v in names.items():
        src = src.replace(k, v)
    return src


def _define(**macros: object) -> str:
    """``#define`` lines for macros production injects as a prefix or as ``-D`` options."""
    return "".join(f"#define {k} {v}\n" for k, v in macros.items())


# key suffix -> builder returning ``[(label, source, options)]`` from the extracted source; the values are the ones the call sites use.
def _both_real(src: str, **extra: str) -> list:
    """One build per value type with ``{REAL}`` set and ``{ACC}`` double; ``extra`` fills any further placeholder."""
    return [(real, _fill(src, **{"{REAL}": real, "{ACC}": "double"}, **extra), ()) for real in _REAL_TYPES]


def _qbin_variants() -> dict:
    """The key type, bit reinterpretation and sign mask the quantile-bin coder fills per column type (read from the module so the recipe cannot drift from production)."""
    from mlframe.feature_selection.filters._fe_batched_qbin import _QBIN_VARIANTS

    return _QBIN_VARIANTS


_RECIPES: dict = {
    "_batch_mi_noise_gate_bits.py::_PACK_SRC": lambda s: [
        (f"DT={c}", _fill(s, **{"{DT}": c}), ("-DCAP=32", "-DNYC=4")) for c in ("unsigned char", "unsigned short", "int")
    ],
    "_family_maxt_cuda_bits.py::_SOURCE_V2": lambda s: [(f"{r} NY1=4", _fill(s, **{"{REAL}": r}), ("-DNY1=4",)) for r in _REAL_TYPES],
    "_family_maxt_cuda_bits.py::_SOURCE_V3": lambda s: [
        (f"{r} NY1={ny1} PB={pb} CAP={cap}", _fill(s, **{"{REAL}": r}), (f"-DNY1={ny1}", f"-DPB={pb}", f"-DCAP={cap}"))
        for r in _REAL_TYPES
        for ny1, pb, cap in ((1, 4, 8), (2, 2, 8), (1, 2, 2), (3, 2, 5), (4, 2, 4), (2, 4, 4), (1, 4, 4))
    ],
    "_fe_accuracy_gate_kernel.py::_SRC_TEMPLATE": lambda s: [
        (r, _fill(s, **{"{REAL}": r, "{ACC}": "double", "__MAXK__": "8", "__WARPS__": "8", "__MAXVA__": "8", "__BLOCK__": "256"}), ("-std=c++11",))
        for r in _REAL_TYPES
    ],
    "_fe_batched_mi_mm_values.py::_MI_MM_FROM_VALUES_TEMPLATE": lambda s: [
        (f"{r}/{e}", _fill(s, **{"{REAL}": r, "{ACC}": "double", "{EDGE}": e}), ()) for r in _REAL_TYPES for e in _REAL_TYPES
    ],
    "_gpu_resident_discretize.py::_BIN_CODES_SRC": lambda s: [
        (f"{t}/edge double", _fill(s.replace("bin_codes_TYPENAME", f"bin_codes_{t}").replace("OUTTYPE", "int").replace("TYPE", t), **{"{EDGE}": "double"}), ())
        for t in _REAL_TYPES
    ],
    "_gpu_resident_discretize.py::_BIN_CODES_F32E_SRC": lambda s: [
        (o, s.replace("bin_codes_f32e_OUTNAME", "bin_codes_f32e_x").replace("OUTTYPE", o), ()) for o in ("signed char", "short", "int", "long long")
    ],
    "_fe_batched_qbin.py::_QBIN_CODER_TEMPLATE": lambda s: [
        (f"{r} out={o}", _fill(s, **{"{REAL}": r, "{OUT}": o, **{"{" + k + "}": v for k, v in _qbin_variants()[r].items()}}), ())
        for r in _REAL_TYPES
        for o in ("long long", "unsigned char")
    ],
    "_pair_mi_perm_cuda.py::_SRC": lambda s: [(f"KB={kb}", _fill(s, **{"{ACC}": "double"}), (f"-DKB={kb}",)) for kb in (1, 4, 8)],
    "_fe_batched_mi_cmi.py::_BATCHED_JOINT_ENTROPY_SRC": lambda s: [(xt, _fill(s, **{"{ACC}": "double", "{XT}": xt}), ()) for xt in ("long long", "unsigned char")],
    "_hsic_gpu.py::_SRC": lambda s: [(r, _fill(s, **{"{REAL}": r, "EXPF": "expf" if r == "float" else "exp"}), ()) for r in _REAL_TYPES],
    "_lof_gpu.py::_SRC": lambda s: [
        (f"{r} k={k} p={p}", _fill(s, **{"{REAL}": r, "{K}": str(k), "{P}": str(p)}), ()) for r in _REAL_TYPES for k, p in ((20, 3), (64, 16))
    ],
    "_modular_codes_gpu.py::_SRC": lambda s: [(t, _define(SRC_T=t) + _fill(s, **{"{REAL}": "float", "{ACC}": "double"}), ()) for t in _REAL_TYPES],
    "_family_maxt_counts_device.py::_SRC": lambda s: [("MAXC=64", _fill(s, **{"MAXC": "64"}), ())],
    "_permutation_null_joint_kernel.py::_SRC": lambda s: [
        (f"{x}/{y}", _fill(s, **{"{XT}": x, "{YT}": y}), ()) for x, y in (("short", "signed char"), ("long long", "int"), ("unsigned char", "unsigned char"))
    ],
    "_null_max_perm_gpu.py::_SRC": lambda s: [(x, _fill(s, **{"{XT}": x, "{REAL}": "float", "{ACC}": "double"}), ()) for x in _REAL_TYPES],
    "_offset_product_gpu.py::_SRC": lambda s: [("default", _fill(s, **{"__NSUB__": "2000", "__NF__": "8", "__THREADS__": "256"}), ())],
    "_orthogonal_univariate_fe/_extra_basis_fourier_cuda.py::_SRC": lambda s: [
        (r, _fill(s, **{"{MUL_RN}": "__fmul_rn" if r == "float" else "__dmul_rn", "{REAL}": r, "{ACC}": "double"}), ()) for r in _REAL_TYPES
    ],
    "_orthogonal_univariate_fe/_fourier_batch_gpu.py::_SRC_TEMPLATE": lambda s: _both_real(s.replace("@VARIANCE_FLOOR@", "1e-12")),
    "_orthogonal_univariate_fe/_fourier_fused_gpu.py::_SRC_TEMPLATE": lambda s: _both_real(s.replace("@VARIANCE_FLOOR@", "1e-12")),
    "_orthogonal_xi_gpu.py::_SRC": lambda s: [(r, _fill(s, **{"{REAL}": r, "{ACC}": "double", "{WIDE}": "double"}), ()) for r in _REAL_TYPES],
    "_quantile_codes_gpu.py::_SRC": lambda s: [(r, _fill(s, **{"{REAL}": r, "{TAG}": "f32" if r == "float" else "f64"}), ()) for r in _REAL_TYPES],
    "discretization/_discretization_cuda.py::_SEARCHSORTED_RIGHT_2D_TEMPLATE": lambda s: [
        (r, _fill(s, **{"{REAL}": r, "{ACC}": "double", "{EDGE}": "double"}), ()) for r in _REAL_TYPES
    ],
    "hermite_fe/_als_fused_gpu.py::_SRC": lambda s: [
        (f"{r} D={d}", _fill(s, **{"{REAL}": r, "{ACC}": "double"}), (f"-DD={d}", f"-DNACC={d * (d + 1) // 2 + d}")) for r in _REAL_TYPES for d in (3, 8)
    ],
    "hermite_fe/_als_kernels_gpu.py::_SRC": lambda s: [
        (f"{r} D={d}", _fill(s, **{"{REAL}": r, "{ACC}": "double"}), (f"-DD={d}", f"-DNACC={d * (d + 1) // 2 + d}")) for r in _REAL_TYPES for d in (3, 8)
    ],
    "../shap_proxied_fs/_shap_proxy_treeshap_gpu.py::_KERNEL_SRC": lambda s: [("depth 6", _define(MAXW=8, MAXSP=28) + s, ())],
    "../shap_proxied_fs/_shap_proxy_treeshap_interactions_gpu.py::_KERNEL_SRC": lambda s: [
        (
            "depth 6",
            _define(MAXW=8, MAXLVL=9, MAXP=16) + _str_constant("feature_selection/shap_proxied_fs/_shap_proxy_treeshap_interactions_gpu.py", "_DEVICE_SRC") + s,
            (),
        )
    ],
}


def _modular_rank_reduces(template_name: str) -> list:
    """The reduction fragments a pairwise-modular-rank template is built with in production (versions 1 and 2 of ``_KERNEL_TEMPLATE``); the other templates have no fragment slot."""
    if template_name != "_KERNEL_TEMPLATE":
        return [("", "")]
    rel = "feature_selection/filters/_pairwise_modular_rank_gpu.py"
    return [("reduce_v1", _str_constant(rel, "_REDUCE_V1")), ("reduce_v2", _str_constant(rel, "_REDUCE_V2"))]


for _name, _bits in (
    ("_KERNEL_TEMPLATE", ("", "__launch_bounds__(128, MLF_THREADS_PER_SM / 128)")),
    ("_KERNEL_V4_TEMPLATE", ("", "__launch_bounds__(128, MLF_THREADS_PER_SM / 128)")),
    ("_KERNEL_V7_TEMPLATE", ("__launch_bounds__(128, MLF_THREADS_PER_SM / 128)",)),
):
    _RECIPES[f"_pairwise_modular_rank_gpu.py::{_name}"] = lambda s, _b=_bits, _n=_name: [
        (f"bounds={b or 'none'} {w} {rname}", _fill(s, **{"__BOUNDS__": b, "__COLT__": w, "__REDUCE__": rtext, "{REAL}": "float", "{ACC}": "double"}), ())
        for b in _b
        for w in ("int", "long long")
        for rname, rtext in _modular_rank_reduces(_n)
    ]


def _default_variant(src: str) -> list:
    """Sources with only ``{REAL}``/``{ACC}``/thread-count placeholders: one build per value type."""
    if "{REAL}" in src:
        return _both_real(src, **{"%THREADS%": "256", "%BLOCK%": "256"})
    return [("default", _fill(src, **{"{ACC}": "double", "%THREADS%": "256", "%BLOCK%": "256"}), ())]


def _arch_header() -> str:
    """The shared threads-per-SM macro header, put in front of every kernel as production does for the kernels that use it."""
    return _str_constant("feature_selection/filters/_kernel_arch_macros.py", "ARCH_THREADS_HEADER")


def all_variants() -> list:
    """Every production kernel as ``KernelVariant`` objects, in a stable order."""
    out = []
    for key, src in sorted(_kernel_constants().items()):
        recipe: Callable | None = next((fn for suffix, fn in _RECIPES.items() if key.endswith(suffix) or key.endswith(suffix.lstrip("./"))), None)
        for label, text, options in (recipe(src) if recipe else _default_variant(src)):
            out.append(KernelVariant(key, label, _arch_header() + text, tuple(options)))
    return out


def leftover_placeholders(variant: KernelVariant) -> list:
    """Placeholders still present in the source text; they must be resolved by a recipe, never skipped."""
    return sorted(set(_PLACEHOLDER_RE.findall(variant.source)) - {"__CUDA_ARCH__", "__LINE__", "__FILE__"})
