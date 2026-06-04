"""Task B feasibility proof: the median-gate pseudo-unary fits the existing recipe
machinery EXACTLY like the prewarp pseudo-unary. We show:
  1. EngineeredRecipe.extra round-trips a single float (the train median).
  2. A gate_med replay (x>median) is bit-identical fit-vs-transform given the same x.
  3. The leak-safety property: median from TRAIN, applied to TEST, never touches y.

This does NOT edit production; it demonstrates the minimal spec needed so the
recommendation to the pipeline owner is concrete.
"""
from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe


def gate_med_apply(x, median):
    return (np.asarray(x, dtype=np.float64) > float(median)).astype(np.float64)


def main():
    rng = np.random.default_rng(0)
    a_tr = rng.normal(3.0, 1.0, size=2000)   # shifted operand, median ~3
    b_tr = rng.normal(0.0, 1.0, size=2000)
    a_te = rng.normal(3.0, 1.0, size=800)
    b_te = rng.normal(0.0, 1.0, size=800)

    # FIT: median from TRAIN only
    med_a = float(np.median(a_tr))

    # Build a recipe carrying ONLY the new state: a single float in extra.
    # (binary_name 'mul' already exists; gate_med is a pseudo-unary on side a.)
    rec = EngineeredRecipe(
        name="mul(gate_med(a),b)",
        kind="unary_binary",
        src_names=("a", "b"),
        unary_names=("gate_med", "identity"),
        binary_name="mul",
        unary_preset="minimal",
        binary_preset="minimal",
        quantization=None,
        extra={"gate_med_a_median": med_a},
    )

    # 1. round-trip the float through extra (and a pickle to confirm serialisability)
    import pickle
    rec2 = pickle.loads(pickle.dumps(rec))
    assert rec2.extra["gate_med_a_median"] == med_a, "median did not round-trip"
    print(f"[1] extra float round-trip OK: median={rec2.extra['gate_med_a_median']:.6f}")

    # 2. bit-identical replay fit-vs-transform on the SAME rows
    z_fit = gate_med_apply(a_tr, rec.extra["gate_med_a_median"]) * b_tr
    z_fit2 = gate_med_apply(a_tr, rec2.extra["gate_med_a_median"]) * b_tr
    assert np.array_equal(z_fit, z_fit2), "replay not bit-identical"
    print(f"[2] replay bit-identical on train rows: {np.array_equal(z_fit, z_fit2)}")

    # 3. leak-safety: transform uses the STORED train median, never recomputes on test
    z_te = gate_med_apply(a_te, rec.extra["gate_med_a_median"]) * b_te
    # If we (wrongly) recomputed median on test, it would differ slightly:
    med_te_wrong = float(np.median(a_te))
    z_te_leaky = gate_med_apply(a_te, med_te_wrong) * b_te
    n_diff = int(np.sum(gate_med_apply(a_te, med_a) != gate_med_apply(a_te, med_te_wrong)))
    print(f"[3] leak-safe vs leaky-recompute differ on {n_diff}/{len(a_te)} test rows "
          f"(train med={med_a:.4f} vs test med={med_te_wrong:.4f}); "
          f"stored-median replay is the leak-safe path.")
    print("\nFEASIBILITY: gate_med needs ONE float in recipe.extra, identical to the "
          "prewarp template. Integration touches _feature_engineering_pairs.py (register "
          "pseudo-unary + per-pair median fit) and engineered_recipes.py "
          "(build_unary_binary_recipe extra + _apply_side replay) -- both shared files.")


if __name__ == "__main__":
    main()
