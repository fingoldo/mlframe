"""Tier 3.3: cat-FE tutorial script.

End-to-end walk-through of cat-FE on a small categorical dataset
(sklearn ``load_iris`` discretized + a synthetic XOR-style hidden
synergy column). Demonstrates:

1. Construction: MRMR with default cat-FE enabled.
2. Fit on train.
3. Inspect ``_cat_fe_state_.recipes`` and ``.diagnostics``.
4. Apply to disjoint test data via ``transform()``.
5. Optional: target encoding emit, k-way greedy, bootstrap CIs.

Run:

    D:/ProgramData/anaconda3/python.exe \\
        mlframe/feature_selection/_benchmarks/cat_fe_tutorial.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR, CatFEConfig


def make_synthetic_dataset(n=2000, n_train=1500, seed=42):
    """Tabular dataset: 2 synergy cols, 2 marginal-signal cols, 4 noise
    cols. Target ``y`` is a binary XOR of the first two columns."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, n).astype(np.int8)
    x2 = rng.integers(0, 2, n).astype(np.int8)
    # Marginal-signal cols: y = m1 with 70% probability
    m1 = rng.integers(0, 2, n).astype(np.int8)
    m2 = rng.integers(0, 2, n).astype(np.int8)
    # Noise cols
    noise = rng.integers(0, 4, size=(n, 4)).astype(np.int8)
    y = (x1 ^ x2).astype(np.int8)
    cols = {
        "synergy_a": pd.Categorical(x1),
        "synergy_b": pd.Categorical(x2),
        "marginal_x": pd.Categorical(m1),
        "marginal_y": pd.Categorical(m2),
    }
    for k in range(4):
        cols[f"noise_{k}"] = pd.Categorical(noise[:, k])
    df = pd.DataFrame(cols)
    y_s = pd.Series(y, name="target")
    return df.iloc[:n_train], y_s.iloc[:n_train], df.iloc[n_train:], y_s.iloc[n_train:]


def section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> None:
    df_train, y_train, df_test, y_test = make_synthetic_dataset()

    section("1. Dataset overview")
    print(f"  train: {df_train.shape}; test: {df_test.shape}")
    print(f"  columns: {list(df_train.columns)}")
    print(f"  target balance (train): {y_train.value_counts().to_dict()}")
    print("  Construction: y = synergy_a XOR synergy_b; rest are noise / marginal")

    section("2. Default cat-FE (enabled by default since 2026-05-11)")
    mrmr_default = MRMR(
        full_npermutations=2, baseline_npermutations=2,
        verbose=0, n_jobs=1,
        # cat_fe_config not passed -> uses default CatFEConfig() which is enabled
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr_default.fit(df_train, y_train)
    state = mrmr_default._cat_fe_state_
    print(f"  cat_fe_state.recipes: {len(state.recipes) if state else 0}")
    if state and state.recipes:
        for r in state.recipes:
            d = state.diagnostics.get(r.name, {})
            print(
                f"    {r.name}: II={d.get('II', float('nan')):.4f}, "
                f"src={r.src_names}"
            )

    section("3. Apply to disjoint test data")
    out_test = mrmr_default.transform(df_test)
    print(f"  transform output shape: {out_test.shape}")
    if hasattr(out_test, "columns"):
        engineered_cols = [c for c in out_test.columns if c.startswith("kway(")]
        print(f"  engineered cols: {engineered_cols}")
        for c in engineered_cols[:3]:
            print(f"    '{c}' unique values: {len(set(out_test[c]))}")

    section("4. Custom config: target encoding emit + bootstrap CIs")
    cfg_custom = CatFEConfig(
        enable=True,
        top_k_pairs=8,
        min_interaction_information=0.05,
        full_npermutations=20,
        fwer_correction="none",
        # Tier 3.1: emit float target-encoded cols
        emit_target_encoding=True,
        target_encoding_oof_folds=5,
        # Tier 2.2: bootstrap CIs
        bootstrap_ci_n_replicates=10,
    )
    mrmr_custom = MRMR(
        full_npermutations=2, baseline_npermutations=2,
        verbose=0, n_jobs=1, cat_fe_config=cfg_custom,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr_custom.fit(df_train, y_train)
    state_c = mrmr_custom._cat_fe_state_
    if state_c:
        factorize_recipes = [r for r in state_c.recipes if r.kind == "factorize"]
        te_recipes = [r for r in state_c.recipes if r.kind == "target_encoding"]
        print(f"  factorize recipes: {len(factorize_recipes)}")
        print(f"  target_encoding recipes: {len(te_recipes)}")
        for r in factorize_recipes[:2]:
            d = state_c.diagnostics.get(r.name, {})
            ci = d.get("bootstrap_ii_ci")
            conf = d.get("joint_dependence_confidence")
            print(
                f"    {r.name}: II={d.get('II', float('nan')):.4f} "
                f"CI={ci} conf={conf}"
            )

    section("5. Restore legacy MRMR (cat-FE disabled)")
    mrmr_legacy = MRMR(
        full_npermutations=2, baseline_npermutations=2,
        verbose=0, n_jobs=1,
        cat_fe_config=CatFEConfig(enable=False),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr_legacy.fit(df_train, y_train)
    print(f"  legacy support_: {list(mrmr_legacy.support_)} "
          f"(no engineered recipes)")
    print(f"  legacy _engineered_recipes_: {mrmr_legacy._engineered_recipes_}")

    section("Done.")


if __name__ == "__main__":
    main()
