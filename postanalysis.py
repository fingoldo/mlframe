import sys


def analyze_xgboost_model(model: object, print_chunk: int = 0) -> dict:
    """Summarize an XGBoost booster (n trees, n leaves, max leaves per tree, dump size).

    Leaf counting uses ``trees_to_dataframe()`` when available (authoritative — returns one
    row per node with Feature=='Leaf' on leaf nodes). Falls back to a regex that matches
    ``\\bleaf=`` only at word boundaries so arbitrary occurrences of the substring "leaf="
    inside feature names or monotone constraints don't inflate the count.
    """
    import re as _re

    booster = model.get_booster()
    trees = booster.get_dump()

    try:
        df = booster.trees_to_dataframe()
        leaves_per_tree = df[df["Feature"] == "Leaf"].groupby("Tree").size()
        # Ensure trees with zero leaves (impossible for well-formed XGBoost trees) do not
        # crash max() on an empty series; fall back to 0 for safety.
        nleaves = [int(leaves_per_tree.get(i, 0)) for i in range(len(trees))]
    except Exception:
        leaf_pattern = _re.compile(r"\bleaf=")
        nleaves = [len(leaf_pattern.findall(tree)) for tree in trees]

    if print_chunk:
        print(trees[0][:print_chunk])
    return dict(
        total_trees=len(trees),
        total_leaves=sum(nleaves),
        max_tree_leaves=max(nleaves) if nleaves else 0,
        model_desc_size_gb=sum([sys.getsizeof(tree) for tree in trees]) / (1024**3),
    )
