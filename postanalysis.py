import sys


def analyze_xgboost_model(model: object, print_chunk: int = 0) -> dict:
    # Get the raw text representation of all trees
    trees = model.get_booster().get_dump()

    nleaves = [tree.count("leaf=") for tree in trees]
    if print_chunk:
        print(trees[0][:print_chunk])
    return dict(
        total_trees=len(trees),
        total_leaves=sum(nleaves),
        max_tree_leaves=max(nleaves),
        model_desc_size_gb=sum([sys.getsizeof(tree) for tree in trees]) / (1024**3),
    )
