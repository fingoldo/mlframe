import pandas as pd
from pyutilz.system import tqdmu


def prepare_df_for_catboost(df: object, columns_to_drop: list = [], text_features: list = [], cat_features: list = [], na_filler: str = "") -> None:
    """
    Catboost needs NAs replaced by a string value.
    Possibly extends cat_features list.
    """
    cols = set(df.columns)

    for var in tqdmu(text_features, desc="Processing textual features for CatBoost...", leave=False):
        if var in cols:
            if var not in columns_to_drop:
                df[var] = df[var].fillna(na_filler)

    for var in tqdmu(cols, desc="Processing categorical features for CatBoost...", leave=False):
        if var in cols:
            if isinstance(df[var].dtype, pd.CategoricalDtype):
                if na_filler not in df[var].cat.categories:
                    df[var].cat.add_categories(na_filler)  # ,inplace=True
                    df[var] = df[var].astype(str)
                df[var] = df[var].fillna(na_filler)
                if var not in cat_features:
                    print(f"{var} appended to cat_features")
                    df[var] = df[var].astype(str)
                    cat_features.append(var)

            else:
                if var in cat_features:
                    df[var] = df[var].fillna(na_filler).astype(str)
