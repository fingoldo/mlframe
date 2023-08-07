# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import pandas as pd
from pyutilz.system import tqdmu


def prepare_df_for_catboost(df: object, columns_to_drop: Sequence = [], text_features: Sequence = [], cat_features: list = [], na_filler: str = "") -> None:
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
        if isinstance(df[var].dtype, pd.CategoricalDtype):
            if df[var].isna().any():
                df[var] = df[var].astype(str).fillna(na_filler).astype('category')
            if var not in cat_features:
                logging.info(f"{var} appended to cat_features")
                #df[var] = df[var].astype(str) #(?)
                cat_features.append(var)
        else:
            if var in cat_features:
                if df[var].isna().any():
                    df[var] = df[var].fillna(na_filler)
                df[var] = df[var].astype('category')


def prepare_df_for_xgboost(df: object, cat_features: Sequence = [], ) -> None:
    """
    Xgboost needs categorical be of category dtype.
    """
    cols = set(df.columns)    
    for var in tqdmu(cols, desc="Processing categorical features for XGBoost...", leave=False):
        if isinstance(df[var].dtype, pd.CategoricalDtype):    
            if var not in cat_features:
                logging.info(f"{var} appended to cat_features")
                #df[var] = df[var].astype(str) #(?)
                cat_features.append(var)            
        else:
            if var in cat_features:
                df[var] = df[var].astype('category')