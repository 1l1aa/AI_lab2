"""Признаки и препроцессинг. Fit только на train (нет утечки)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Weight"
ALL_NUMERIC = ["Length1", "Length2", "Length3", "Height", "Width"]
CATEGORICAL = "Species"


def load_data(path, drop_zero_weight: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if drop_zero_weight:
        df = df[df[TARGET] > 0].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reset_index(drop=True)


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Volume. Идемпотентно: если Volume уже есть, не пересоздаёт."""
    df = df.copy()
    if "Volume" not in df.columns:
        df["Volume"] = df["Length3"] * df["Height"] * df["Width"]
    else:
        df["Volume"] = df["Volume"].fillna(
            df["Length3"] * df["Height"] * df["Width"]
        )
    return df


def make_group_column(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    df = df.copy()
    df["group"] = (
        df[ALL_NUMERIC].round(decimals).astype(str).agg("|".join, axis=1)
    )
    return df


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.25,
    seed: int = 42,
    decimals: int = 2,
):
    df = make_group_column(df, decimals)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["group"]))
    train_val = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val["group"]))
    train = train_val.iloc[train_idx].copy().reset_index(drop=True)
    val = train_val.iloc[val_idx].copy().reset_index(drop=True)
    return train, val, test


def build_preprocessor(
    numeric_features: list[str],
    use_categorical: bool,
    use_scaling: bool,
) -> ColumnTransformer:
    """ColumnTransformer: imputer + (scaler) для чисел, one-hot для категорий."""
    # Species — категориальный, не должен попасть в numeric-ветку
    numeric_only = [c for c in numeric_features if c != CATEGORICAL]

    transformers = []
    if numeric_only:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if use_scaling:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(num_steps)
        transformers.append(("num", num_pipe, numeric_only))

    if use_categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, [CATEGORICAL]))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_xy(
    df: pd.DataFrame,
    numeric_features: list[str],
    use_categorical: bool,
):
    df = add_volume(df)
    cols = list(numeric_features) + ([CATEGORICAL] if use_categorical else [])
    seen = set()
    cols_unique = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            cols_unique.append(c)
    return df[cols_unique], df[TARGET]