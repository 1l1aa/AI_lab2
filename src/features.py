"""Признаки и препроцессинг. Fit только на train (нет утечки)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Weight"


def load_data(path: str, drop_zero_weight: bool = True) -> pd.DataFrame:
    """Загружает данные и убирает строки с нулевым весом."""
    df = pd.read_csv(path)
    if drop_zero_weight and TARGET in df.columns:
        df = df[df[TARGET] > 0].copy()
    return df.reset_index(drop=True)


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Создает производный признак Vproxy = Length3 * Height * Width."""
    df = df.copy()
    needed = {"Length3", "Height", "Width"}
    if needed.issubset(df.columns) and "Vproxy" not in df.columns:
        df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    return df


def _make_groups(df: pd.DataFrame, decimals: int = 2) -> pd.Series:
    """Создает группы на основе уникальных комбинаций габаритов."""
    cols = ["Length1", "Length2", "Length3", "Height", "Width"]
    cols = [c for c in cols if c in df.columns]
    return (
        df[cols].round(decimals).astype(str).agg("|".join, axis=1)
    )


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.25,
    seed: int = 42,
    decimals: int = 2,
):
    """Групповое разбиение train/val/test с защитой от утечки данных."""
    from sklearn.model_selection import GroupShuffleSplit

    df = df.copy().reset_index(drop=True)
    df["group"] = _make_groups(df, decimals)

    # 1. Отделяем Test (20%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv_idx, te_idx = next(gss1.split(df, groups=df["group"]))
    train_val = df.iloc[tv_idx].copy()
    test = df.iloc[te_idx].copy()

    # 2. Отделяем Val от Train (25% от оставшихся 80% = 20% от всех)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss2.split(train_val, groups=train_val["group"]))
    train = train_val.iloc[tr_idx].copy()
    val = train_val.iloc[va_idx].copy()

    # Защита от утечки
    assert set(train["group"]) & set(val["group"]) == set()
    assert set(train["group"]) & set(test["group"]) == set()
    assert set(val["group"]) & set(test["group"]) == set()
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def split_xy(df: pd.DataFrame, numeric_features, use_cat: bool = False):
    """Разделяет данные на X (признаки) и y (целевую переменную)."""
    feats = list(numeric_features)
    
    # Если попросили категории и колонка Species реально есть в df — добавляем её
    if use_cat and "Species" in df.columns and "Species" not in feats:
        feats.append("Species")
        
    X = df[feats].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor(numeric_features, use_cat: bool, use_scaling: bool):
    """Создает пайплайн предобработки. В текущей версии — только числовые признаки."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaling:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # Мы полностью убрали блок с 'Species', чтобы не было ошибки KeyError.
    transformers = [("num", num_pipe, list(numeric_features))]
    
    return ColumnTransformer(transformers, remainder="drop")