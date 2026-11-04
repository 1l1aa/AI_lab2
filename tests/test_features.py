"""Обязательные тесты (раздел 7 ТЗ ЛР№2). Часть 1: признаки и разбиение."""
import numpy as np
import pandas as pd
from src.features import (
    add_volume, split_train_val_test, build_preprocessor,
)


def test_split_indices_do_not_overlap():
    """Пункт 1 ТЗ: индексы train/val/test не пересекаются (по group)."""
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "Species": ["Perch"] * n,
        "Weight": rng.uniform(10, 1000, n),
        "Length1": rng.uniform(10, 30, n),
        "Length2": rng.uniform(11, 31, n),
        "Length3": rng.uniform(12, 32, n),
        "Height": rng.uniform(3, 10, n),
        "Width": rng.uniform(2, 8, n),
    })
    df = add_volume(df)
    train, val, test = split_train_val_test(df, seed=42)

    g_tr, g_va, g_te = set(train["group"]), set(val["group"]), set(test["group"])
    assert g_tr.isdisjoint(g_va), "train и val содержат одинаковые группы"
    assert g_tr.isdisjoint(g_te), "train и test содержат одинаковые группы"
    assert g_va.isdisjoint(g_te), "val и test содержат одинаковые группы"

    # Количество строк в сумме = исходному датасету
    assert len(train) + len(val) + len(test) == n

def test_preprocessor_accepts_unknown_category():
    """Пункт 2 ТЗ: цепочка обработки принимает строку с неизвестной категорией."""
    df = pd.DataFrame({
        "Species": ["Bream", "Perch", "Roach"],
        "Volume": [100.0, 200.0, 300.0],
        "Length3": [10.0, 20.0, 30.0],
        "Height": [5.0, 6.0, 7.0],
        "Width": [2.0, 3.0, 4.0],
    })
    pre = build_preprocessor(
        ["Volume", "Length3", "Height", "Width", "Species"],
        use_categorical=True,
        use_scaling=True,
    )
    pre.fit(df, [100, 200, 300])

    df_new = df.iloc[[0]].copy()
    df_new["Species"] = "UnknownSpecies"
    out = pre.transform(df_new)
    assert out.shape[0] == 1
    assert np.all(np.isfinite(out))