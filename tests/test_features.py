import pandas as pd
import pytest
import numpy as np
from src.features import load_data, add_volume, split_train_val_test


def test_add_volume():
    df = pd.DataFrame({"Length3": [2.0], "Height": [3.0], "Width": [4.0]})
    out = add_volume(df)
    assert out.loc[0, "Volume"] == pytest.approx(24.0)


def test_drop_zero_weight(tmp_path):
    csv = tmp_path / "fish.csv"
    csv.write_text(
        "Species,Weight,Length1,Length2,Length3,Height,Width\n"
        "Roach,0,19,20.5,22.8,6.47,3.35\n"
        "Roach,150,20,22,24,6.0,3.5\n",
        encoding="utf-8",
    )
    df = load_data(csv, drop_zero_weight=True)
    assert len(df) == 1
    assert df.iloc[0]["Weight"] == 150


def test_split_sizes():
    # Создаём синтетический датафрейм на 158 строк
    n = 158
    df = pd.DataFrame({
        "Species": ["Perch"] * n,
        "Weight": np.linspace(10, 1000, n),
        "Length1": np.linspace(10, 30, n),
        "Length2": np.linspace(11, 31, n),
        "Length3": np.linspace(12, 32, n),
        "Height": np.linspace(3, 10, n),
        "Width": np.linspace(2, 8, n),
    })
    df = add_volume(df)
    train, val, test = split_train_val_test(df, test_size=0.2, val_size=0.25, seed=42)
    assert len(train) == 94
    assert len(val) == 32
    assert len(test) == 32