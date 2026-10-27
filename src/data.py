import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str = "fish_participant.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def remove_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Weight"])
    y = df["Weight"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Удаляем выбросы только из обучающей выборки
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df = remove_outliers(train_df, ["Weight", "Length3", "Height", "Width"])
    X_train = train_df.drop(columns=["Weight"])
    y_train = train_df["Weight"]
    return X_train, X_test, y_train, y_test