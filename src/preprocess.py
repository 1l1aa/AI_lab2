import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def build_preprocessor(numeric_cols, categorical_cols, use_log_features=True):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_log_features:
        numeric_steps.append(("log", FunctionTransformer(np.log1p, validate=False)))
    numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])