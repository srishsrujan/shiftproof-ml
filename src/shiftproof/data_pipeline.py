from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, TIME_COL


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if TIME_COL not in data:
        raise ValueError(f"Missing required column: {TIME_COL}")

    data[TIME_COL] = pd.to_datetime(data[TIME_COL], errors="coerce")
    if data[TIME_COL].isna().any():
        raise ValueError("created_at contains unparseable timestamps")

    for col in NUMERIC_FEATURES:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in data:
            data[col] = data[col].astype("string").str.strip().str.lower()
            data[col] = data[col].astype(object).where(pd.notna(data[col]), np.nan)

    if "priority" in data:
        priority_map = {
            "urgent": "critical",
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "med": "medium",
            "low": "low",
        }
        data["priority"] = data["priority"].map(priority_map).fillna(data["priority"])

    # Required feature columns are allowed to be missing at row level; the preprocessor will impute them.
    for col in FEATURES:
        if col not in data:
            data[col] = np.nan

    return data


def time_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15) -> DatasetSplits:
    data = clean_input(df).sort_values(TIME_COL).reset_index(drop=True)
    if TARGET not in data:
        raise ValueError(f"Missing target column: {TARGET}")

    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end < 10 or val_end <= train_end or val_end >= n:
        raise ValueError("Dataset is too small for requested time split")

    return DatasetSplits(
        train=data.iloc[:train_end].copy(),
        validation=data.iloc[train_end:val_end].copy(),
        test=data.iloc[val_end:].copy(),
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    data = clean_input(df)
    if TARGET not in data:
        raise ValueError("Target column not present")
    x = data[FEATURES].copy()
    y = data[TARGET].astype(int).to_numpy()
    return x, y
