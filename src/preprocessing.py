import numpy as np
import pandas as pd

from src.config import DROPPED_COLUMNS


def load_data(path: str) -> pd.DataFrame:
    """Load stockout dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data before feature engineering.

    Decisions:
    - convert date to datetime;
    - replace physically impossible temperatures with NaN;
    - impute missing sales_qty and temperature;
    - remove leakage-prone columns.
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Physically implausible temperatures are treated as missing values.
    invalid_temperature = (
        ~df["temperature"].between(-40, 40)
        & df["temperature"].notna()
    )
    df.loc[invalid_temperature, "temperature"] = np.nan

    # Median imputation is robust and simple to maintain.
    df["sales_qty"] = df["sales_qty"].fillna(df["sales_qty"].mean())
    df["temperature"] = df["temperature"].fillna(df["temperature"].mean())

    # Exclude leakage-prone columns if present.
    df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple business-oriented features."""
    df = df.copy()

    df["sales_stock_gap"] = df["sales_qty"] - df["stock_level"]

    # Avoid division by zero while keeping the feature interpretable.
    df["days_of_stock"] = df["stock_level"] / df["sales_qty"].replace(0, np.nan)
    df["days_of_stock"] = df["days_of_stock"].replace([np.inf, -np.inf], np.nan)
    df["days_of_stock"] = df["days_of_stock"].fillna(df["days_of_stock"].median())

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline before train/test split."""
    df = clean_data(df)
    df = add_features(df)
    return df