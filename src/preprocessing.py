import numpy as np
import pandas as pd

from src.config import DROPPED_COLUMNS


def load_data(path: str) -> pd.DataFrame:
    """Load stockout dataset from CSV."""
    return pd.read_csv(path)


def clean_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage structurel : ne dépend d'aucune statistique calculée sur les
    données (mean, median...), donc strictement identique à l'entraînement
    et à l'inférence. Ne fait AUCUNE imputation ici -- l'imputation nécessite
    des valeurs calculées sur train_df, appliquées séparément par apply_imputation().
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Physiquement impossible -> NaN, mais pas encore imputé.
    invalid_temperature = (
        ~df["temperature"].between(-40, 40)
        & df["temperature"].notna()
    )
    df.loc[invalid_temperature, "temperature"] = np.nan

    # Exclusion des colonnes à risque de fuite de données.
    df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])

    return df


def apply_imputation(df: pd.DataFrame, imputation_values: dict) -> pd.DataFrame:
    """
    Applique des valeurs d'imputation déjà calculées à l'entraînement (sur
    train_df uniquement). Ne recalcule jamais rien ici -- c'est ce qui évite
    l'écart train/inférence (ex. moyenne calculée sur une seule ligne reçue
    par l'API, ce qui ne veut rien dire).
    """
    df = df.copy()

    df["sales_qty"] = df["sales_qty"].fillna(imputation_values["sales_qty"])
    df["temperature"] = df["temperature"].fillna(imputation_values["temperature"])

    return df


def add_features(df: pd.DataFrame, imputation_values: dict) -> pd.DataFrame:
    """
    Feature engineering métier. days_of_stock reçoit aussi une valeur
    d'imputation figée (cas sales_qty == 0 -> division impossible).
    """
    df = df.copy()

    # Feature la plus discriminante identifiée en EDA (52.6% vs 0.9% de rupture).
    df["demand_exceeds_stock"] = (df["sales_qty"] > df["stock_level"]).astype(int)

    df["sales_stock_gap"] = df["sales_qty"] - df["stock_level"]

    df["days_of_stock"] = df["stock_level"] / df["sales_qty"].replace(0, np.nan)
    df["days_of_stock"] = df["days_of_stock"].replace([np.inf, -np.inf], np.nan)
    df["days_of_stock"] = df["days_of_stock"].fillna(imputation_values["days_of_stock"])

    return df


def preprocess_data(df: pd.DataFrame, imputation_values: dict) -> pd.DataFrame:
    """
    Pipeline complet. imputation_values est OBLIGATOIRE et vient toujours de
    l'extérieur (calculé une seule fois à l'entraînement sur train_df, puis
    réutilisé tel quel à l'inférence) -- jamais recalculé ici.
    """
    df = clean_structure(df)
    df = apply_imputation(df, imputation_values)
    df = add_features(df, imputation_values)
    return df