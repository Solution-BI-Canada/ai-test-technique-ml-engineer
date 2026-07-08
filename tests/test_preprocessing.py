import numpy as np
import pandas as pd

from src.preprocessing import clean_structure, apply_imputation, add_features, preprocess_data


# Valeurs d'imputation factices pour les tests -- simulent ce que
# compute_imputation_values() produirait à partir de train_df en production.
IMPUTATION_VALUES = {
    "sales_qty": 20.0,
    "temperature": 10.0,
    "days_of_stock": 3.0,
}


def test_preprocessing_creates_expected_features():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "sku_id": ["SKU_001"],
            "store_id": ["STORE_1"],
            "sales_qty": [25.0],
            "stock_level": [10],
            "promotion_flag": [0],
            "temperature": [5.0],
            "day_of_week": [0],
            "stockout_next_3d": [1],
            "stock_risk_score": [0.99],
        }
    )

    processed = preprocess_data(df, IMPUTATION_VALUES)

    assert "sales_stock_gap" in processed.columns
    assert "days_of_stock" in processed.columns
    assert "demand_exceeds_stock" in processed.columns
    assert "stock_risk_score" not in processed.columns
    assert processed.loc[0, "sales_stock_gap"] == 15.0
    # sales_qty (25) > stock_level (10) -> le flag doit être à 1.
    assert processed.loc[0, "demand_exceeds_stock"] == 1


def test_preprocessing_handles_invalid_temperature_and_missing_values():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "sku_id": ["SKU_001", "SKU_001"],
            "store_id": ["STORE_1", "STORE_1"],
            "sales_qty": [np.nan, 20.0],
            "stock_level": [10, 30],
            "promotion_flag": [0, 1],
            "temperature": [65.0, 5.0],
            "day_of_week": [0, 1],
            "stockout_next_3d": [1, 0],
            "stock_risk_score": [0.99, 0.01],
        }
    )

    processed = preprocess_data(df, IMPUTATION_VALUES)

    assert processed["sales_qty"].isna().sum() == 0
    assert processed["temperature"].isna().sum() == 0
    assert processed["days_of_stock"].replace([np.inf, -np.inf], np.nan).notna().all()

    # La valeur manquante de sales_qty doit avoir été remplacée par la valeur
    # d'imputation fournie, pas recalculée sur ce batch de 2 lignes.
    assert processed.loc[0, "sales_qty"] == IMPUTATION_VALUES["sales_qty"]
    # La température physiquement impossible (65°C) doit avoir été remplacée
    # par la valeur d'imputation fournie, pas laissée telle quelle.
    assert processed.loc[0, "temperature"] == IMPUTATION_VALUES["temperature"]


def test_imputation_uses_provided_values_not_batch_statistics():
    """
    Test critique : reproduit le scénario de bug identifié (moyenne calculée
    sur un batch d'une seule ligne à l'inférence). Vérifie que la valeur
    d'imputation utilisée est bien celle fournie en paramètre, jamais une
    statistique recalculée sur ce batch -- même avec un batch de taille 1.
    """
    single_row = pd.DataFrame(
        {
            "date": ["2024-12-15"],
            "sku_id": ["SKU_001"],
            "store_id": ["STORE_1"],
            "sales_qty": [np.nan],
            "stock_level": [10],
            "promotion_flag": [0],
            "temperature": [999.0],  # hors domaine physique
            "day_of_week": [6],
        }
    )

    df = clean_structure(single_row)
    df = apply_imputation(df, IMPUTATION_VALUES)

    # Sans les valeurs figées, une moyenne calculée sur 1 ligne contenant un NaN
    # resterait NaN et casserait la prédiction -- ce test garantit que ça n'arrive plus.
    assert df.loc[0, "sales_qty"] == IMPUTATION_VALUES["sales_qty"]
    assert df.loc[0, "temperature"] == IMPUTATION_VALUES["temperature"]
    assert not df["sales_qty"].isna().any()
    assert not df["temperature"].isna().any()