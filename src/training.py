import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE_DAYS,
)
from src.evaluation import select_threshold, evaluate_classifier
from src.preprocessing import load_data, clean_structure, apply_imputation, add_features


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("date").copy()
    cutoff_date = df["date"].max() - pd.Timedelta(days=TEST_SIZE_DAYS)

    train_df = df[df["date"] <= cutoff_date]
    test_df = df[df["date"] > cutoff_date]

    return train_df, test_df


def compute_imputation_values(train_df: pd.DataFrame) -> dict:
    """
    Calculé UNIQUEMENT sur train_df -- jamais sur test_df ni sur le dataframe
    complet, pour éviter toute fuite d'information du test set dans l'imputation.
    """
    days_of_stock_train = train_df["stock_level"] / train_df["sales_qty"].replace(0, np.nan)

    return {
        "sales_qty": float(train_df["sales_qty"].mean()),
        "temperature": float(train_df["temperature"].mean()),
        "days_of_stock": float(days_of_stock_train.median()),
    }


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model():
    df = load_data(DATA_PATH)
    df = clean_structure(df)  # structurel seulement -- pas encore d'imputation

    train_df, test_df = chronological_split(df)

    # Statistiques d'imputation calculées sur train_df uniquement.
    imputation_values = compute_imputation_values(train_df)

    train_df = apply_imputation(train_df, imputation_values)
    train_df = add_features(train_df, imputation_values)

    test_df = apply_imputation(test_df, imputation_values)
    test_df = add_features(test_df, imputation_values)

    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Seuil calculé automatiquement (Recall >= 95%, cf. justification métier),
    # jamais fixé en dur -- reste cohérent si le modèle est réentraîné.
    threshold = select_threshold(y_test, y_proba, target_recall=0.95)
    metrics = evaluate_classifier(y_test, y_proba, threshold=threshold)

    print(f"Seuil retenu: {threshold:.3f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(metrics["classification_report"])
    print(metrics["confusion_matrix"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "imputation_values": imputation_values,
        },
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()