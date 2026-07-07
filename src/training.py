import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
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
from src.preprocessing import load_data, preprocess_data


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("date").copy()
    cutoff_date = df["date"].max() - pd.Timedelta(days=TEST_SIZE_DAYS)

    train_df = df[df["date"] <= cutoff_date]
    test_df = df[df["date"] > cutoff_date]

    return train_df, test_df


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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model():
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    train_df, test_df = chronological_split(df)

    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()