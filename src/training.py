from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import warnings
from typing import Any, Final

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_model


TARGET_COLUMN: Final[str] = "stockout_next_3d"
SELECTION_METRIC: Final[str] = "f1_score"


MODEL_CONFIGS: Final[dict[str, Any]] = {
    "logistic_regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "logistic_regression_balanced": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "random_forest_balanced": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def get_project_root() -> Path:
    current_path = Path.cwd().resolve()

    for candidate in [current_path] + list(current_path.parents):
        expected_path = candidate / "data" / "processed" / "train_preprocessed.csv"

        if expected_path.is_file():
            return candidate

    raise FileNotFoundError(
        "Project root could not be detected. "
        "Expected file: data/processed/train_preprocessed.csv"
    )


def load_dataset(file_path: Path, name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"{name} file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"{name} path is not a file: {file_path}")

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{name} CSV file is empty: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"{name} CSV file could not be parsed: {file_path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} CSV encoding could not be decoded: {file_path}") from exc


def validate_ml_dataset(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{name} target column missing: {TARGET_COLUMN}")

    if df.isna().any().any():
        raise ValueError(f"{name} dataset contains missing values.")

    non_numeric_columns = [
        column for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            f"{name} dataset contains non-numeric columns: {non_numeric_columns}"
        )

    target_values = set(df[TARGET_COLUMN].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"{name} invalid target values: {target_values}")

    if len(target_values) < 2:
        raise ValueError(f"{name} target contains only one class: {target_values}")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if TARGET_COLUMN in X.columns:
        raise ValueError("Target column found in features.")

    return X, y


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns do not match.")

    constant_columns = [
        column for column in X_train.columns
        if X_train[column].nunique() <= 1
    ]

    if constant_columns:
        print(
            f"Warning: {len(constant_columns)} constant feature columns detected. "
            "They will remain in the dataset."
        )




def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    print("\n=== 5.3 TRAIN AND EVALUATE MODELS ===")

    trained_models: dict[str, Any] = {}
    evaluation_results: dict[str, Any] = {}

    for model_name, model in MODEL_CONFIGS.items():
        print(f"\nTraining model: {model_name}")

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(X_train, y_train)

        trained_models[model_name] = model
        evaluation_results[model_name] = evaluate_model(model, X_test, y_test)

        print(evaluation_results[model_name])

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name][SELECTION_METRIC],
    )

    print(f"\nBest model by {SELECTION_METRIC}: {best_model_name}")

    return trained_models, evaluation_results, best_model_name


def save_models(
    models: dict[str, Any],
    best_model_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Model file was not created: {model_path}")

    best_model_path = output_dir / "best_model.joblib"
    joblib.dump(models[best_model_name], best_model_path)

    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model file was not created: {best_model_path}")


def save_training_metadata(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    evaluation_results: dict[str, Any],
    best_model_name: str,
    output_path: Path,
    train_path: Path,
    test_path: Path,
) -> None:
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "selection_metric": SELECTION_METRIC,
        "best_model": best_model_name,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_columns": X_train.columns.tolist(),
        "train_target_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_target_distribution": y_test.value_counts().sort_index().to_dict(),
        "model_configs": list(MODEL_CONFIGS.keys()),
        "metrics": evaluation_results,
        "library_versions": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Training metadata saved to: {output_path}")


def main() -> None:
    try:
        project_root = get_project_root()

        train_path = project_root / "data" / "processed" / "train_preprocessed.csv"
        test_path = project_root / "data" / "processed" / "test_preprocessed.csv"

        models_dir = project_root / "models"
        metadata_path = models_dir / "training_metadata.json"

        train_df = load_dataset(train_path, "Train")
        test_df = load_dataset(test_path, "Test")

        validate_ml_dataset(train_df, "Train")
        validate_ml_dataset(test_df, "Test")

        X_train, y_train = split_features_target(train_df)
        X_test, y_test = split_features_target(test_df)

        validate_feature_alignment(X_train, X_test)

        trained_models, evaluation_results, best_model_name = train_and_evaluate_models(
            X_train,
            y_train,
            X_test,
            y_test,
        )

        save_models(trained_models, best_model_name, models_dir)
        save_training_metadata(
            X_train,
            y_train,
            X_test,
            y_test,
            evaluation_results,
            best_model_name,
            metadata_path,
            train_path,
            test_path,
        )

        print("\nTraining status: PASSED")
        print(f"Best model: {best_model_name}")

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        ConvergenceWarning,
        pd.errors.ParserError,
    ) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
