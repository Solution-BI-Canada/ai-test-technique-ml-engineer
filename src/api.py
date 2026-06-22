from pathlib import Path
from typing import Any, Final
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TARGET_COLUMN: Final[str] = "stockout_next_3d"
DATE_COLUMN: Final[str] = "date"
MODEL_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
PARAMS_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "data" / "processed" / "preprocessing_params.json"
METADATA_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "models" / "training_metadata.json"

CATEGORICAL_COLUMNS: Final[list[str]] = ["sku_id", "store_id"]


class StockObservation(BaseModel):
    date: str
    sku_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    sales_qty: float = Field(ge=0)
    stock_level: int = Field(ge=0)
    promotion_flag: int = Field(ge=0, le=1)
    temperature: float
    day_of_week: int = Field(ge=0, le=6)


class PredictionRequest(BaseModel):
    instances: list[StockObservation] = Field(min_length=1)


class PredictionItem(BaseModel):
    prediction: int
    probability: float | None


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]


app = FastAPI(title="Stockout Prediction API", version="1.0.0")

model: Any | None = None
preprocessing_params: dict[str, Any] | None = None
feature_columns: list[str] | None = None


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON artifact not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.on_event("startup")
def startup() -> None:
    global model, preprocessing_params, feature_columns

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessing_params = load_json(PARAMS_PATH)

    metadata = load_json(METADATA_PATH)
    feature_columns = metadata.get("feature_columns")

    if not feature_columns:
        raise ValueError("Training metadata missing feature_columns.")


@app.get("/health")
def health() -> dict[str, str]:
    if model is None or preprocessing_params is None or feature_columns is None:
        return {"status": "not_ready"}

    return {"status": "ok"}


def preprocess_for_inference(input_df: pd.DataFrame) -> pd.DataFrame:
    if preprocessing_params is None or feature_columns is None:
        raise RuntimeError("Model artifacts are not loaded.")

    df = input_df.copy()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Invalid date value detected.")

    sku_medians = df["sku_id"].map(preprocessing_params["sku_sales_medians"])

    df["sales_qty"] = df["sales_qty"].fillna(sku_medians)
    df["sales_qty"] = df["sales_qty"].fillna(
        preprocessing_params["global_sales_median"]
    )
    df["temperature"] = df["temperature"].fillna(
        preprocessing_params["temperature_median"]
    )

    df["year"] = df[DATE_COLUMN].dt.year
    df["month"] = df[DATE_COLUMN].dt.month
    df["day"] = df[DATE_COLUMN].dt.day
    df["day_of_year"] = df[DATE_COLUMN].dt.dayofyear
    df = df.drop(columns=[DATE_COLUMN])

    for column in CATEGORICAL_COLUMNS:
        known_categories = preprocessing_params["categories"][column]
        df[column] = df[column].astype(str)

        for category in known_categories:
            encoded_column = f"{column}_{category}"
            df[encoded_column] = (df[column] == category).astype(int)

        df = df.drop(columns=[column])

    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    extra_columns = [column for column in df.columns if column not in feature_columns]
    if extra_columns:
        df = df.drop(columns=extra_columns)

    df = df[feature_columns]

    if df.isna().any().any():
        raise ValueError("Missing values remain after inference preprocessing.")

    non_numeric_columns = [
        column for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(f"Non-numeric features after preprocessing: {non_numeric_columns}")

    return df


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_df = pd.DataFrame(
            [instance.model_dump() for instance in request.instances]
        )

        features = preprocess_for_inference(input_df)

        predictions = model.predict(features)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[:, 1]
        else:
            probabilities = [None] * len(predictions)

        return PredictionResponse(
            predictions=[
                PredictionItem(
                    prediction=int(prediction),
                    probability=None if probability is None else float(probability),
                )
                for prediction, probability in zip(predictions, probabilities)
            ]
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc