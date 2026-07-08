import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.preprocessing import clean_structure, apply_imputation, add_features


def load_model():
    return joblib.load(MODEL_PATH)  # dict: pipeline, threshold, imputation_values


def predict_instances(model_bundle: dict, instances: list[dict]) -> list[dict]:
    df = pd.DataFrame(instances)

    imputation_values = model_bundle["imputation_values"]
    df = clean_structure(df)
    df = apply_imputation(df, imputation_values)
    df = add_features(df, imputation_values)

    probabilities = model_bundle["pipeline"].predict_proba(df)[:, 1]
    predictions = (probabilities >= model_bundle["threshold"]).astype(int)

    return [
        {"stockout_prediction": int(pred), "stockout_probability": round(float(prob), 4)}
        for pred, prob in zip(predictions, probabilities)
    ]