import joblib
import pandas as pd

from src.config import MODEL_PATH, DECISION_THRESHOLD
from src.preprocessing import preprocess_data


def load_model():
    return joblib.load(MODEL_PATH)


def predict_instances(model, instances: list[dict]) -> list[dict]:
    df = pd.DataFrame(instances)
    df = preprocess_data(df)

    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    return [
        {
            "stockout_prediction": int(pred),
            "stockout_probability": round(float(prob), 4),
        }
        for pred, prob in zip(predictions, probabilities)
    ]