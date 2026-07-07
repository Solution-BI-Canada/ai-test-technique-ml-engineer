from fastapi import FastAPI, HTTPException

from src.inference import load_model, predict_instances
from src.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Stockout Prediction API",
    description="Predict stockout risk within the next 3 days.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        model = None


@app.get("/health")
def health():
    if model is None:
        return {"status": "degraded", "reason": "model_not_loaded"}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    predictions = predict_instances(
        model=model,
        instances=[item.model_dump() for item in request.instances],
    )

    return {"predictions": predictions}