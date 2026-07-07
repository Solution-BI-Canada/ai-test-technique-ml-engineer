from typing import List
from pydantic import BaseModel, Field


class StockObservation(BaseModel):
    date: str
    sku_id: str
    store_id: str
    sales_qty: float = Field(..., ge=0)
    stock_level: int = Field(..., ge=0)
    promotion_flag: int = Field(..., ge=0, le=1)
    temperature: float
    day_of_week: int = Field(..., ge=0, le=6)


class PredictionRequest(BaseModel):
    instances: List[StockObservation]


class PredictionResult(BaseModel):
    stockout_prediction: int
    stockout_probability: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]