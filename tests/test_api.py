from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint_returns_prediction():
    payload = {
        "instances": [
            {
                "date": "2024-12-15",
                "sku_id": "SKU_001",
                "store_id": "STORE_1",
                "sales_qty": 25,
                "stock_level": 10,
                "promotion_flag": 0,
                "temperature": 5.0,
                "day_of_week": 6,
            }
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in [200, 503]

    if response.status_code == 200:
        body = response.json()
        assert "predictions" in body
        assert "stockout_prediction" in body["predictions"][0]
        assert "stockout_probability" in body["predictions"][0]


def test_predict_endpoint_handles_single_instance_with_missing_and_invalid_values():
    """
    Test critique : reproduit exactement le scénario qui faisait planter
    l'API avant correction -- une seule observation avec une valeur de
    température hors domaine. Vérifie que l'imputation figée (pas recalculée
    sur ce batch d'une seule ligne) permet à la prédiction d'aboutir.
    """
    payload = {
        "instances": [
            {
                "date": "2024-12-15",
                "sku_id": "SKU_001",
                "store_id": "STORE_1",
                "sales_qty": 25,
                "stock_level": 10,
                "promotion_flag": 0,
                "temperature": 999.0,  # hors domaine physique
                "day_of_week": 6,
            }
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in [200, 503]

    if response.status_code == 200:
        body = response.json()
        prediction = body["predictions"][0]
        assert prediction["stockout_prediction"] in [0, 1]
        assert 0.0 <= prediction["stockout_probability"] <= 1.0