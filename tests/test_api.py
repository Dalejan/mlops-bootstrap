from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_null_prediction():
    response = client.post(
        "/v1/prediction",
        json={
            "opening_gross": 0,
            "screens": 0,
            "production_budget": 0,
            "title_year": 2002,
            "aspect_ratio": 185,
            "duration": 113,
            "budget": 0,
            "imdb_score": 0,
        },
    )
    assert response.status_code == 200
    assert response.json()["worldwide_gross"] == 0


def test_random_prediction():
    response = client.post(
        "v1/prediction",
        json={
            "opening_gross": 2074929.0,
            "screens": 77.0,
            "production_budget": 30000000,
            "title_year": 2002.0,
            "aspect_ratio": 1.85,
            "duration": 113.0,
            "budget": 45000000.0,
            "imdb_score": 7.2,
        },
    )

    assert response.status_code == 200
    assert response.json()["worldwide_gross"] != 0
