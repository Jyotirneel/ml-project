import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os

# Mock Predictor before importing app
with patch("src.inference.Predictor") as MockPredictor:
    from src.app import app

client = TestClient(app)

def test_predict_allowed_extension():
    with patch("src.app.predictor.predict") as mock_predict:
        mock_predict.return_value = {"breed": "Jersey", "confidence": 0.9, "info": {}}

        # Small valid image
        content = b"fake image content"
        files = {"file": ("test.jpg", content, "image/jpeg")}
        response = client.post("/predict", files=files)

        assert response.status_code == 200
        assert response.json()["breed"] == "Jersey"

def test_predict_disallowed_extension():
    content = b"fake script content"
    files = {"file": ("test.exe", content, "application/x-msdownload")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "File extension not allowed" in response.json()["detail"]

def test_predict_file_too_large():
    # 6MB file (larger than 5MB limit)
    content = b"a" * (6 * 1024 * 1024)
    files = {"file": ("test.png", content, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]

def test_predict_no_extension():
    content = b"some content"
    files = {"file": ("testfile", content, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "File extension not allowed" in response.json()["detail"]
