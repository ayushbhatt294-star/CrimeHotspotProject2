import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_status_ok():
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data.get("api") == "running"


def test_register_and_login():
    username = f"testuser_{pytest.main.__name__ if hasattr(pytest, '__name__') else 'x'}"
    password = "testpassword"

    # register
    r = client.post("/api/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200
    data = r.json()
    assert data.get("username") == username

    # login
    r = client.post("/api/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200
    token_data = r.json()
    assert "access_token" in token_data

    access_token = token_data["access_token"]

    # get current user
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert r.status_code == 200
    me = r.json()
    assert me.get("username") == username
