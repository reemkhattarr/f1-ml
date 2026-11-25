"""Strategy endpoints of the local app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tire_strategy.serving.app import app
from tire_strategy.strategy import params as params_module
from tire_strategy.strategy.params import fit_strategy_params, save_params

from .strategy_fixtures import synthetic_race_laps


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "strategy_params.json"
    save_params(fit_strategy_params(synthetic_race_laps()), path)
    monkeypatch.setattr(params_module, "PARAMS_PATH", path)
    return TestClient(app)


def test_plan_endpoint(client) -> None:
    r = client.post("/api/plan", json={"circuit": "Testville"})
    assert r.status_code == 200
    body = r.json()
    assert sum(s["laps"] for s in body["best"]["stints"]) == 50
    assert client.post("/api/plan", json={"circuit": "Nowhere"}).status_code == 422
    assert client.post("/api/plan", json={"circuit": "Nowhere", "laps": 44}).status_code == 200


def test_advise_endpoint(client) -> None:
    r = client.post("/api/advise", json={"circuit": "Testville", "lap": 30, "compound": "MEDIUM",
                                          "tyre_age": 29, "caution": "sc"})  # fmt: skip
    assert r.status_code == 200 and r.json()["decision"] == "box"
    late = {"circuit": "Testville", "lap": 99, "compound": "SOFT", "tyre_age": 1, "total_laps": 50}
    assert client.post("/api/advise", json=late).status_code == 422


def test_local_health(client) -> None:
    assert client.get("/api/health").json()["local"] is True
