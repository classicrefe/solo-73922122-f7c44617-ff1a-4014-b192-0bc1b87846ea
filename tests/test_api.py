"""API 与 SQLite 版本化存储的端到端测试。"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("SURVEY_DB_PATH", os.path.join(tmp_dir, "test.db"))
    # 延迟导入，使环境变量生效
    from app import main

    monkeypatch.setattr(main, "store", main.SurveyStore(os.environ["SURVEY_DB_PATH"]))
    with TestClient(main.app) as c:
        yield c


def _payload(cap=100):
    return {
        "first_round": [{"tag": f"T{i}"} for i in range(6)],
        "second_round": [
            {"record_id": "R1", "candidate_tags": ["T0"], "can_be_new": False},
            {"record_id": "R2", "candidate_tags": ["T0", "T1", "T2"], "can_be_new": True},
            {"record_id": "R3", "candidate_tags": ["T1"], "can_be_new": False},
            {"record_id": "R4", "candidate_tags": [], "can_be_new": True},
        ],
        "population_cap": cap,
    }


def test_health_and_docs(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_submit_and_estimate_shape(client):
    resp = client.post("/surveys/lake-a/estimates", json=_payload())
    assert resp.status_code == 201
    body = resp.json()
    assert body["version"] == 1 and body["status"] == "ok"
    assert body["recapture_min"] == 2  # R1、R3 必须重捕
    assert body["recapture_max"] == 3  # R2 还可占用 T0/T1 之一，但与 R1/R3 冲突 => 最多 3
    assert body["feasible_recaptures"] == [2, 3]
    assert body["confidence_set_min"] is not None
    assert body["confidence_set_max"] <= 100
    assert body["computation"]["tail_evaluations"] > 0


def test_correction_creates_new_version_and_keeps_history(client):
    client.post("/surveys/lake-b/estimates", json=_payload())
    corrected = _payload()
    corrected["second_round"][1]["candidate_tags"] = ["T3"]  # 标牌辨认更正
    resp = client.post("/surveys/lake-b/estimates", json=corrected)
    assert resp.status_code == 201 and resp.json()["version"] == 2

    current = client.get("/surveys/lake-b/current").json()
    assert current["version"] == 2 and current["is_current"] is True
    assert current["payload"]["second_round"][1]["candidate_tags"] == ["T3"]

    versions = client.get("/surveys/lake-b/versions").json()
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["is_current"] is False
    old = client.get("/surveys/lake-b/versions/1").json()
    assert old["payload"]["second_round"][1]["candidate_tags"] == ["T0", "T1", "T2"]


def test_failed_correction_keeps_original_current(client):
    client.post("/surveys/lake-c/estimates", json=_payload())
    bad = _payload()
    # 两条必须重捕的记录只可能对应同一标牌 => 无可行配对
    bad["second_round"] = [
        {"record_id": "X1", "candidate_tags": ["T0"], "can_be_new": False},
        {"record_id": "X2", "candidate_tags": ["T0"], "can_be_new": False},
    ]
    resp = client.post("/surveys/lake-c/estimates", json=bad)
    assert resp.status_code == 422
    assert "无可行配对" in resp.json()["detail"]["error"]

    # 现行版本仍是 v1，原版结论可查
    current = client.get("/surveys/lake-c/current").json()
    assert current["version"] == 1 and current["status"] == "ok"

    versions = client.get("/surveys/lake-c/versions").json()
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[1]["status"] == "failed"
    assert versions[1]["is_current"] is False
    assert "无可行配对" in versions[1]["error"]
    assert versions[1]["payload"] == bad  # 失败单据也完整留痕


def test_cap_floor_rejection_persists_failed_version(client):
    bad = _payload(cap=7)  # M+C=10，要求 r>=3 但配对约束下可行 r 为 2..3，cap 下限恰好可满足 -> 改用更小
    bad["population_cap"] = 6  # 要求 r>=4 > r_max=3
    resp = client.post("/surveys/lake-d/estimates", json=bad)
    assert resp.status_code == 422
    assert "上限" in resp.json()["detail"]["error"]
    # 首次提交即失败：无现行版本，但失败单据可在版本列表中查到
    assert client.get("/surveys/lake-d/current").status_code == 404
    versions = client.get("/surveys/lake-d/versions").json()
    assert versions[0]["status"] == "failed" and versions[0]["error"]


def test_zero_recapture_flagged_in_response(client):
    payload = {
        "first_round": [{"tag": "T0"}, {"tag": "T1"}],
        "second_round": [
            {"record_id": "R1", "candidate_tags": [], "can_be_new": True},
        ],
        "population_cap": 500,
    }
    body = client.post("/surveys/lake-e/estimates", json=payload).json()
    assert body["recapture_min"] == 0
    assert any("零重捕" in w for w in body["warnings"])


def test_validation_errors(client):
    # 超过 80 条
    too_many = _payload()
    too_many["first_round"] = [{"tag": f"T{i}"} for i in range(81)]
    assert client.post("/surveys/x/estimates", json=too_many).status_code == 422

    # 标牌重复
    dup = _payload()
    dup["first_round"][1]["tag"] = "T0"
    assert client.post("/surveys/x/estimates", json=dup).status_code == 422

    # 上限超 10 亿
    over_cap = _payload()
    over_cap["population_cap"] = 1_000_000_001
    assert client.post("/surveys/x/estimates", json=over_cap).status_code == 422

    # 候选标牌不在首轮清单
    ghost = _payload()
    ghost["second_round"][0]["candidate_tags"] = ["NOPE"]
    assert client.post("/surveys/x/estimates", json=ghost).status_code == 422

    # 未知调查单
    assert client.get("/surveys/unknown/current").status_code == 404
    assert client.get("/surveys/unknown/versions").status_code == 404
