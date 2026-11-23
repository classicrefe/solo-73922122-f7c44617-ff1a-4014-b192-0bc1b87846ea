"""补录复核 API 与最低代价建议的测试。"""

from __future__ import annotations

import itertools
import os
import random
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.estimation import EstimationError, estimate_population
from app.review import Clue, find_min_cost_suggestion


@pytest.fixture()
def client(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("SURVEY_DB_PATH", os.path.join(tmp_dir, "test.db"))
    # 延迟导入，使环境变量生效
    from app import main

    monkeypatch.setattr(main, "store", main.SurveyStore(os.environ["SURVEY_DB_PATH"]))
    with TestClient(main.app) as c:
        yield c


def _conflict_payload():
    """X1、X2 两条必重捕记录都只认出 T0；X3、X4 分别锁定 T1、T3。"""
    return {
        "first_round": [{"tag": f"T{i}"} for i in range(6)],
        "second_round": [
            {"record_id": "X1", "candidate_tags": ["T0"], "can_be_new": False},
            {"record_id": "X2", "candidate_tags": ["T0"], "can_be_new": False},
            {"record_id": "X3", "candidate_tags": ["T1"], "can_be_new": False},
            {"record_id": "X4", "candidate_tags": ["T3"], "can_be_new": False},
            {"record_id": "R5", "candidate_tags": [], "can_be_new": True},
        ],
        "population_cap": 5000,
    }


def _ok_payload():
    return {
        "first_round": [{"tag": f"T{i}"} for i in range(6)],
        "second_round": [
            {"record_id": "R1", "candidate_tags": ["T0"], "can_be_new": False},
            {"record_id": "R2", "candidate_tags": ["T1", "T2"], "can_be_new": True},
        ],
        "population_cap": 100,
    }


def _clue(record_id, tag, cost, status):
    return {"record_id": record_id, "tag": tag, "cost": cost, "status": status}


def _submit_failed(client, survey, payload):
    resp = client.post(f"/surveys/{survey}/estimates", json=payload)
    assert resp.status_code == 422
    return resp


# --------------------------------------------------------------------------- #
# 主流程：找线索 -> 建议 -> 核实 -> 发布
# --------------------------------------------------------------------------- #


def test_full_review_journey(client):
    # 两条必重捕记录都只认出同一个标牌 => 整单被拒，v1 失败，无现行版
    _submit_failed(client, "lake-r", _conflict_payload())

    # 已核实 X1<->T3 仍不够（与 X4 争 T3）；待核三条；排除一条更便宜的干扰线索
    review_req = {
        "failed_version": 1,
        "expected_current_version": None,  # 尚无现行版
        "clues": [
            _clue("X1", "T3", 4, "verified"),
            _clue("X4", "T4", 2, "pending"),
            _clue("X2", "T5", 3, "pending"),
            _clue("X1", "T4", 8, "pending"),
            _clue("X2", "T4", 1, "excluded"),  # 若被误用，总代价会变成 1
        ],
    }
    resp = client.post("/surveys/lake-r/reviews", json=review_req)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "suggestion"
    assert body["total_cost"] == 2  # 最低代价：只核实 X4<->T4
    assert [(c["record_id"], c["tag"]) for c in body["recommended_clues"]] == [("X4", "T4")]
    pairing = {(p["record_id"], p["tag"]): p["via_recommended_clue"] for p in body["pairing"]}
    assert pairing == {
        ("X1", "T3"): False,  # 来自已核实线索
        ("X2", "T0"): False,
        ("X3", "T1"): False,
        ("X4", "T4"): True,  # 来自建议核实的线索
    }

    # 建议不写库：版本仍只有 v1
    assert [v["version"] for v in client.get("/surveys/lake-r/versions").json()] == [1]

    # 核实建议线索后再次复核 => 发布新现行版
    review_req["clues"][1]["status"] = "verified"  # X4<->T4 已核实
    review_req["clues"][2]["status"] = "excluded"
    review_req["clues"][3]["status"] = "excluded"
    resp = client.post("/surveys/lake-r/reviews", json=review_req)
    assert resp.status_code == 201
    body = resp.json()
    assert body["outcome"] == "published" and body["version"] == 2
    assert body["estimate"]["recapture_min"] == body["estimate"]["recapture_max"] == 4
    assert {c["tag"] for c in body["verified_clues_applied"]} == {"T3", "T4"}

    # 新现行版 v2：已核实线索已补入原清单；失败版 v1 仍可查
    current = client.get("/surveys/lake-r/current").json()
    assert current["version"] == 2 and current["status"] == "ok"
    assert current["payload"]["second_round"][0]["candidate_tags"] == ["T0", "T3"]
    assert current["payload"]["second_round"][3]["candidate_tags"] == ["T3", "T4"]
    assert current["result"]["review"]["failed_version"] == 1
    versions = client.get("/surveys/lake-r/versions").json()
    assert [(v["version"], v["status"]) for v in versions] == [(1, "failed"), (2, "ok")]
    old = client.get("/surveys/lake-r/versions/1").json()
    assert old["status"] == "failed" and "无可行配对" in old["error"]


def test_verified_clues_alone_publish(client):
    _submit_failed(client, "lake-v", _conflict_payload())
    # 已核实两条线索即可直接补足证据（X1->T3、X4->T4）
    req = {
        "failed_version": 1,
        "expected_current_version": None,
        "clues": [
            _clue("X1", "T3", 4, "verified"),
            _clue("X4", "T4", 2, "verified"),
            _clue("X2", "T5", 1, "pending"),  # 已无需核实
        ],
    }
    resp = client.post("/surveys/lake-v/reviews", json=req)
    assert resp.status_code == 201
    body = resp.json()
    assert body["outcome"] == "published" and body["version"] == 2
    assert body["estimate"]["status"] == "ok"
    assert client.get("/surveys/lake-v/current").json()["version"] == 2


# --------------------------------------------------------------------------- #
# 乐观并发：计算期间同单可继续更正，发布时核对现行版号
# --------------------------------------------------------------------------- #


def test_publish_conflict_when_corrected_meanwhile(client):
    client.post("/surveys/lake-c1/estimates", json=_ok_payload())  # v1 ok
    _submit_failed(client, "lake-c1", _conflict_payload())  # v2 failed
    # 复核计算期间，同单被整单更正 => v3 成为现行版
    client.post("/surveys/lake-c1/estimates", json=_ok_payload())

    req = {
        "failed_version": 2,
        "expected_current_version": 1,  # 期望已过时
        "clues": [_clue("X1", "T3", 4, "verified"), _clue("X4", "T4", 2, "verified")],
    }
    resp = client.post("/surveys/lake-c1/reviews", json=req)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["expected_current_version"] == 1 and detail["actual_current_version"] == 3
    # 冲突不留新版本
    assert [v["version"] for v in client.get("/surveys/lake-c1/versions").json()] == [1, 2, 3]
    assert client.get("/surveys/lake-c1/current").json()["version"] == 3


def test_expected_current_version_must_be_null_when_no_current(client):
    _submit_failed(client, "lake-c2", _conflict_payload())  # v1 failed，无现行版
    req = {
        "failed_version": 1,
        "expected_current_version": 1,  # 尚无现行版时只能填 null
        "clues": [_clue("X1", "T3", 4, "verified"), _clue("X4", "T4", 2, "verified")],
    }
    resp = client.post("/surveys/lake-c2/reviews", json=req)
    assert resp.status_code == 409
    assert resp.json()["detail"]["actual_current_version"] is None
    assert client.get("/surveys/lake-c2/versions").json() == [
        client.get("/surveys/lake-c2/versions/1").json()
    ]


def test_expected_null_rejected_once_current_exists(client):
    _submit_failed(client, "lake-c3", _conflict_payload())
    req = {
        "failed_version": 1,
        "expected_current_version": None,
        "clues": [_clue("X1", "T3", 4, "verified"), _clue("X4", "T4", 2, "verified")],
    }
    assert client.post("/surveys/lake-c3/reviews", json=req).status_code == 201  # v2 现行
    # 现行版已存在后再填 null => 409，且不留新版本
    resp = client.post("/surveys/lake-c3/reviews", json=req)
    assert resp.status_code == 409
    assert resp.json()["detail"]["actual_current_version"] == 2
    assert [v["version"] for v in client.get("/surveys/lake-c3/versions").json()] == [1, 2]


def test_publish_with_matching_expected_version(client):
    client.post("/surveys/lake-c4/estimates", json=_ok_payload())  # v1 ok
    _submit_failed(client, "lake-c4", _conflict_payload())  # v2 failed
    req = {
        "failed_version": 2,
        "expected_current_version": 1,
        "clues": [_clue("X1", "T3", 4, "verified"), _clue("X4", "T4", 2, "verified")],
    }
    resp = client.post("/surveys/lake-c4/reviews", json=req)
    assert resp.status_code == 201 and resp.json()["version"] == 3
    assert client.get("/surveys/lake-c4/current").json()["version"] == 3


# --------------------------------------------------------------------------- #
# 无解与提示（均不写库）
# --------------------------------------------------------------------------- #


def test_infeasible_when_required_unmatchable(client):
    payload = {
        "first_round": [{"tag": "T0"}, {"tag": "T1"}],
        "second_round": [
            {"record_id": "X1", "candidate_tags": ["T0"], "can_be_new": False},
            {"record_id": "X2", "candidate_tags": ["T0"], "can_be_new": False},
            {"record_id": "X3", "candidate_tags": ["T0"], "can_be_new": False},
        ],
        "population_cap": 100,
    }
    _submit_failed(client, "lake-i1", payload)
    req = {
        "failed_version": 1,
        "expected_current_version": None,
        "clues": [_clue("X1", "T1", 1, "pending")],  # 全核实也只能配 2/3 条
    }
    body = client.post("/surveys/lake-i1/reviews", json=req).json()
    assert body["outcome"] == "infeasible"
    assert "必重捕" in body["reason"] and "3" in body["reason"] and "2" in body["reason"]
    assert [v["version"] for v in client.get("/surveys/lake-i1/versions").json()] == [1]


def test_infeasible_without_pending_clues(client):
    _submit_failed(client, "lake-i2", _conflict_payload())
    req = {
        "failed_version": 1,
        "expected_current_version": None,
        "clues": [_clue("X1", "T3", 4, "verified"), _clue("X2", "T4", 1, "excluded")],
    }
    body = client.post("/surveys/lake-i2/reviews", json=req).json()
    assert body["outcome"] == "infeasible"
    assert "未提供待核线索" in body["reason"]
    assert [v["version"] for v in client.get("/surveys/lake-i2/versions").json()] == [1]


def test_cap_floor_suggestion_and_no_solution(client):
    # M=2, C=3，全部可能是新个体且无候选；cap=4 要求 r>=1，但无候选可配 => 失败
    payload = {
        "first_round": [{"tag": "T0"}, {"tag": "T1"}],
        "second_round": [
            {"record_id": "R1", "candidate_tags": [], "can_be_new": True},
            {"record_id": "R2", "candidate_tags": [], "can_be_new": True},
            {"record_id": "R3", "candidate_tags": [], "can_be_new": True},
        ],
        "population_cap": 4,
    }
    _submit_failed(client, "lake-i3", payload)
    req = {
        "failed_version": 1,
        "expected_current_version": None,
        "clues": [_clue("R1", "T0", 5, "pending"), _clue("R2", "T1", 2, "pending")],
    }
    body = client.post("/surveys/lake-i3/reviews", json=req).json()
    assert body["outcome"] == "suggestion"
    assert body["total_cost"] == 2  # 只需核实最便宜的 R2<->T1
    assert [(c["record_id"], c["tag"]) for c in body["recommended_clues"]] == [("R2", "T1")]
    assert body["pairing"] == [{"record_id": "R2", "tag": "T1", "via_recommended_clue": True}]

    # cap=3 要求 r>=2，只有一条线索可用 => 无解并说明上限原因
    payload["population_cap"] = 3
    _submit_failed(client, "lake-i4", payload)
    req["clues"] = [_clue("R1", "T0", 5, "pending")]
    body = client.post("/surveys/lake-i4/reviews", json=req).json()
    assert body["outcome"] == "infeasible"
    assert "上限" in body["reason"]
    assert [v["version"] for v in client.get("/surveys/lake-i4/versions").json()] == [1]


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #


def test_clue_validation(client):
    _submit_failed(client, "lake-val", _conflict_payload())
    base = {"failed_version": 1, "expected_current_version": None}

    def post(clues):
        return client.post("/surveys/lake-val/reviews", json={**base, "clues": clues})

    # 标牌已在原候选清单
    assert post([_clue("X1", "T0", 1, "pending")]).status_code == 422
    # 标牌不在首轮清单
    assert post([_clue("X1", "T99", 1, "pending")]).status_code == 422
    # 记录号不存在
    assert post([_clue("X9", "T2", 1, "pending")]).status_code == 422
    # 重复线索
    assert post([_clue("X1", "T3", 1, "pending"), _clue("X1", "T3", 2, "verified")]).status_code == 422
    # 代价非正整数
    assert post([_clue("X1", "T3", 0, "pending")]).status_code == 422
    # 非法状态
    assert post([_clue("X1", "T3", 1, "maybe")]).status_code == 422
    # 全部 422 均不写库
    assert [v["version"] for v in client.get("/surveys/lake-val/versions").json()] == [1]


def test_review_target_must_be_failed_version(client):
    client.post("/surveys/lake-val2/estimates", json=_ok_payload())  # v1 ok
    req = {
        "failed_version": 1,
        "expected_current_version": 1,
        "clues": [_clue("R1", "T3", 1, "pending")],
    }
    resp = client.post("/surveys/lake-val2/reviews", json=req)
    assert resp.status_code == 422 and "失败版" in resp.json()["detail"]["error"]
    # 版本不存在 / 调查单不存在 => 404
    assert client.post("/surveys/lake-val2/reviews", json={**req, "failed_version": 9}).status_code == 404
    assert client.post("/surveys/unknown/reviews", json=req).status_code == 404


def test_suggestion_scales_to_80_records(client):
    # 80 条必重捕记录共享同一标牌；线索给出各自专属标牌
    payload = {
        "first_round": [{"tag": f"T{i}"} for i in range(80)],
        "second_round": [
            {"record_id": f"R{j}", "candidate_tags": ["T0"], "can_be_new": False}
            for j in range(80)
        ],
        "population_cap": 1_000_000_000,
    }
    _submit_failed(client, "lake-80", payload)
    req = {
        "failed_version": 1,
        "expected_current_version": None,
        "clues": [_clue(f"R{j}", f"T{j}", j, "pending") for j in range(1, 80)],
    }
    body = client.post("/surveys/lake-80/reviews", json=req).json()
    assert body["outcome"] == "suggestion"
    assert body["total_cost"] == sum(range(1, 80))  # 79 条线索缺一不可
    assert len(body["pairing"]) == 80
    assert len({p["tag"] for p in body["pairing"]}) == 80  # 一牌一记录


# --------------------------------------------------------------------------- #
# 引擎级：最小费用流 vs 暴力枚举
# --------------------------------------------------------------------------- #


def test_min_cost_suggestion_matches_brute_force():
    rng = random.Random(20261123)
    checked = 0
    while checked < 60:
        m_count = rng.randint(1, 5)
        c_count = rng.randint(1, 5)
        tags = [f"T{i}" for i in range(m_count)]
        records = [f"R{j}" for j in range(c_count)]
        candidates = [[t for t in tags if rng.random() < 0.4] for _ in range(c_count)]
        flags = [rng.random() < 0.5 for _ in range(c_count)]
        cap = rng.randint(m_count, m_count + c_count + 3)
        pending = [
            Clue(records[j], t, rng.randint(1, 9))
            for j in range(c_count)
            for t in tags
            if t not in candidates[j] and rng.random() < 0.3
        ]
        if not pending or len(pending) > 10:
            continue
        try:
            estimate_population(tags, candidates, flags, cap)
            continue  # 基线可行，跳过
        except EstimationError:
            pass
        full = [list(c) for c in candidates]
        for clue in pending:
            full[records.index(clue.record_id)].append(clue.tag)
        try:
            estimate_population(tags, full, flags, cap)
        except EstimationError:
            continue  # 全部核实仍无解（NoSolution 路径由 API 测试覆盖）

        # 暴力枚举全部子集求最低总代价
        best = None
        for r in range(len(pending) + 1):
            for subset in itertools.combinations(range(len(pending)), r):
                merged = [list(c) for c in candidates]
                for i in subset:
                    merged[records.index(pending[i].record_id)].append(pending[i].tag)
                try:
                    estimate_population(tags, merged, flags, cap)
                except EstimationError:
                    continue
                cost = sum(pending[i].cost for i in subset)
                if best is None or cost < best:
                    best = cost
        assert best is not None

        sug = find_min_cost_suggestion(tags, records, candidates, flags, cap, pending)
        assert sug.total_cost == best
        assert sum(c.cost for c in sug.recommended) == sug.total_cost

        # 配对本身合法：覆盖全部必重捕记录、一牌一记录、边来自原清单或建议线索
        required = {records[j] for j, flag in enumerate(flags) if not flag}
        paired_records = [rid for rid, _, _ in sug.pairing]
        assert required <= set(paired_records)
        assert len(paired_records) == len(set(paired_records))
        assert len({t for _, t, _ in sug.pairing}) == len(sug.pairing)
        recommended = {(c.record_id, c.tag) for c in sug.recommended}
        for rid, tag, via in sug.pairing:
            original = tag in candidates[records.index(rid)]
            assert original or (rid, tag) in recommended
            assert via == ((rid, tag) in recommended)
        checked += 1
