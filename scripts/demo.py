"""端到端演示脚本：在 /docs 演示核实标牌前后的估计与补录复核。

用法：
    python scripts/demo.py            # 默认 http://localhost:8000
    BASE_URL=http://其他地址 python scripts/demo.py

场景：
    1. 提交「标牌模糊」版本：R2 看不清，候选 T1/T2/T3，得到重捕数区间；
    2. 整单更正为「标牌核实」版本：R2 确认是 T3，重捕数被唯一确定；
    3. 再提交一份无可行配对的错误更正，确认失败留痕、原版仍可查；
    4. 零重捕 + 紧上限案例，演示空置信集提示；
    5. 补录复核全流程：两条必重捕记录只认出同一标牌被拒 -> 提交线索 ->
       已核实不足时给出最低代价核查建议（不写库）-> 核实建议线索后发布
       新现行版 -> 现行版号冲突时 409 且不留新版本。
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
# 每次运行使用新调查单，保证脚本可重复执行、版本号断言稳定
SURVEY = f"demo-lake-{int(time.time())}"


def show(title: str, body: dict) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(json.dumps(body, ensure_ascii=False, indent=2))


def first_payload() -> dict:
    return {
        "first_round": [{"tag": f"T{i:02d}"} for i in range(1, 11)],  # M=10
        "second_round": [
            {"record_id": "R1", "candidate_tags": ["T01"], "can_be_new": False},
            {"record_id": "R2", "candidate_tags": ["T01", "T02", "T03"], "can_be_new": True},
            {"record_id": "R3", "candidate_tags": ["T02"], "can_be_new": False},
            {"record_id": "R4", "candidate_tags": [], "can_be_new": True},
            {"record_id": "R5", "candidate_tags": [], "can_be_new": True},
            {"record_id": "R6", "candidate_tags": [], "can_be_new": True},
        ],  # C=6
        "population_cap": 5000,
    }


def corrected_payload() -> dict:
    payload = first_payload()
    # 标牌核实：R2 确认对应 T03
    payload["second_round"][1] = {
        "record_id": "R2",
        "candidate_tags": ["T03"],
        "can_be_new": False,
    }
    return payload


def impossible_payload() -> dict:
    payload = first_payload()
    # R2 与 R1 都被指定为 T01 且都不可能是新个体 => 一牌不能配两条
    payload["second_round"][1] = {
        "record_id": "R2",
        "candidate_tags": ["T01"],
        "can_be_new": False,
    }
    return payload


def review_conflict_payload() -> dict:
    """两条必重捕记录（X1、X2）都只认出 T01，整单因无可行配对被拒。"""
    return {
        "first_round": [{"tag": f"T{i:02d}"} for i in range(1, 11)],  # M=10
        "second_round": [
            {"record_id": "X1", "candidate_tags": ["T01"], "can_be_new": False},
            {"record_id": "X2", "candidate_tags": ["T01"], "can_be_new": False},
            {"record_id": "X3", "candidate_tags": ["T02"], "can_be_new": False},
            {"record_id": "X4", "candidate_tags": ["T03"], "can_be_new": False},
            {"record_id": "R5", "candidate_tags": [], "can_be_new": True},
            {"record_id": "R6", "candidate_tags": [], "can_be_new": True},
        ],  # C=6
        "population_cap": 5000,
    }


def review_request() -> dict:
    """补录复核请求：1 条已核实 + 3 条待核 + 1 条排除（更便宜的干扰项）。"""
    return {
        "failed_version": 1,
        "expected_current_version": None,  # 该单尚无现行版
        "clues": [
            {"record_id": "X1", "tag": "T03", "cost": 4, "status": "verified"},
            {"record_id": "X4", "tag": "T04", "cost": 2, "status": "pending"},
            {"record_id": "X2", "tag": "T05", "cost": 3, "status": "pending"},
            {"record_id": "X1", "tag": "T04", "cost": 8, "status": "pending"},
            {"record_id": "X2", "tag": "T04", "cost": 1, "status": "excluded"},
        ],
    }


def main() -> int:
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    print(">>> 健康检查:", client.get("/health").json())

    # 1) 标牌模糊版本
    r1 = client.post(f"/surveys/{SURVEY}/estimates", json=first_payload())
    r1.raise_for_status()
    show("① 标牌辨认前（R2 模糊，候选 T01/T02/T03）", r1.json())
    assert r1.json()["recapture_min"] == 2 and r1.json()["recapture_max"] == 3

    # 2) 标牌核实后的整单更正
    r2 = client.post(f"/surveys/{SURVEY}/estimates", json=corrected_payload())
    r2.raise_for_status()
    show("② 标牌核实后（R2 确认为 T03，整单换版）", r2.json())
    assert r2.json()["version"] == 2
    assert r2.json()["recapture_min"] == r2.json()["recapture_max"] == 3

    # 3) 错误更正：无可行配对，应被拒且保留原版
    r3 = client.post(f"/surveys/{SURVEY}/estimates", json=impossible_payload())
    show("③ 错误更正（两记录争同一标牌）预期 422", r3.json())
    assert r3.status_code == 422

    versions = client.get(f"/surveys/{SURVEY}/versions").json()
    show("④ 版本列表：失败留痕，现行版本仍为 v2", versions)
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert versions[2]["status"] == "failed"
    current = client.get(f"/surveys/{SURVEY}/current").json()
    assert current["version"] == 2 and current["status"] == "ok"

    # 4) 零重捕 + 紧上限 => 空置信集提示
    empty_case = {
        "first_round": [{"tag": f"T{i:02d}"} for i in range(1, 11)],
        "second_round": [
            {"record_id": f"R{i}", "candidate_tags": [], "can_be_new": True}
            for i in range(1, 11)
        ],
        "population_cap": 20,
    }
    r4 = client.post(f"/surveys/{SURVEY}-zero/estimates", json=empty_case)
    show("⑤ 可能零重捕且上限过紧：空置信集明确提示", r4.json())
    assert r4.status_code == 201
    body = r4.json()
    assert body["confidence_set_min"] is None and body["confidence_set_max"] is None
    assert any("零重捕" in w for w in body["warnings"])
    assert any("置信集为空" in w for w in body["warnings"])

    # ------------------------------------------------------------------ #
    # 补录复核：从找线索到补录发布
    # ------------------------------------------------------------------ #
    review_survey = f"{SURVEY}-review"

    # 6) 两条必重捕记录都只认出 T01 => 无可行配对，整单被拒（v1 失败）
    r6 = client.post(f"/surveys/{review_survey}/estimates", json=review_conflict_payload())
    show("⑥ 两条必重捕记录只认出同一标牌，整单被拒（失败留痕）", r6.json())
    assert r6.status_code == 422

    # 7) 提交线索：已核实 X1<->T03 仍不足；待核三条；排除一条更便宜的干扰线索
    req = review_request()
    r7 = client.post(f"/surveys/{review_survey}/reviews", json=req)
    show("⑦ 补录复核：证据仍不足，给出最低代价核查建议", r7.json())
    assert r7.status_code == 200
    body = r7.json()
    assert body["outcome"] == "suggestion"
    # 最低代价是 2（只核实 X4<->T04）；排除的 X2<->T04（代价 1）不得被采用
    assert body["total_cost"] == 2
    assert [(c["record_id"], c["tag"]) for c in body["recommended_clues"]] == [("X4", "T04")]
    assert len(body["pairing"]) == 4  # 覆盖 X1..X4 四条必重捕记录
    # 建议不写库：版本仍只有 v1
    assert [v["version"] for v in client.get(f"/surveys/{review_survey}/versions").json()] == [1]

    # 8) 按建议核实 X4<->T04 后再次复核 => 直接发布为新现行版
    req["clues"][1]["status"] = "verified"  # X4<->T04 已核实
    req["clues"][2]["status"] = "excluded"
    req["clues"][3]["status"] = "excluded"
    r8 = client.post(f"/surveys/{review_survey}/reviews", json=req)
    show("⑧ 核实建议线索后再次复核：发布新现行版", r8.json())
    assert r8.status_code == 201
    body = r8.json()
    assert body["outcome"] == "published" and body["version"] == 2
    assert body["estimate"]["recapture_min"] == body["estimate"]["recapture_max"] == 4

    current = client.get(f"/surveys/{review_survey}/current").json()
    show("⑨ 现行版 v2：已核实线索已补入原清单；失败版 v1 仍可查", current)
    assert current["version"] == 2
    assert current["payload"]["second_round"][0]["candidate_tags"] == ["T01", "T03"]
    assert current["payload"]["second_round"][3]["candidate_tags"] == ["T03", "T04"]
    old = client.get(f"/surveys/{review_survey}/versions/1").json()
    assert old["status"] == "failed" and "无可行配对" in old["error"]

    # 10) 现行版已是 v2，仍按「尚无现行版」发布 => 409 冲突且不留新版本
    r10 = client.post(f"/surveys/{review_survey}/reviews", json=req)
    show("⑩ 现行版号冲突：409，本次不留新版本", r10.json())
    assert r10.status_code == 409
    versions = client.get(f"/surveys/{review_survey}/versions").json()
    assert [v["version"] for v in versions] == [1, 2]

    print("\n全部演示断言通过 ✅")
    print(f"交互式文档地址：{BASE_URL}/docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
