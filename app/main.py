"""FastAPI 入口：封闭水域鱼类种群估计服务。"""

from __future__ import annotations

import os
from collections import Counter

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from .estimation import EstimationError, estimate_population
from .review import (
    Clue,
    NoSolutionError,
    apply_verified_clues,
    find_min_cost_suggestion,
)
from .schemas import (
    ClueStatus,
    EstimateResponse,
    IntervalOut,
    PairingOut,
    ReviewClue,
    ReviewRequest,
    ReviewResponse,
    SurveyRequest,
    VersionEnvelope,
)
from .storage import CurrentVersionMismatch, SaveConflictError, SurveyStore

DB_PATH = os.getenv("SURVEY_DB_PATH", "/data/surveys.db")

app = FastAPI(
    title="封闭水域鱼类种群估计 API",
    description=(
        "两轮标志重捕（Petersen）种群估计。首轮唯一标牌，次轮记录可标注模糊标牌的"
        "候选集合与「是否可能是新个体」；一个标牌最多配给一条次轮记录。"
        "返回可行重捕数上下界，以及反演双侧 2.5% 等尾超几何检验得到的 95% 置信集。"
        "整单因无可行配对被拒后，可经补录复核接口提交线索：已核实线索补入原清单"
        "重新估计并发布新现行版；证据仍不足时，从待核线索中求核查总代价最低、"
        "核实后即可通过的一组，并给出可行配对（建议不写库）。"
    ),
    version="1.1.0",
)
store = SurveyStore(DB_PATH)


def _run_estimation(req: SurveyRequest) -> dict:
    first_tags = [item.tag for item in req.first_round]
    second_records = [item.candidate_tags for item in req.second_round]
    can_be_new = [item.can_be_new for item in req.second_round]

    # 存储层之外的业务校验：候选集合内不允许重复标牌
    for record in req.second_round:
        dup = [tag for tag, count in Counter(record.candidate_tags).items() if count > 1]
        if dup:
            raise EstimationError(f"记录 {record.record_id} 的候选标牌重复: {', '.join(dup)}")
        if not record.candidate_tags and not record.can_be_new:
            raise EstimationError(
                f"记录 {record.record_id} 无候选标牌且不可能是新个体，必然无可行配对"
            )

    estimate = estimate_population(first_tags, second_records, can_be_new, req.population_cap)
    return {
        "first_round_count": estimate.first_round_count,
        "second_round_count": estimate.second_round_count,
        "population_cap": estimate.population_cap,
        "recapture_min": estimate.r_min,
        "recapture_max": estimate.r_max,
        "feasible_recaptures": estimate.feasible_recaptures,
        "confidence_set_min": estimate.population_min,
        "confidence_set_max": estimate.population_max,
        "intervals": [
            IntervalOut.from_domain(item).model_dump() for item in estimate.intervals
        ],
        "warnings": estimate.warnings,
        "computation": {
            "tail_evaluations": estimate.tail_evaluations,
            "support_points_evaluated": estimate.support_points_evaluated,
            "complexity": "O(k^2 log U)，k<=80 为第二轮记录数，U=总体上限；对 U 次线性",
        },
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/surveys/{survey_id}/estimates",
    response_model=EstimateResponse,
    status_code=201,
    summary="提交（或整单更正）两轮调查记录并估计种群",
)
def submit_survey(survey_id: str, req: SurveyRequest) -> EstimateResponse:
    payload = req.model_dump()
    try:
        result = _run_estimation(req)
    except EstimationError as exc:
        # 失败留痕：写入 failed 版本，现行版本指针保持不变，仍可查原版
        store.save_version(survey_id, payload, result=None, error=str(exc))
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc

    saved = store.save_version(survey_id, payload, result=result, error=None)
    return EstimateResponse(
        survey_id=saved["survey_id"],
        version=saved["version"],
        status=saved["status"],
        **result,
    )


@app.get(
    "/surveys/{survey_id}/current",
    response_model=VersionEnvelope,
    summary="查询现行清单与结论",
)
def get_current(survey_id: str) -> VersionEnvelope:
    row = store.get_current(survey_id)
    if row is None:
        raise HTTPException(status_code=404, detail="该调查单尚无成功估计的现行版本")
    return VersionEnvelope(**row)


@app.get(
    "/surveys/{survey_id}/versions",
    response_model=list[VersionEnvelope],
    summary="列出全部历史版本（含被拒的更正单与失败原因）",
)
def list_versions(survey_id: str) -> list[VersionEnvelope]:
    rows = store.list_versions(survey_id)
    if not rows:
        raise HTTPException(status_code=404, detail="该调查单不存在任何提交记录")
    return [VersionEnvelope(**row) for row in rows]


@app.get(
    "/surveys/{survey_id}/versions/{version}",
    response_model=VersionEnvelope,
    summary="查询指定版本（失败后仍可查原版）",
)
def get_version(survey_id: str, version: int) -> VersionEnvelope:
    row = store.get_version(survey_id, version)
    if row is None:
        raise HTTPException(status_code=404, detail="指定版本不存在")
    return VersionEnvelope(**row)


# --------------------------------------------------------------------------- #
# 补录复核
# --------------------------------------------------------------------------- #


def _validate_clues(base_req: SurveyRequest, clues: list[ReviewClue]) -> None:
    """线索必须指向失败版中存在的次轮记录、首轮清单中存在的标牌，
    且该标牌不在此记录的原候选清单内；同一记录+标牌组合只允许一条。"""
    records = {rec.record_id: rec for rec in base_req.second_round}
    first_tags = {item.tag for item in base_req.first_round}
    seen: set[tuple[str, str]] = set()
    for clue in clues:
        key = (clue.record_id, clue.tag)
        if key in seen:
            raise HTTPException(
                status_code=422,
                detail={"error": f"重复线索：记录 {clue.record_id} 与标牌 {clue.tag} 的组合出现多次"},
            )
        seen.add(key)
        record = records.get(clue.record_id)
        if record is None:
            raise HTTPException(
                status_code=422,
                detail={"error": f"线索指向的次轮记录号 {clue.record_id!r} 在失败版清单中不存在"},
            )
        if clue.tag not in first_tags:
            raise HTTPException(
                status_code=422,
                detail={"error": f"线索标牌 {clue.tag!r} 不在首轮标牌清单中"},
            )
        if clue.tag in record.candidate_tags:
            raise HTTPException(
                status_code=422,
                detail={"error": f"线索标牌 {clue.tag!r} 已在记录 {clue.record_id} 的原候选清单中"},
            )


def _suggestion_response(
    survey_id: str,
    req: ReviewRequest,
    merged_req: SurveyRequest,
    pending: list[ReviewClue],
    verified: list[ReviewClue],
    base_error: str,
) -> ReviewResponse:
    """已核实线索仍不足：求最低代价核查建议或说明无解原因（均不写库）。"""
    common = dict(
        survey_id=survey_id,
        failed_version=req.failed_version,
        verified_clues_applied=verified,
    )
    if not pending:
        return ReviewResponse(
            **common,
            outcome="infeasible",
            reason=f"已核实线索补入后仍无可行配对（{base_error}），且未提供待核线索，无法给出核查建议",
        )
    try:
        suggestion = find_min_cost_suggestion(
            [item.tag for item in merged_req.first_round],
            [rec.record_id for rec in merged_req.second_round],
            [list(rec.candidate_tags) for rec in merged_req.second_round],
            [rec.can_be_new for rec in merged_req.second_round],
            merged_req.population_cap,
            [Clue(record_id=c.record_id, tag=c.tag, cost=c.cost) for c in pending],
        )
    except NoSolutionError as exc:
        return ReviewResponse(**common, outcome="infeasible", reason=str(exc))
    return ReviewResponse(
        **common,
        outcome="suggestion",
        reason=f"已核实线索补入后仍无可行配对（{base_error}）；建议优先核查以下待核线索，核实后整单即可通过",
        total_cost=suggestion.total_cost,
        recommended_clues=[
            ReviewClue(record_id=c.record_id, tag=c.tag, cost=c.cost, status=ClueStatus.pending)
            for c in suggestion.recommended
        ],
        pairing=[
            PairingOut(record_id=rid, tag=tag, via_recommended_clue=via)
            for rid, tag, via in suggestion.pairing
        ],
    )


@app.post(
    "/surveys/{survey_id}/reviews",
    response_model=ReviewResponse,
    summary="补录复核：合并已核实线索尝试发布，或为待核线索给出最低代价核查建议",
)
def review_survey(survey_id: str, req: ReviewRequest, response: Response) -> ReviewResponse:
    failed = store.get_version(survey_id, req.failed_version)
    if failed is None:
        raise HTTPException(status_code=404, detail="调查单或指定版本不存在")
    if failed["status"] != "failed":
        raise HTTPException(
            status_code=422, detail={"error": "指定版本不是失败版，补录复核仅针对被拒的提交"}
        )
    try:
        base_req = SurveyRequest(**failed["payload"])
    except ValidationError as exc:  # 历史数据损坏，正常流程不会触发
        raise HTTPException(
            status_code=500, detail={"error": f"失败版数据无法解析: {exc}"}
        ) from exc

    _validate_clues(base_req, req.clues)
    verified = [c for c in req.clues if c.status is ClueStatus.verified]
    pending = [c for c in req.clues if c.status is ClueStatus.pending]

    # 已核实线索补入原清单，两轮全部记录与原规则（必重捕、一牌一记录、总体上限）不变
    merged_payload = apply_verified_clues(
        failed["payload"],
        [Clue(record_id=c.record_id, tag=c.tag, cost=c.cost) for c in verified],
    )
    merged_req = SurveyRequest(**merged_payload)
    try:
        result = _run_estimation(merged_req)
    except EstimationError as exc:
        if "无可行配对" not in str(exc):
            raise HTTPException(
                status_code=422,
                detail={"error": f"失败版含数据错误，补录复核无法修复，请整单更正：{exc}"},
            ) from exc
        return _suggestion_response(survey_id, req, merged_req, pending, verified, str(exc))

    # 已核实线索补足证据：发布新现行版。计算期间同单可继续更正，
    # 提交前在单事务内核对请求中的现行版号；冲突或保存失败则本次不留新版本。
    provenance = {
        "review": {
            "failed_version": req.failed_version,
            "verified_clues": [c.model_dump() for c in verified],
        }
    }
    try:
        saved = store.save_version_if_current(
            survey_id, req.expected_current_version, merged_payload, {**result, **provenance}
        )
    except CurrentVersionMismatch as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": str(exc),
                "expected_current_version": exc.expected,
                "actual_current_version": exc.actual,
            },
        ) from exc
    except SaveConflictError as exc:
        raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc

    response.status_code = status.HTTP_201_CREATED
    return ReviewResponse(
        survey_id=survey_id,
        failed_version=req.failed_version,
        outcome="published",
        verified_clues_applied=verified,
        version=saved["version"],
        estimate=EstimateResponse(
            survey_id=survey_id, version=saved["version"], status="ok", **result
        ),
    )
