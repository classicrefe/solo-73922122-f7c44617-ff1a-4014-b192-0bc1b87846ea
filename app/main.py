"""FastAPI 入口：封闭水域鱼类种群估计服务。"""

from __future__ import annotations

import os
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .estimation import EstimationError, estimate_population
from .schemas import EstimateResponse, IntervalOut, SurveyRequest, VersionEnvelope
from .storage import SurveyStore

DB_PATH = os.getenv("SURVEY_DB_PATH", "/data/surveys.db")

app = FastAPI(
    title="封闭水域鱼类种群估计 API",
    description=(
        "两轮标志重捕（Petersen）种群估计。首轮唯一标牌，次轮记录可标注模糊标牌的"
        "候选集合与「是否可能是新个体」；一个标牌最多配给一条次轮记录。"
        "返回可行重捕数上下界，以及反演双侧 2.5% 等尾超几何检验得到的 95% 置信集。"
    ),
    version="1.0.0",
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
