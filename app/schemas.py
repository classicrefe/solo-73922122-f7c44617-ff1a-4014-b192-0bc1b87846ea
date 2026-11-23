"""请求/响应数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .estimation import MAX_POPULATION_CAP, MAX_RECORDS_PER_ROUND, RecaptureInterval


class FirstRoundRecord(BaseModel):
    """第一轮记录：唯一标牌。"""

    tag: str = Field(..., min_length=1, max_length=64, description="第一轮唯一标牌号")


class SecondRoundRecord(BaseModel):
    """第二轮记录：唯一记录号 + 可能的首轮标牌 + 是否可能是新个体。"""

    record_id: str = Field(..., min_length=1, max_length=64, description="第二轮唯一记录号")
    candidate_tags: list[str] = Field(
        ..., max_length=MAX_RECORDS_PER_ROUND, description="该记录可能对应的第一轮标牌（可空=全新个体）"
    )
    can_be_new: bool = Field(..., description="该个体是否可能是第一轮未捕获的新个体")

    @field_validator("candidate_tags")
    @classmethod
    def _no_blank_tag(cls, value: list[str]) -> list[str]:
        if any(not tag.strip() for tag in value):
            raise ValueError("候选标牌号不能为空字符串")
        return value


class SurveyRequest(BaseModel):
    """整单提交（首轮 + 次轮 + 总体上限）。"""

    first_round: list[FirstRoundRecord] = Field(..., min_length=1, max_length=MAX_RECORDS_PER_ROUND)
    second_round: list[SecondRoundRecord] = Field(..., min_length=1, max_length=MAX_RECORDS_PER_ROUND)
    population_cap: int = Field(..., ge=1, le=MAX_POPULATION_CAP)

    @field_validator("first_round")
    @classmethod
    def _unique_tags(cls, value: list[FirstRoundRecord]) -> list[FirstRoundRecord]:
        tags = [item.tag for item in value]
        if len(set(tags)) != len(tags):
            raise ValueError("第一轮标牌号必须唯一")
        return value

    @field_validator("second_round")
    @classmethod
    def _unique_record_ids(cls, value: list[SecondRoundRecord]) -> list[SecondRoundRecord]:
        ids = [item.record_id for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("第二轮记录号必须唯一")
        return value


class IntervalOut(BaseModel):
    recapture: int
    population_low: int
    population_high: int

    @classmethod
    def from_domain(cls, item: RecaptureInterval) -> "IntervalOut":
        return cls(
            recapture=item.recapture,
            population_low=item.population_low,
            population_high=item.population_high,
        )


class VersionEnvelope(BaseModel):
    """一次整单提交产生的版本（成功或失败均留痕）。"""

    survey_id: str
    version: int
    status: str = Field(..., description="ok=估计成功；failed=整单更正被拒，现行版本保持不变")
    created_at: str
    payload: dict
    result: dict | None = None
    error: str | None = None
    is_current: bool


class EstimateResponse(BaseModel):
    survey_id: str
    version: int
    status: str
    first_round_count: int
    second_round_count: int
    population_cap: int
    recapture_min: int = Field(..., description="可行重捕数下界")
    recapture_max: int = Field(..., description="可行重捕数上界")
    feasible_recaptures: list[int]
    confidence_set_min: int | None = Field(..., description="95% 置信集中最小总体数量；空集为 null")
    confidence_set_max: int | None = Field(..., description="95% 置信集中最大总体数量；空集为 null")
    intervals: list[IntervalOut]
    warnings: list[str]
    computation: dict = Field(..., description="计算量审计：尾概率求值次数与支撑点总数")
