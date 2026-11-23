"""请求/响应数据模型。"""

from __future__ import annotations

from enum import Enum

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


# --------------------------------------------------------------------------- #
# 补录复核
# --------------------------------------------------------------------------- #


class ClueStatus(str, Enum):
    """线索核查状态。"""

    verified = "verified"  # 已核实：直接补入原清单
    pending = "pending"  # 待核：参与最低代价建议的求解
    excluded = "excluded"  # 排除：不参与任何计算


class ReviewClue(BaseModel):
    """一条补录线索：某次轮记录还可能对应的首轮标牌及其核查代价与状态。"""

    record_id: str = Field(..., min_length=1, max_length=64, description="次轮记录号（须存在于失败版清单）")
    tag: str = Field(..., min_length=1, max_length=64, description="原清单未列出的首轮候选标牌")
    cost: int = Field(..., ge=1, le=1_000_000_000, description="正整数核查代价")
    status: ClueStatus = Field(..., description="verified=已核实 / pending=待核 / excluded=排除")


class ReviewRequest(BaseModel):
    """补录复核请求：针对无可行配对的失败版补充线索。"""

    failed_version: int = Field(..., ge=1, description="无可行配对的失败版号")
    expected_current_version: int | None = Field(
        ...,
        description="发布时核对的现行版号（乐观并发）；该单尚无现行版时填 null",
    )
    clues: list[ReviewClue] = Field(
        ...,
        min_length=1,
        max_length=MAX_RECORDS_PER_ROUND * MAX_RECORDS_PER_ROUND,
        description="补录线索（同一记录号+标牌组合只能出现一次）",
    )


class PairingOut(BaseModel):
    """建议配对中的一条记录-标牌对应。"""

    record_id: str
    tag: str
    via_recommended_clue: bool = Field(..., description="该配对是否来自建议核实的线索")


class ReviewResponse(BaseModel):
    """补录复核结果：发布 / 建议 / 无解 三选一。"""

    survey_id: str
    failed_version: int
    outcome: str = Field(
        ...,
        description="published=已核实线索补足证据，已发布新现行版；"
        "suggestion=证据仍不足，给出最低代价核查建议（不写库）；"
        "infeasible=全部待核线索核实后仍无解（不写库）",
    )
    reason: str | None = Field(None, description="suggestion / infeasible 时的说明")
    verified_clues_applied: list[ReviewClue] = Field(
        default_factory=list, description="本次补入原清单的已核实线索"
    )
    # outcome = published
    version: int | None = Field(None, description="published 时的新现行版号")
    estimate: EstimateResponse | None = Field(None, description="published 时按原规则估计的结果")
    # outcome = suggestion
    total_cost: int | None = Field(None, description="suggestion 时的最低核查总代价")
    recommended_clues: list[ReviewClue] | None = Field(
        None, description="suggestion 时建议核实（补入）的待核线索"
    )
    pairing: list[PairingOut] | None = Field(
        None, description="suggestion 时核实建议线索后的一份可行配对"
    )
