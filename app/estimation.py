"""标志重捕种群估计引擎。

场景（两轮 Petersen 设计，封闭水域）：

* 第一轮捕获 M 尾鱼，全部挂上唯一标牌后放回；
* 第二轮共 C 条记录，每条记录要么能读到某个第一轮标牌，要么标牌模糊，
  只能列出若干个「可能的第一轮标牌」，并注明该个体是否可能是新个体
  （第一轮未被捕获过）；
* 一个第一轮标牌最多只能配给一条第二轮记录（一一匹配）。

设重捕数为 r，则两轮涉及的不同个体数为 M + C - r，总体数量 N 必须满足
M + C - r <= N，即 r >= M + C - N。在给定总体上限 U 下，仅保留
r >= M + C - U 的配对方案。

统计模型：第二轮从总体 N 中抽取 C 尾，其中带标个体数 X 服从超几何分布
X ~ Hypergeom(N, K=M, draws=C)（总体中带标个体恰为首轮的 M 尾）。
95% 等尾置信集通过反演检验得到：N 被纳入当且仅当
P(X <= r) >= 2.5% 且 P(X >= r) >= 2.5%（两尾均含观测值 r）。

计算复杂度：可行 r 至多 80 个，每个 r 用两次二分查找定位 N 的边界，
每次尾概率计算只遍历至多 81 个支撑点，故总体为 O(80^2 * log U)，
对 U 次线性（U 只出现在对数项中）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_RECORDS_PER_ROUND = 80
MAX_POPULATION_CAP = 1_000_000_000
ALPHA_TAIL = 0.025


class EstimationError(ValueError):
    """无可行配对（或违反总体上限）时抛出。"""


@dataclass
class RecaptureInterval:
    """单个可行重捕数 r 对应的总体数量 95% 区间。"""

    recapture: int
    population_low: int
    population_high: int


@dataclass
class Estimate:
    """一次完整估计的结论。"""

    first_round_count: int
    second_round_count: int
    population_cap: int
    r_min: int
    r_max: int
    feasible_recaptures: list[int]
    intervals: list[RecaptureInterval]
    population_min: int | None
    population_max: int | None
    warnings: list[str] = field(default_factory=list)
    tail_evaluations: int = 0
    support_points_evaluated: int = 0


# --------------------------------------------------------------------------- #
# 二分图匹配（Kuhn 增广路算法，左节点=第二轮记录，右节点=第一轮标牌）
# --------------------------------------------------------------------------- #


def _maximum_matching(
    adjacency: list[list[int]], n_left: int, n_right: int, left_subset: list[int] | None = None
) -> int:
    """计算二分图最大匹配，返回匹配边数。

    left_subset 不为 None 时，仅允许左节点子集中的节点参与匹配，
    用于检验「必须配对」的记录是否全部能配上。
    """
    match_right = [-1] * n_right

    def augment(left: int, seen: list[bool]) -> bool:
        for right in adjacency[left]:
            if seen[right]:
                continue
            seen[right] = True
            if match_right[right] == -1 or augment(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    size = 0
    lefts = left_subset if left_subset is not None else range(n_left)
    for left in lefts:
        seen = [False] * n_right
        if augment(left, seen):
            size += 1
    return size


# --------------------------------------------------------------------------- #
# 超几何分布尾概率（对数空间 + 递推，避免组合数溢出）
# --------------------------------------------------------------------------- #


def _log_pmf(N: int, K: int, n: int, x: int) -> float:
    """log P(X=x)，X ~ Hypergeom(N, K, n)。"""
    return (
        math.lgamma(K + 1)
        - math.lgamma(x + 1)
        - math.lgamma(K - x + 1)
        + math.lgamma(N - K + 1)
        - math.lgamma(n - x + 1)
        - math.lgamma(N - K - n + x + 1)
        - math.lgamma(N + 1)
        + math.lgamma(n + 1)
        + math.lgamma(N - n + 1)
    )


def _log_pmf_series(N: int, K: int, n: int) -> tuple[int, int, list[float]]:
    """返回支撑下界 lo、上界 hi 及各点的未归一化对数概率。

    锚点用 lgamma 计算一次，其余点按相邻概率比值递推，
    把每次尾概率计算的 lgamma 调用从 O(支撑点数 * 9) 降到常数次。
    """
    lo = max(0, n - (N - K))
    hi = min(n, K)
    logs = [0.0] * (hi - lo + 1)
    logs[0] = _log_pmf(N, K, n, lo)
    for i, x in enumerate(range(lo, hi), start=1):
        # P(x+1)/P(x) = (K-x)/(x+1) * (n-x)/(N-K-n+x+1)
        ratio = ((K - x) / (x + 1)) * ((n - x) / (N - K - n + x + 1))
        logs[i] = logs[i - 1] + math.log(ratio)
    return lo, hi, logs


def hypergeom_tails(
    N: int, K: int, n: int, x: int, counters: dict[str, int] | None = None
) -> tuple[float, float]:
    """返回 (P(X<=x), P(X>=x))，两尾均含观测值 x。

    若 x 落在支撑之外，按尾概率的极限语义返回 0/1。
    """
    lo = max(0, n - (N - K))
    hi = min(n, K)
    if x < lo:
        if counters is not None:
            counters["tail_evaluations"] = counters.get("tail_evaluations", 0) + 1
        return 0.0, 1.0
    if x > hi:
        if counters is not None:
            counters["tail_evaluations"] = counters.get("tail_evaluations", 0) + 1
        return 1.0, 0.0

    if counters is not None:
        counters["tail_evaluations"] = counters.get("tail_evaluations", 0) + 1
        counters["support_points_evaluated"] = counters.get("support_points_evaluated", 0) + (
            hi - lo + 1
        )

    _, _, logs = _log_pmf_series(N, K, n)
    peak = max(logs)
    weights = [math.exp(v - peak) for v in logs]
    total = sum(weights)
    probs = [w / total for w in weights]

    idx = x - lo
    lower_tail = sum(probs[: idx + 1])
    upper_tail = sum(probs[idx:])
    return lower_tail, upper_tail


# --------------------------------------------------------------------------- #
# 单个重捕数 r 的置信区间（二分查找；尾概率关于 N 单调）
# --------------------------------------------------------------------------- #


def _interval_for_r(
    r: int, M: int, C: int, cap: int, counters: dict[str, int]
) -> tuple[int, int] | None:
    """反演等尾检验，返回使 r 通过检验的 N 的最小/最大值。

    对固定 r，总体 N 的可行下界为 M+C-r（两轮不同个体数上限），
    上界为用户给定的 cap。
    """
    n_floor = M + C - r  # N 至少要能容纳两轮涉及的全部不同个体

    lower_at_cap, _ = hypergeom_tails(cap, M, C, r, counters)
    if lower_at_cap < ALPHA_TAIL:
        # 下限随 N 单调不减；连在上限处都不足 2.5%，整个范围无解
        return None
    _, upper_at_floor = hypergeom_tails(n_floor, M, C, r, counters)
    if upper_at_floor < ALPHA_TAIL:
        # 上限随 N 单调不增；连在最小 N 处都不足 2.5%，整个范围无解
        return None

    # 最小 N：P(X<=r) >= 2.5%（谓词随 N 单调不减）
    lo, hi = n_floor, cap
    while lo < hi:
        mid = (lo + hi) // 2
        lower, _ = hypergeom_tails(mid, M, C, r, counters)
        if lower >= ALPHA_TAIL:
            hi = mid
        else:
            lo = mid + 1
    population_low = lo

    # 最大 N：P(X>=r) >= 2.5%（谓词随 N 单调不增，找最后一个真值）
    lo, hi = n_floor, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, upper = hypergeom_tails(mid, M, C, r, counters)
        if upper >= ALPHA_TAIL:
            lo = mid
        else:
            hi = mid - 1
    population_high = lo

    if population_low > population_high:
        return None
    return population_low, population_high


# --------------------------------------------------------------------------- #
# 顶层估计流程
# --------------------------------------------------------------------------- #


def estimate_population(
    first_tags: list[str],
    second_records: list[list[str]],
    can_be_new: list[bool],
    cap: int,
) -> Estimate:
    """根据两轮记录与总体上限计算可行重捕数范围与 95% 置信集。

    参数：
        first_tags: 第一轮唯一标牌列表，长度 M（1~80）。
        second_records: 每条第二轮记录可能对应的第一轮标牌列表，长度 C（1~80）。
        can_be_new: 与 second_records 等长，该记录是否可能是新个体。
        cap: 用户给定的总体数量上限（1 ~ 10 亿）。

    抛出 EstimationError：不存在满足约束与上限的配对方案。
    """
    M = len(first_tags)
    C = len(second_records)
    if not 1 <= M <= MAX_RECORDS_PER_ROUND or not 1 <= C <= MAX_RECORDS_PER_ROUND:
        raise EstimationError("每轮记录数必须在 1~80 条之间")
    if not 1 <= cap <= MAX_POPULATION_CAP:
        raise EstimationError("总体上限必须在 1~1,000,000,000 之间")
    if len(can_be_new) != C:
        raise EstimationError("can_be_new 与第二轮记录数不一致")

    tag_index = {tag: i for i, tag in enumerate(first_tags)}
    adjacency: list[list[int]] = []
    required_left: list[int] = []
    for left, candidates in enumerate(second_records):
        if len(set(candidates)) != len(candidates):
            raise EstimationError(f"第二轮记录 {left} 的候选标牌存在重复")
        rights = []
        for tag in candidates:
            if tag not in tag_index:
                raise EstimationError(f"候选标牌 {tag!r} 在第一轮标牌清单中不存在")
            rights.append(tag_index[tag])
        adjacency.append(rights)
        if not can_be_new[left]:
            required_left.append(left)

    # r_min：所有「不可能是新个体」的记录必须配上标牌
    required_matched = _maximum_matching(adjacency, C, M, required_left)
    if required_matched != len(required_left):
        raise EstimationError(
            "无可行配对：存在必须重捕的记录，但受一牌一记录约束无法全部配上标牌"
        )
    r_min = len(required_left)

    # r_max：整图最大匹配
    r_max = _maximum_matching(adjacency, C, M)

    # 总体上限约束：M + C - r <= cap  <=>  r >= M + C - cap
    cap_floor = M + C - cap
    r_low = max(r_min, cap_floor)
    if r_low > r_max:
        raise EstimationError(
            f"无可行配对：上限 {cap} 下要求重捕数 >= {cap_floor}，"
            f"但配对最多只能达到 {r_max}"
        )

    counters: dict[str, int] = {}
    intervals: list[RecaptureInterval] = []
    for r in range(r_low, r_max + 1):
        bounds = _interval_for_r(r, M, C, cap, counters)
        if bounds is not None:
            intervals.append(
                RecaptureInterval(recapture=r, population_low=bounds[0], population_high=bounds[1])
            )

    warnings: list[str] = []
    if r_low == 0:
        warnings.append(
            "可能零重捕：可行配对允许 r=0，此时经典 Petersen 估计不可用，"
            "置信集仅由等尾超几何检验反演给出"
        )
    if not intervals:
        population_min = population_max = None
        warnings.append(
            "95% 置信集为空：在给定总体上限内，没有任何总体数量能使任一可行"
            "重捕数通过双侧 2.5% 等尾检验，可考虑放宽总体上限后重试"
        )
    else:
        population_min = min(item.population_low for item in intervals)
        population_max = max(item.population_high for item in intervals)

    return Estimate(
        first_round_count=M,
        second_round_count=C,
        population_cap=cap,
        r_min=r_low,
        r_max=r_max,
        feasible_recaptures=list(range(r_low, r_max + 1)),
        intervals=intervals,
        population_min=population_min,
        population_max=population_max,
        warnings=warnings,
        tail_evaluations=counters.get("tail_evaluations", 0),
        support_points_evaluated=counters.get("support_points_evaluated", 0),
    )
