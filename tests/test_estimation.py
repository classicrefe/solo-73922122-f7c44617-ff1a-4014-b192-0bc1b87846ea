"""核心引擎测试：含 scipy 交叉验证与暴力扫描对照。"""

from __future__ import annotations

import math
import time

import pytest
from scipy.stats import hypergeom as scipy_hypergeom

from app.estimation import (
    ALPHA_TAIL,
    EstimationError,
    estimate_population,
    hypergeom_tails,
)


# --------------------------------------------------------------------------- #
# 超几何尾概率：与 scipy 对照（含 10 亿量级参数）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "N,K,n",
    [
        (20, 7, 10),
        (100, 50, 30),
        (1000, 10, 500),
        (50, 49, 25),
        (10_000, 80, 160),
        (1_000_000_000, 80, 160),
        (1_000_000_000, 1, 2),
    ],
)
def test_hypergeom_tails_match_scipy(N, K, n):
    lo = max(0, n - (N - K))
    hi = min(n, K)
    for x in range(lo, hi + 1):
        lower, upper = hypergeom_tails(N, K, n, x)
        # scipy 参数：M=总体，n=成功状态数，N=抽取数
        exp_lower = scipy_hypergeom.cdf(x, N, K, n)
        exp_upper = scipy_hypergeom.sf(x - 1, N, K, n)
        assert lower == pytest.approx(exp_lower, abs=1e-10)
        assert upper == pytest.approx(exp_upper, abs=1e-10)


def test_hypergeom_tails_outside_support():
    # N=10, K=4, n=10 => 全量抽取，X 恒等于 4
    assert hypergeom_tails(10, 4, 10, 0) == (0.0, 1.0)
    assert hypergeom_tails(10, 4, 10, 9) == (1.0, 0.0)
    lower, upper = hypergeom_tails(10, 4, 10, 4)
    assert lower == 1.0 and upper == 1.0


def test_tail_monotonicity_in_N():
    """CDF 随 N 单调不减、上尾随 N 单调不增——二分查找正确性的前提。"""
    K, n, x = 30, 50, 12
    prev_lower, prev_upper = 0.0, 1.0
    for N in range(n, 400):
        lower, upper = hypergeom_tails(N, K, n, x)
        assert lower >= prev_lower - 1e-15
        assert upper <= prev_upper + 1e-15
        prev_lower, prev_upper = lower, upper


# --------------------------------------------------------------------------- #
# 置信集：与 scipy 暴力扫描逐点对照
# --------------------------------------------------------------------------- #


def _brute_force_set(M, C, feasible_r, cap):
    accepted = set()
    for r in feasible_r:
        for N in range(M + C - r, cap + 1):
            lower = scipy_hypergeom.cdf(r, N, M, C)
            upper = scipy_hypergeom.sf(r - 1, N, M, C)
            if lower >= ALPHA_TAIL and upper >= ALPHA_TAIL:
                accepted.add(N)
    return (min(accepted), max(accepted)) if accepted else (None, None)


def _check_against_brute_force(first, second, new_flags, cap):
    est = estimate_population(first, second, new_flags, cap)
    M, C = est.first_round_count, est.second_round_count
    expected_min, expected_max = _brute_force_set(M, C, est.feasible_recaptures, cap)
    assert est.population_min == expected_min
    assert est.population_max == expected_max
    # 每个 r 的区间也逐一对照
    for item in est.intervals:
        r = item.recapture
        accepted = [
            N
            for N in range(M + C - r, cap + 1)
            if scipy_hypergeom.cdf(r, N, M, C) >= ALPHA_TAIL
            and scipy_hypergeom.sf(r - 1, N, M, C) >= ALPHA_TAIL
        ]
        if accepted:
            assert item.population_low == min(accepted)
            assert item.population_high == max(accepted)
    return est


def test_classic_petersen_exact_recapture():
    # 5 条明确重捕 + 3 条明确新个体，重捕数被唯一确定为 5
    tags = [f"T{i}" for i in range(10)]
    second = [[f"T{i}"] for i in range(5)] + [[], [], []]
    flags = [False] * 5 + [True] * 3
    est = _check_against_brute_force(tags, second, flags, 200)
    assert est.r_min == est.r_max == 5
    assert est.feasible_recaptures == [5]
    assert est.population_min is not None and est.population_max is not None
    assert not est.warnings


def test_ambiguous_tags_give_recapture_range():
    # R1 必须配 T1；R2 候选 {T1,T2} 可新；R3 候选 {T2} 可新
    tags = ["T1", "T2", "T3"]
    second = [["T1"], ["T1", "T2"], ["T2"]]
    flags = [False, True, True]
    est = _check_against_brute_force(tags, second, flags, 100)
    # r_min=1（仅强制记录），r_max=2（T1、T2 各配一条，R1 锁定 T1）
    assert est.r_min == 1 and est.r_max == 2
    assert est.feasible_recaptures == [1, 2]


def test_population_cap_floor_removes_low_recaptures():
    # M=6, C=4，全部可新、无候选 => 若无上限约束 r 可为 0；
    # cap=8 要求 r >= 10-8=2，但最大匹配只有 0 => 无可行配对
    tags = [f"T{i}" for i in range(6)]
    with pytest.raises(EstimationError, match="上限"):
        estimate_population(tags, [[], [], [], []], [True] * 4, 8)


def test_infeasible_required_conflict():
    # 两条记录都必须重捕，但候选只有同一个标牌，一牌不能配两条
    tags = ["T1", "T2"]
    with pytest.raises(EstimationError, match="无可行配对"):
        estimate_population(tags, [["T1"], ["T1"]], [False, False], 50)


def test_zero_recapture_warning():
    tags = [f"T{i}" for i in range(4)]
    est = _check_against_brute_force(tags, [[], []], [True, True], 300)
    assert est.r_min == est.r_max == 0
    assert any("零重捕" in w for w in est.warnings)
    assert est.population_min is not None  # r=0 仍有置信区间


def test_empty_confidence_set_when_cap_too_tight():
    # 零重捕 + 紧上限：M=C=10、r=0，cap=20 时在整个可行 N 范围 [20,20] 内
    # P(X=0)=5.4e-6 < 2.5%，没有任何 N 能通过检验 => 95% 置信集为空
    tags = [f"T{i}" for i in range(10)]
    second = [[] for _ in range(10)]
    est = estimate_population(tags, second, [True] * 10, 20)
    assert est.feasible_recaptures == [0]
    assert est.intervals == []
    assert est.population_min is None and est.population_max is None
    assert any("置信集为空" in w for w in est.warnings)
    assert any("零重捕" in w for w in est.warnings)
    # 放宽上限后置信集重新出现
    relaxed = estimate_population(tags, second, [True] * 10, 100_000)
    assert relaxed.population_min is not None and relaxed.population_max is not None


def test_unknown_tag_rejected():
    with pytest.raises(EstimationError, match="不存在"):
        estimate_population(["T1"], [["T9"]], [True], 50)


def test_cap_boundary_feasibility():
    # M=C=5，r 固定 4 => 不同个体数 6，cap=6 恰好可行（r 下限=4）
    tags = [f"T{i}" for i in range(5)]
    second = [[f"T{i}"] for i in range(4)] + [[]]
    est = estimate_population(tags, second, [False] * 4 + [True], 6)
    assert est.r_min == 4
    # N 只能为 6；r=4 在 N=6 下检验应通过，置信集恰为 {6}
    assert est.population_min == est.population_max == 6


# --------------------------------------------------------------------------- #
# 次线性计算量：10 亿上限的计时与求值计数
# --------------------------------------------------------------------------- #


def test_sublinear_performance_at_one_billion():
    tags = [f"T{i}" for i in range(80)]
    # 完全模糊：每条次轮记录候选为全部 80 个标牌，均可新 => r 取 0..80
    second = [tags[:] for _ in range(80)]
    start = time.perf_counter()
    est = estimate_population(tags, second, [True] * 80, 1_000_000_000)
    elapsed = time.perf_counter() - start
    assert est.r_min == 0 and est.r_max == 80
    assert len(est.feasible_recaptures) == 81
    # 尾概率求值次数只与记录数和 log2(U) 有关，与 U 本身无关
    assert est.tail_evaluations < 81 * 2 * 32
    assert elapsed < 5.0, f"10 亿上限耗时 {elapsed:.2f}s，超出次线性预算"


def test_runtime_independent_of_cap_magnitude():
    """U 从 1e5 提到 1e9（4 个数量级）：求值次数只应按 log2(U) 增长。"""
    tags = [f"T{i}" for i in range(40)]
    second = [tags[:] for _ in range(40)]
    small = estimate_population(tags, second, [True] * 40, 100_000)
    huge = estimate_population(tags, second, [True] * 40, 1_000_000_000)
    log_growth = math.log2(1e9 / 1e5)  # ~13.3 个二分步
    # 每个 r 两次二分，故求值次数增量上界约 2 * 41 个可行 r * 14 步
    assert huge.tail_evaluations - small.tail_evaluations <= 2 * 41 * math.ceil(log_growth)
    # 关键：次数是 O(r * log U) 量级，而不是 O(U)；两者都只有几千次
    assert huge.tail_evaluations < 41 * 2 * 32
