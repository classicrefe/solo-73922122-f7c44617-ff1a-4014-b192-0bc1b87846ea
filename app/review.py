"""补录复核：失败版的线索合并与最低代价核查建议。

业务背景：两条必重捕记录只认出同一个标牌时，整单因「无可行配对」被拒。
调查员翻查照片后得到若干线索——某条次轮记录还可能对应某个原清单未列出
的首轮标牌，并标为已核实 / 待核 / 排除。

本模块负责两件纯计算（均不写库）：

* 把已核实线索补入对应次轮记录的候选清单，得到新的整单 payload；
* 已核实线索仍不足时，从待核线索中求一组**核查总代价最低**、核实后即可
  让整单通过（存在覆盖全部必重捕记录、大小 >= r_low 的配对）的线索组合，
  并给出一份可行配对。

数学形式：候选边（原清单 + 已核实）代价为 0，待核线索边代价为其核查代价。
目标配对所用待核边的代价之和最小，等价于最小费用流：源点到必重捕记录连
代价 -B 的边（B 大于全部待核代价之和，保证必重捕优先配上），记录到标牌
连线索代价边，求流量恰为 r_low 的最小费用流。边代价非负时，流量超过
r_low 的可行流费用不会更低，故 r_low 处即全局最优。求解用势函数
（Johnson）+ Dijkstra 的连续最短路增广，规模 80x80 时瞬间完成。
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass

from .estimation import maximum_matching


class NoSolutionError(ValueError):
    """即使核实全部待核线索整单仍无可行配对；携带给人看的原因说明。"""


@dataclass
class Clue:
    """一条线索：次轮记录号 + 原清单未列出的首轮候选标牌 + 正整数核查代价。"""

    record_id: str
    tag: str
    cost: int


@dataclass
class Suggestion:
    """最低代价核查建议及对应的一份可行配对。"""

    total_cost: int
    recommended: list[Clue]  # 建议核实（补入）的待核线索
    pairing: list[tuple[str, str, bool]]  # (次轮记录号, 首轮标牌, 是否来自建议线索)


def apply_verified_clues(payload: dict, verified: list[Clue]) -> dict:
    """把已核实线索补入对应次轮记录的候选清单，返回新 payload（不改原对象）。"""
    merged = copy.deepcopy(payload)
    by_record = {rec["record_id"]: rec for rec in merged["second_round"]}
    for clue in verified:
        by_record[clue.record_id]["candidate_tags"].append(clue.tag)
    return merged


def _min_cost_flow(
    adj_costs: list[list[tuple[int, int]]],
    n_left: int,
    n_right: int,
    required: set[int],
    flow_target: int,
    big: int,
) -> tuple[int, list[tuple[int, int]]] | None:
    """流量恰为 flow_target 的最小费用流（连续最短路增广）。

    返回 (线索总代价, 匹配边列表[(left, right)])；不可行返回 None。
    源点到必重捕左节点的边代价为 -big，故返回的总费用已折算回线索代价。
    """
    n_nodes = n_left + n_right + 2
    src, sink = 0, n_nodes - 1
    graph: list[list[list[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: int) -> None:
        graph[u].append([v, len(graph[v]), cap, cost])
        graph[v].append([u, len(graph[u]) - 1, 0, -cost])

    for left in range(n_left):
        add_edge(src, 1 + left, 1, -big if left in required else 0)
    used_edges: dict[tuple[int, int], int] = {}
    active_rights: set[int] = set()
    for left, edges in enumerate(adj_costs):
        for right, cost in edges:
            add_edge(1 + left, 1 + n_left + right, 1, cost)
            used_edges[(left, right)] = len(graph[1 + left]) - 1
            active_rights.add(right)
    for right in active_rights:
        add_edge(1 + n_left + right, sink, 1, 0)

    # 初始势：图是「源 -> 记录 -> 标牌 -> 汇」的分层 DAG，一遍松弛即可让
    # 所有约化费用非负（负边只有源点出边，无负环）
    inf = 1 << 62
    potential = [0] * n_nodes
    for left in range(n_left):
        potential[1 + left] = -big if left in required else 0
    for right in active_rights:
        potential[1 + n_left + right] = inf
    for left, edges in enumerate(adj_costs):
        for right, cost in edges:
            node = 1 + n_left + right
            potential[node] = min(potential[node], potential[1 + left] + cost)
    if active_rights:
        potential[sink] = min(potential[1 + n_left + r] for r in active_rights)

    flow = 0
    total_cost = 0
    while flow < flow_target:
        dist = [inf] * n_nodes
        prev_node = [-1] * n_nodes
        prev_edge = [-1] * n_nodes
        dist[src] = 0
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for idx, (to, _, cap, cost) in enumerate(graph[u]):
                if cap <= 0:
                    continue
                nd = d + cost + potential[u] - potential[to]
                if nd < dist[to]:
                    dist[to] = nd
                    prev_node[to] = u
                    prev_edge[to] = idx
                    heapq.heappush(heap, (nd, to))
        if dist[sink] == inf:
            return None
        for v in range(n_nodes):
            if dist[v] < inf:
                potential[v] += dist[v]
        v = sink
        while v != src:
            edge = graph[prev_node[v]][prev_edge[v]]
            edge[2] -= 1
            graph[v][edge[1]][2] += 1
            total_cost += edge[3]
            v = prev_node[v]
        flow += 1

    matching = [
        (left, right)
        for (left, right), idx in used_edges.items()
        if graph[1 + left][idx][2] == 0  # 容量被用尽 => 该边在匹配中
    ]
    matched_required = sum(1 for left, _ in matching if left in required)
    clue_cost = total_cost + big * matched_required
    return clue_cost, matching


def find_min_cost_suggestion(
    first_tags: list[str],
    record_ids: list[str],
    candidate_lists: list[list[str]],
    can_be_new: list[bool],
    cap: int,
    pending: list[Clue],
) -> Suggestion:
    """从待核线索中找总代价最低、核实后整单即可通过的一组，并给出可行配对。

    candidate_lists 为已补入已核实线索后的候选清单。沿用原规则：必重捕
    （can_be_new=false 的记录必须配上）、一牌一记录、总体上限
    （r >= M + C - cap）。无解时抛 NoSolutionError 并说明原因。
    """
    m_count = len(first_tags)
    c_count = len(record_ids)
    tag_index = {tag: i for i, tag in enumerate(first_tags)}
    rec_index = {rid: i for i, rid in enumerate(record_ids)}

    pending_cost: dict[tuple[int, int], int] = {}
    pending_clue: dict[tuple[int, int], Clue] = {}
    for clue in pending:
        key = (rec_index[clue.record_id], tag_index[clue.tag])
        pending_cost[key] = clue.cost
        pending_clue[key] = clue

    # 全量图 = 原清单 + 已核实 + 全部待核线索
    full_adj: list[list[int]] = []
    adj_costs: list[list[tuple[int, int]]] = []
    for left, candidates in enumerate(candidate_lists):
        edges = {tag_index[tag] for tag in candidates}
        edges.update(right for (l, right) in pending_cost if l == left)
        ordered = sorted(edges)
        full_adj.append(ordered)
        adj_costs.append([(right, pending_cost.get((left, right), 0)) for right in ordered])

    required = [left for left, flag in enumerate(can_be_new) if not flag]

    matched_required = maximum_matching(full_adj, c_count, m_count, required)
    if matched_required != len(required):
        raise NoSolutionError(
            f"即使核实全部待核线索，{len(required)} 条必重捕记录在一牌一记录约束下"
            f"最多也只能配上 {matched_required} 条，补录无法补足证据，需重新核查原始记录"
        )

    r_max_full = maximum_matching(full_adj, c_count, m_count)
    cap_floor = m_count + c_count - cap
    r_low = max(len(required), cap_floor)
    if r_low > r_max_full:
        raise NoSolutionError(
            f"即使核实全部待核线索，最大重捕数 {r_max_full} 仍低于总体上限 {cap} "
            f"要求的 {cap_floor}，补录无法补足证据，可考虑放宽总体上限或补充更多线索"
        )

    big = 1 + sum(clue.cost for clue in pending)
    result = _min_cost_flow(adj_costs, c_count, m_count, set(required), r_low, big)
    if result is None:  # 前置检查已保证可行，此处为防御性兜底
        raise NoSolutionError("内部不一致：全量图可行但最小费用流未找到配对")
    total_cost, matching = result

    recommended: list[Clue] = []
    pairing: list[tuple[str, str, bool]] = []
    for left, right in sorted(matching):
        key = (left, right)
        if key in pending_clue:
            recommended.append(pending_clue[key])
            pairing.append((record_ids[left], first_tags[right], True))
        else:
            pairing.append((record_ids[left], first_tags[right], False))
    return Suggestion(total_cost=total_cost, recommended=recommended, pairing=pairing)
