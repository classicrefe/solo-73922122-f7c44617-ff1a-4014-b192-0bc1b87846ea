# 封闭水域鱼类种群估计 API

两轮标志重捕（Petersen 设计）的种群估计服务。第二轮部分标牌模糊时，通过二分图
配对求出**可行重捕数的上下界**，再对每个可行重捕数反演**双侧等尾超几何检验**
得到总体数量的 95% 置信集。FastAPI + SQLite，Docker Compose 一键启动。

## 业务模型

记首轮捕获并标记 **M** 尾（标牌唯一），次轮 **C** 条记录（记录号唯一）。每条次轮
记录给出：

* `candidate_tags`：可能对应的首轮标牌列表（空列表 = 完全读不到）；
* `can_be_new`：该个体是否可能是首轮未捕获的新个体。

约束：**一个首轮标牌最多配给一条次轮记录**（一一匹配）。

### 1. 可行重捕数区间（二分图匹配）

把次轮记录当左节点、首轮标牌当右节点，候选关系即边：

* `can_be_new=false` 的记录**必须**配上标牌。先在这些强制节点上求最大匹配，
  若不能全部匹配 → **无可行配对**，整单拒绝（422）；
* 强制节点数即重捕数下界 `r_min`；整图最大匹配（Kuhn 增广路）给出 `r_max`。
  区间内每个整数 r 都可由某个合法配对达到。

### 2. 总体上限约束

重捕 r 时两轮涉及的不同个体数为 `M + C - r`，故总体 N 必须满足
`M + C - r ≤ N`，即 `r ≥ M + C - cap`。该下限若高于 `r_max` → 无可行配对。

### 3. 95% 置信集（等尾超几何检验反演）

次轮从总体 N 中抽 C 尾，其中带标数
`X ~ Hypergeom(N, K=M, draws=C)`。对观测重捕数 r，双侧各 2.5%、且两尾均含
观测值：

* 下尾 `P(X ≤ r) ≥ 0.025`
* 上尾 `P(X ≥ r) ≥ 0.025`

对每个可行 r，在 N 的可行范围 `[M+C-r, cap]` 内反演检验。超几何 CDF 关于 N
单调不减、上尾单调不增（已有单调性测试守护），故用**两次二分查找**定位通过检验
的 N 区间；所有可行 r 的区间并集即 95% 置信集，返回其中最小/最大总体数量。

### 4. 次线性计算量

* 可行 r 至多 80 个；每个 r 两次二分，各 `O(log cap)` 次尾概率求值；
* 每次尾概率只遍历超几何支撑点（≤ 81 个），对数概率用锚点 lgamma + 相邻项
  比值递推，避免组合数溢出并把 lgamma 调用降为常数次；
* 总复杂度 **O(k² · log U)**（k ≤ 80 为记录数，U 为总体上限），U 只出现在
  对数项中。实测 U=10 亿、80×80 完全模糊图：约 0.6 秒、5009 次尾概率求值。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/surveys/{survey_id}/estimates` | 提交/整单更正，返回估计（201）或拒绝（422） |
| GET | `/surveys/{survey_id}/current` | 现行清单与结论 |
| GET | `/surveys/{survey_id}/versions` | 全部历史版本（含失败单据与原因） |
| GET | `/surveys/{survey_id}/versions/{v}` | 指定版本（失败后仍可查原版） |

交互式文档：启动后访问 `http://localhost:8000/docs`。

### 请求示例

```json
{
  "first_round": [{"tag": "T01"}, {"tag": "T02"}, {"tag": "T03"}],
  "second_round": [
    {"record_id": "R1", "candidate_tags": ["T01"], "can_be_new": false},
    {"record_id": "R2", "candidate_tags": ["T01", "T02", "T03"], "can_be_new": true},
    {"record_id": "R3", "candidate_tags": [], "can_be_new": true}
  ],
  "population_cap": 5000
}
```

校验规则：每轮 1～80 条；首轮标牌、次轮记录号各自唯一；`population_cap`
为 1～1,000,000,000；候选标牌必须存在于首轮清单且单条记录内不重复。

## SQLite 版本化存储

* `surveys`：调查单；`survey_versions`：不可变版本（提交原文 + 结果/错误 +
  `is_current` 指针）。
* **整单更正 = 追加新版本**。估计成功时在单事务内切指针、归档旧版；
  估计失败也写入 `status=failed` 版本并保留原始提交，**不改动现行指针**，
  失败后原版结论照常可查。
* WAL 模式 + busy_timeout，保证读写并发下的锁安全。

## 运行方式

### Docker Compose（推荐）

```bash
docker compose up --build
# 数据持久化在命名卷 survey-data，容器内 /data/surveys.db
```

### 本地直接运行

```bash
pip install -r requirements.txt
SURVEY_DB_PATH=./data/surveys.db uvicorn app.main:app --port 8000
```

### 在 /docs 演示「标牌辨认前后」的估计对比

服务启动后另开终端：

```bash
pip install -r requirements-dev.txt   # 仅演示/测试需要 httpx
python scripts/demo.py
```

脚本依次演示：① 标牌模糊（重捕数是区间 2..3）→ ② 整单更正、标牌核实
（重捕数确定为 3，换版）→ ③ 错误更正被 422 拒绝但失败留痕、现行版本不变 →
④ 可能零重捕且上限过紧时置信集为空的明确提示。同样可在 Swagger UI 手工操作。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

26 个测试覆盖：尾概率与 **scipy 精确对照**（含 10 亿参数）、CDF 单调性、
置信区间与 scipy 暴力扫描逐点对照、配对上下界、上限约束、无可行配对、
零重捕/空置信集提示、10 亿上限耗时与求值计数、API 全流程与版本化存储。

## 目录结构

```
app/
  estimation.py   # 二分图匹配 + 超几何尾概率 + 二分反演（纯函数引擎）
  schemas.py      # Pydantic 请求/响应模型与输入校验
  storage.py      # SQLite 版本化存储（事务换版、失败留痕）
  main.py         # FastAPI 路由
scripts/demo.py   # /docs 配套端到端演示
tests/            # pytest（含 scipy 交叉验证）
Dockerfile, docker-compose.yml
```

## 已知边界与取舍

* 置信集是各可行 r 对应区间的并集，按要求只返回并集的最小/最大值，同时响应中
  保留每个 r 的独立区间便于追溯；
* 尾概率在对数空间计算，10 亿量级参数无溢出；递推累积误差与 scipy 的偏差
  < 1e-10；
* 当前环境未安装 Docker，Dockerfile/compose 已编写但**未在本机实际构建运行**，
  服务与演示均通过本地 uvicorn 实测。
