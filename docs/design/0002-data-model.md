# 设计文档 0002：数据模型契约

| 项 | 值 |
|---|---|
| 状态 | Draft |
| 版本 | 0.3 |
| 上游 | [PRD-0001](../prd/0001-contract-clause-review.md)、[设计 0001](0001-architecture.md) |

## 1. 约束

本文件定义跨模块传递的全部数据结构。三条硬性约束：

1. **不可变**：构造后不得修改，任何"修改"必须产生新对象
2. **构造即校验**：违反不变式立即抛错，不允许非法状态流入下游
3. **gold 隔离**：`Query` 与 `Judgment` 中不存在任何 gold 字段

## 2. 枚举

```python
class ClauseKey(Enum):
    """六类条款。全项目唯一允许出现 CUAD 原始名称的地方。"""
    GOVERNING_LAW = "Governing Law"
    CAP_ON_LIABILITY = "Cap On Liability"
    UNCAP_LIABILITY = "Uncapped Liability"
    ANTI_ASSIGNMENT = "Anti-Assignment"
    TERMINATION_FOR_CONVENIENCE = "Termination For Convenience"
    NOTICE_PERIOD_TO_TERMINATE_RENEWAL = "Notice Period To Terminate Renewal"


class Presence(Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Route(Enum):
    AUTO_PASS = "auto_pass"
    TO_HUMAN = "to_human"


class Status(Enum):
    OK = "ok"
    LLM_ERROR = "llm_error"
    CONTEXT_OVERFLOW = "context_overflow"
```

**规则**：CUAD 的原始条款名只出现在 `ClauseKey` 的定义里。其他模块一律使用 `ClauseKey`，不得再写字符串字面量。这样可以保证条款名的变更只改一处。

## 3. 核心类型

```python
@dataclass(frozen=True, slots=True)
class Query:
    """模型看到的全部信息。刻意不含 gold。"""
    query_id: str        # f"{contract_id}::{clause.name}"
    contract_id: str     # CUAD 合同标题
    clause: ClauseKey
    contract_text: str


@dataclass(frozen=True, slots=True)
class Judgment:
    """模型输出的解析结果。"""
    present: Presence
    quotes: tuple[str, ...]
    confidence: float
    raw_output: str      # 模型原始输出，保留以供审计
    parse_ok: bool       # False = 解析失败，此时 present 被强制为 UNCERTAIN


@dataclass(frozen=True, slots=True)
class Span:
    """由 quote 在原文中定位得出的区间。"""
    quote: str
    start: int | None
    end: int | None
    valid: bool
    occurrences: int     # 该 quote 在原文中出现的次数
```

## 4. 落盘单元

```python
@dataclass(frozen=True, slots=True)
class Record:
    """JSONL 的一行。每条记录自包含，可独立解析。"""
    run_id: str
    query_id: str
    contract_id: str
    clause: ClauseKey
    status: Status
    judgment: Judgment | None
    spans: tuple[Span, ...]
    route: Route | None
    model: str
    commit: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool
    created_at: str      # ISO 8601，UTC
    error_detail: str | None
```

`status != OK` 时，`judgment`、`spans`、`route` 必须为空 —— 失败的条目不允许产生半成品结果。

## 5. 评测侧类型

```python
@dataclass(frozen=True, slots=True)
class GoldSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class GoldLabel:
    """只由 metrics 加载，不进入 Query。"""
    query_id: str
    has_answer: bool
    spans: tuple[GoldSpan, ...]
```

## 6. 运行元信息

```python
@dataclass(frozen=True, slots=True)
class RunMeta:
    run_id: str
    commit: str
    model: str
    config_digest: str   # 运行配置的哈希
    seed: int
    split: str           # "dev" | "report"
    n_contracts: int
    strategy: str
    started_at: str      # ISO 8601，UTC
    n_queries: int
```

## 7. 不变式

检查顺序：parse_ok 相关约束优先于 present / quotes 约束。这样解析失败的记录报出的是「解析失败」，而不是被下游约束抢先报错。
违反任一条立即抛 `ValueError`，错误信息必须指出违反的字段与期望。

| 类型 | 不变式 |
|---|---|
| `Query` | `query_id == f"{contract_id}::{clause.name}"` |
| `Query` | `contract_id` 非空；`contract_text` 非空 |
| `Query` | `contract_id` 不得包含 `::` |
| `Judgment` | `0.0 ≤ confidence ≤ 1.0` |
| `Judgment` | `present == ABSENT` → `quotes == ()` |
| `Judgment` | `present == PRESENT` → `len(quotes) ≥ 1` |
| `Judgment` | `parse_ok == False` → `present is UNCERTAIN and confidence == 0.0` |
| `Span` | `valid` → `start`、`end` 均非 None 且 `0 ≤ start < end` |
| `Span` | `not valid` → `start is None and end is None` |
| `Span` | `occurrences ≥ 0`；`valid` → `occurrences ≥ 1` |
| `Record` | `status == OK` → `judgment` 与 `route` 均非 None |
| `Record` | `status != OK` → `judgment is None and spans == () and route is None` |
| `Record` | `input_tokens ≥ 0`、`output_tokens ≥ 0`、`latency_ms ≥ 0` |
| `Record` | `run_id`、`model`、`commit`、`prompt_hash`、`created_at` 均非空 |
| `Record` | `status == OK` → `len(spans) == len(judgment.quotes)` |
| `GoldLabel` | `has_answer == (len(spans) ≥ 1)` |
| `GoldSpan` | `0 ≤ start < end`，且 `text` 非空 |
| `RunMeta` | 0 ≤ seed < 2**63；split ∈ {"dev","report"}；n_contracts > 0；strategy 非空 |

> `Judgment` 对 `UNCERTAIN` **不设 quotes 约束** —— 模糊态允许带引用，也允许不带。

## 8. 序列化约定

**文件布局**：一次运行产出两个文件，同目录同前缀。

```
<run_id>.meta.json     RunMeta 单对象
<run_id>.jsonl         一行一个 Record
```

**JSONL 规则**

- 一行一个 `Record`，单行可独立解析，无跨行依赖
- 字段顺序固定，便于 diff
- 枚举序列化为其字符串值（如 `"uncertain"`）
- `tuple` 序列化为数组；`None` 序列化为 `null`
- 浮点：`confidence` 保留 4 位，`latency_ms` 保留 1 位；**在序列化时取整，内存中保持全精度**
- 时间：ISO 8601，UTC，形如 `2026-10-21T12:00:00Z`
- `prompt_hash`：对实际发送的完整 prompt（system 与 user 用两个换行拼接）取 SHA-256
- `commit`：运行开始时 `git rev-parse HEAD` 的结果；不可得时写 `"unknown"`

**`created_at` 与 `started_at` 是唯一允许的不确定字段。** NFR-1 的逐位一致比较必须排除它们，实现时按字段名排除，不按位置。

## 9. 不做什么

- 不在数据模型中放业务逻辑（路由计算、指标计算都放在各自模块）
- 不引入 ORM / 数据库
- 不做 schema 版本迁移：字段只增不改，改名视为破坏性变更，需走契约评审
- 不允许 `Query` 出现 gold 相关字段，即使标注为可选

## 10. 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-10-21 | 初稿 |
| 0.2 | 2026-10-26 | `Judgment` 增加 `parse_ok`，区分解析失败与模型真不确定（见设计 0004 §3） |
