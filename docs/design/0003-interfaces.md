# 设计文档 0003：接口契约

| 项 | 值 |
|---|---|
| 状态 | Draft |
| 版本 | 0.2 |
| 上游 | [设计 0001](0001-architecture.md)、[设计 0002](0002-data-model.md) |

## 1. 通用约定

适用于本文件中的全部函数：

- 所有公开函数必须带完整类型注解
- 参数默认值不得依赖当前时间、随机数或环境变量
- 纯函数模块（`locate` / `route` / `metrics` / `calibration`）不得 import `llm` / `storage` / `runner`
- 抛错只用两种：`ValueError`（契约违反）、`RuntimeError`（环境或 IO 失败）。暂不引入自定义异常层级

## 2. locate

```python
def locate(quotes: Sequence[str], text: str) -> tuple[Span, ...]
```

- **纯度**：纯函数，无 IO
- **前置**：`text` 非空
- **后置**
  - `len(返回值) == len(quotes)`，且顺序一一对应
  - 空字符串 quote → `valid=False`、`occurrences=0`，不抛错
  - `valid` 为真当且仅当 `occurrences ≥ 1`
  - `valid` 时 `start = text.find(quote)`、`end = start + len(quote)`
  - `occurrences = text.count(quote)`

> 空串不抛错是刻意容忍：模型可能输出空引用，这属于可统计的失败，不应中断运行。

## 3. route

```python
def apply_route(judgment: Judgment, threshold: float) -> Route
```

- **纯度**：纯函数
- **前置**：`0.0 ≤ threshold ≤ 1.0`
- **后置**
  - `judgment.present is UNCERTAIN` → `TO_HUMAN`
  - 否则 `confidence ≥ threshold` → `AUTO_PASS`，反之为 `TO_HUMAN`
- 引用质量**不参与**路由决策（降级已在 `judge` 完成）

```python
def select_thresholds(
    records: Sequence[Record],
    target_review_rate: float,
) -> dict[ClauseKey, float]
```

- **纯度**：纯函数
- **前置**：`records` 全部为 `status == OK`；`0 < target_review_rate ≤ 1`
- **后置**
  - 返回恰好 6 个键
  - 用返回的阈值重算，转人工率 ≤ `target_review_rate + 1e-9`
  - **确定性**：同一输入两次调用，返回完全相同的字典
  - 每个类别的阈值必须取自该类别实际出现过的 `confidence` 值
- **目标**：在满足转人工率约束的前提下，最大化自动通过条目的准确率
- **实现自由**：算法由实现决定（建议确定性贪心），但必须在模块 docstring 中写明，且满足上述确定性要求

## 4. metrics

```python
@dataclass(frozen=True, slots=True)
class ClauseMetrics:
    clause: ClauseKey
    n_total: int
    n_decided: int                 # present ∈ {PRESENT, ABSENT}
    n_uncertain: int
    n_correct: int
    accuracy: float | None         # n_decided == 0 时为 None
    accuracy_ci: tuple[float, float] | None
    recall: float | None
    precision: float | None
    citation_iou: float | None
    citation_contained: float | None
    citation_valid_rate: float | None
    ece: float | None


@dataclass(frozen=True, slots=True)
class Report:
    run_id: str
    split: str
    seed: int
    iou_threshold: float
    by_clause: tuple[ClauseMetrics, ...]   # 恰好 6 项，顺序同 ClauseKey 定义
    macro_accuracy: float | None
    n_total: int
    n_uncertain: int
    uncertain_rate: float


def compute(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    seed: int,
    iou_threshold: float,
) -> Report
```

- **纯度**：纯函数
- **前置**：`records` 中的 `query_id` 与 `gold` 的键完全对应
- **后置**
  - `by_clause` 恰好 6 项，顺序与 `ClauseKey` 定义一致
  - `macro_accuracy` = 各类 `accuracy` 的算术平均，**`None` 的类别跳过**
  - 相同输入 + 相同 `seed` → 返回对象逐字段相等
  - 分母口径遵循 PRD 7.1、FR-7
- `Report` 与 `ClauseMetrics` 定义在 `metrics` 模块中，不进 `models`（只有 `metrics` 产出、`cli` 消费）

## 5. judge（唯一模型出口，也是测试注入点）

```python
@dataclass(frozen=True, slots=True)
class CallMeta:
    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool


class Judge(Protocol):
    def judge(self, query: Query) -> tuple[Judgment, CallMeta]: ...
```

- **后置**
  - 返回的 `Judgment` 必须已通过不变式校验
  - 模型输出无法解析时，返回 `Presence.UNCERTAIN`、`confidence=0.0`、`quotes=()`，**不抛错**
  - 解析成功时 `parse_ok=True`
  - 模型输出无法解析时，返回 `Presence.UNCERTAIN`、`confidence=0.0`、`quotes=()`、`parse_ok=False`，**不抛错**
- **`Protocol` 是刻意选择的**：`runner` 只依赖这个协议，测试时可注入假实现，无需真实模型（P2.5 的自检依赖这一点）

## 6. runner

```python
def run(
    *,
    queries: Sequence[Query],
    judge: Judge,
    run_meta: RunMeta,
    thresholds: Mapping[ClauseKey, float],
    out_dir: Path,
    resume: bool = True,
) -> Path
```

- **后置**
  - 产出 `out_dir/{run_id}.jsonl` 与 `out_dir/{run_id}.meta.json`，返回 jsonl 路径
  - 记录数等于 `len(queries)`（`resume=True` 时包含运行前已存在的记录）
  - `resume=True` 时，已存在于文件中的 `query_id` 不再调用 `judge`
  - 单条 `judge` 抛 `RuntimeError` → 该条记为 `status=LLM_ERROR`，**继续处理后续条目**
  - 单条超出上下文 → 该条记为 `status=CONTEXT_OVERFLOW`，不调用模型
  - 被中断时，已写入的记录必须完整可解析

## 7. dataset

```python
def build_queries(*, split: str, n_contracts: int, seed: int) -> tuple[Query, ...]
def load_gold(*, split: str, n_contracts: int, seed: int) -> dict[str, GoldLabel]
```

- **后置**
  - 同参数两次调用，返回的 `query_id` 序列完全相同（含顺序）
  - `build_queries` 的 `query_id` 集合 == `load_gold` 的键集合
  - `Query` 中不含任何 gold 字段
  - `split ∈ {"dev", "report"}`；两个 split 的合同集合不相交

## 8. storage

```python
def append(path: Path, record: Record) -> None
def read(path: Path) -> Iterator[Record]
def write_meta(path: Path, meta: RunMeta) -> None
def read_meta(path: Path) -> RunMeta
```

- **后置**：`read(append(r))` 得到的对象与 `r` 等值（浮点按 0002 §8 的取整规则）
- `append` 必须单行写入并立即 flush，保证中断时已写内容完整

## 9. calibration

```python
@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    pairs: Sequence[tuple[float, bool]],
    *,
    n_bins: int = 10,
    min_bin_size: int = 30,
) -> tuple[CalibrationBin, ...]


def expected_calibration_error(
    bins: Sequence[CalibrationBin],
) -> float | None
```

- **纯度**：纯函数
- **前置**：`pairs` 中每个 confidence 落在 [0, 1]
- **后置**
  - **等频分桶**：按 confidence 升序排序后等分，每桶样本数相同（最后一桶可略多）
  - 桶内样本数 < `min_bin_size` 时与相邻桶合并
  - `mean_confidence` 为桶内 confidence 均值；`accuracy` 为桶内布尔值为真的比例
  - `expected_calibration_error` = Σ(nᵢ/N)·|accᵢ − confᵢ|，N 为所有桶样本总数
  - 空输入 → `reliability_bins` 返回空元组，`expected_calibration_error` 返回 `None`
- **调用关系**：`metrics.compute` 按条款类别调用 `reliability_bins` 与 `expected_calibration_error`，结果填入 `ClauseMetrics.ece`

## 10. 契约自检要求

以下测试是 P2.5 的交付内容，不是可选项：

1. **假 Judge 端到端**：用返回固定结果的 `FakeJudge` 驱动 `runner`，产出合法 JSONL，全程不触碰真实模型
2. **每条不变式一个测试**：0002 §7 的每一行对应至少一个断言
3. **每个后置条件一个测试**：尤其是 `select_thresholds` 的确定性与 `dataset` 的两次调用一致性
4. **往返一致**：`storage.append` → `read` 等值

## 11. 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-10-21 | 初稿 |
| 0.2 | 2026-10-26 | `judge` 后置条件补充 `parse_ok` 语义（见设计 0004 §3） |