# 设计文档 0003：接口契约

| 项 | 值 |
|---|---|
| 状态 | Draft |
| 版本 | 0.8 |
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
@dataclass(frozen=True, slots=True)
class RoutingPoint:
    target_review_rate: float
    actual_review_rate: float
    auto_pass_accuracy: float | None    # 自动通过条目中判对的比例；无自动通过条目时 None
    post_review_accuracy: float         # 1 − 自动通过且错误数 / 全部记录数
    thresholds: dict[ClauseKey, float]

def select_thresholds(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    target_review_rate: float,
) -> dict[ClauseKey, float]

def sweep(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    review_rates: Sequence[float],
) -> tuple[RoutingPoint, ...]
```

- **纯度**：纯函数（route 不 import metrics；「判定正确」口径同 §4，实现中复制该纯函数）
- **前置**：`0 < target_review_rate ≤ 1`；`records` 可含失败记录（失败记录必然转人工）。
  `gold` 用于判定「自动通过后是否仍正确」，故为必选参数
- **候选集**：每类别 = 该类别 OK 记录实际出现过的 `confidence` 值（去重升序）∪ `{0.0, 1.0}`。
  加入 `1.0` 是为了让 `target_review_rate == 1.0` 时能把各类别最高置信度的判定记录也推给人工
- **后置**
  - `select_thresholds` 返回恰好 6 个键，值取自该类别候选集
  - 用返回的阈值重算，转人工率 ≤ `target_review_rate + 1e-9`；转人工率 =
    转人工条数 / 全部记录数，分母含失败记录
  - **确定性**：同一输入两次调用，返回完全相同的字典
  - `target_review_rate` 低于不可约复核率（UNCERTAIN 与失败记录占比）时抛 `ValueError`，
    消息含目标值与实际下界；`sweep` 遇不可满足点同样整体抛 `ValueError`（不跳过、不截断）
  - `sweep` 返回长度等于 `review_rates` 的 tuple 且顺序一致；`review_rates` 升序时
    `actual_review_rate` 单调不减
- **目标**：在满足转人工率约束的前提下，最大化自动通过条目的准确率
- **实现自由**：算法由实现决定（确定性贪心），但必须在模块 docstring 中写明，且满足上述确定性要求
- **已知边界**：`confidence` 恰为 `1.0` 的记录无法被任何候选阈值推给人工（`apply_route` 的 `≥` 语义）

> `select_thresholds` 只应在开发集上调用；评估集只应用已确定的阈值，不得重新选择。
> 否则阈值会拟合到评估数据上，M4 的结果不可信。

## 4. metrics

```python
@dataclass(frozen=True, slots=True)
class ClauseMetrics:
    clause: ClauseKey
    n_total: int                    # 该类别全部记录
    n_ok: int                       # status == OK
    n_failed: int                   # status != OK
    n_decided: int                  # OK 且 present ∈ {PRESENT, ABSENT}
    n_uncertain: int                # OK 且 present == UNCERTAIN
    n_correct: int
    accuracy: float | None          # n_correct / n_decided；n_decided==0 时 None
    accuracy_ci: tuple[float, float] | None
    recall: float | None            # TP / (TP + FN)；分母为 0 时 None
    precision: float | None         # TP / (TP + FP)；分母为 0 时 None
    citation_iou: float | None      # 全部 gold span 的平均最大 IoU；无 gold span 时 None
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
    ece: float | None                      # 全局校准误差（见 §9）
    n_total: int
    n_ok: int
    n_failed: int
    n_uncertain: int
    uncertain_rate: float
    total_cost_cny: float
    mean_input_tokens: float
    mean_output_tokens: float
    p95_latency_ms: float


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float


def compute(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    seed: int,
    iou_threshold: float = 0.5,
    split: str = "",
) -> Report
```

- **纯度**：纯函数（唯一外部依赖 `config.prices_for`，按每条记录的 model 查单价）
- **前置**
  - `records` 的 `query_id` 唯一且全部在 `gold` 中，违反抛 `ValueError`
    （重复的消息含 id 与出现次数；缺失的消息含该 id）；`gold` 有多余条目不报错
    （允许只评测子集）
- **后置**
  - `by_clause` 恰好 6 项，顺序与 `ClauseKey` 定义一致
  - `macro_accuracy` = 各类 `accuracy` 的算术平均，**`None` 的类别跳过**；全为 None 时为 None
  - 相同输入 + 相同 `seed` → 返回对象逐字段相等；`seed` 只影响 `accuracy_ci`
  - 分母口径遵循 PRD 7.1、FR-7：`accuracy` 分母为 `n_decided`；`uncertain_rate`
    分母为 `n_total`（含失败）；恒等式 `n_total == n_decided + n_uncertain + n_failed`
    对每个类别与全局成立
  - 判定正确 = `gold.has_answer` 与 `present`（PRESENT/ABSENT）一致（PRD §7.1 四象限）
  - 引用指标只用 `span.valid == True` 的预测区间；`citation_iou` 对该类别每个 gold span
    取与全部有效预测区间的最大 IoU 后求平均（未命中记 0），`citation_contained`
    为有效预测区间完全落在某 gold span 内的比例，两者无 gold span 时为 None；
    `citation_valid_rate` 分母为该类别全部 span 数（含 invalid），无 span 时为 None
  - `iou_threshold` 本期不参与主指标计算，仅透传到 `Report` 备后续诊断口径使用
  - `Report.ece` 为全局校准误差：pairs 为全部 `n_decided` 条记录的
    (confidence, 是否判对)，跨条款类别聚合后按 §9 计算
  - bootstrap：对各类别 `n_decided` 条判定有放回重采样 1000 次，取 2.5% 与 97.5%
    分位（nearest-rank）；随机源 `random.Random(f"clauseproof:bootstrap:{seed}:{clause.value}")`
  - `total_cost_cny`：`Record` 无 cost 字段，由 `prices_for(record.model)` 按
    input/output token 重算；缓存命中（`cached=True`）与失败记录记 0
    （见 §10：缓存命中未发生计费，token 字段仅用于追溯）
  - `mean_input_tokens` / `mean_output_tokens` 分母为 `n_ok`，分子只含 OK 记录
  - `p95_latency_ms` 仅对 OK 记录取 nearest-rank 分位（`ceil(0.95·n)−1`）；
    失败记录的 `latency_ms` 为占位 0，非测量值，不参与
  - `split` 无法从 `Record` / `GoldLabel` 推出（只有 `RunMeta` 有），由调用方
    （cli）从 `RunMeta` 传入，缺省空串
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
def load_gold_with_stats(*, split: str, n_contracts: int, seed: int) -> tuple[dict[str, GoldLabel], int]
```

- **后置**
  - 同参数两次调用，返回的 `query_id` 序列完全相同（含顺序）
  - `build_queries` 的 `query_id` 集合 == `load_gold` 的键集合
  - `load_gold_with_stats` 返回标注与被跳过的 gold span 数；`load_gold` 为其薄包装
  - `Query` 中不含任何 gold 字段
  - `split ∈ {"dev", "report"}`；两个 split 的合同集合不相交

## 8. storage

```python
def append_record(path: Path, record: Record) -> None
def read_records(path: Path) -> Iterator[Record]
def write_meta(path: Path, meta: RunMeta) -> None
def read_meta(path: Path) -> RunMeta
```

- **后置**：`read_records(append_record(r))` 得到的对象与 `r` 等值（浮点按 0002 §8 的取整规则）
- `append_record` 必须单行写入并立即 flush，保证中断时已写内容完整
- 函数名以本文件为准；实现与文档不一致时，以文档为准并修正实现

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
- **前置**：`pairs` 中每个 confidence 落在 [0, 1]，越界抛 `ValueError`；`n_bins ≥ 1`、`min_bin_size ≥ 1`
- **后置**
  - **等频分桶**：按 confidence 升序排序后等分，除最后一桶可略多外每桶样本数相同；
    样本数不足 `n_bins` 时收缩桶数，保证每桶非空
  - 桶内样本数 < `min_bin_size` 时合并：从头累积，累积量达到 `min_bin_size` 即封桶；
    尾部残余并入前一个桶；全部样本不足时返回单桶。由此除单桶情形外每桶样本数
    ≥ `min_bin_size`
  - `lo` / `hi` 为桶内 confidence 的最小 / 最大值；`mean_confidence` 为桶内
    confidence 均值；`accuracy` 为桶内布尔值为真的比例
  - `expected_calibration_error` = Σ(nᵢ/N)·|accᵢ − confᵢ|，N 为所有桶样本总数；
    非空桶数 < 2 时返回 `None`——单桶时该值退化为 |平均准确率 − 平均置信度|，
    不构成校准误差，返回 `None` 以显式表示「样本不足以评估校准」
  - 空输入 → `reliability_bins` 返回空元组，`expected_calibration_error` 返回 `None`
- **调用关系**：`metrics.compute` 按条款类别调用 `reliability_bins` 与 `expected_calibration_error`（`n_bins=10`、`min_bin_size=30`），结果填入 `ClauseMetrics.ece`；分类别 ECE 在 `n_decided` 不足以形成 ≥2 个桶时为 `None`，对外报告的校准指标使用 `Report.ece`（全局）

## 10. llm（模型调用层）

模型调用层的接口，分布在 `config` 与 `llm` 两个模块：

```python
@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    reasoning_effort: str      # 合法取值随 protocol 而定，见下
    temperature: float
    timeout_s: float
    cache_dir: Path
    protocol: str              # "anthropic" | "openai"，由 CLAUSEPROOF_PROTOCOL 决定


@dataclass(frozen=True, slots=True)
class Prices:
    input_cny_per_mtok: float
    output_cny_per_mtok: float
    cached_cny_per_mtok: float


def load_config(env_file: Path | None = None) -> LLMConfig: ...
def prices_for(model: str) -> Prices: ...


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_ms: float
    cached: bool
    cache_written: bool
    stop_reason: str | None
    cost_cny: float


class Transport(Protocol):
    def post(self, url: str, payload: dict, headers: dict[str, str]) -> dict: ...


class HttpxTransport:
    """真实传输（httpx 直发）。项目中唯一允许发起网络请求的类（0001 §5 原则二）。"""
    def __init__(self, timeout_s: float) -> None: ...
    def post(self, url: str, payload: dict, headers: dict[str, str]) -> dict: ...


@dataclass(frozen=True, slots=True)
class BackendResult:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    stop_reason: str | None


class Backend(Protocol):
    def request(self, config: LLMConfig, prompt: str, system: str | None,
                max_tokens: int) -> tuple[str, dict, dict[str, str]]:
        """返回 (url, payload, headers)。"""

    def parse(self, response: dict) -> BackendResult:
        """把端点响应解析为统一结构。"""


class AnthropicBackend: ...
class OpenAIBackend: ...


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: Transport | None = None,
        backend: Backend | None = None,   # 缺省按 config.protocol 选择
        cache_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None: ...

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...
```

- **后置**
  - 缓存命中时不发起请求，返回 `cached=True`
  - 磁盘缓存命中时 `cost_cny=0.0`（未发生计费），`latency_ms` 为本次本地读取耗时；input/output token 字段保留原始响应值，仅用于追溯，不参与成本计算
  - 缓存写失败（磁盘满、权限等）时 `cache_written=False`，结果照常返回；写入成功或缓存命中时为 `True`。`cache_written=False` 只有一种情形：写入失败（异常），不影响本次结果的有效性
  - 缓存目录来自运行时配置（`LLMConfig.cache_dir`），默认为 `<项目根>/.cache/llm`；不存在「默认关闭缓存」的路径
  - 重试耗尽抛 `RuntimeError`；4xx（非 429）不重试
  - 成本按「新输入 / 缓存命中 / 输出」三段计算，异常与日志不含 `api_key`
  - 缓存键包含 `protocol`，不含 backend 类型与 `base_url` / `api_key`（后三者不改变
    同一协议下同一模型的应答语义）
- **协议与端点（T-006）**
  - `protocol` 由 `CLAUSEPROOF_PROTOCOL` 决定，默认 `"openai"`；未知值在 `load_config` 抛 `RuntimeError`，`LLMClient` 构造期兜底校验
  - **anthropic**：URL = `base_url + /v1/messages`；头含 `Authorization: Bearer`、`anthropic-version`、`content-type`；system 为顶层 `system` 字段；`reasoning_effort` 合法值 `{low, high, max}`，实测被该端点忽略（发送无害），实际档位为服务端默认（max）。因此成本以响应 `usage` 实测为准，不按档位估算
  - **openai**：URL 由代码内置 `https://open.bigmodel.cn/api/paas/v4/chat/completions`，`ANTHROPIC_BASE_URL` 可缺省；头含 `Authorization: Bearer`、`content-type`（无 `anthropic-version`）；system 进 `messages` 首条；`reasoning_effort` 合法值 `{none, minimal, low, medium, high, xhigh, max}`，该端点会校验，非法值返回 400
  - 两种协议下 `ANTHROPIC_AUTH_TOKEN` 均为必填
  - **响应归一化**：`OpenAIBackend.parse` 把 `finish_reason == "length"` 归一化为 `"max_tokens"`（上层空正文保护依赖此值）；`content` 为 null 按空串处理；`usage.prompt_tokens_details.cached_tokens` 缺失按 0 处理。字段映射详见 tasks/T-006.md
  - 缓存键包含 protocol；切换协议即视为不同请求，不会跨协议命中缓存

> 网络传输遵循 httpx 默认的代理环境变量行为；代理配置错误会表现为构造期失败，而非请求期失败。

> 认证头使用 `Authorization: Bearer`，实测可被智谱 Anthropic 兼容端点接受（2026-10-26）。若将来切换端点后返回 401，优先改为 `x-api-key`。

## 11. 契约自检要求

以下测试是 P2.5 的交付内容，不是可选项：

1. **假 Judge 端到端**：用返回固定结果的 `FakeJudge` 驱动 `runner`，产出合法 JSONL，全程不触碰真实模型
2. **每条不变式一个测试**：0002 §7 的每一行对应至少一个断言
3. **每个后置条件一个测试**：尤其是 `select_thresholds` 的确定性与 `dataset` 的两次调用一致性
4. **往返一致**：`storage.append_record` → `read_records` 等值

## 12. 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-10-21 | 初稿 |
| 0.2 | 2026-10-26 | `judge` 后置条件补充 `parse_ok` 语义（见设计 0004 §3） |
| 0.3 | 2026-10-26 | 补 llm 模块接口与 `cache_written` 语义 |
| 0.4 | 2026-10-27 | §10 增加 Backend 抽象（Anthropic / OpenAI）与 `LLMConfig.protocol`，端点与归一化规则见「协议与端点」（T-006） |
| 0.5 | 2026-10-27 | 缓存键加入 protocol（跨协议不得命中旧缓存）；`LLMConfig.protocol` 默认值与 load_config 统一为 "openai" |
| 0.6 | 2026-10-27 | §3 修正 `select_thresholds` 签名（增加 `gold` 参数），补 `RoutingPoint` 与 `sweep`；候选集含 `{0.0, 1.0}`，记录 confidence==1.0 的已知边界（T-008） |
| 0.6 | 2026-10-27 | §4 Report/ClauseMetrics 扩展 T-007 全部字段并落定口径（成本由 token 重算、缓存命中记 0；p95 仅 OK 记录；split 由调用方传入）；§9 细化等频切分与桶合并规则 |
| 0.7 | 2026-10-27 | §9 ECE 在非空桶数 <2 时返回 None（单桶不构成校准误差）；§4 Report 新增全局 `ece`，对外报告的校准指标以全局口径为准 |
| 0.8 | 2026-10-29 | §3 增补阈值与评估集分离约束：`select_thresholds` 只应在开发集上调用（T-009） |