# 协议切换 A/B 对照：anthropic / 实际 max vs openai / low

| 字段 | smoke-t005 | smoke-t006 |
|---|---|---|
| run_id | smoke-t005 | smoke-t006 |
| 日期 | 2026-10-27 | 2026-10-27 |
| commit | dev（T-005 快照，未提交） | dev（T-006 协议切换后快照，未提交） |
| 模型 | glm-5.3 | glm-5.3 |
| 端点协议 | anthropic（`/v1/messages`） | openai（chat/completions，代码内置 URL） |
| reasoning_effort | 发送 low 但被端点忽略，实际 max | low，生效 |
| 数据集 | dev / 3 份合同 / seed=42 / 18 条（6 类 × 3） | 同左 |
| 结论 | — | 采用 openai 协议 + reasoning_effort=low |

## 背景

T-005 冒烟发现：`reasoning_effort` 在 Anthropic 兼容端点上被静默忽略，实际恒为 max 档，
18 条里 5 条因思考耗尽 8192 token 导致正文为空、1 条传输超时。
实测 OpenAI 兼容端点会校验该字段（非法值返回 400 并列出合法取值）。
T-006 因此抽象出 Backend 并切换协议，本记录是切换前后的对照。

两次运行使用完全相同的 18 条查询（同 split、同 seed、同 prompt），唯一变量是端点协议
及其带来的 reasoning_effort 实际档位。

## 对比表

两个口径：**全量**（分母固定为 18，对应 PRD FR-7）与**已判定**（分母为明确判定条数）。

| 指标 | smoke-t005 | smoke-t006 |
|---|---|---|
| OK / LLM_ERROR | 12 / 6 | 18 / 0 |
| 明确判定（有/无） | 10 | 14 |
| 判对 | 9 | 12 |
| 模糊（uncertain） | 2 | 4 |
| 全量正确率（分母 18） | 50% | 67% |
| 已判定准确率 | 90% | 86% |
| 平均 input token（OK 条目） | 7152 | 7861 |
| 平均 output token（OK 条目） | 1479 | 94 |
| output token 中位数 / p95（OK 条目） | 890 / 3565 | 74.5 / 240 |
| 平均延迟（OK 条目） | 26.4 s | 6.6 s |
| 总成本（OK 条目 usage 重算） | ¥1.18 | ¥1.18 |
| 单条成本 | ¥0.10 | ¥0.07 |

判对口径：gold 只有「有/无」两态（`GoldLabel.has_answer`），判对 = 明确判定
（present/absent）与 has_answer 一致；uncertain 计为模糊，不计对也不计错。

### 分类别明细（每类 3 条）

| 条款 | smoke-t005 | smoke-t006 |
|---|---|---|
| GOVERNING_LAW | 失败 对 对 | 模糊 对 对 |
| CAP_ON_LIABILITY | 对 失败 失败 | 对 模糊 模糊 |
| UNCAP_LIABILITY | 对 错 失败 | 对 错 对 |
| ANTI_ASSIGNMENT | 对 失败 失败 | 模糊 对 对 |
| TERMINATION_FOR_CONVENIENCE | 对 模糊 模糊 | 对 错 对 |
| NOTICE_PERIOD_TO_TERMINATE_RENEWAL | 对 对 对 | 对 对 对 |
| 宏平均准确率（已判定口径，无判定类别跳过） | 92% | 89% |

95% CI：n=18（每类仅 3 条）不适用，留待 M1（300 条）。

## 耗时

两次运行均未单独记录墙钟时间；以 OK 条目延迟合计为参考下界：
smoke-t005 约 317 s（12 × 26.4 s），smoke-t006 约 120 s（18 × 6.6 s），
不含失败条目的等待与进程开销。

## 分析

### 1）分母陷阱：90% 与 86% 不可直接比较

t005 有 6 条失败、只对 10 条给出明确判定；t006 对 14 条给出判定。
少答会推高「已判定准确率」——把分母固定为全部 18 条后，判对数从 9 提升到 12
（全量正确率 50% → 67%）。这条对应 PRD FR-7「分母口径必须固定」。

### 2）t005 的 6 条失败构成

- 5 条 `stop_reason=max_tokens` 且正文为空（思考耗尽 8192 token）
- 1 条传输层读超时（180 s）

协议切换后两类失败全部消失（t006 为 18/18 OK）。

### 3）逐条差异（18 条中 9 条不同）

- 6 条为 t005 失败、t006 给出判定，属于「失败转成功」，无矛盾
- 3 条双 OK 但判定翻转：
  - RISEED ANTI_ASSIGNMENT：present(0.9) → uncertain(0.6)
  - INTERSECTENT TERMINATION_FOR_CONVENIENCE：uncertain(0.8) → present(0.75)
  - BEYONDCOM TERMINATION_FOR_CONVENIENCE：uncertain(0.85) → absent(0.8)

（两次都 OK 的 12 条中判定一致 9 条，即上表 3 条翻转。）

### 4）t006 的 2 条判错

两条均为假阳性（gold = 无，模型判 present），但**不在同一条款**（此处与本次复核前的
口头结论不同，以 JSONL + gold 复核为准）：

| 条目 | 条款 | t005 | t006 | 性质 |
|---|---|---|---|---|
| INTERSECTENT | UNCAP_LIABILITY | 同样判错（present） | present(0.85) | **稳定错误**：两次运行一致，与协议/档位无关 |
| INTERSECTENT | TERMINATION_FOR_CONVENIENCE | uncertain(0.8) | present(0.75) | **新增假阳性**：低档下从模糊转为误判 |

TERMINATION_FOR_CONVENIENCE 的假阳性对应 PRD §3.2 该条款的边界「无违约、无需理由」，
推测模型把「协商不成可终止」误判为任意终止权。

## 结论

采用 openai 协议 + reasoning_effort=low。依据按权重：

1. 失败率 33% → 0
2. 全量判对数 9 → 12
3. 延迟 −75%、输出 token −94%
4. 已判定准确率的差异在 n=10/14 下不显著

## 风险与后续观察

- 样本极小（18 条、可比仅 12 条），结论待 M1（300 条）复核
- TERMINATION_FOR_CONVENIENCE 的新增假阳性需在 M1 时按类别单独观察；
  若显著高于其他类别，需收紧提示词中的边界表述
- low 档在贴边条款上的判定稳定性可能弱于 max 档（3 条翻转均发生在 t005 本就
  低置信 ≤0.9 的贴边条款上），需在 M1 的分类别准确率上验证
- UNCAP_LIABILITY 的稳定假阳性与档位无关，属提示词/边界问题，M1 一并观察
- 总成本几乎持平是口径造成的：t005 的 6 条失败调用未计入（失败记录计量为 0），
  真实开销 t005 更高

## 复现方式

两条 run 均为：`build_queries(split="dev", n_contracts=3, seed=42)` 的 18 条，
阈值统一 0.8，`runner.run` 落盘到 `results/`。

- **smoke-t005**（anthropic 协议 / reasoning_effort 发送 low 被忽略、实际 max）：
  `CLAUSEPROOF_PROTOCOL=anthropic`（T-006 之前的代码直接运行即可），run_id=smoke-t005
- **smoke-t006**（openai 协议 / reasoning_effort=low 生效）：
  `CLAUSEPROOF_PROTOCOL=openai`、`CLAUSEPROOF_REASONING_EFFORT=low`，
  run_id=smoke-t006，缓存目录用独立的 `.cache/llm-smoke-t006`（缓存键含 protocol，
  且避免命中 t005 写入的旧条目）

对应文件：`results/smoke-t005.jsonl`、`results/smoke-t006.jsonl`
（各配 `results/<run_id>.meta.json`）。

## 复核说明

本记录全部数字对照 `results/smoke-t005.jsonl`、`results/smoke-t006.jsonl` 与
`load_gold(split="dev", n_contracts=3, seed=42)` 复算。与任务指令给出的数字相比，
发现一处不一致并已按文件修正：

- 指令称 t006 的 2 条错误「均在 Termination For Convenience」；实际为
  UNCAP_LIABILITY 与 TERMINATION_FOR_CONVENIENCE 各 1 条（见「分析」第 4 节）。
  其中 UNCAP_LIABILITY 在 t005 中同样判错，是稳定错误而非协议切换引入。

其余指标（状态分布、判定分布、判对数、token、延迟、成本）与指令给出的全部一致。
