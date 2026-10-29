# ClauseProof

> 合同条款审阅 Agent：在固定人工复核预算下，把审阅准确率推到最高。

给一份合同 + 一份审阅清单，逐条输出：判定、**原文引用（字符级区间）**、置信度、是否转人工。
每条结论都能回原文逐字校验 —— 这让「可审计」从口号变成可自动打分的指标。

数据基础：[CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad)（510 份真实英文合同，13,823 条 span 标注）。

## 结果

M1 基线（模型 glm-5.3，openai 协议 / reasoning_effort=low，2026-10-29）：

| 指标 | 值 | 口径 |
|---|---|---|
| 判定准确率（宏平均） | **93.2%** | 6 类算术平均，95% CI [0.900, 0.961] |
| 判定准确率（池化） | 93.8% | 255/272，分母为明确判定条目 |
| 模糊率 | 9.0% | 27/300，分母为全部条目 |
| 引用有效率 | **96.0%** | 214/223 条引用可在原文定位 |
| 引用 IoU 宏平均 | 0.615 | 有效引用与 gold span 的平均 IoU |
| 转人工率（阈值 0.8） | 11.3% | 34/300 |
| 自动通过条目准确率 | **94.7%** | 266 条自动通过中判对的比例 |
| 单条成本 | ¥0.074 | 300 条共 ¥22.29 |
| 单条延迟 | 2.77 s | 总耗时 14m32s |

数据集：CUAD v1，dev 划分，50 份合同 × 6 类 = 300 条查询，seed=42。

数据来源：`results/m1-baseline-dev.spans-v2.jsonl`（300 条 dev 运行，已含引用定位修复）。
分类别数字与运行方式详见[实验记录](docs/experiments/2026-10-28-m1-baseline.md)。

**不需要 API key 即可复算上表**：

```bash
clauseproof report --run results/m1-baseline-dev.spans-v2.jsonl
```

读上表必须连同三个口径：

1. 「判定准确率」的分母是**明确判定**（有/无）条目，模糊与失败不计入 —— 因此必须同时看模糊率
2. 「引用有效率」指**能在原文中定位**，不等于引用质量；引用质量看 IoU
3. 以上数字来自 **dev 划分**，报告集（report）尚未使用

## 它跑起来是什么样

一条真实调用（`clauseproof ask`，9 行输出）：

```
合同   2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement
条款   Governing Law
判定   有   confidence 1.000
路由   自动通过

引用（1 处，已校验）
  [21927:22102] 12.1 GOVERNING LAW.  This Agreement will be governed and construed
  in accordance with the laws of the State of California without giving effect to
  conflict of laws principles.

输入 6818 tokens · 输出 56 tokens · 耗时 2.2s · 成本 ¥0.056
```

`[21927:22102]` 是引用在合同全文中的字符区间，可用它把结论逐字回验。
当引用无法在原文定位时，该条会被强制转人工 —— 系统不假装自己找到了证据。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.12。

```bash
make install        # 安装依赖
make check          # lint + 443 个测试，提交前必须通过
```

**看结果（不需要 API key）**

```bash
clauseproof report --run results/m1-baseline-dev.spans-v2.jsonl
```

**跑单条（需要 API key）**

```bash
bash scripts/download_data.sh                # 下载 CUAD（约 38MB）到 data/cuad/
cp .env.example .env                         # 填入 ANTHROPIC_AUTH_TOKEN
clauseproof ask --contract-id "<任意合同标题>" --clause "Governing Law"
```

`--clause` 接受六类之一：Governing Law / Cap On Liability / Uncapped Liability /
Anti-Assignment / Termination For Convenience / Notice Period To Terminate Renewal。

## 已知限制

- 仅 dev 划分，报告集尚未运行，因此没有真正的留出集结论
- 单种子、单次运行，未做重复实验；置信区间只反映抽样不确定性
- `Cap On Liability` 是最弱的一类：准确率 80.6%、引用 IoU 0.28，且占据全部模糊条目的过半（14/27）
- 引用 IoU 的标注口径为 CUAD gold span；不同标注规范下该数字会变化

## 怎么做的

13 个任务拆解，每个任务先写契约（接口 + 不变量 + 验收标准）再实现；
契约变更先于代码提交，实现者与评审者分离。
关键决策记录在 6 份 [ADR](docs/adr/README.md)，踩过的坑沉淀为 16 条[工程心得](docs/lessons.md)。

值得单说的两件事：

- **引用定位修复**：M1 首次结果只有 80.3% 的引用能精确定位。诊断发现 15.7% 是空白归一化差异
  （CUAD 原文含连续空格），根因在定位层而非模型。改为三层匹配后有效率 96.0%，
  修复零 API 成本（离线重算）。详见[实验记录](docs/experiments/2026-10-28-m1-baseline.md)。
- **被支配曲线的修复**：路由阈值的贪心算法曾产出「复核率更高但准确率更低」的解。
  定位为「零收益升档」，修复后拟合侧曲线在 0.355 复核率处达到 100% 并平台化。

## 状态

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 项目治理 | ✅ | 工具链、协作约定 |
| P1 需求 | ✅ | [PRD-0001](docs/prd/0001-contract-clause-review.md) |
| P2 设计与契约 | ✅ | [设计 0001–0004](docs/design/) · [ADR 0001–0006](docs/adr/README.md) |
| P3 测试策略 | ✅ | 并入各任务书的验收标准 |
| P4 实现 | ✅ | T-001~T-011：数据模型、模型调用、判定器、数据加载、落盘、协议切换、指标、路由、CLI、引用定位修复 |
| P5 交付 | ✅ | T-012 / T-013：结果与限制（本文件）、工程心得、实验记录、M4 曲线、路由贪心修复 |

**当前进度**：代码、实验与交付物均已完成（443 个测试通过，12 个模块）。
报告集留出验证尚未运行，配额保留。

## 文档

| 文件 | 内容 |
|---|---|
| [PROJECT.md](PROJECT.md) | 项目章程：为什么做、边界、评测哲学 |
| [PRD](docs/prd/) | 需求与验收标准 |
| [design](docs/design/) | 架构、数据模型、接口契约、错误语义 |
| [experiments](docs/experiments/README.md) | 实验记录与报告集配额 |
| [ADR](docs/adr/README.md) | 架构决策记录（含索引） |
| [lessons](docs/lessons.md) | 16 条工程心得 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 协作约定与提交规范 |
| [CLAUDE.md](CLAUDE.md) | 给 AI 实现者的约束 |

## 复现

| 产物 | 文件 | 是否需 API key |
|---|---|---|
| M1 基线（300 条 dev） | `results/m1-baseline-dev.spans-v2.jsonl` | 否 |
| 协议 A/B（18 条对照） | `results/smoke-t005.jsonl` / `smoke-t006.jsonl` | 否 |
| M4 曲线（拟合/评估分组） | `results/m1-fit.jsonl` / `m1-eval.jsonl` | 否 |