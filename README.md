# ClauseProof

> 合同条款审阅 Agent：在固定人工复核预算下，把审阅准确率推到最高。

给一份合同 + 一份审阅清单，逐条输出：判定、**原文引用（字符级区间）**、置信度、是否转人工。
每条结论都能回原文逐字校验 —— 这让「可审计」从口号变成可自动打分的指标。

数据基础：[CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad)（510 份真实英文合同，13,823 条 span 标注）。

## 状态

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 项目治理 | ✅ | 工具链、协作约定、ADR |
| P1 需求 | ✅ | [PRD-0001](docs/prd/0001-contract-clause-review.md) |
| P2 设计与契约 | 🚧 | [架构](docs/design/0001-architecture.md) · [数据模型](docs/design/0002-data-model.md) · [接口](docs/design/0003-interfaces.md) · [错误语义](docs/design/0004-error-semantics.md) |
| P3 测试策略 | ⏭ | — |
| P4 实现 | ⏭ | — |

**代码尚未开始。** 契约定稿前不写实现 —— 否则契约和测试都会变成给既有代码找理由。

## 快速开始

```bash
make install   # 安装依赖（uv，Python 3.12）
make check     # lint + 测试，提交前必须通过
```

## 文档

| 文件 | 内容 |
|---|---|
| [PROJECT.md](PROJECT.md) | 项目章程：为什么做、边界、评测哲学 |
| [PRD](docs/prd/) | 需求与验收标准 |
| [design](docs/design/) | 架构、数据模型、接口契约、错误语义 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 协作约定与提交规范 |
| [CLAUDE.md](CLAUDE.md) | 给 AI 实现者的约束 |