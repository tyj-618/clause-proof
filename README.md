# ClauseProof

> 合同条款审阅 Agent：在固定人工复核预算下，把审阅准确率推到最高。

给一份合同 + 一份审阅清单，逐条输出：判定、**原文引用（字符级区间）**、置信度、是否转人工。
每条结论都能回原文逐字校验 —— 这让「可审计」从口号变成可自动打分的指标。

数据基础：[CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad)（510 份真实英文合同，13,823 条 span 标注）。

## 状态

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 项目治理 | ✅ | 工具链、协作约定 |
| P1 需求 | ✅ | [PRD-0001](docs/prd/0001-contract-clause-review.md) |
| P2 设计与契约 | ✅ | [设计 0001–0004](docs/design/) · [ADR 0001–0006](docs/adr/README.md) |
| P3 测试策略 | ✅ | 并入各任务书的验收标准 |
| P4 实现 | 🚧 | T-001~T-007 完成：数据模型、模型调用、判定器、数据加载与定位、落盘与批量执行、协议切换、指标与校准；下一步路由阈值与成本曲线 |

**当前进度**：指标层已完成，可在 18 条真实数据上产出分类别指标与校准结果；下一步实现路由阈值与人工复核率曲线。

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
| [ADR](docs/adr/README.md) | 架构决策记录（含索引） |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 协作约定与提交规范 |
| [CLAUDE.md](CLAUDE.md) | 给 AI 实现者的约束 |
