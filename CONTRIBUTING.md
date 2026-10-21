# 协作与开发约定

本项目由「人 + 两个 AI 工具」协作开发。本文约定角色、决策权与提交规范。
针对 AI 实现者的代码级约束见 [CLAUDE.md](CLAUDE.md)；需求与设计见 [docs/](docs/)。

## 角色与决策权

| 角色 | 谁 | 决定什么 | 不做什么 |
|---|---|---|---|
| Owner | Tan | 需求取舍、验收、最终决策 | — |
| Tech Lead | DSH | 架构、契约、测试策略、代码评审 | 不写实现代码 |
| Implementer | Claude Code | 按任务书实现、跑测试、报告结果 | 不改契约 |

## 两条铁律

1. **契约是唯一真相源。** 接口与数据模型的变更顺序只能是：改契约 → 改测试 → 改实现。
2. **实现者与评审者分离。** 实现由 Implementer 完成，评审由 Tech Lead 完成，两者不互换。原因是自回归模型会为自己刚生成的输出辩护，自评存在系统性盲区。

## 分支与提交

- `main` 受保护：只有 `make check` 通过才合入
- 任务分支：`feat/<slug>`、`fix/<slug>`、`docs/<slug>`
- 提交信息用 Conventional Commits：`<type>(<scope>): <描述>`，type ∈ {feat, fix, test, docs, refactor, chore}
- 一个 commit 一件事；**契约变更单独提交**

## 工作流转

```
需求（docs/prd/） → 设计契约（docs/design/） → 任务书（tasks/） → 实现 → 评审 → 提交
```

每个任务书必须含四件事：契约片段、验收标准、不变量、禁止事项。模板见 [tasks/T-000-template.md](tasks/T-000-template.md)。

## 提交前检查

- [ ] `make check` 通过
- [ ] 暂存清单里没有 `.env`、`data/`、`.venv/`
- [ ] 接口或数据模型有变更时，`docs/design/` 已同步
- [ ] 若做了评测，`docs/experiments/` 有记录