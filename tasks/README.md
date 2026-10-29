| 编号 | 任务 | 状态 | 依赖 | 主要产出 |
|---|---|---|---|---|
| T-001 | 数据模型与不变式 | ✅ 完成 | — | `models.py` |
| T-002 | 模型调用层 | ✅ 完成 | — | `llm.py` `config.py` |
| T-003 | 判定器实现 | ✅ 完成 | T-001, T-002 | `judge.py` |
| T-004 | 数据加载与引用定位 | ✅ 完成 | T-001 | `dataset.py` `locate.py` |
| T-005 | 落盘与批量执行 | ✅ 完成 | T-002, T-004 | `storage.py` `runner.py` |
| T-006 | 协议抽象与端点切换 | ✅ 完成 | T-002, T-005 | `llm.Backend` × 2 |
| T-007 | 指标与校准 | ✅ 完成 | T-005 | `metrics.py` `calibration.py` |
| T-008 | 路由阈值与成本曲线 | ✅ 完成 | T-005, T-007 | `route.select_thresholds` `sweep` |
| T-009 | CLI 与运行摘要 | ✅ 完成 | T-005, T-007, T-008 | `cli.py` |
| T-010 | M1 基线实验 | ⏭ 下一步 | T-009 | `results/` `docs/experiments/` |
| T-011 | 引用定位修复 | ✅ 完成 | T-010 | `locate.py` `scripts/recompute_spans.py` |
| T-012 | 交付物整理 | ✅ 完成 | T-011 | `README.md` `docs/lessons.md` |
| T-013 | 修复路由贪心的零收益提升 | ✅ 完成 | T-008 | `route.select_thresholds` |
| T-014 | 补齐契约缺口 | ✅ 完成 | PRD FR-2、FR-3、FR-7、M6 | `runner.py` 降级、`metrics.py` M6 列、spans-v3 |
| T-015 | 契约同步与 M4 重派生 | ✅ 完成 | T-014 | `0003` v1.1、`m1-fit` / `m1-eval`（v3 派生） |