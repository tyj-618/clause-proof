# ADR-0001：技术栈与工具链选型

- 状态：Accepted（2026-10-26 补记，决策发生于 2026-10-20）
- 日期：2026-10-26
- 决策者：Owner + Tech Lead
- 关联：[pyproject.toml](../../pyproject.toml)、[Makefile](../../Makefile)、[CONTRIBUTING](../../CONTRIBUTING.md)

## 背景

本项目的交付形态是 CLI + notebook，核心资产是**可复现的评测数字**：PRD NFR-7 要求陌生人在新机器上仅凭仓库内容、通过一条安装命令与一条最小评测命令完成复现。工具链是 P0 阶段（项目治理）的决策，先于一切实现代码。

当时的候选空间：语言（Python / TypeScript）、包管理（uv / Poetry）、lint 与格式化（ruff / flake8+isort+black 组合）、测试框架、以及是否引入 agent 编排框架。

## 决策

1. **Python 3.12**。`requires-python = ">=3.12,<3.13"` 钉死版本。理由：数据处理与评测生态成熟；3.12 的 wheel 覆盖已无缺口；他人复现时行为可预期。
2. **uv**。`uv.lock` 提供可靠锁文件；`uv python install` 连解释器版本本身也纳入管理（Makefile 的 `install` 目标），新机器不需要预装 Python；所有命令经 `uv run` 执行，保证跑在锁定的环境里。
3. **ruff**。一个工具同时承担 lint 与 format（`ruff check` + `ruff format`），替代 flake8 / isort / black 三件套；配置集中在 `pyproject.toml`（E/F/I/UP/B/SIM，line-length 100），无额外配置文件。
4. **pytest**。fixture 与参数化适合按数据子集组织测试；`--strict-markers` + `slow` 标记把「需下载数据或调用真实模型」的测试隔离在默认运行之外；测试替身注入（如 `FakeJudge`）支撑无模型的端到端自检。
5. **不引入编排框架（LangGraph 等）**。本项目要展示的是**可控过程与可测模块**：设计 0003 把 `locate` / `route` / `metrics` / `calibration` 全部定义为纯函数，`runner` 只依赖 `Judge` 协议。框架会把这条流水线变成黑盒，与「契约是唯一真相源」的治理原则冲突。依赖清单（httpx / typer / rich / tqdm）里没有任何 agent 框架；PROJECT §6 也把「多 agent 角色扮演」列为非目标。

## 后果

**正面**

- 工具链四件套各司其职，`make check` 一条命令覆盖 lint + format 检查 + 测试，构成 `main` 分支的合入门禁
- 纯函数模块 + 协议注入让大部分测试不需要模型与网络

**负面**

- Python 版本钉死 `<3.13`，升级需改 `pyproject.toml` 并重新锁定依赖
- 工具链里没有类型检查器（mypy / pyright 不在 dev 依赖中）：「所有公开函数带类型注解」目前只靠约定，`make check` 抓不住注解缺失
- 不用框架意味着重试、缓存、断点续跑（FR-6）都要自己实现，工程量落在本项目头上

**风险**

- ruff 规则集偏保守，替代不了设计评审；工具链的「全绿」可能被误读为「代码正确」

## 被否决的方案

| 方案 | 否决理由 |
|---|---|
| TypeScript / Node | 生态偏移：评测与数据处理工具链在 Python 侧更成熟，CUAD 处理与 notebook 交付形态也都以 Python 为中心 |
| Poetry | 安装与解析速度慢；不管理 Python 解释器版本，复现链条缺一环，与 NFR-7「一条安装命令完成复现」冲突 |
| 直接调模型厂商官方 SDK | 本项目经兼容端点调用模型，SDK 对非官方端点的字段容忍度不一，升级可能静默破坏；用 httpx 直接发请求，请求体完全可控、可按 `prompt_hash` 落盘审计 |
