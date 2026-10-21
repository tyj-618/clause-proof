# 项目约定（给实现者的约束）

## 上下文
合同条款审阅 Agent。需求见 docs/prd/，设计与契约见 docs/design/。
当前阶段见 README 的状态表。

## 必须做
- 动手前先读 tasks/ 下对应的任务书；任务书没写的事不要做。
- 完成前必须自己跑 `make check`，并如实报告输出。
- 只改任务书「允许改动的文件」范围内的文件。

## 绝对禁止
- 修改 docs/design/ 下的契约（有异议就停下来报告）
- 修改 .env、.gitignore、Makefile（需 Owner 同意）
- 下载数据集或任何大文件
- 提交 .env、密钥、data/

## 风格
- Python 3.12，带类型注解，函数单一职责
- 不硬编码路径、模型名、阈值

## 提交（仅在 Owner 明确要求时执行）
不要自行 commit 或 push。提交格式见 CONTRIBUTING.md「分支与提交」。
