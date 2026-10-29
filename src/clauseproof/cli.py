"""命令行入口：ask（单合同演示）、eval（批量评测）、report（指标与曲线）。

契约见 tasks/T-009.md，上游为 docs/design/0003-interfaces.md §5–§9、
docs/design/0004-error-semantics.md §6/§8 与 PRD FR-8。CLI 只做参数校验、
既有模块的编排与输出格式化，不实现业务逻辑；异常映射遵循 0004 §6/§7：
ValueError → 退出码 1（输入或校验失败），RuntimeError → 退出码 2（配置缺失
或环境不可用）。typer.Exit / typer.Abort 都是 RuntimeError 的子类，必须先于
RuntimeError 放行，否则会把正常退出改判成环境错误。

ask 的路由在 route.apply_route 之上叠加任务书规定的演示级规则：任一引用未能
在原文中定位即强制「转人工」。批量评测的路由不受引用质量影响（0003 §3），
两种口径并存是刻意的：单合同演示里不可校验的结论一律交给人。
"""

from __future__ import annotations

import functools
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any

import typer

from clauseproof import dataset, metrics, route, runner, storage
from clauseproof.config import LLMConfig, load_config, prices_for
from clauseproof.judge import CallMeta, Judge, LLMJudge
from clauseproof.llm import LLMClient
from clauseproof.locate import locate
from clauseproof.models import (
    ClauseKey,
    GoldLabel,
    Judgment,
    Presence,
    Query,
    Record,
    Route,
    RunMeta,
    Span,
    Status,
)

# 任务书规定的默认值与展示上限，非可调参数
_DEFAULT_THRESHOLD = 0.8
_DEFAULT_OUT_DIR = Path("results")
_DEFAULT_REVIEW_RATES = "0.25,0.5,0.75,1.0"
_MAX_CONSECUTIVE_FAILURES = 5
_STRATEGY = "length_stratified"  # 0002 §6：抽样策略名，dataset 的长度分层抽样
_QUOTE_DISPLAY_LIMIT = 200  # 引用超过 200 字符截断并标注总长（任务书）
_MAX_QUOTE_LINES = 30  # ask 输出总行数 ≤ 40：固定结构占 8 行后的引用行上限

# 条款中文名（PRD §3.1）；--clause 大小写不敏感，接受英文原名或中文名
_CN_NAMES: dict[ClauseKey, str] = {
    ClauseKey.GOVERNING_LAW: "管辖法律",
    ClauseKey.CAP_ON_LIABILITY: "责任上限",
    ClauseKey.UNCAP_LIABILITY: "无上限责任",
    ClauseKey.ANTI_ASSIGNMENT: "转让限制",
    ClauseKey.TERMINATION_FOR_CONVENIENCE: "任意终止权",
    ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: "续约通知期",
}
_CLAUSE_ALIASES: dict[str, ClauseKey] = {c.value.casefold(): c for c in ClauseKey} | {
    name: clause for clause, name in _CN_NAMES.items()
}
_PRESENCE_LABEL: dict[Presence, str] = {
    Presence.PRESENT: "有",
    Presence.ABSENT: "没有",
    Presence.UNCERTAIN: "模糊",
}
_ROUTE_LABEL: dict[Route, str] = {Route.AUTO_PASS: "自动通过", Route.TO_HUMAN: "转人工"}
_CLAUSE_COL = max(len(clause.value) for clause in ClauseKey)  # 指标表的条款列宽

app = typer.Typer(invoke_without_command=True)


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """命令级异常映射（0004 §6/§7）：ValueError → 1，RuntimeError → 2。

    错误信息由底层模块按 0004 §8 的四要素格式给出，这里只加归类前缀；
    CLI 不新增判定逻辑。typer.Exit / typer.Abort 先放行（见模块 docstring）。
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except ValueError as exc:
            typer.echo(f"输入或校验失败（退出码 1）：{exc}", err=True)
            raise typer.Exit(code=1) from exc
        except RuntimeError as exc:
            typer.echo(f"配置或环境不可用（退出码 2）：{exc}", err=True)
            raise typer.Exit(code=2) from exc

    return wrapper


@app.callback()
def _root(ctx: typer.Context) -> None:
    """ClauseProof 合同条款审阅：引用可溯源、不确定性可量化。

    三个子命令：

    - ask：单合同演示，恰好一次模型调用
    - eval：批量评测，抽样合同 × 六类条款
    - report：指标与曲线，只读已有运行，不调用模型
    """
    if ctx.invoked_subcommand is None:
        # 任务书：无参数运行不抛异常，打印用法并以退出码 0 结束
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


# ---------- 通用辅助 ----------


def _now_utc() -> str:
    """ISO 8601 UTC 时间戳（0002 §8），与 runner 落盘格式一致。"""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    """当前 commit（git rev-parse HEAD）；取不到写 "unknown"（任务书）。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _config_digest(config: LLMConfig) -> str:
    """运行配置摘要（0002 §6「运行配置的哈希」）。

    只覆盖影响模型行为的字段；不含 api_key（0004 §8 禁止凭据出现在任何输出）
    与本地缓存目录（不影响应答语义）。
    """
    payload = "|".join(
        (
            config.protocol,
            config.base_url,
            config.model,
            str(config.max_tokens),
            config.reasoning_effort,
            repr(config.temperature),
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_judge(config: LLMConfig) -> Judge:
    """构造真实判定器；测试以 monkeypatch 替换本函数注入假 judge（任务书工程项）。"""
    return LLMJudge(LLMClient(config))


def _call_cost_cny(call: CallMeta) -> float:
    """单次调用的成本，口径同 metrics：缓存命中记 0（0003 §10）。"""
    if call.cached:
        return 0.0
    prices = prices_for(call.model)
    return (
        call.input_tokens * prices.input_cny_per_mtok
        + call.output_tokens * prices.output_cny_per_mtok
    ) / 1e6


def _parse_clause(raw: str) -> ClauseKey:
    """--clause 解析：大小写不敏感，接受英文原名或中文名（PRD §3.1）。"""
    key = _CLAUSE_ALIASES.get(raw.strip().casefold())
    if key is None:
        legal = "、".join(f"{clause.value}（{_CN_NAMES[clause]}）" for clause in ClauseKey)
        raise ValueError(f"cli: --clause 违反 合法取值，期望 {legal}，实际 {raw!r}")
    return key


def _parse_review_rates(raw: str) -> tuple[float, ...]:
    """--review-rates 解析：逗号分隔的小数列表；取值范围由 route.select_thresholds 校验。"""
    try:
        rates = tuple(float(item) for item in raw.split(",") if item.strip())
    except ValueError:
        raise ValueError(
            f"cli: --review-rates 违反 逗号分隔的小数列表，期望 如 0.25,0.5,1.0，实际 {raw!r}"
        ) from None
    if not rates:
        raise ValueError(f"cli: --review-rates 违反 非空，期望 至少一个小数，实际 {raw!r}")
    return rates


def _fmt_opt(value: float | None) -> str:
    """可选浮点的展示：None 记 "-"（样本不足时不硬编数字）。

    比率统一保留 2 位小数（FR-7，仅显示层；内部计算与 bootstrap 不取整）。
    """
    return "-" if value is None else f"{value:.2f}"


def _fmt_duration(seconds: float) -> str:
    """耗时展示：不足一分钟保留一位小数，超过则 2m14s 样式（任务书示例）。"""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60)}s"


def _meta_path(run_path: Path) -> Path:
    """同名 meta 文件：foo.jsonl → foo.meta.json（0003 §8 的 <run_id>.meta.json）。"""
    return run_path.with_name(run_path.stem + ".meta.json")


# ---------- ask：单合同演示 ----------


@app.command()
@_guard
def ask(
    clause: Annotated[str, typer.Option(help="条款类别：六类之一的英文原名或中文名，大小写不敏感")],
    contract_id: Annotated[str | None, typer.Option(help="CUAD 合同标题，与 --file 二选一")] = None,
    file: Annotated[
        Path | None, typer.Option(help="合同文本文件路径，与 --contract-id 二选一")
    ] = None,
) -> None:
    """单合同演示：对一份合同的一类条款做恰好一次判定。

    输出判定、引用（原文区间与文本）、置信度与路由，总行数不超过 40。

    示例：

        clauseproof ask --contract-id "<CUAD 合同标题>" --clause "Cap On Liability"

        clauseproof ask --file path/to/contract.txt --clause "Governing Law"

        clauseproof ask --file contract.txt --clause 责任上限
    """
    if (contract_id is None) == (file is None):
        raise ValueError(
            "cli: --contract-id/--file 违反 二选一，期望 恰好提供一个，实际 "
            + ("都未提供" if contract_id is None else "都提供了")
        )
    parsed_clause = _parse_clause(clause)
    contract_source, contract_text = _load_contract_text(contract_id, file)
    query = Query(
        query_id=f"{contract_source}::{parsed_clause.name}",
        contract_id=contract_source,
        clause=parsed_clause,
        contract_text=contract_text,
    )
    config = load_config()
    judgment, call = _build_judge(config).judge(query)
    spans = locate(judgment.quotes, query.contract_text)
    forced_human = any(not span.valid for span in spans)
    route_result = (
        Route.TO_HUMAN if forced_human else route.apply_route(judgment, _DEFAULT_THRESHOLD)
    )
    _print_ask(query, judgment, spans, route_result, forced_human, call)


def _load_contract_text(contract_id: str | None, file: Path | None) -> tuple[str, str]:
    """加载合同全文，返回 (合同标识, 全文)。

    --contract-id 时标识为 CUAD 标题；--file 时标识为所给路径（作为 query_id 的
    contract_id 部分，含 "::" 时由 Query 的不变式校验拒绝）。
    """
    if contract_id is not None:
        for title, text in dataset.load_contracts():
            if title == contract_id:
                return title, text
        raise ValueError(
            "cli: --contract-id 违反 必须是 CUAD 中已有的合同标题，"
            f"期望 既有标题之一，实际 {contract_id!r}"
        )
    if file is None:  # 互斥已在命令入口校验，此处仅为穷尽分支
        raise ValueError("cli: --contract-id/--file 违反 二选一，期望 恰好提供一个，实际 都未提供")
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cli: --file 违反 可读，期望 存在且可读的文本文件，实际 {file}（{exc}）"
        ) from exc
    if not text.strip():
        raise ValueError(f"cli: --file 违反 非空，期望 非空文本文件，实际 {file}")
    return str(file), text


def _print_ask(
    query: Query,
    judgment: Judgment,
    spans: Sequence[Span],
    route_result: Route,
    forced_human: bool,
    call: CallMeta,
) -> None:
    """按任务书参考格式输出；行数 = 固定 8 行 + 引用行（至多 _MAX_QUOTE_LINES + 1）。"""
    lines: list[str] = [
        f"合同   {query.contract_id}",
        f"条款   {query.clause.value}",
    ]
    verdict = _PRESENCE_LABEL[judgment.present]
    if not judgment.parse_ok:
        verdict += "（模型输出解析失败）"
    lines.append(f"判定   {verdict}   confidence {judgment.confidence:.3f}")
    route_label = _ROUTE_LABEL[route_result]
    if forced_human and judgment.present is not Presence.UNCERTAIN:
        # UNCERTAIN 本就转人工，无需标注；这里只标注「因引用不可校验而被强制」
        route_label += "（引用定位失败，已强制）"
    lines.append(f"路由   {route_label}")
    lines.append("")
    if not spans:
        lines.append("引用（无）")
    else:
        n_valid = sum(1 for span in spans if span.valid)
        if n_valid == len(spans):
            detail = "已校验"
        elif n_valid == 0:
            detail = "定位失败"
        else:
            detail = f"已校验 {n_valid}，定位失败 {len(spans) - n_valid}"
        lines.append(f"引用（{len(spans)} 处，{detail}）")
        for span in spans[:_MAX_QUOTE_LINES]:
            if span.valid:
                quote = span.quote
                if len(quote) > _QUOTE_DISPLAY_LIMIT:
                    quote = quote[:_QUOTE_DISPLAY_LIMIT] + f"…（共 {len(quote)} 字符）"
                lines.append(f"  [{span.start}:{span.end}] {quote}")
            else:
                lines.append("  引用无效：该片段未在原文中找到")
        if len(spans) > _MAX_QUOTE_LINES:
            lines.append(f"  ……另有 {len(spans) - _MAX_QUOTE_LINES} 处引用未显示")
    lines.append("")
    cached_note = " · 缓存命中" if call.cached else ""
    lines.append(
        f"输入 {call.input_tokens} tokens · 输出 {call.output_tokens} tokens · "
        f"耗时 {call.latency_ms / 1000:.1f}s · 成本 ¥{_call_cost_cny(call):.3f}{cached_note}"
    )
    typer.echo("\n".join(lines))


# ---------- eval：批量评测 ----------


@app.command("eval")
@_guard
def eval_run(
    split: Annotated[str, typer.Option(help="数据划分：dev 或 report")],
    n_contracts: Annotated[int, typer.Option(help="抽样的合同数")],
    seed: Annotated[int, typer.Option(help="抽样种子，同参数产出同一集合")],
    run_id: Annotated[str, typer.Option(help="运行标识，用作输出文件名前缀")],
    out_dir: Annotated[Path, typer.Option(help="输出目录")] = _DEFAULT_OUT_DIR,
    threshold: Annotated[float, typer.Option(help="路由置信度阈值，六类同值")] = _DEFAULT_THRESHOLD,
    max_consecutive_failures: Annotated[
        int, typer.Option(help="连续失败熔断阈值")
    ] = _MAX_CONSECUTIVE_FAILURES,
) -> None:
    """批量评测：抽样合同 × 六类条款，逐条判定并落盘。

    结束后打印运行摘要（条数、OK/失败/模糊、耗时、成本、平均单条耗时）。
    同参数重复执行会跳过已完成的 query_id（断点续跑，不重复计费）。

    示例：

        clauseproof eval --split dev --n-contracts 3 --seed 42 --run-id my-run

        clauseproof eval --split dev --n-contracts 3 --seed 42 --run-id my-run --threshold 0.9
    """
    if not run_id.strip():
        raise ValueError(f"cli: --run-id 违反 非空，期望 非空字符串，实际 {run_id!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"cli: --threshold 违反 0.0<=threshold<=1.0，期望 [0.0, 1.0]，实际 {threshold!r}"
        )
    if max_consecutive_failures < 1:
        raise ValueError(
            f"cli: --max-consecutive-failures 违反 >=1，期望 >=1 的整数，"
            f"实际 {max_consecutive_failures}"
        )
    queries = dataset.build_queries(split=split, n_contracts=n_contracts, seed=seed)
    config = load_config()
    meta = RunMeta(
        run_id=run_id,
        commit=_git_commit(),
        model=config.model,
        config_digest=_config_digest(config),
        seed=seed,
        split=split,
        n_contracts=n_contracts,
        strategy=_STRATEGY,
        started_at=_now_utc(),
        n_queries=len(queries),
    )
    started = time.monotonic()
    jsonl_path = runner.run(
        queries=queries,
        judge=_build_judge(config),
        run_meta=meta,
        thresholds=dict.fromkeys(ClauseKey, threshold),
        out_dir=out_dir,
        max_consecutive_failures=max_consecutive_failures,
    )
    elapsed = time.monotonic() - started
    records = list(storage.read_records(jsonl_path))
    gold = dataset.load_gold(split=split, n_contracts=n_contracts, seed=seed)
    rep = metrics.compute(records, gold, seed=seed, split=split)
    _print_eval_summary(meta, records, rep, elapsed, jsonl_path)


def _print_eval_summary(
    meta: RunMeta,
    records: Sequence[Record],
    rep: metrics.Report,
    elapsed: float,
    jsonl_path: Path,
) -> None:
    """运行摘要（任务书参考格式）。成本复用 metrics.compute，与 report 的口径一致。"""
    n_total = len(records)
    n_ok = sum(1 for record in records if record.status is Status.OK)
    n_failed = n_total - n_ok
    n_uncertain = sum(
        1
        for record in records
        if record.status is Status.OK and record.judgment.present is Presence.UNCERTAIN
    )
    n_parse_fail = sum(
        1 for record in records if record.status is Status.OK and not record.judgment.parse_ok
    )
    parse_note = f"（含解析失败 {n_parse_fail}）" if n_parse_fail else ""
    ok_latencies = [record.latency_ms for record in records if record.status is Status.OK]
    mean_s = sum(ok_latencies) / len(ok_latencies) / 1000 if ok_latencies else None
    average = f"{mean_s:.1f}s/条" if mean_s is not None else "-"
    typer.echo(f"运行 {meta.run_id} 完成")
    typer.echo(f"  {n_total} 条 · OK {n_ok} · 失败 {n_failed} · 模糊 {n_uncertain}{parse_note}")
    typer.echo(f"  耗时 {_fmt_duration(elapsed)} · 成本 ¥{rep.total_cost_cny:.3f} · 平均 {average}")
    typer.echo(f"  JSONL  {jsonl_path}")
    typer.echo(f"  meta   {jsonl_path.parent / (meta.run_id + '.meta.json')}")


# ---------- report：指标与曲线 ----------


@app.command()
@_guard
def report(
    run: Annotated[Path, typer.Option(help="要评测的运行 JSONL 路径")],
    thresholds_from: Annotated[
        Path | None,
        typer.Option(help="阈值来源运行的 JSONL，--curve 时必填"),
    ] = None,
    curve: Annotated[bool, typer.Option(help="输出复核率–准确率曲线")] = False,
    review_rates: Annotated[
        str | None,
        typer.Option(help="逗号分隔的目标复核率列表，默认 0.25,0.5,0.75,1.0"),
    ] = None,
) -> None:
    """报告：从运行 JSONL 与同名 meta 重算指标，可选输出复核率–准确率曲线。

    默认输出分类别指标表（n_total / n_decided / n_correct / accuracy /
    引用准确率 / ECE / M6 用量四列：平均输入输出 token、单条成本、p95 延迟）
    与宏平均、模糊率、全局 ECE，不调用模型；所有比率在显示层保留 2 位小数
    （FR-7）。--curve 的阈值必须来自另一个运行（--thresholds-from）：阈值只应
    在开发集上选择，在同一批数据上既拟合又评估会让结果虚高（0003 §3）。

    示例：

        clauseproof report --run results/my-run.jsonl

        clauseproof report --run results/eval-run.jsonl --thresholds-from results/dev.jsonl --curve

        clauseproof report --run results/eval-run.jsonl \\
            --thresholds-from results/dev.jsonl --curve --review-rates 0.5,1.0
    """
    if not run.is_file():
        raise ValueError(f"cli: --run 违反 存在，期望 既有 JSONL 文件，实际 {run}")
    meta_path = _meta_path(run)
    if not meta_path.is_file():
        raise ValueError(f"cli: --run 违反 同名 meta 存在，期望 {meta_path}，实际 缺失")
    if curve and thresholds_from is None:
        raise ValueError(
            "cli: --curve 违反 必须提供 --thresholds-from，期望 另一个运行的 JSONL，"
            "实际 未提供。阈值必须在开发集上选择：在同一批数据上既拟合又评估，"
            "结果会虚高（0003 §3）"
        )
    if not curve and thresholds_from is not None:
        raise ValueError(
            "cli: --thresholds-from 违反 仅与 --curve 同用，期望 指定 --curve，实际 未指定"
        )
    if not curve and review_rates is not None:
        raise ValueError(
            "cli: --review-rates 违反 仅与 --curve 同用，期望 指定 --curve，实际 未指定"
        )
    meta = storage.read_meta(meta_path)
    records = list(storage.read_records(run))
    gold = dataset.load_gold(split=meta.split, n_contracts=meta.n_contracts, seed=meta.seed)
    rep = metrics.compute(records, gold, seed=meta.seed, split=meta.split)
    _print_report(meta, rep)
    if curve and thresholds_from is not None:
        raw_rates = _DEFAULT_REVIEW_RATES if review_rates is None else review_rates
        rates = _parse_review_rates(raw_rates)
        source_records, source_gold = _load_threshold_source(thresholds_from)
        points = _curve_points(source_records, source_gold, records, gold, rates)
        _print_curve(run, thresholds_from, points)


def _print_report(meta: RunMeta, rep: metrics.Report) -> None:
    """指标表：分类别判定/引用/ECE 列 + M6 用量四列 + 宏平均 + 模糊率 + 全局 ECE。

    所有比率在显示层统一保留 2 位小数（FR-7）；M6 用量列非比率：token 与
    p95 为整数、成本保留 3 位（同运行摘要口径）。
    """
    typer.echo(
        f"运行 {meta.run_id} · split {meta.split} · {meta.n_contracts} contracts · seed {meta.seed}"
    )
    typer.echo(f"总计 {rep.n_total} 条 · OK {rep.n_ok} · 失败 {rep.n_failed}")
    typer.echo(
        f"{'条款':<{_CLAUSE_COL}}  {'n_total':>7} {'n_decided':>9} {'n_correct':>9}"
        f" {'accuracy':>8} {'引用准确率':>8} {'ECE':>5}"
        f" {'in_tok':>7} {'out_tok':>8} {'cost¥':>7} {'p95_ms':>7}"
    )
    for clause_metrics in rep.by_clause:
        typer.echo(
            f"{clause_metrics.clause.value:<{_CLAUSE_COL}}  "
            f"{clause_metrics.n_total:>7} {clause_metrics.n_decided:>9}"
            f" {clause_metrics.n_correct:>9} {_fmt_opt(clause_metrics.accuracy):>8}"
            f" {_fmt_opt(clause_metrics.citation_iou):>8} {_fmt_opt(clause_metrics.ece):>5}"
            f" {clause_metrics.mean_input_tokens:>7.0f}"
            f" {clause_metrics.mean_output_tokens:>8.0f}"
            f" {clause_metrics.mean_cost_cny:>7.3f}"
            f" {clause_metrics.p95_latency_ms:>7.0f}"
        )
    typer.echo(f"宏平均 accuracy {_fmt_opt(rep.macro_accuracy)}")
    typer.echo(
        f"模糊率 {round(rep.uncertain_rate, 2) * 100:.1f}%（{rep.n_uncertain}/{rep.n_total}）"
    )
    typer.echo(f"全局 ECE {_fmt_opt(rep.ece)}")


def _load_threshold_source(path: Path) -> tuple[list[Record], dict[str, GoldLabel]]:
    """读取阈值来源运行：记录 + 按其自身 meta 重建的 gold。

    route.select_thresholds 直接以 gold[query_id] 取标注，来源记录必须全部
    在其 gold 中；缺失在此前置校验为 ValueError（退出码 1），避免 KeyError
    以堆栈形式裸奔。
    """
    if not path.is_file():
        raise ValueError(f"cli: --thresholds-from 违反 存在，期望 既有 JSONL 文件，实际 {path}")
    source_meta_path = _meta_path(path)
    if not source_meta_path.is_file():
        raise ValueError(
            f"cli: --thresholds-from 违反 同名 meta 存在，期望 {source_meta_path}，实际 缺失"
        )
    source_meta = storage.read_meta(source_meta_path)
    source_records = list(storage.read_records(path))
    source_gold = dataset.load_gold(
        split=source_meta.split, n_contracts=source_meta.n_contracts, seed=source_meta.seed
    )
    missing = [record.query_id for record in source_records if record.query_id not in source_gold]
    if missing:
        raise ValueError(
            "cli: --thresholds-from 违反 记录与 gold 对应，"
            f"期望 全部 query_id 在来源 gold 中，实际 缺失 {missing[0]}"
        )
    return source_records, source_gold


def _curve_points(
    source_records: Sequence[Record],
    source_gold: Mapping[str, GoldLabel],
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    rates: Sequence[float],
) -> tuple[route.RoutingPoint, ...]:
    """曲线点：阈值取自来源运行（route.sweep → select_thresholds），统计在 --run 数据上重算。

    拟合与评估分属两批数据是任务书的核心约束：阈值看着来源数据的 gold 拟合，
    应用到 --run 这批记录上统计。
    """
    selected = route.sweep(source_records, source_gold, review_rates=rates)
    return tuple(
        _point_on(records, gold, point.thresholds, point.target_review_rate) for point in selected
    )


def _point_on(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    thresholds: Mapping[ClauseKey, float],
    target_review_rate: float,
) -> route.RoutingPoint:
    """给定阈值在 records 上重算曲线点（统计口径同 route.sweep）。

    route._routing_stats 是模块私有实现，这里基于公开的 route.apply_route
    重算；「判定正确」的口径与 metrics / route 一致（0003 §4），按 T-008 的
    先例以复制纯函数的方式实现，不触私有符号。
    """
    n_total = len(records)
    n_human = n_auto = n_auto_wrong = 0
    for record in records:
        if (
            record.status is Status.OK
            and route.apply_route(record.judgment, thresholds[record.clause]) is Route.AUTO_PASS
        ):
            n_auto += 1
            if gold[record.query_id].has_answer != (record.judgment.present is Presence.PRESENT):
                n_auto_wrong += 1
        else:
            n_human += 1
    return route.RoutingPoint(
        target_review_rate=target_review_rate,
        actual_review_rate=n_human / n_total if n_total else 0.0,
        auto_pass_accuracy=(n_auto - n_auto_wrong) / n_auto if n_auto else None,
        post_review_accuracy=(1.0 - n_auto_wrong / n_total if n_total else 1.0),
        thresholds=dict(thresholds),
    )


def _print_curve(run_path: Path, source_path: Path, points: Sequence[route.RoutingPoint]) -> None:
    """曲线表：每行 = 目标复核率、实际复核率、自动通过准确率、复核后准确率。

    比率在显示层统一保留 2 位小数（FR-7）。
    """
    typer.echo(f"曲线：阈值拟合自 {source_path.stem}，评估于 {run_path.stem}")
    typer.echo(f"{'目标复核率':>8} {'实际复核率':>8} {'自动通过准确率':>8} {'复核后准确率':>8}")
    for point in points:
        typer.echo(
            f"{point.target_review_rate:>8.2f} {point.actual_review_rate:>8.2f}"
            f" {_fmt_opt(point.auto_pass_accuracy):>8} {point.post_review_accuracy:>8.2f}"
        )
