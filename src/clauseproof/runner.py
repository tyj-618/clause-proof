"""批量编排：逐条调用 judge → 定位引用 → FR-2 降级 → 路由决策 → 落盘，支持续跑与连续失败熔断。

契约见 docs/design/0003-interfaces.md §6、docs/design/0004-error-semantics.md §2 与 §5，
函数签名以 tasks/T-005.md 为准（单阈值 threshold；select_thresholds 留待 T-006）。

CONTEXT_OVERFLOW 的判定不在本任务范围：当前端点上下文 1M，最长合同约 106k token，
实际不会触发；该状态保留在枚举中，等出现可识别的溢出信号再接入，调用失败先统一
记为 LLM_ERROR。本模块不做重试（属 llm 层）、不做指标计算（属 metrics 层）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from clauseproof.judge import CallMeta, Judge
from clauseproof.locate import locate
from clauseproof.models import (
    ClauseKey,
    Judgment,
    Presence,
    Query,
    Record,
    RunMeta,
    Span,
    Status,
)
from clauseproof.route import apply_route
from clauseproof.storage import append_record, read_records, write_meta

# judge 抛错时未发生成功调用，实际 prompt 的哈希不可得；与 0002 §8 的 commit="unknown" 先例一致
_UNKNOWN_PROMPT_HASH = "unknown"


def _now_utc() -> str:
    """ISO 8601 UTC 时间戳，形如 2026-10-27T12:00:00Z（0002 §8）。"""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _failed_record(run_meta: RunMeta, query: Query, error_detail: str) -> Record:
    """LLM_ERROR 记录：无半成品结果（0002 §4），计量字段为零。"""
    return Record(
        run_id=run_meta.run_id,
        query_id=query.query_id,
        contract_id=query.contract_id,
        clause=query.clause,
        status=Status.LLM_ERROR,
        judgment=None,
        spans=(),
        route=None,
        model=run_meta.model,
        commit=run_meta.commit,
        prompt_hash=_UNKNOWN_PROMPT_HASH,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        cached=False,
        created_at=_now_utc(),
        error_detail=error_detail,
    )


def downgrade_unverifiable(judgment: Judgment, spans: Sequence[Span]) -> Judgment:
    """FR-2 降级：判定为「有」但全部引用定位失败时改为「模糊」。

    quotes 原样保留（失效率统计需要；Judgment 允许 UNCERTAIN 携带 quotes），
    只改 present。降级后的判定经 apply_route 必然转人工（UNCERTAIN 恒转人工）。
    PRESENT 蕴含 quotes 非空（0002 §7），故 spans 非空；非 PRESENT 或存在任一
    有效引用时原样返回。
    """
    if judgment.present is Presence.PRESENT and spans and all(not span.valid for span in spans):
        return replace(judgment, present=Presence.UNCERTAIN)
    return judgment


def _ok_record(
    run_meta: RunMeta,
    query: Query,
    judgment: Judgment,
    call: CallMeta,
    threshold: float,
) -> Record:
    """OK 记录：judgment / spans / route 齐全，计量字段来自 CallMeta。

    写入前先做 FR-2 降级（present==PRESENT 且全部 span 无效 → UNCERTAIN），
    路由在降级后的判定上计算。
    """
    spans = locate(judgment.quotes, query.contract_text)
    judgment = downgrade_unverifiable(judgment, spans)
    return Record(
        run_id=run_meta.run_id,
        query_id=query.query_id,
        contract_id=query.contract_id,
        clause=query.clause,
        status=Status.OK,
        judgment=judgment,
        spans=spans,
        route=apply_route(judgment, threshold),
        model=call.model,
        commit=run_meta.commit,
        prompt_hash=call.prompt_hash,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        latency_ms=call.latency_ms,
        cached=call.cached,
        created_at=_now_utc(),
        error_detail=None,
    )


def run(
    *,
    queries: Sequence[Query],
    judge: Judge,
    run_meta: RunMeta,
    thresholds: Mapping[ClauseKey, float],
    out_dir: Path,
    resume: bool = True,
    max_consecutive_failures: int = 5,
) -> Path:
    """批量执行并落盘，返回 JSONL 路径。

    - thresholds：分条款类别的置信度阈值。键必须覆盖全部 ClauseKey（缺失在启动前
      抛 ValueError，不产生任何记录）；值的取值范围 [0.0, 1.0] 由 route.apply_route
      校验，本函数不重复校验
    - 产出 out_dir/{run_id}.jsonl 与 out_dir/{run_id}.meta.json
    - resume=True：跳过文件中已存在的 query_id，不重复调用 judge
    - resume=False：清空同名 JSONL 重新开始
    - 单条 judge 抛 RuntimeError → 该条记 LLM_ERROR 后继续（0004 §2 单条层）
    - 连续失败达 max_consecutive_failures → 该条已写盘后抛 RuntimeError，
      已写入的记录完整可读，resume=True 可续跑（0004 §5 熔断）
    """
    missing = [clause.name for clause in ClauseKey if clause not in thresholds]
    if missing:
        raise ValueError(
            "runner: thresholds 违反 必须覆盖全部条款类别，"
            f"期望 6 个键，实际 缺失 {', '.join(missing)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_meta.run_id}.jsonl"
    meta_path = out_dir / f"{run_meta.run_id}.meta.json"
    if resume and jsonl_path.exists():
        existing = list(read_records(jsonl_path))
    else:
        if not resume:
            jsonl_path.unlink(missing_ok=True)
        existing = []
    done_ids = {record.query_id for record in existing}
    write_meta(meta_path, run_meta)

    consecutive_failures = 0
    written = 0
    for query in queries:
        if query.query_id in done_ids:
            continue
        try:
            judgment, call = judge.judge(query)
        except RuntimeError as exc:
            append_record(jsonl_path, _failed_record(run_meta, query, str(exc)))
            written += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"runner: 连续失败 {consecutive_failures} 条，达到熔断阈值"
                    f" max_consecutive_failures={max_consecutive_failures}，已停止；"
                    f"文件中已有 {len(existing) + written} 条记录（{jsonl_path}），"
                    "resume=True 可从断点续跑"
                ) from exc
            continue
        append_record(
            jsonl_path, _ok_record(run_meta, query, judgment, call, thresholds[query.clause])
        )
        written += 1
        consecutive_failures = 0
    return jsonl_path
