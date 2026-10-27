"""落盘层：Record 的 JSONL 往返序列化与 RunMeta 的 JSON 读写。

契约见 docs/design/0002-data-model.md §8（序列化约定）、docs/design/0003-interfaces.md §8，
函数签名以 tasks/T-005.md 为准。枚举序列化为 .value 字符串、按值还原为枚举；
tuple → JSON 数组，None → null；confidence 保留 4 位、latency_ms 保留 1 位小数，
仅在序列化时取整，内存中保持全精度。字段顺序与 dataclass 定义序一致（便于 diff）。
append 单行写入并立即 flush，保证中断时已写内容完整可解析。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from clauseproof.models import (
    ClauseKey,
    Judgment,
    Presence,
    Record,
    Route,
    RunMeta,
    Span,
    Status,
)


def _judgment_to_json(judgment: Judgment) -> dict:
    return {
        "present": judgment.present.value,
        "quotes": list(judgment.quotes),
        "confidence": round(judgment.confidence, 4),
        "raw_output": judgment.raw_output,
        "parse_ok": judgment.parse_ok,
    }


def _span_to_json(span: Span) -> dict:
    return {
        "quote": span.quote,
        "start": span.start,
        "end": span.end,
        "valid": span.valid,
        "occurrences": span.occurrences,
    }


def _record_to_json(record: Record) -> dict:
    """字段顺序固定为 Record 定义序（0002 §8）。"""
    return {
        "run_id": record.run_id,
        "query_id": record.query_id,
        "contract_id": record.contract_id,
        "clause": record.clause.value,
        "status": record.status.value,
        "judgment": None if record.judgment is None else _judgment_to_json(record.judgment),
        "spans": [_span_to_json(span) for span in record.spans],
        "route": None if record.route is None else record.route.value,
        "model": record.model,
        "commit": record.commit,
        "prompt_hash": record.prompt_hash,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "latency_ms": round(record.latency_ms, 1),
        "cached": record.cached,
        "created_at": record.created_at,
        "error_detail": record.error_detail,
    }


def _judgment_from_json(data: object) -> Judgment:
    if not isinstance(data, dict):
        raise TypeError(f"judgment 期望 JSON 对象，实际 {type(data).__name__}")
    return Judgment(
        present=Presence(data["present"]),
        quotes=tuple(data["quotes"]),
        confidence=data["confidence"],
        raw_output=data["raw_output"],
        parse_ok=data["parse_ok"],
    )


def _span_from_json(data: object) -> Span:
    if not isinstance(data, dict):
        raise TypeError(f"spans 元素期望 JSON 对象，实际 {type(data).__name__}")
    return Span(
        quote=data["quote"],
        start=data["start"],
        end=data["end"],
        valid=data["valid"],
        occurrences=data["occurrences"],
    )


def _record_from_json(data: object) -> Record:
    if not isinstance(data, dict):
        raise TypeError(f"Record 期望 JSON 对象，实际 {type(data).__name__}")
    return Record(
        run_id=data["run_id"],
        query_id=data["query_id"],
        contract_id=data["contract_id"],
        clause=ClauseKey(data["clause"]),
        status=Status(data["status"]),
        judgment=None if data["judgment"] is None else _judgment_from_json(data["judgment"]),
        spans=tuple(_span_from_json(item) for item in data["spans"]),
        route=None if data["route"] is None else Route(data["route"]),
        model=data["model"],
        commit=data["commit"],
        prompt_hash=data["prompt_hash"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        latency_ms=data["latency_ms"],
        cached=data["cached"],
        created_at=data["created_at"],
        error_detail=data["error_detail"],
    )


def _meta_to_json(meta: RunMeta) -> dict:
    """字段顺序固定为 RunMeta 定义序（0002 §8）。"""
    return {
        "run_id": meta.run_id,
        "commit": meta.commit,
        "model": meta.model,
        "config_digest": meta.config_digest,
        "seed": meta.seed,
        "split": meta.split,
        "n_contracts": meta.n_contracts,
        "strategy": meta.strategy,
        "started_at": meta.started_at,
        "n_queries": meta.n_queries,
    }


def _meta_from_json(data: object) -> RunMeta:
    if not isinstance(data, dict):
        raise TypeError(f"RunMeta 期望 JSON 对象，实际 {type(data).__name__}")
    return RunMeta(
        run_id=data["run_id"],
        commit=data["commit"],
        model=data["model"],
        config_digest=data["config_digest"],
        seed=data["seed"],
        split=data["split"],
        n_contracts=data["n_contracts"],
        strategy=data["strategy"],
        started_at=data["started_at"],
        n_queries=data["n_queries"],
    )


def append_record(path: Path, record: Record) -> None:
    """单行追加一条 Record 并立即 flush：中断时已写内容保持完整（0003 §8）。"""
    line = json.dumps(_record_to_json(record), ensure_ascii=False, separators=(",", ":"))
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except OSError as exc:
        raise RuntimeError(f"storage: 追加记录失败，期望 可写的 {path}，实际 {exc}") from exc


def read_records(path: Path) -> Iterator[Record]:
    """逐行读取 Record。

    空文件（或仅空白行）返回空迭代器；某行不是合法 JSON 或结构不符时抛
    RuntimeError，消息含行号。文件不存在同样抛 RuntimeError（IO 失败）。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"storage: 读取记录失败，期望 可读的 {path}，实际 {exc}") from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = _record_from_json(json.loads(line))
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"storage: {path.name} 第 {lineno} 行 违反 合法 Record，"
                f"期望 一行一个合法 Record 的 JSON，实际 {exc}"
            ) from exc
        yield record


def write_meta(path: Path, meta: RunMeta) -> None:
    """把 RunMeta 写为单个 JSON 对象（<run_id>.meta.json，不混进 JSONL）。"""
    line = json.dumps(_meta_to_json(meta), ensure_ascii=False, separators=(",", ":"))
    try:
        path.write_text(line + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"storage: 写入元信息失败，期望 可写的 {path}，实际 {exc}") from exc


def read_meta(path: Path) -> RunMeta:
    """读取 write_meta 写出的 RunMeta；文件缺失、非法 JSON 或结构不符抛 RuntimeError。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"storage: 读取元信息失败，期望 可读的 {path}，实际 {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(
            f"storage: {path.name} 违反 合法 JSON，期望 单个 RunMeta 对象，实际 解析失败：{exc}"
        ) from exc
    try:
        return _meta_from_json(data)
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"storage: {path.name} 违反 合法 RunMeta，期望 RunMeta 的全部字段，实际 {exc}"
        ) from exc
